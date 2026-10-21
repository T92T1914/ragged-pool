"""A bounded, detached CG example for a fixed symmetric linear HVP operator.

This is an example-local diagnostic, not a package solver or an optimizer.
Positive damping does not prove that an arbitrary Hessian is positive definite.
"""
import math

import torch


def _dot(left, right):
    return float((left.to(torch.float64) * right.to(torch.float64)).sum())


def _norm(value):
    # Squaring first can erase tiny nonzero residuals or overflow before the
    # square root. The bounded vector has at most 32 CPU entries.
    return math.hypot(*value.detach().to(torch.float64).tolist())


class _ProductFailure(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def damped_cg(hvp, rhs, *, damping, max_iterations, rtol, atol):
    """Approximately solve (H+damping*I)x=rhs from zero, without updating a model.

    The caller supplies one fixed linear symmetric operator. CG's usual
    convergence argument additionally needs positive definiteness. Checks on
    encountered directions cannot certify either property for all directions.
    Every accepted solution has a fresh true-residual check, including zero RHS.
    There are at most max_iterations+1 callback attempts. A refused iterate is
    diagnostic partial work, never an accepted update.
    """
    if (not isinstance(rhs, torch.Tensor) or rhs.device.type != "cpu"
            or rhs.layout != torch.strided or rhs.dtype not in (torch.float32, torch.float64)
            or rhs.dim() != 1 or not 1 <= rhs.numel() <= 32):
        raise ValueError("rhs must be a CPU float32/float64 vector with 1..32 entries")
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("rhs must be finite")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 32:
        raise ValueError("max_iterations must be an integer in [1,32]")
    for name, number in (("damping", damping), ("rtol", rtol), ("atol", atol)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{name} must be a finite real scalar")
    if damping < 0 or not 0 < rtol < 1 or atol <= 0:
        raise ValueError("Require damping >=0, 0<rtol<1 and atol>0")
    if not callable(hvp):
        raise TypeError("hvp must be callable")
    b = rhs.detach().clone().contiguous()
    x, r = torch.zeros_like(b), b.clone()
    p = r.clone()
    norm_b = _norm(b)
    threshold = float(atol + rtol * norm_b)
    if not math.isfinite(threshold):
        raise ValueError("The residual threshold must be finite")
    report = {"status": None, "iterations": 0, "products": 0,
              "max_products": max_iterations + 1, "max_iterations": max_iterations,
              "damping": float(damping), "rtol": float(rtol), "atol": float(atol),
              "rhs_norm": norm_b, "threshold": threshold,
              "recursive_residual_norm": norm_b, "true_residual_norm": None,
              "history": [], "solution_applied": False}

    def product(direction):
        assert report["products"] < report["max_products"]
        report["products"] += 1
        owned = direction.detach().clone().contiguous()
        before, version = owned.clone(), owned._version
        try:
            result = hvp(owned)
        except RuntimeError as error:
            raise _ProductFailure("operator_error", str(error)) from error
        if owned._version != version or not torch.equal(owned, before):
            raise _ProductFailure("operator_error", "HVP callback mutated its direction")
        if (not isinstance(result, torch.Tensor) or result.device != b.device
                or result.layout != torch.strided or result.dtype != b.dtype or result.shape != b.shape):
            raise _ProductFailure("operator_error", "HVP callback returned invalid metadata")
        result = result.detach().clone().contiguous()
        if not bool(torch.isfinite(result).all()):
            raise _ProductFailure("nonfinite", "HVP callback returned nonfinite entries")
        result = (result + damping * direction).detach()
        if not bool(torch.isfinite(result).all()):
            raise _ProductFailure("nonfinite", "Damped operator returned nonfinite entries")
        return result

    def finish(status, *, check_residual=False):
        if check_residual:
            residual = b - product(x)
            if not bool(torch.isfinite(residual).all()):
                raise _ProductFailure("nonfinite", "True residual has nonfinite entries")
            norm_residual = _norm(residual)
            if not math.isfinite(norm_residual):
                raise _ProductFailure("nonfinite", "True residual norm is nonfinite")
            report["true_residual_norm"] = norm_residual
            if status == "converged" and report["true_residual_norm"] > threshold:
                status = "residual_mismatch"
        report["status"] = status
        return x.detach().clone(), report

    try:
        if norm_b <= threshold:
            return finish("converged", check_residual=True)
        rr = _dot(r, r)
        if not math.isfinite(rr):
            return finish("nonfinite")
        if rr <= 0:
            return finish("curvature_breakdown")
        for iteration in range(1, max_iterations + 1):
            ap = product(p)
            curvature = _dot(p, ap)
            scale = _norm(p) * _norm(ap)
            if not all(math.isfinite(value) for value in (curvature, scale, rr)):
                return finish("nonfinite")
            row = {"iteration": iteration, "p_dot_Ap": curvature,
                   "curvature_scale": scale,
                   "curvature_floor": 32 * torch.finfo(b.dtype).eps * scale}
            report["history"].append(row)
            if curvature <= 0:
                return finish("nonpositive_curvature")
            if curvature <= row["curvature_floor"]:
                return finish("curvature_breakdown")
            alpha = rr / curvature
            if not math.isfinite(alpha):
                return finish("nonfinite")
            next_x, next_r = (x + alpha * p).detach(), (r - alpha * ap).detach()
            if not bool(torch.isfinite(next_x).all() & torch.isfinite(next_r).all()):
                return finish("nonfinite")
            x, r = next_x, next_r
            next_rr = _dot(r, r)
            if not math.isfinite(next_rr):
                return finish("nonfinite")
            norm_r = _norm(r)
            report["iterations"] = iteration
            report["recursive_residual_norm"] = norm_r
            row["alpha"], row["recursive_residual_norm"] = alpha, norm_r
            if norm_r <= threshold:
                return finish("converged", check_residual=True)
            if iteration == max_iterations:
                return finish("iteration_limit", check_residual=True)
            if next_rr <= 0:
                return finish("curvature_breakdown")
            beta = next_rr / rr
            if not math.isfinite(beta):
                return finish("nonfinite")
            p, rr = (r + beta * p).detach(), next_rr
            if not bool(torch.isfinite(p).all()):
                return finish("nonfinite")
    except _ProductFailure as error:
        report["message"] = error.message
        return finish(error.status)
    raise AssertionError("The bounded iteration loop must terminate")
