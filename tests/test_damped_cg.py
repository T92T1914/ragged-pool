"""Independent linear-system math and failure controls for the local example."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_solve_example_tests", ROOT / "examples/solve_curvature.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
cg = example.cg_example.damped_cg
DTYPES = (torch.float32, torch.float64)


def solve(operator, rhs, **options):
    settings = dict(damping=0, max_iterations=8, rtol=1e-5 if rhs.dtype == torch.float32 else 1e-10, atol=1e-12)
    settings.update(options)
    return cg(operator, rhs, **settings)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("expanded", (False, True))
def test_analytic_damped_spd_and_owned_detached_result(dtype, expanded):
    h = torch.tensor([[2., 1.], [1., 1.]], dtype=dtype)
    rhs = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
    if expanded:
        rhs = torch.tensor([1., 2., 3., 4.], dtype=dtype, requires_grad=True)[::2]
    before, version = rhs.detach().clone(), rhs._version
    calls = []
    def operator(p):
        assert not p.requires_grad and p.grad_fn is None
        calls.append(p)
        return h @ p
    result, report = solve(operator, rhs, damping=2)
    expected = (torch.tensor([1/11, 7/11], dtype=dtype) if not expanded
                else torch.tensor([0., 1.], dtype=dtype))
    torch.testing.assert_close(result, expected)
    assert report["status"] == "converged" and report["iterations"] == 2
    assert report["products"] == len(calls) == 3
    assert report["true_residual_norm"] <= report["threshold"]
    assert result.grad_fn is None and not result.requires_grad
    assert result.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr()
    torch.testing.assert_close(rhs, before, rtol=0, atol=0)
    assert rhs._version == version


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_rhs_executes_fresh_residual_without_division(dtype):
    calls = []
    result, report = solve(lambda p: calls.append(p.clone()) or p, torch.zeros(3, dtype=dtype))
    assert report["status"] == "converged" and report["iterations"] == 0
    assert report["products"] == len(calls) == 1 and report["true_residual_norm"] == 0
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("damping", (0., 1.))
def test_positive_damping_is_not_definiteness_proof(dtype, damping):
    h = torch.diag(torch.tensor([-2., 3.], dtype=dtype))
    result, report = solve(lambda p: h @ p, torch.tensor([1., 0.], dtype=dtype), damping=damping)
    assert report["status"] == "nonpositive_curvature" and report["products"] == 1
    assert report["iterations"] == 0 and report["true_residual_norm"] is None
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singular_and_nearly_orthogonal_curvature_refuse(dtype):
    rhs = torch.tensor([1., 0.], dtype=dtype)
    _, singular = solve(lambda p: torch.zeros_like(p), rhs)
    h = torch.tensor([[torch.finfo(dtype).eps, 1.], [1., 1.]], dtype=dtype)
    _, near = solve(lambda p: h @ p, rhs)
    assert singular["status"] == "nonpositive_curvature"
    assert near["status"] == "curvature_breakdown"
    assert all(r["products"] == 1 and r["iterations"] == 0 for r in (singular, near))


@pytest.mark.parametrize("dtype", DTYPES)
def test_iteration_exhaustion_is_not_convergence(dtype):
    h = torch.diag(torch.tensor([1., 4.], dtype=dtype))
    _, report = solve(lambda p: h @ p, torch.ones(2, dtype=dtype), max_iterations=1)
    assert report["status"] == "iteration_limit"
    assert report["iterations"] == 1 and report["products"] == report["max_products"] == 2
    assert report["true_residual_norm"] > report["threshold"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_fresh_residual_rejects_misleading_recursive_zero(dtype):
    calls = []
    def changed_operator(p):
        calls.append(p.clone())
        return p if len(calls) == 1 else 2 * p
    _, report = solve(changed_operator, torch.ones(1, dtype=dtype), max_iterations=1)
    assert report["status"] == "residual_mismatch" and report["products"] == 2
    assert report["recursive_residual_norm"] == 0 and report["true_residual_norm"] == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ("shape", "dtype", "layout", "nan", "infinity", "mutation", "admission"))
def test_callback_refusals_and_attempt_count(dtype, kind):
    rhs = torch.ones(2, dtype=dtype)
    def callback(p):
        if kind == "shape": return p[:1]
        if kind == "dtype": return p.to(torch.float64 if dtype == torch.float32 else torch.float32)
        if kind == "layout": return p.to_sparse()
        if kind == "nan": return p * float("nan")
        if kind == "infinity": return p * float("inf")
        if kind == "mutation": p.add_(1); return p
        raise RuntimeError("values direction must be finite and within [-1000,1000]")
    result, report = solve(callback, rhs)
    assert report["status"] == ("nonfinite" if kind in ("nan", "infinity") else "operator_error")
    assert report["products"] == 1 and report["true_residual_norm"] is None
    assert torch.equal(rhs, torch.ones_like(rhs)) and torch.isfinite(result).all()
    if kind == "admission": assert "[-1000,1000]" in report["message"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_callback_alias_is_owned_and_programming_exception_propagates(dtype):
    rhs = torch.ones(2, dtype=dtype)
    result, report = solve(lambda p: p, rhs)
    assert report["status"] == "converged"
    assert result.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr()
    def broken(p): raise TypeError("intentional callback programming error")
    with pytest.raises(TypeError, match="programming error"):
        solve(broken, rhs)


@pytest.mark.parametrize("kwargs", ({"damping": -1}, {"damping": float("nan")}, {"rtol": 1},
                                    {"rtol": 0}, {"atol": 0}, {"max_iterations": 0},
                                    {"max_iterations": 33}, {"max_iterations": True}))
def test_invalid_policy_is_rejected_before_callback(kwargs):
    def forbidden(p): raise AssertionError("Policy must be validated first")
    with pytest.raises(ValueError): solve(forbidden, torch.ones(2), **kwargs)


@pytest.mark.parametrize("rhs", (torch.ones(0), torch.ones(33), torch.ones(2, 2),
                                  torch.ones(2, dtype=torch.int64), torch.tensor([float("nan")]),
                                  torch.ones(2).to_sparse()))
def test_invalid_rhs_is_rejected(rhs):
    with pytest.raises(ValueError): solve(lambda p: p, rhs)


def test_final_norm_overflow_is_a_finite_serializable_refusal():
    calls = []
    def callback(p):
        calls.append(1)
        return p if len(calls) == 1 else torch.full_like(p, 1.3e308)
    _, report = solve(callback, torch.ones(3, dtype=torch.float64))
    assert report["status"] == "nonfinite" and report["products"] == 2
    assert report["true_residual_norm"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("scale", (1., 1e300))
def test_tiny_nonzero_rhs_cannot_pass_through_a_squared_norm_underflow(scale):
    rhs = torch.tensor([1e-200], dtype=torch.float64)
    _, report = solve(lambda p: scale * p, rhs, rtol=1e-3, atol=1e-300)
    assert report["rhs_norm"] == 1e-200
    assert report["threshold"] == pytest.approx(1e-203, rel=1e-12, abs=0)
    # Dot products can still underflow. Refusal is honest; a zero solution
    # with a claimed zero residual would be false convergence.
    assert report["status"] != "converged"
    assert report["true_residual_norm"] is None
    json.dumps(report, allow_nan=False)


def test_finite_spd_products_can_overflow_recursive_dot_arithmetic():
    matrix = torch.tensor([[1e-66, 1e-53], [1e-53, 1.]], dtype=torch.float64)
    _, report = solve(lambda p: matrix @ p, torch.tensor([1e153, 0.], dtype=torch.float64), max_iterations=1)
    assert report["status"] == "nonfinite" and report["products"] == 1
    assert report["true_residual_norm"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_full_model_solve_uses_native_hvps_and_independent_dense_oracle(dtype_name):
    template, parameters, raw, offsets, targets, reference_pool, _, _ = example.fixture(dtype_name)
    callbacks, _, gradients = example.make_hvp_callbacks(template, parameters, raw, offsets, targets, reference_pool)
    point = example.flatten(parameters).requires_grad_()
    def independent_loss(vector):
        # Unlike consumer direction unpacking, this test-only map keeps AD.
        blocks, cursor = {}, 0
        for name in example.PARAMETER_NAMES:
            primal = parameters[name]
            blocks[name] = vector[cursor:cursor + primal.numel()].reshape(primal.shape)
            cursor += primal.numel()
        return example.curvature_example.loss_function(template, raw, offsets, targets, reference_pool)(blocks)
    hessian = torch.autograd.functional.hessian(independent_loss, point)
    independent_gradient, = torch.autograd.grad(independent_loss(point), point)
    matrix = hessian.detach() + 2 * torch.eye(25, dtype=point.dtype)
    assert float(torch.linalg.eigvalsh(matrix).min()) > 0
    rhs = -example.flatten(gradients["bridge"])
    torch.testing.assert_close(rhs, -independent_gradient)
    rtol, atol = (2e-5, 1e-7) if dtype_name == "float32" else (1e-9, 1e-12)
    native = torch.ops.ragged_pool._double_backward.default
    with patch.object(torch.ops.ragged_pool._double_backward, "default", wraps=native) as delegated:
        result, report = solve(callbacks["bridge"], rhs, damping=2, max_iterations=32, rtol=rtol, atol=atol)
    assert report["status"] == "converged"
    # Each HVP uses native second backward for the primal pool JVP and
    # again for the tangent of backward. Count both actual traversals.
    assert delegated.call_count == 2 * report["products"] > 2
    expected = torch.linalg.solve(matrix, -independent_gradient.detach())
    tolerance = dict(rtol=1e-4, atol=2e-6) if dtype_name == "float32" else dict(rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(result, expected, **tolerance)
    assert float(torch.linalg.vector_norm((matrix @ result - rhs).to(torch.float64))) <= report["threshold"]
    omitted = rhs / 2
    assert float(torch.linalg.vector_norm((matrix @ omitted - rhs).to(torch.float64))) > report["threshold"]
