"""Solve one fixed damped loss-curvature system using installed native HVPs.

The retained curvature example supplies the model and its two matched SGD
updates. At that fixed trained point, bounded conjugate gradients solves
(H + 2*I)s = -g without constructing a Hessian or applying s to the model.
Ordinary native reverse-over-reverse and independent primitive PyTorch check
the solve and freshly recomputed residuals. This demonstrates numerical
integration, with no optimizer, model-quality, security or performance claim.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch


def load_neighbor(filename, module_name):
    """Load a named example beside this script without changing sys.path."""
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the required neighboring example: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curvature_example = load_neighbor("train_curvature.py", "_ragged_pool_curvature_example")
cg_example = load_neighbor("damped_cg.py", "_ragged_pool_damped_cg_example")

PARAMETER_NAMES = ("encoder.weight", "encoder.bias", "scorer.weight", "head.weight", "head.bias")
DTYPES = curvature_example.DTYPES


def flatten(parameters):
    """Own one detached flat vector in the fixed module parameter order."""
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("Expected the five declared model parameter blocks")
    return torch.cat([parameters[name].detach().reshape(-1) for name in PARAMETER_NAMES]).detach().clone()


def unflatten(vector, parameters):
    """Own detached blocks matching the fixed parameter shapes and order."""
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("Expected the five declared model parameter blocks")
    example = parameters[PARAMETER_NAMES[0]]
    count = sum(parameters[name].numel() for name in PARAMETER_NAMES)
    if (vector.layout != torch.strided or vector.device.type != "cpu" or vector.dim() != 1
            or vector.numel() != count or vector.dtype != example.dtype):
        raise ValueError("A direction must be a matching strided CPU flat parameter vector")
    result, cursor = {}, 0
    for name in PARAMETER_NAMES:
        primal = parameters[name]
        result[name] = vector[cursor:cursor + primal.numel()].detach().reshape(primal.shape).clone()
        cursor += primal.numel()
    return result


def norm(vector):
    return math.hypot(*vector.detach().to(torch.float64).tolist())


def fixture(dtype_name):
    """Return the unchanged N16 model fixture after its two actual SGD steps."""
    dtype = DTYPES[dtype_name]
    torch.manual_seed(curvature_example.SEED)
    tolerance = ({"rtol": 3e-5, "atol": 3e-6} if dtype_name == "float32"
                 else {"rtol": 1e-10, "atol": 2e-12})
    raw = 0.3 * torch.randn(6, 3, dtype=dtype, device="cpu")
    offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64, device="cpu")
    targets = torch.tensor([-0.4, 0.3, 0.8, -0.2], dtype=dtype, device="cpu")
    reference_pool = curvature_example.primitive_pool(tuple(offsets.tolist()))
    initial = curvature_example.SetRegressor().to(device="cpu", dtype=dtype)
    original = {"raw": raw, "offsets": offsets, "targets": targets,
                **{f"initial_parameter:{name}": value for name, value in initial.named_parameters()}}
    before = curvature_example.snapshot(original)
    trained, steps = curvature_example.train(initial, raw, offsets, targets, reference_pool, tolerance)
    curvature_example.unchanged(original, before)
    template = copy.deepcopy(trained).eval()
    template.requires_grad_(False)
    template.zero_grad(set_to_none=True)
    parameters = {name: value.detach().clone() for name, value in template.named_parameters()}
    if tuple(parameters) != PARAMETER_NAMES or sum(value.numel() for value in parameters.values()) != 25:
        raise AssertionError("The retained fixture must have the declared five blocks and 25 parameters")
    return template, parameters, raw, offsets, targets, reference_pool, steps, tolerance


def native_gradient(loss, parameters):
    owned = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    derivatives = torch.autograd.grad(loss(owned), tuple(owned.values()))
    return {name: value.detach().clone() for name, value in zip(owned, derivatives)}


def make_hvp_callbacks(template, parameters, raw, offsets, targets, reference_pool):
    """Return detached matrix-free products, call counts and initial gradients.

    Counts describe callback traversals. Separate mathematical tests check native
    delegation at the operator boundary. This consumer does not replace kernels.
    """
    bridge_loss = curvature_example.loss_function(
        template, raw, offsets, targets, curvature_example.ragged_pool_forward_ad)
    native_loss = curvature_example.loss_function(
        template, raw, offsets, targets, curvature_example.ragged_pool)
    reference_loss = curvature_example.loss_function(template, raw, offsets, targets, reference_pool)
    bridge_grad = torch.func.grad(bridge_loss)
    reference_grad = torch.func.grad(reference_loss)
    gradients = {"bridge": bridge_grad(parameters),
                 "original_native_reverse_over_reverse": native_gradient(native_loss, parameters),
                 "reference_hvp": reference_grad(parameters)}
    counts = {name: 0 for name in gradients}

    def bridge(vector):
        counts["bridge"] += 1
        direction = unflatten(vector, parameters)
        _, product = torch.func.jvp(bridge_grad, (parameters,), (direction,))
        return flatten(product)

    def native(vector):
        counts["original_native_reverse_over_reverse"] += 1
        direction = unflatten(vector, parameters)
        _, product = curvature_example.reverse_over_reverse(native_loss, parameters, direction)
        return flatten(product)

    def reference(vector):
        counts["reference_hvp"] += 1
        direction = unflatten(vector, parameters)
        _, product = torch.func.jvp(reference_grad, (parameters,), (direction,))
        return flatten(product)

    return {"bridge": bridge, "original_native_reverse_over_reverse": native,
            "reference_hvp": reference}, counts, gradients


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = curvature_example.installed_identity()
    # The shared identity checker belongs to another example. Record this caller
    # and both loaded neighbor files separately, with their actual copied bytes.
    identity["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    identity["helper_sha256"] = {
        name: sha256(Path(__file__).resolve().with_name(name).read_bytes()).hexdigest()
        for name in ("train_curvature.py", "damped_cg.py")}
    template, parameters, raw, offsets, targets, reference_pool, steps, training_tolerance = fixture(dtype_name)
    fixed = {"raw": raw, "offsets": offsets, "targets": targets,
             **{f"parameter:{name}": value for name, value in parameters.items()},
             **{f"template:{name}": value for name, value in template.named_parameters()}}
    before = curvature_example.snapshot(fixed)
    callbacks, counts, gradients = make_hvp_callbacks(
        template, parameters, raw, offsets, targets, reference_pool)
    gradient_errors = {name: curvature_example.compare(gradients["bridge"], gradient, training_tolerance)
                       for name, gradient in gradients.items() if name != "bridge"}
    rhs = -flatten(gradients["bridge"])
    if not bool(torch.isfinite(rhs).all()) or norm(rhs) == 0:
        raise AssertionError("The declared model fixture needs a finite nonzero right-hand side")
    rhs_before = curvature_example.snapshot({"rhs": rhs})
    damping, max_iterations = 2.0, 32
    rtol, atol = (2e-5, 1e-7) if dtype_name == "float32" else (1e-9, 1e-12)
    expected_threshold = atol + rtol * norm(rhs)
    solutions, reports, solve_counts = {}, {}, {}
    for name, callback in callbacks.items():
        count_before = counts[name]
        solution, report = cg_example.damped_cg(
            callback, rhs, damping=damping, max_iterations=max_iterations, rtol=rtol, atol=atol)
        solve_counts[name] = counts[name] - count_before
        if report["status"] != "converged":
            raise AssertionError(f"The frozen {name} solve did not converge: {report}")
        if (solve_counts[name] != report["products"] or report["products"] > max_iterations + 1
                or report["max_products"] != max_iterations + 1):
            raise AssertionError("Every reported solve product must match an actual bounded callback call")
        if abs(report["threshold"] - expected_threshold) > 1e-14 * max(1, expected_threshold):
            raise AssertionError("The stopping threshold must use the declared right-hand side")
        if (solution.requires_grad or solution.grad_fn is not None
                or solution.shape != rhs.shape or solution.dtype != rhs.dtype
                or not bool(torch.isfinite(solution).all())):
            raise AssertionError("Every solve must return a finite detached matching solution")
        solutions[name], reports[name] = solution, report
    # This comparison supplements the separately deciding residual checks. It
    # allows the declared finite-precision stopping errors in all three routes.
    solution_tolerance = ({"rtol": 1e-4, "atol": 2e-6} if dtype_name == "float32"
                          else {"rtol": 1e-8, "atol": 1e-10})
    solution_errors = {
        name: curvature_example.compare(unflatten(solutions["bridge"], parameters),
                                        unflatten(solution, parameters), solution_tolerance)
        for name, solution in solutions.items() if name != "bridge"}
    validation_before = dict(counts)
    residuals = {}
    for name, callback in callbacks.items():
        residual = rhs - (callback(solutions["bridge"]) + damping * solutions["bridge"])
        absolute = norm(residual)
        if not math.isfinite(absolute) or not math.isfinite(expected_threshold) or absolute > expected_threshold:
            raise AssertionError(f"The independent {name} residual exceeds the same stopping threshold")
        residuals[name] = {"norm": absolute, "normalized_norm": absolute / norm(rhs),
                           "threshold": expected_threshold, "accepted": True}
    omitted = rhs / damping
    wrong_residual = rhs - (callbacks["reference_hvp"](omitted) + damping * omitted)
    wrong_norm = norm(wrong_residual)
    if not math.isfinite(wrong_norm) or not wrong_norm > expected_threshold:
        raise AssertionError("The lambda-only control must fail the independent full-system residual gate")
    validation_counts = {name: counts[name] - validation_before[name] for name in counts}
    forbidden = {value.untyped_storage().data_ptr() for value in (*fixed.values(), rhs) if value.numel()}
    addresses = [value.untyped_storage().data_ptr() for value in solutions.values()]
    if any(address in forbidden for address in addresses) or len(addresses) != len(set(addresses)):
        raise AssertionError("The three detached solutions must own fresh separate storage")
    versions = curvature_example.unchanged(fixed, before)
    rhs_versions = curvature_example.unchanged({"rhs": rhs}, rhs_before)
    if any(value.grad is not None for value in (*parameters.values(), *template.parameters(), raw)):
        raise AssertionError("The fixed solve must not accumulate parameter or input .grad fields")
    return {
        "status": "passed", "method": "damped_cg.packed_parameter_hvp", "dtype": dtype_name, "device": "cpu",
        "purpose": "One bounded fixed-point numerical solve using the installed packed HVP",
        "scope": "Example-local detached CPU solve. No model update from the solution or optimizer-improvement, model-quality, security, speed or memory-benefit result.",
        "seed": curvature_example.SEED, "raw_shape": list(raw.shape), "group_lengths": [2, 0, 3, 1],
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
        "parameter_tensors": 5, "scalar_parameters": 25,
        "optimizer": "SGD", "learning_rate": 0.03,
        "optimizer_steps": 2, "original_native_optimizer_steps": 2, "reference_optimizer_steps": 2,
        "comparison_tolerance": training_tolerance, "steps": steps,
        "solve": {"status": "converged", "system": "(H_loss + lambda*I)s = -g",
                  "parameters_fixed": True, "optimizer_steps": 0, "model_updated_with_solution": False,
                  "full_hessian_materialized": False, "preconditioned": False,
                  "damping": damping, "max_iterations": max_iterations, "rtol": rtol, "atol": atol,
                  "rhs_norm": norm(rhs), "threshold": expected_threshold,
                  "gradient_errors": gradient_errors, "solve_reports": reports,
                  "callback_counts": counts, "solve_callback_counts": solve_counts,
                  "validation_callback_counts": validation_counts,
                  "solution_tolerance": solution_tolerance, "solution_errors": solution_errors,
                  "solution_norms": {name: norm(value) for name, value in solutions.items()},
                  "independent_residuals": residuals,
                  "omitted_h_control": {"status": "rejected_by_residual", "solution": "rhs/lambda",
                                        "residual_norm": wrong_norm, "threshold": expected_threshold,
                                        "same_independent_operator": "reference_hvp",
                                        "exceeds_threshold": True},
                  "fresh_detached_solution_storage": True, "parameter_grad_fields_unset": True,
                  "inputs_and_parameters_unchanged": True, "input_versions": versions,
                  "rhs_unchanged": True, "rhs_versions": rhs_versions},
        "inputs_unchanged": True, "initial_parameters_unchanged": True,
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "python_executable": sys.executable, "working_directory": str(Path.cwd()),
        "installed_package": True, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), **identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt file")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    text = json.dumps(run(arguments.dtype), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")

