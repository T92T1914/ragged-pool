"""Train a packed model, then check two parameter curvature directions.

Two real SGD updates through the installed bridge are compared with the
original native API and independent PyTorch pooling. Fixed trained parameters
then pass through functional_call and jvp(grad(loss)). Native reverse-over-
reverse, primitive PyTorch and centered differences of reference gradients
check every parameter block. This demonstrates integration, with no model-
quality, optimizer-improvement, robustness, privacy or performance result.
"""
import argparse
import copy
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_forward_ad


SEED = 1609
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A common scalar score bias cancels within each group.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, pool):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def primitive_pool(boundaries):
    """Capture Python boundaries before entering a function transform."""
    def pool(values, logits, offsets):
        rows = []
        for start, stop in zip(boundaries, boundaries[1:]):
            selected_values, selected_logits = values[start:stop], logits[start:stop]
            if start == stop:
                rows.append(selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0)
            else:
                weights = torch.softmax(selected_logits, dim=0)
                rows.append((weights[:, None] * selected_values).sum(dim=0))
        return torch.stack(rows)
    return pool


def difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def snapshot(tensors):
    return {name: (tensor.detach().clone(), tensor._version) for name, tensor in tensors.items()}


def unchanged(tensors, snapshots):
    versions = {}
    for name, tensor in tensors.items():
        before, version = snapshots[name]
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
        if tensor._version != version:
            raise AssertionError(f"The consumer mutated {name}")
        versions[name] = {"before": version, "after": tensor._version}
    return versions


def compare(first, second, tolerance):
    if first.keys() != second.keys():
        raise AssertionError("Comparison keys differ")
    errors = {}
    for name in first:
        torch.testing.assert_close(first[name], second[name], **tolerance)
        errors[name] = difference(first[name], second[name])
    return errors


def finite(tensors):
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise AssertionError("Every compared output, gradient, HVP and parameter must be finite")


def installed_identity():
    installed = distribution("ragged-pool")
    direct = installed.read_text("direct_url.json")
    prefix, imported = Path(sys.prefix).resolve(), Path(package.__file__).resolve(strict=True)
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed wheel")
    if (imported != Path(installed.locate_file("ragged_pool/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != package.__version__):
        raise AssertionError("The import must match this environment's installed distribution")
    source = Path(__file__).resolve().parents[1]
    if (source / "pyproject.toml").is_file() and prefix.is_relative_to(source):
        raise AssertionError("Use an environment outside the checkout")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if len(libraries) != 1 or libraries[0].resolve() not in recorded or libraries[0].resolve() not in loaded:
        raise AssertionError("Expected one recorded and loaded installed native library")
    return {"package_path": str(imported), "package_version": package.__version__,
            "environment_prefix": str(prefix), "native_library_loaded": True,
            "native_library_path": str(libraries[0].resolve()),
            "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest(),
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def training_result(model, raw, offsets, targets, pool):
    model.zero_grad(set_to_none=True)
    raw.grad = None
    inputs = {"raw": raw, "offsets": offsets, "targets": targets}
    before = snapshot(inputs)
    predictions, pooled = model(raw, offsets, pool)
    loss = (predictions - targets).square().mean()
    loss.backward()
    if raw.grad is None or any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("Inputs and every parameter block must receive gradients")
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    result = {"predictions": predictions.detach(), "pooled": pooled.detach(),
              "loss": loss.detach(), "input_gradient": raw.grad.detach().clone()}
    finite((*result.values(), *gradients.values()))
    torch.testing.assert_close(result["pooled"][1], torch.zeros_like(result["pooled"][1]), rtol=0, atol=0)
    versions = unchanged(inputs, before)
    return result, gradients, versions


def train(initial, raw, offsets, targets, reference_pool, tolerance):
    routes = {"bridge": ragged_pool_forward_ad, "original_native": ragged_pool,
              "reference": reference_pool}
    models = {name: copy.deepcopy(initial) for name in routes}
    inputs = {name: raw.detach().clone().requires_grad_() for name in routes}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03) for name, model in models.items()}
    steps = []
    for step in (1, 2):
        before = {name: {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
                  for name, model in models.items()}
        evaluations = {name: training_result(models[name], inputs[name], offsets, targets, pool)
                       for name, pool in routes.items()}
        output_errors = {name: compare(evaluations["bridge"][0], evaluations[name][0], tolerance)
                         for name in ("original_native", "reference")}
        gradient_errors = {name: compare(evaluations["bridge"][1], evaluations[name][1], tolerance)
                           for name in ("original_native", "reference")}
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
                  for name, model in models.items()}
        update_errors = {name: compare(states["bridge"], states[name], tolerance)
                         for name in ("original_native", "reference")}
        changes = {name: {key: difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(value == 0 for route in changes.values() for value in route.values()):
            raise AssertionError("Both actual SGD steps must change every parameter block in all three models")
        finite(value for state in states.values() for value in state.values())
        steps.append({"step": step,
                      "losses": {name: float(result[0]["loss"]) for name, result in evaluations.items()},
                      "output_and_input_gradient_errors": output_errors,
                      "parameter_gradient_errors": gradient_errors,
                      "updated_parameter_errors": update_errors, "parameter_update_max_abs": changes,
                      "all_routes_all_parameter_blocks_changed": True, "empty_group_pooled_zero": True,
                      "inputs_unchanged": True,
                      "input_versions": {name: result[2] for name, result in evaluations.items()}})
    return models["bridge"], steps


def dot(first, second):
    return sum((first[name] * second[name]).sum() for name in first)


def parameter_directions(parameters):
    """Two fixed, bounded directions in the complete 25-parameter space."""
    count = sum(value.numel() for value in parameters.values())
    example = next(iter(parameters.values()))
    indices = torch.arange(count, dtype=example.dtype, device=example.device)
    first = torch.linspace(-0.2, 0.2, count, dtype=example.dtype, device=example.device)
    second = 0.15 * torch.cos(0.7 * indices) + 0.04 * torch.sin(0.3 * indices)
    directions = []
    for flat in (first, second):
        cursor, direction = 0, {}
        for name, value in parameters.items():
            direction[name] = flat[cursor:cursor + value.numel()].reshape(value.shape).clone()
            cursor += value.numel()
        directions.append(direction)
    ratio = second[0] / first[0]
    nonproportional = float((second - ratio * first).abs().max())
    if nonproportional <= 1e-4 or max(float(value.abs().max()) for row in directions for value in row.values()) > 1:
        raise AssertionError("The fixture requires two bounded nonproportional parameter directions")
    return directions, nonproportional


def loss_function(template, raw, offsets, targets, pool):
    def loss(parameters):
        predictions, _ = torch.func.functional_call(template, parameters, (raw, offsets, pool))
        return (predictions - targets).square().mean()
    return loss


def reverse_over_reverse(loss, parameters, direction):
    """Differentiate the original native first gradient using ordinary autograd."""
    owned = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    first = torch.autograd.grad(loss(owned), tuple(owned.values()), create_graph=True)
    gradient = dict(zip(owned, first))
    second = torch.autograd.grad(dot(gradient, direction), tuple(owned.values()))
    return ({name: value.detach() for name, value in gradient.items()},
            {name: value.detach() for name, value in zip(owned, second)})


def curvature(model, raw, offsets, targets, reference_pool, tolerance, dtype_name):
    """Evaluate curvature without another optimizer update or a full Hessian."""
    template = copy.deepcopy(model).eval()
    template.requires_grad_(False)
    template.zero_grad(set_to_none=True)
    parameters = {name: value.detach().clone() for name, value in template.named_parameters()}
    directions, nonproportional = parameter_directions(parameters)
    tensors = {"raw": raw, "offsets": offsets, "targets": targets,
               **{f"parameter:{name}": value for name, value in parameters.items()},
               **{f"template:{name}": value for name, value in template.named_parameters()},
               **{f"direction{index}:{name}": value for index, direction in enumerate(directions)
                  for name, value in direction.items()}}
    before = snapshot(tensors)
    bridge_loss = loss_function(template, raw, offsets, targets, ragged_pool_forward_ad)
    native_loss = loss_function(template, raw, offsets, targets, ragged_pool)
    reference_loss = loss_function(template, raw, offsets, targets, reference_pool)
    bridge_grad, reference_grad = torch.func.grad(bridge_loss), torch.func.grad(reference_loss)
    h = 1e-2 if dtype_name == "float32" else 1e-5
    fd_tolerance = ({"rtol": 5e-3, "atol": 3e-5} if dtype_name == "float32"
                    else {"rtol": 3e-6, "atol": 3e-9})
    rows, actual_hvps, actual_gradients = [], [], []
    forbidden = {value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()}
    for index, direction in enumerate(directions):
        gradient, actual = torch.func.jvp(bridge_grad, (parameters,), (direction,))
        native_gradient, native = reverse_over_reverse(native_loss, parameters, direction)
        reference_gradient, reference = torch.func.jvp(reference_grad, (parameters,), (direction,))
        plus = {name: value + h * direction[name] for name, value in parameters.items()}
        minus = {name: value - h * direction[name] for name, value in parameters.items()}
        upper, lower = reference_grad(plus), reference_grad(minus)
        finite_difference = {name: (upper[name] - lower[name]) / (2 * h) for name in parameters}
        gradient_errors = {
            "original_native_reverse_over_reverse": compare(gradient, native_gradient, tolerance),
            "reference": compare(gradient, reference_gradient, tolerance)}
        errors = {
            "original_native_reverse_over_reverse": compare(actual, native, tolerance),
            "reference_hvp": compare(actual, reference, tolerance),
            "reference_gradient_finite_difference": compare(actual, finite_difference, fd_tolerance)}
        finite(value for result in (gradient, native_gradient, reference_gradient, actual,
                                   native, reference, finite_difference) for value in result.values())
        if max(float(value.abs().max()) for value in actual.values()) <= 1e-5:
            raise AssertionError("Both fixed directions must produce nontrivial parameter curvature")
        if any(value.numel() and value.untyped_storage().data_ptr() in forbidden
               for result in (gradient, actual) for value in result.values()):
            raise AssertionError("Functional gradients and HVPs must not alias the fixed inputs")
        curvature_values = {name: float(dot(direction, result).detach())
                            for name, result in (("bridge", actual), ("original_native_reverse_over_reverse", native),
                                                 ("reference_hvp", reference),
                                                 ("reference_gradient_finite_difference", finite_difference))}
        torch.testing.assert_close(dot(direction, actual), dot(direction, native), **tolerance)
        torch.testing.assert_close(dot(direction, actual), dot(direction, reference), **tolerance)
        rows.append({"direction": index + 1, "gradient_errors": gradient_errors, "errors": errors,
                     "directional_curvature": curvature_values,
                     "hvp_max_abs": {name: float(value.abs().max()) for name, value in actual.items()},
                     "direction_max_abs": {name: float(value.abs().max()) for name, value in direction.items()},
                     "fresh_gradient_and_hvp_storage": True})
        actual_hvps.append(actual)
        actual_gradients.append(gradient)
    # Both directions are evaluated at exactly the same trained parameter point.
    compare(actual_gradients[0], actual_gradients[1], tolerance)
    left, right = dot(directions[0], actual_hvps[1]), dot(directions[1], actual_hvps[0])
    torch.testing.assert_close(left, right, **tolerance)
    combination = {name: directions[0][name] + 0.5 * directions[1][name] for name in parameters}
    combined_gradient, combined_hvp = torch.func.jvp(bridge_grad, (parameters,), (combination,))
    expected_combination = {name: actual_hvps[0][name] + 0.5 * actual_hvps[1][name] for name in parameters}
    linearity_errors = compare(combined_hvp, expected_combination, tolerance)
    compare(combined_gradient, actual_gradients[0], tolerance)
    zero = {name: torch.zeros_like(value) for name, value in parameters.items()}
    zero_gradient, zero_hvp = torch.func.jvp(bridge_grad, (parameters,), (zero,))
    zero_errors = compare(zero_hvp, zero, {"rtol": 0, "atol": 0})
    compare(zero_gradient, actual_gradients[0], tolerance)
    finite((*combined_hvp.values(), *zero_hvp.values(), left, right))
    versions = unchanged(tensors, before)
    if any(value.grad is not None for value in (*parameters.values(), *template.parameters())):
        raise AssertionError("Curvature evaluation must not accumulate parameter .grad fields")
    return {"status": "passed", "method": "torch.func.jvp.grad.functional_call",
            "parameters_fixed": True, "optimizer_steps": 0, "directions": 2,
            "nonproportional_max_abs_residual": nonproportional,
            "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
            "finite_difference_step": h, "finite_difference_tolerance": fd_tolerance,
            "direction_results": rows,
            "bilinear_symmetry": {"status": "passed", "u_dot_Hv": float(left.detach()),
                                 "v_dot_Hu": float(right.detach()), "absolute_error": difference(left, right)},
            "linearity": {"status": "passed", "combination": "u + 0.5*v", "errors": linearity_errors},
            "zero_direction": {"status": "passed", "errors": zero_errors},
            "inputs_and_parameters_unchanged": True, "input_versions": versions,
            "parameter_grad_fields_unset": True, "full_hessian_materialized": False}


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    torch.manual_seed(SEED)
    dtype = DTYPES[dtype_name]
    tolerance = ({"rtol": 3e-5, "atol": 3e-6} if dtype_name == "float32"
                 else {"rtol": 1e-10, "atol": 2e-12})
    raw = 0.3 * torch.randn(6, 3, dtype=dtype, device="cpu")
    offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64, device="cpu")
    targets = torch.tensor([-0.4, 0.3, 0.8, -0.2], dtype=dtype, device="cpu")
    reference_pool = primitive_pool(tuple(offsets.tolist()))
    initial = SetRegressor().to(device="cpu", dtype=dtype)
    inputs = {"raw": raw, "offsets": offsets, "targets": targets,
              **{f"initial_parameter:{name}": value for name, value in initial.named_parameters()}}
    before = snapshot(inputs)
    trained, steps = train(initial, raw, offsets, targets, reference_pool, tolerance)
    result = curvature(trained, raw, offsets, targets, reference_pool, tolerance, dtype_name)
    versions = unchanged(inputs, before)
    return {"status": "passed", "purpose": "Synthetic training and directional parameter curvature integration",
            "scope": "Packed eager CPU. No third derivative, mapped HVP, optimizer-improvement, model-quality, robustness, privacy or performance result.",
            "method": "torch.func.jvp.grad.functional_call", "dtype": dtype_name, "device": "cpu", "seed": SEED,
            "raw_shape": list(raw.shape), "group_lengths": [2, 0, 3, 1],
            "architecture": "encoder 3->4, tanh, scorer 4->1 without bias, pool, head 4->1",
            "parameter_tensors": len(tuple(initial.parameters())),
            "scalar_parameters": sum(parameter.numel() for parameter in initial.parameters()),
            "optimizer": "SGD", "learning_rate": 0.03, "objective": "MSE(predictions, targets)",
            "optimizer_steps": 2, "native_optimizer_steps": 2,
            "original_native_optimizer_steps": 2, "reference_optimizer_steps": 2,
            "comparison_tolerance": tolerance, "steps": steps, "curvature": result,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "input_versions": versions,
            "gradients_and_hvps_finite": True, "python_version": sys.version.split()[0],
            "torch_version": torch.__version__, "python_executable": sys.executable,
            "working_directory": str(Path.cwd()), "installed_package": True,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


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
