"""Train a packed model, then check its first-order input sensitivity.

Two SGD updates through the installed forward-AD bridge are compared with the
original native API and independent PyTorch pooling. At the trained parameters,
explicit dual tensors and torch.func.jvp check one raw-input direction against
primitive PyTorch math and centered finite differences. This is integration,
not a model-quality, robustness, privacy or performance result.
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


SEED = 20261021
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A common scalar score bias would cancel within each group.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, pool):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def primitive_pool(boundaries):
    """Capture ordinary Python offsets before entering a transform."""
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
            "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest(),
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def forward_backward(model, raw, offsets, targets, pool):
    model.zero_grad(set_to_none=True)
    raw.grad = None
    tensors = {"raw": raw, "offsets": offsets, "targets": targets}
    before = snapshot(tensors)
    predictions, pooled = model(raw, offsets, pool)
    loss = nn.functional.mse_loss(predictions, targets)
    loss.backward()
    if raw.grad is None or any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("Inputs and every connected parameter must receive gradients")
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    results = {"predictions": predictions.detach(), "pooled": pooled.detach(),
               "loss": loss.detach(), "input_gradient": raw.grad.detach().clone()}
    if not all(bool(torch.isfinite(value).all()) for value in (*results.values(), *gradients.values())):
        raise AssertionError("Training values and gradients must remain finite")
    versions = unchanged(tensors, before)
    return results, gradients, versions


def train(initial, features, offsets, targets, reference_pool, tolerance):
    routes = {"bridge": ragged_pool_forward_ad, "original_native": ragged_pool,
              "reference": reference_pool}
    models = {name: copy.deepcopy(initial) for name in routes}
    inputs = {name: features.clone().requires_grad_() for name in routes}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03) for name, model in models.items()}
    steps = []
    for index in (1, 2):
        before = {name: {key: p.detach().clone() for key, p in model.named_parameters()}
                  for name, model in models.items()}
        evaluations = {name: forward_backward(models[name], inputs[name], offsets, targets, pool)
                       for name, pool in routes.items()}
        output_errors, gradient_errors = {}, {}
        for name in ("original_native", "reference"):
            output_errors[name] = compare(evaluations["bridge"][0], evaluations[name][0], tolerance)
            gradient_errors[name] = compare(evaluations["bridge"][1], evaluations[name][1], tolerance)
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: {key: p.detach().clone() for key, p in model.named_parameters()}
                  for name, model in models.items()}
        update_errors = {name: compare(states["bridge"], states[name], tolerance)
                         for name in ("original_native", "reference")}
        changes = {name: {key: difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(value == 0 for route in changes.values() for value in route.values()):
            raise AssertionError("Both actual SGD steps must change every parameter block in all three models")
        if not all(bool(torch.isfinite(value).all()) for state in states.values() for value in state.values()):
            raise AssertionError("Updated parameters must remain finite")
        steps.append({"step": index, "losses": {name: float(result[0]["loss"]) for name, result in evaluations.items()},
                      "output_and_input_gradient_errors": output_errors,
                      "parameter_gradient_errors": gradient_errors,
                      "updated_parameter_errors": update_errors, "parameter_update_max_abs": changes,
                      "inputs_unchanged": True, "input_versions": evaluations["bridge"][2]})
    return models["bridge"], steps


def sensitivity(model, features, offsets, direction, reference_pool, tolerance, dtype_name):
    """Use fixed trained parameters for one directional derivative of the model."""
    model = copy.deepcopy(model).eval()
    model.requires_grad_(False)
    tensors = {"raw": features, "direction": direction, "offsets": offsets,
               **{f"parameter:{name}": parameter for name, parameter in model.named_parameters()}}
    before = snapshot(tensors)
    def native(raw):
        return model(raw, offsets, ragged_pool_forward_ad)
    def reference(raw):
        return model(raw, offsets, reference_pool)
    with torch.no_grad():
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(features, direction)
            outputs = native(dual)
            unpacked = [torch.autograd.forward_ad.unpack_dual(output) for output in outputs]
            if any(output.tangent is None for output in unpacked):
                raise AssertionError("The explicit dual route must carry both output tangents")
            dual_primal = tuple(output.primal.clone() for output in unpacked)
            dual_tangent = tuple(output.tangent.clone() for output in unpacked)
        func_primal, func_tangent = torch.func.jvp(native, (features,), (direction,))
        reference_primal, reference_tangent = torch.func.jvp(reference, (features,), (direction,))
        original_primal = model(features, offsets, ragged_pool)
        h = 1e-2 if dtype_name == "float32" else 1e-5
        plus, minus = native(features + h * direction), native(features - h * direction)
        finite_difference = tuple((upper - lower) / (2 * h) for upper, lower in zip(plus, minus))
    names = ("predictions", "pooled")
    named = lambda values: dict(zip(names, values))
    errors = {
        "dual_vs_func_primal": compare(named(dual_primal), named(func_primal), tolerance),
        "dual_vs_func_tangent": compare(named(dual_tangent), named(func_tangent), tolerance),
        "dual_vs_reference_tangent": compare(named(dual_tangent), named(reference_tangent), tolerance),
        "func_vs_reference_primal": compare(named(func_primal), named(reference_primal), tolerance),
        "func_vs_reference_tangent": compare(named(func_tangent), named(reference_tangent), tolerance),
        "func_vs_original_native_primal": compare(named(func_primal), named(original_primal), tolerance),
    }
    fd_tolerance = ({"rtol": 2e-3, "atol": 2e-5} if dtype_name == "float32"
                    else {"rtol": 2e-6, "atol": 2e-9})
    errors["func_vs_centered_finite_difference"] = compare(named(func_tangent), named(finite_difference), fd_tolerance)
    if max(float(value.abs().max()) for value in func_tangent) <= 1e-5:
        raise AssertionError("The declared direction must produce a nontrivial sensitivity")
    for outputs in (dual_primal, dual_tangent, func_primal, func_tangent, reference_primal,
                    reference_tangent, original_primal, finite_difference):
        if not all(bool(torch.isfinite(output).all()) for output in outputs):
            raise AssertionError("Sensitivity comparisons must remain finite")
    # The fixture's second bag is empty. Its pooled value and input sensitivity are zero.
    torch.testing.assert_close(func_primal[1][1], torch.zeros_like(func_primal[1][1]), rtol=0, atol=0)
    torch.testing.assert_close(func_tangent[1][1], torch.zeros_like(func_tangent[1][1]), rtol=0, atol=0)
    torch.testing.assert_close(func_tangent[0][1], torch.zeros_like(func_tangent[0][1]), rtol=0, atol=0)
    forbidden = {value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()}
    output_addresses = [value.untyped_storage().data_ptr() for value in (*func_primal, *func_tangent)]
    if any(address in forbidden for address in output_addresses) or len(output_addresses) != len(set(output_addresses)):
        raise AssertionError("Transformed model outputs and tangents must use independent fresh storage")
    versions = unchanged(tensors, before)
    return {"status": "passed", "routes": ["forward_ad.dual_level", "torch.func.jvp"],
            "direction": "One fixed raw-input direction at the trained fixed parameters",
            "parameters_fixed": True, "optimizer_steps": 0, "finite_difference_step": h,
            "finite_difference_tolerance": fd_tolerance, "errors": errors,
            "dual_max_abs_error": max(errors["dual_vs_reference_tangent"].values()),
            "func_max_abs_error": max(errors["func_vs_reference_tangent"].values()),
            "finite_difference_max_abs_error": max(errors["func_vs_centered_finite_difference"].values()),
            "tangent_max_abs": {name: float(value.abs().max()) for name, value in named(func_tangent).items()},
            "empty_bag_pooled_and_input_tangent_zero": True, "fresh_output_and_tangent_storage": True,
            "inputs_unchanged": True, "input_versions": versions}


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    torch.manual_seed(SEED)
    dtype = DTYPES[dtype_name]
    tolerance = ({"rtol": 2e-5, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 1e-10, "atol": 1e-12})
    features = 0.3 * torch.randn(6, 3, dtype=dtype, device="cpu")
    direction = 0.2 * torch.randn(6, 3, dtype=dtype, device="cpu")
    offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64, device="cpu")
    targets = torch.tensor([-0.4, 0.3, 0.8, -0.2], dtype=dtype, device="cpu")
    # Tensor-to-Python conversion remains outside both transformation routes.
    reference_pool = primitive_pool(tuple(offsets.tolist()))
    inputs = {"raw": features, "direction": direction, "offsets": offsets, "targets": targets}
    before = snapshot(inputs)
    initial = SetRegressor().to(device="cpu", dtype=dtype)
    trained, steps = train(initial, features, offsets, targets, reference_pool, tolerance)
    sensitivity_result = sensitivity(trained, features, offsets, direction, reference_pool, tolerance, dtype_name)
    versions = unchanged(inputs, before)
    return {"status": "passed", "purpose": "Synthetic training and first-order directional sensitivity integration",
            "scope": "Packed eager CPU bridge. No nested transform, model-quality, robustness, privacy or performance result.",
            "method": "ragged_pool_forward_ad", "dtype": dtype_name, "device": "cpu", "seed": SEED,
            "raw_shape": list(features.shape), "group_lengths": [2, 0, 3, 1],
            "architecture": "encoder 3->4, tanh, scorer 4->1 without bias, pool, head 4->1",
            "parameter_tensors": len(tuple(initial.parameters())),
            "scalar_parameters": sum(parameter.numel() for parameter in initial.parameters()),
            "optimizer": "SGD", "learning_rate": 0.03, "objective": "MSE(predictions, targets)",
            "optimizer_steps": 2, "native_optimizer_steps": 2,
            "original_native_optimizer_steps": 2, "reference_optimizer_steps": 2,
            "comparison_tolerance": tolerance, "steps": steps, "sensitivity": sensitivity_result,
            "inputs_unchanged": True, "input_versions": versions, "gradients_finite": True,
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
