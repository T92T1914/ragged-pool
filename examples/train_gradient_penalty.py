"""Train a packed CPU model with a small input-gradient penalty.

The penalty needs derivatives of the first backward pass. A detached penalty
has the same scalar value but contributes no parameter gradient. This example
checks that difference and performs two SGD steps through the installed wheel.
"""
import argparse
import copy
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_reference


SEED = 20261020


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # A shared scalar score bias would cancel within each softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, values, offsets, pool):
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def maximum_difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def forward_backward(model, values, offsets, targets, pool, mode="proper"):
    """Leave parameter gradients ready for SGD and return numerical comparisons."""
    if mode not in ("mse_only", "detached", "proper"):
        raise ValueError("Unknown gradient-penalty mode")
    model.zero_grad(set_to_none=True)
    values.grad = None
    snapshots = {"raw_values": (values.detach().clone(), values._version),
                 "offsets": (offsets.detach().clone(), offsets._version)}
    predictions, pooled = model(values, offsets, pool)
    mse = nn.functional.mse_loss(predictions, targets)
    first_gradient, penalty = None, None
    loss = mse
    if mode != "mse_only":
        first_gradient, = torch.autograd.grad(
            predictions.sum(), values, create_graph=True, retain_graph=True)
        penalty = first_gradient.square().mean()
        # This control keeps the same scalar while deleting its derivative path.
        contribution = penalty.detach() if mode == "detached" else penalty
        loss = mse + 0.5 * contribution
    loss.backward()
    finite = [predictions, pooled, mse, loss]
    if first_gradient is not None:
        finite.extend((first_gradient, penalty))
    if values.grad is None:
        raise AssertionError("The raw inputs must receive a gradient")
    finite.append(values.grad)
    parameter_gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing parameter gradient for {name}")
        finite.append(parameter.grad)
        parameter_gradients[name] = parameter.grad.detach().clone()
    if not all(bool(torch.isfinite(value).all()) for value in finite):
        raise AssertionError("The gradient-penalty integration must remain finite")
    versions = {}
    for name, value in (("raw_values", values), ("offsets", offsets)):
        snapshot, version = snapshots[name]
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        if value._version != version:
            raise AssertionError(f"The gradient-penalty consumer mutated {name}")
        versions[name] = {"before": version, "after": value._version}
    return {
        "predictions": predictions.detach(), "pooled": pooled.detach(),
        "mse": mse.detach(), "loss": loss.detach(),
        "penalty": penalty.detach() if penalty is not None else None,
        "first_input_gradient": first_gradient.detach() if first_gradient is not None else None,
        "total_input_gradient": values.grad.detach().clone(),
        "parameter_gradients": parameter_gradients,
        "inputs_unchanged": True, "input_versions": versions,
    }


def compare_results(native, reference, tolerance):
    errors = {}
    for key in ("pooled", "predictions", "mse", "loss", "penalty",
                "first_input_gradient", "total_input_gradient"):
        torch.testing.assert_close(native[key], reference[key], **tolerance)
        errors[key + "_max_abs_difference"] = maximum_difference(native[key], reference[key])
    parameter_errors = {}
    for name, gradient in native["parameter_gradients"].items():
        other = reference["parameter_gradients"][name]
        torch.testing.assert_close(gradient, other, **tolerance)
        parameter_errors[name] = maximum_difference(gradient, other)
    errors["parameter_gradient_max_abs_differences"] = parameter_errors
    return errors


def check_controls(initial, features, offsets, targets, tolerance):
    """Compare a constant-valued penalty with a penalty that retains its graph."""
    results = {}
    for name in ("mse_only", "detached", "proper"):
        model = copy.deepcopy(initial)
        values = features.clone().requires_grad_()
        results[name] = forward_backward(model, values, offsets, targets, ragged_pool, name)
    reference = forward_backward(
        copy.deepcopy(initial), features.clone().requires_grad_(), offsets,
        targets, ragged_pool_reference)
    base, detached, proper = (results[name] for name in ("mse_only", "detached", "proper"))
    torch.testing.assert_close(detached["penalty"], proper["penalty"], rtol=0, atol=0)
    torch.testing.assert_close(detached["loss"], proper["loss"], rtol=0, atol=0)
    torch.testing.assert_close(base["total_input_gradient"], detached["total_input_gradient"], rtol=0, atol=0)
    detached_errors, proper_differences = {}, {}
    for name, gradient in base["parameter_gradients"].items():
        detached_gradient = detached["parameter_gradients"][name]
        torch.testing.assert_close(gradient, detached_gradient, rtol=0, atol=0)
        detached_errors[name] = maximum_difference(gradient, detached_gradient)
        proper_differences[name] = maximum_difference(gradient, proper["parameter_gradients"][name])
    for name in ("scorer.weight", "head.weight"):
        if proper_differences[name] == 0:
            raise AssertionError(f"The proper penalty must contribute to {name}")
    return {
        "optimizer_steps": 0,
        "controls": ["MSE only", "MSE plus detached penalty", "MSE plus differentiable penalty"],
        "mse_only_loss": float(base["loss"]), "detached_penalty_loss": float(detached["loss"]),
        "proper_penalty_loss": float(proper["loss"]), "penalty": float(proper["penalty"]),
        "detached_and_proper_penalty_values_equal": True,
        "mse_only_vs_detached_parameter_gradient_max_abs_differences": detached_errors,
        "mse_only_vs_detached_total_input_gradient_max_abs_difference": maximum_difference(
            base["total_input_gradient"], detached["total_input_gradient"]),
        "proper_vs_mse_only_parameter_gradient_max_abs_differences": proper_differences,
        "proper_penalty_changes_gradients_for": [name for name, difference in proper_differences.items() if difference != 0],
        "proper_native_vs_reference": compare_results(proper, reference, tolerance),
        "inputs_unchanged": True,
    }


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    prefix = Path(sys.prefix).resolve()
    package_path = Path(package.__file__).resolve(strict=True)
    installed_path = Path(distribution("ragged-pool").locate_file("ragged_pool/__init__.py")).resolve(strict=True)
    if package_path != installed_path or not package_path.is_relative_to(prefix):
        raise AssertionError("This consumer requires the package installed in its Python environment")
    source = Path(__file__).resolve().parents[1]
    if (source / "pyproject.toml").is_file() and prefix.is_relative_to(source):
        raise AssertionError("Use an environment outside the source checkout")

    lengths = [3, 0, 5, 1, 4, 2]
    offsets = torch.tensor([0, 3, 3, 8, 9, 13, 15], dtype=torch.int64, device="cpu")
    features = torch.randn(15, 4, dtype=dtype, device="cpu")
    targets = torch.tensor([0.25, -0.1, 0.75, -0.5, 0.3, 0.1], dtype=dtype, device="cpu")
    initial = SetRegressor().to(device="cpu", dtype=dtype)
    controls = check_controls(initial, features, offsets, targets, tolerance)
    native_model, reference_model = copy.deepcopy(initial), copy.deepcopy(initial)
    native_values = features.clone().requires_grad_()
    reference_values = features.clone().requires_grad_()
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.03)
    reference_parameters = dict(reference_model.named_parameters())
    steps = []
    for step in range(1, 3):
        before = {name: parameter.detach().clone() for name, parameter in native_model.named_parameters()}
        reference_before = {name: parameter.detach().clone() for name, parameter in reference_model.named_parameters()}
        native = forward_backward(native_model, native_values, offsets, targets, ragged_pool)
        reference = forward_backward(reference_model, reference_values, offsets, targets, ragged_pool_reference)
        errors = compare_results(native, reference, tolerance)
        native_optimizer.step()
        reference_optimizer.step()
        native_updates, reference_updates, update_errors = {}, {}, {}
        for name, parameter in native_model.named_parameters():
            other = reference_parameters[name]
            torch.testing.assert_close(parameter, other, **tolerance)
            if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(other).all()):
                raise AssertionError(f"The updated parameter {name} must remain finite")
            native_updates[name] = maximum_difference(parameter, before[name])
            reference_updates[name] = maximum_difference(other, reference_before[name])
            update_errors[name] = maximum_difference(parameter, other)
            if native_updates[name] == 0 or reference_updates[name] == 0:
                raise AssertionError(f"SGD step {step} must change trainable parameter {name} in both models")
        steps.append({"step": step, "native_loss": float(native["loss"]),
                      "reference_loss": float(reference["loss"]),
                      "native_penalty": float(native["penalty"]),
                      "reference_penalty": float(reference["penalty"]),
                      **errors, "native_parameter_update_max_abs": native_updates,
                      "reference_parameter_update_max_abs": reference_updates,
                      "updated_parameter_max_abs_differences": update_errors,
                      "inputs_unchanged": True, "input_versions": native["input_versions"]})
    return {
        "purpose": "Synthetic second-order training integration, not a model quality, robustness or performance study",
        "execution": "Eager packed CPU pooling", "seed": SEED, "dtype": dtype_name, "device": "cpu",
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "package_version": package.__version__, "package_path": str(package_path),
        "python_executable": sys.executable, "environment_prefix": str(prefix),
        "working_directory": str(Path.cwd()), "installed_package": True,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "raw_shape": list(features.shape), "group_lengths": lengths,
        "objective": "MSE(predictions, targets) + 0.5 * mean((d sum(predictions) / d raw_values)^2)",
        "input_gradient_create_graph": True, "penalty_coefficient": 0.5,
        "optimizer": "SGD", "learning_rate": 0.03,
        "optimizer_steps": 2, "reference_optimizer_steps": 2,
        "controls": controls, "steps": steps,
        "pooled_max_abs_difference": max(step["pooled_max_abs_difference"] for step in steps),
        "predictions_max_abs_difference": max(step["predictions_max_abs_difference"] for step in steps),
        "first_input_gradient_max_abs_difference": max(step["first_input_gradient_max_abs_difference"] for step in steps),
        "total_input_gradient_max_abs_difference": max(step["total_input_gradient_max_abs_difference"] for step in steps),
        "parameter_gradient_max_abs_difference": max(max(step["parameter_gradient_max_abs_differences"].values()) for step in steps),
        "updated_parameter_max_abs_difference": max(max(step["updated_parameter_max_abs_differences"].values()) for step in steps),
        "gradients_finite": True, "inputs_unchanged": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
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
