"""Take two SGD steps with masked padded inputs through an installed CPU wheel.

Padding is cleaned before the encoder. Removing bad padding only at pooling can
leave a finite prediction while earlier layers acquire nonfinite gradients.
The optional compiled route uses aot_eager and does not measure performance.
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
from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


SEED = 20261020


class PaddedRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A shared scalar score bias would cancel inside the softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, mask, pool, *, clean_padding=True):
        # Clean before either linear layer can multiply a padded NaN or Inf.
        values = torch.where(mask[..., None], raw, 0) if clean_padding else raw
        encoded = torch.tanh(self.encoder(values))
        logits = self.scorer(encoded).squeeze(-1).contiguous()
        pooled = pool(encoded, logits, mask)
        return self.head(pooled).squeeze(-1), pooled


def maximum_difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def debug_counterexample(model, raw, mask, targets):
    """Show why ignoring padding in the pool is too late for upstream layers."""
    unsafe = copy.deepcopy(model)
    unsafe_raw = raw.detach().clone().requires_grad_()
    predictions, _ = unsafe(unsafe_raw, mask, ragged_pool_padded, clean_padding=False)
    if not bool(torch.isfinite(predictions).all()):
        raise AssertionError("The debugging case must first produce finite predictions")
    nn.functional.mse_loss(predictions, targets).backward()
    nonfinite = [name for name, parameter in unsafe.named_parameters()
                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    if not nonfinite:
        raise AssertionError("Expected the unsafe encoder path to expose nonfinite parameter gradients")
    return {
        "status": "observed_expected_nonfinite_gradient",
        "explanation": "Pooling discards padding in forward, but linear backward still multiplies zero gradients by padded NaNs. Clean raw padding before the encoder.",
        "forward_finite": True,
        "nonfinite_parameter_gradients": nonfinite,
        "optimizer_steps": 0,
    }


def compare_compiled_gradients(model, reference, values, reference_values, mask, tolerance):
    """Compare first derivatives, including ignored padding and every parameter."""
    for raw in (values, reference_values):
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise AssertionError("The compiled comparison needs finite raw input gradients")
        if bool(torch.count_nonzero(raw.grad[~mask])):
            raise AssertionError("Ignored raw padding must receive exactly zero gradients")
    torch.testing.assert_close(values.grad, reference_values.grad, **tolerance)
    reference_parameters = dict(reference.named_parameters())
    errors = {}
    for name, parameter in model.named_parameters():
        other = reference_parameters[name]
        if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in (parameter, other)):
            raise AssertionError(f"Missing or nonfinite compiled comparison gradient for {name}")
        torch.testing.assert_close(parameter.grad, other.grad, **tolerance)
        errors[name] = maximum_difference(parameter.grad, other.grad)
    return {"input_gradient_max_abs_difference": maximum_difference(values.grad, reference_values.grad),
            "parameter_gradient_max_abs_differences": errors}


def check_compiled_masks(compiled_call, model, reference, dtype, tolerance):
    """Reuse one compiled model while changing mask contents at fixed shapes."""
    positions = torch.arange(30, device="cpu").reshape(5, 6)
    masks = (
        ("all_false", torch.zeros(5, 6, dtype=torch.bool, device="cpu")),
        ("alternating_holes", (positions % 2 == 0).contiguous()),
        ("singletons", (torch.arange(6, device="cpu")[None, :] == torch.arange(5, device="cpu")[:, None]).contiguous()),
        ("all_true", torch.ones(5, 6, dtype=torch.bool, device="cpu")),
    )
    finite_raw = torch.linspace(-0.9, 0.9, 90, dtype=dtype, device="cpu").reshape(5, 6, 3)
    targets = torch.tensor([-0.3, 0.8, -0.5, 0.25, -0.1], dtype=dtype, device="cpu")
    cases = []
    for name, mask in masks:
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        padded_raw = torch.where(mask[..., None], finite_raw, float("nan"))
        values = padded_raw.clone().requires_grad_()
        reference_values = padded_raw.clone().requires_grad_()
        inputs = (values, reference_values, mask)
        snapshots = [(value.detach().clone(), value._version) for value in inputs]
        predictions, pooled = compiled_call(values, mask, ragged_pool_padded)
        reference_predictions, reference_pooled = reference(reference_values, mask, ragged_pool_padded_reference)
        for value in (predictions, pooled, reference_predictions, reference_pooled):
            if not bool(torch.isfinite(value).all()):
                raise AssertionError(f"The {name} mask comparison returned a nonfinite result")
        torch.testing.assert_close(pooled, reference_pooled, **tolerance)
        torch.testing.assert_close(predictions, reference_predictions, **tolerance)
        empty = ~mask.any(dim=1)
        torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
        nn.functional.mse_loss(predictions, targets).backward()
        nn.functional.mse_loss(reference_predictions, targets).backward()
        gradients = compare_compiled_gradients(model, reference, values, reference_values, mask, tolerance)
        for value, (snapshot, version) in zip(inputs, snapshots):
            torch.testing.assert_close(value, snapshot, rtol=0, atol=0, equal_nan=True)
            if value._version != version:
                raise AssertionError(f"The {name} comparison mutated an input")
        cases.append({"name": name, "selected_items_per_row": mask.sum(dim=1).tolist(),
                      "selected_items": int(mask.sum()), "optimizer_steps": 0,
                      "pooled_max_abs_difference": maximum_difference(pooled, reference_pooled),
                      "prediction_max_abs_difference": maximum_difference(predictions, reference_predictions),
                      **gradients, "padding_raw_gradients_zero": True, "inputs_unchanged": True})
    return cases


def run(dtype_name, compiled=False):
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

    mask = torch.tensor([
        [True, False, True, False, False, True],
        [False, True, False, True, False, False],
        [False, False, False, False, False, False],
        [True, True, False, False, True, False],
        [False, False, True, False, True, True],
    ], dtype=torch.bool, device="cpu")
    raw = torch.randn(5, 6, 3, dtype=dtype, device="cpu")
    raw[~mask] = float("nan")
    raw[0, 1, 0], raw[1, 0, 1] = float("inf"), -float("inf")
    targets = torch.tensor([-0.3, 0.8, -0.5, 0.25, -0.1], dtype=dtype, device="cpu")
    native_values = raw.clone().requires_grad_()
    reference_values = raw.clone().requires_grad_()
    inputs = {"native_raw": native_values, "reference_raw": reference_values, "mask": mask}
    if compiled:
        compiled_values = raw.clone().requires_grad_()
        inputs["compiled_raw"] = compiled_values
    snapshots = {name: (value.detach().clone(), value._version) for name, value in inputs.items()}
    native_model = PaddedRegressor().to(device="cpu", dtype=dtype)
    reference_model = copy.deepcopy(native_model)
    if compiled:
        compiled_model = copy.deepcopy(native_model)
        compiled_call = torch.compile(compiled_model, backend="aot_eager", fullgraph=True, dynamic=False)
        compiled_optimizer = torch.optim.SGD(compiled_model.parameters(), lr=0.05)
    negative_control = debug_counterexample(native_model, raw, mask, targets)
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    reference_parameters = dict(reference_model.named_parameters())
    steps = []

    for step in range(1, 3):
        native_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        native_values.grad = reference_values.grad = None
        if compiled:
            compiled_optimizer.zero_grad(set_to_none=True)
            compiled_values.grad = None
            compiled_before = {name: parameter.detach().clone() for name, parameter in compiled_model.named_parameters()}
        before = {name: parameter.detach().clone() for name, parameter in native_model.named_parameters()}
        predictions, pooled = native_model(native_values, mask, ragged_pool_padded)
        reference_predictions, reference_pooled = reference_model(reference_values, mask, ragged_pool_padded_reference)
        if compiled:
            compiled_predictions, compiled_pooled = compiled_call(compiled_values, mask, ragged_pool_padded)
            for value in (compiled_predictions, compiled_pooled):
                if not bool(torch.isfinite(value).all()):
                    raise AssertionError("The compiled clean forward path must remain finite")
            torch.testing.assert_close(compiled_pooled, reference_pooled, **tolerance)
            torch.testing.assert_close(compiled_predictions, reference_predictions, **tolerance)
        for value in (predictions, pooled, reference_predictions, reference_pooled):
            if not bool(torch.isfinite(value).all()):
                raise AssertionError("Cleaning padding must keep the ordinary forward path finite")
        torch.testing.assert_close(pooled, reference_pooled, **tolerance)
        torch.testing.assert_close(predictions, reference_predictions, **tolerance)
        empty = ~mask.any(dim=1)
        torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
        nn.functional.mse_loss(predictions, targets).backward()
        nn.functional.mse_loss(reference_predictions, targets).backward()
        if compiled:
            nn.functional.mse_loss(compiled_predictions, targets).backward()
            compiled_gradients = compare_compiled_gradients(
                compiled_model, reference_model, compiled_values, reference_values, mask, tolerance)
        for values in (native_values, reference_values):
            if values.grad is None or not bool(torch.isfinite(values.grad).all()):
                raise AssertionError("The cleaned raw input gradients must all be finite")
            if bool(torch.count_nonzero(values.grad[~mask])):
                raise AssertionError("Ignored raw padding must receive exactly zero gradients")
        torch.testing.assert_close(native_values.grad, reference_values.grad, **tolerance)
        gradient_errors = {}
        for name, parameter in native_model.named_parameters():
            other = reference_parameters[name]
            if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in (parameter, other)):
                raise AssertionError(f"Missing or nonfinite clean parameter gradient for {name}")
            torch.testing.assert_close(parameter.grad, other.grad, **tolerance)
            gradient_errors[name] = maximum_difference(parameter.grad, other.grad)
        native_optimizer.step()
        reference_optimizer.step()
        if compiled:
            compiled_optimizer.step()
        updates, update_errors = {}, {}
        for name, parameter in native_model.named_parameters():
            torch.testing.assert_close(parameter, reference_parameters[name], **tolerance)
            updates[name] = maximum_difference(parameter, before[name])
            update_errors[name] = maximum_difference(parameter, reference_parameters[name])
            if updates[name] == 0:
                raise AssertionError(f"SGD step {step} must update trainable parameter {name}")
        steps.append({
            "step": step,
            "pooled_max_abs_difference": maximum_difference(pooled, reference_pooled),
            "prediction_max_abs_difference": maximum_difference(predictions, reference_predictions),
            "input_gradient_max_abs_difference": maximum_difference(native_values.grad, reference_values.grad),
            "parameter_gradient_max_abs_differences": gradient_errors,
            "native_parameter_update_max_abs": updates,
            "updated_parameter_max_abs_differences": update_errors,
        })
        if compiled:
            compiled_updates, compiled_update_errors = {}, {}
            for name, parameter in compiled_model.named_parameters():
                torch.testing.assert_close(parameter, reference_parameters[name], **tolerance)
                compiled_updates[name] = maximum_difference(parameter, compiled_before[name])
                compiled_update_errors[name] = maximum_difference(parameter, reference_parameters[name])
                if compiled_updates[name] == 0:
                    raise AssertionError(f"Compiled SGD step {step} must update trainable parameter {name}")
            steps[-1]["compiled"] = {
                "pooled_max_abs_difference": maximum_difference(compiled_pooled, reference_pooled),
                "prediction_max_abs_difference": maximum_difference(compiled_predictions, reference_predictions),
                **compiled_gradients, "parameter_update_max_abs": compiled_updates,
                "updated_parameter_max_abs_differences": compiled_update_errors,
            }

    versions = {}
    for name, value in inputs.items():
        snapshot, version = snapshots[name]
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0, equal_nan=True)
        if value._version != version:
            raise AssertionError(f"The consumer mutated {name}")
        versions[name] = {"before": version, "after": value._version}
    receipt = {
        "purpose": "Synthetic padded-input training integration, not a model quality or performance study",
        "seed": SEED, "dtype": dtype_name, "device": "cpu",
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "package_version": package.__version__, "package_path": str(package_path),
        "python_executable": sys.executable, "environment_prefix": str(prefix),
        "working_directory": str(Path.cwd()), "installed_package": True,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "raw_shape": list(raw.shape), "selected_items_per_row": mask.sum(dim=1).tolist(),
        "optimizer": "SGD", "learning_rate": 0.05, "optimizer_steps": 2,
        "reference_optimizer_steps": 2,
        "pooled_max_abs_difference": max(step["pooled_max_abs_difference"] for step in steps),
        "prediction_max_abs_difference": max(step["prediction_max_abs_difference"] for step in steps),
        "input_gradient_max_abs_difference": max(step["input_gradient_max_abs_difference"] for step in steps),
        "parameter_gradient_max_abs_difference": max(max(step["parameter_gradient_max_abs_differences"].values()) for step in steps),
        "updated_parameter_max_abs_difference": max(max(step["updated_parameter_max_abs_differences"].values()) for step in steps),
        "padding_raw_gradients_zero": True, "clean_parameter_gradients_finite": True,
        "inputs_unchanged": True, "input_versions": versions,
        "steps": steps, "negative_control": negative_control,
    }
    if compiled:
        receipt["compiled"] = {
            "backend": "aot_eager", "fullgraph": True, "dynamic": False,
            "optimizer_steps": 2, "same_callable_for_training_and_mask_checks": True,
            "comparisons": "Compiled native model against the independent eager reference",
            "pooled_max_abs_difference": max(step["compiled"]["pooled_max_abs_difference"] for step in steps),
            "prediction_max_abs_difference": max(step["compiled"]["prediction_max_abs_difference"] for step in steps),
            "input_gradient_max_abs_difference": max(step["compiled"]["input_gradient_max_abs_difference"] for step in steps),
            "parameter_gradient_max_abs_difference": max(max(step["compiled"]["parameter_gradient_max_abs_differences"].values()) for step in steps),
            "updated_parameter_max_abs_difference": max(max(step["compiled"]["updated_parameter_max_abs_differences"].values()) for step in steps),
            "mask_checks": check_compiled_masks(compiled_call, compiled_model, reference_model, dtype, tolerance),
        }
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--compiled", action="store_true", help="Also check a fullgraph aot_eager training path")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt file")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    text = json.dumps(run(arguments.dtype, compiled=arguments.compiled), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
