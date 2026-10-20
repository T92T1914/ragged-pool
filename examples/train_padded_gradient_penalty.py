"""Take two eager CPU SGD steps through a padding-invariant gradient penalty.

The chosen penalty averages over selected raw features. Keeping the same items
and targets while adding ignored padding must preserve the objective and its
gradients. A separate dense-mean control illustrates the denominator choice.
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
from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


SEED = 20261020
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class PaddedRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A shared scalar score bias would cancel inside each softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, mask, pool):
        clean = torch.where(mask[..., None], raw, 0)
        encoded = torch.tanh(self.encoder(clean))
        logits = self.scorer(encoded).squeeze(-1).contiguous()
        pooled = pool(encoded, logits, mask)
        return self.head(pooled).squeeze(-1), pooled


def maximum_difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def selected_gradient_penalty(first_gradient, mask):
    """Average selected squared derivatives, with a connected zero if empty."""
    selected = first_gradient[mask]
    if selected.numel() == 0:
        return selected.sum() * 0
    return selected.square().sum() / selected.numel()


def make_layouts(dtype_name):
    """Keep per-row selected data and order fixed across four padding layouts."""
    torch.manual_seed(SEED)
    dtype = DTYPES[dtype_name]
    dense = torch.randn(5, 6, 3, dtype=dtype, device="cpu")
    mask = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1]], dtype=torch.bool, device="cpu")

    def layout(raw, selected, changed=False):
        raw, selected = raw.clone().contiguous(), selected.clone().contiguous()
        raw[~selected] = 1e30 if changed else float("nan")
        padding = (~selected).nonzero()
        first, last = padding[0].tolist(), padding[-1].tolist()
        raw[first[0], first[1], 0] = -float("inf") if changed else float("inf")
        raw[last[0], last[1], 2] = float("nan") if changed else -float("inf")
        return {"raw": raw, "mask": selected}

    moved_mask = torch.roll(mask, 1, dims=1).contiguous()
    moved = torch.zeros_like(dense)
    for row in range(mask.shape[0]):
        moved[row][moved_mask[row]] = dense[row][mask[row]]
    doubled = torch.zeros(5, 12, 3, dtype=dtype, device="cpu")
    doubled_mask = torch.zeros(5, 12, dtype=torch.bool, device="cpu")
    doubled[:, :6], doubled_mask[:, :6] = dense, mask
    layouts = {"fixed_holes": layout(dense, mask), "moved_holes": layout(moved, moved_mask),
               "doubled_width": layout(doubled, doubled_mask),
               "changed_ignored": layout(dense, mask, changed=True)}
    for variant in layouts.values():
        for row in range(mask.shape[0]):
            torch.testing.assert_close(variant["raw"][row][variant["mask"][row]], dense[row][mask[row]], rtol=0, atol=0)
    return layouts


def _snapshot(raw, mask):
    return [(value.detach().view(torch.uint8).clone(), value._version) for value in (raw, mask)]


def _check_unchanged(raw, mask, snapshot):
    versions = {}
    for name, value, (original, version) in zip(("raw", "mask"), (raw, mask), snapshot):
        if value._version != version or not torch.equal(value.detach().view(torch.uint8), original):
            raise AssertionError(f"The gradient-penalty consumer changed {name} bytes or its version")
        versions[name] = {"before": version, "after": value._version}
    return versions


def forward_backward(model, raw, mask, targets, pool, mode="proper"):
    """Leave gradients ready for SGD and retain detached numerical witnesses."""
    if mode not in ("proper", "detached", "mse_only"):
        raise ValueError("Unknown padded gradient-penalty mode")
    model.zero_grad(set_to_none=True)
    raw.grad = None
    snapshot = _snapshot(raw, mask)
    predictions, pooled = model(raw, mask, pool)
    mse = nn.functional.mse_loss(predictions, targets)
    first, penalty = None, None
    loss = mse
    if mode != "mse_only":
        first, = torch.autograd.grad(predictions.sum(), raw, create_graph=True, retain_graph=True)
        penalty = selected_gradient_penalty(first, mask)
        if not penalty.requires_grad:
            raise AssertionError("The selected penalty, including an empty zero, must retain its graph")
        loss = mse + 0.5 * (penalty.detach() if mode == "detached" else penalty)
    loss.backward()
    finite = [predictions, pooled, mse, loss]
    if raw.grad is None:
        raise AssertionError("Raw inputs must receive a total gradient")
    finite.append(raw.grad)
    if bool(torch.count_nonzero(raw.grad[~mask])):
        raise AssertionError("Ignored raw features must receive zero total gradients")
    if first is not None:
        finite.extend((first, penalty))
        if bool(torch.count_nonzero(first[~mask])):
            raise AssertionError("Ignored raw features must receive zero first gradients")
    parameters = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing parameter gradient for {name}")
        finite.append(parameter.grad)
        parameters[name] = parameter.grad.detach().clone()
    if not all(bool(torch.isfinite(value).all()) for value in finite):
        raise AssertionError("Padded predictions, objectives and gradients must remain finite")
    if bool(torch.count_nonzero(pooled[~mask.any(dim=1)])):
        raise AssertionError("Empty rows must have exactly zero pooled features")
    versions = _check_unchanged(raw, mask, snapshot)
    return {"predictions": predictions.detach().clone(), "pooled": pooled.detach().clone(),
        "mse": mse.detach(), "loss": loss.detach(),
        "penalty": penalty.detach() if penalty is not None else None,
        "penalty_requires_grad": penalty.requires_grad if penalty is not None else False,
        "first_raw_gradient": first.detach().clone() if first is not None else None,
        "total_raw_gradient": raw.grad.detach().clone(),
        "selected_first_gradient": first[mask].detach().clone() if first is not None else None,
        "selected_total_gradient": raw.grad[mask].detach().clone(),
        "parameter_gradients": parameters, "selected_items": int(mask.sum()),
        "selected_raw_features": int(mask.sum()) * raw.shape[-1],
        "inputs_unchanged": True, "input_versions": versions}


def compare_results(left, right, tolerance, *, selected_only=False):
    keys = ("predictions", "pooled", "mse", "penalty", "loss")
    gradient_keys = (("selected_first_gradient", "selected_total_gradient") if selected_only
                     else ("first_raw_gradient", "total_raw_gradient"))
    errors = {}
    for key in keys + gradient_keys:
        torch.testing.assert_close(left[key], right[key], **tolerance)
        errors[key + "_max_abs_difference"] = maximum_difference(left[key], right[key])
    parameter_errors = {}
    for name, value in left["parameter_gradients"].items():
        other = right["parameter_gradients"][name]
        torch.testing.assert_close(value, other, **tolerance)
        parameter_errors[name] = maximum_difference(value, other)
    errors["parameter_gradient_max_abs_differences"] = parameter_errors
    return errors


def trajectory(initial, layout, targets, tolerance):
    native_model, reference_model = copy.deepcopy(initial), copy.deepcopy(initial)
    native_raw = layout["raw"].clone().requires_grad_()
    reference_raw = layout["raw"].clone().requires_grad_()
    mask = layout["mask"]
    optimizer = torch.optim.SGD(native_model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.03)
    history = []
    for step in range(1, 3):
        before = {name: value.detach().clone() for name, value in native_model.named_parameters()}
        reference_before = {name: value.detach().clone() for name, value in reference_model.named_parameters()}
        native = forward_backward(native_model, native_raw, mask, targets, ragged_pool_padded)
        reference = forward_backward(reference_model, reference_raw, mask, targets, ragged_pool_padded_reference)
        errors = compare_results(native, reference, tolerance)
        optimizer.step()
        reference_optimizer.step()
        native_state = {name: value.detach().clone() for name, value in native_model.named_parameters()}
        reference_state = {name: value.detach().clone() for name, value in reference_model.named_parameters()}
        updates, reference_updates, update_errors = {}, {}, {}
        for name, value in native_state.items():
            other = reference_state[name]
            if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(other).all()):
                raise AssertionError("Updated padded parameters must remain finite")
            torch.testing.assert_close(value, other, **tolerance)
            updates[name], reference_updates[name] = maximum_difference(value, before[name]), maximum_difference(other, reference_before[name])
            if updates[name] == 0 or reference_updates[name] == 0:
                raise AssertionError(f"Both models must update {name} at nonempty training step {step}")
            update_errors[name] = maximum_difference(value, other)
        metrics = {"step": step, "native_loss": float(native["loss"]), "reference_loss": float(reference["loss"]),
            "native_penalty": float(native["penalty"]), "reference_penalty": float(reference["penalty"]),
            **errors, "native_parameter_update_max_abs": updates,
            "reference_parameter_update_max_abs": reference_updates,
            "updated_parameter_max_abs_differences": update_errors,
            "inputs_unchanged": True, "input_versions": native["input_versions"]}
        history.append({"native": native, "reference": reference, "native_state": native_state,
                        "reference_state": reference_state, "metrics": metrics})
    return history


def check_controls(initial, layouts, targets, tolerance):
    fixed = layouts["fixed_holes"]
    results = {mode: forward_backward(copy.deepcopy(initial), fixed["raw"].clone().requires_grad_(),
                fixed["mask"], targets, ragged_pool_padded, mode) for mode in ("mse_only", "detached", "proper")}
    base, detached, proper = (results[mode] for mode in ("mse_only", "detached", "proper"))
    reference = forward_backward(copy.deepcopy(initial), fixed["raw"].clone().requires_grad_(),
                                 fixed["mask"], targets, ragged_pool_padded_reference)
    torch.testing.assert_close(detached["penalty"], proper["penalty"], rtol=0, atol=0)
    torch.testing.assert_close(detached["loss"], proper["loss"], rtol=0, atol=0)
    torch.testing.assert_close(base["total_raw_gradient"], detached["total_raw_gradient"], rtol=0, atol=0)
    detached_errors, proper_differences = {}, {}
    for name, gradient in base["parameter_gradients"].items():
        torch.testing.assert_close(gradient, detached["parameter_gradients"][name], rtol=0, atol=0)
        detached_errors[name] = maximum_difference(gradient, detached["parameter_gradients"][name])
        proper_differences[name] = maximum_difference(gradient, proper["parameter_gradients"][name])
    for name in ("scorer.weight", "head.weight"):
        if proper_differences[name] == 0:
            raise AssertionError(f"The proper padded penalty must contribute to {name}")
    doubled = layouts["doubled_width"]
    doubled_result = forward_backward(copy.deepcopy(initial), doubled["raw"].clone().requires_grad_(),
                                      doubled["mask"], targets, ragged_pool_padded)
    # This control averages real and ignored slots together. It is not optimized.
    dense = proper["first_raw_gradient"].square().mean()
    doubled_dense = doubled_result["first_raw_gradient"].square().mean()
    if float(dense) <= 0:
        raise AssertionError("The dense normalization control needs a nonzero penalty")
    torch.testing.assert_close(doubled_dense, dense / 2, **tolerance)
    torch.testing.assert_close(doubled_result["penalty"], proper["penalty"], **tolerance)
    return {"status": "passed", "optimizer_steps": 0,
        "detached_and_proper_penalty_values_equal": True,
        "mse_only_vs_detached_parameter_gradient_max_abs_differences": detached_errors,
        "mse_only_vs_detached_total_raw_gradient_max_abs_difference": maximum_difference(base["total_raw_gradient"], detached["total_raw_gradient"]),
        "proper_vs_mse_parameter_gradient_max_abs_differences": proper_differences,
        "proper_penalty_changes_gradients_for": [name for name, error in proper_differences.items() if error != 0],
        "proper_native_vs_reference": compare_results(proper, reference, tolerance),
        "dense_mean": {"status": "passed", "base_t": 6, "doubled_t": 12,
            "base_penalty": float(dense), "doubled_penalty": float(doubled_dense),
            "doubled_over_base": float(doubled_dense / dense),
            "selected_normalization_max_abs_difference": maximum_difference(proper["penalty"], doubled_result["penalty"]),
            "interpretation": "A mean over all raw slots is padding-sensitive for this chosen objective. It is not universally invalid."}}


def check_all_empty(initial, dtype, targets, tolerance):
    raw = torch.full((5, 6, 3), float("nan"), dtype=dtype, device="cpu")
    raw[0, 0, 0], raw[-1, -1, -1], raw[1, 1, 1] = float("inf"), -float("inf"), 1e30
    mask = torch.zeros(5, 6, dtype=torch.bool, device="cpu")
    native = forward_backward(copy.deepcopy(initial), raw.clone().requires_grad_(), mask, targets, ragged_pool_padded)
    reference = forward_backward(copy.deepcopy(initial), raw.clone().requires_grad_(), mask, targets, ragged_pool_padded_reference)
    for result in (native, reference):
        if (float(result["penalty"]) != 0 or not result["penalty_requires_grad"]
                or bool(torch.count_nonzero(result["first_raw_gradient"]))
                or bool(torch.count_nonzero(result["total_raw_gradient"]))):
            raise AssertionError("The all-empty case needs a differentiable zero penalty and zero raw gradients")
        for name, gradient in result["parameter_gradients"].items():
            if name != "head.bias" and bool(torch.count_nonzero(gradient)):
                raise AssertionError("Only the prediction bias can receive an all-empty MSE gradient")
    return {"status": "passed", "optimizer_steps": 0, "selected_items": 0,
        "selected_raw_features": 0, "penalty": 0.0, "penalty_requires_grad": True,
        "first_and_total_raw_gradients_zero": True, "inputs_unchanged": True,
        "input_versions": native["input_versions"], "native_vs_reference": compare_results(native, reference, tolerance)}


def installed_identity():
    installed = distribution("ragged-pool")
    direct_url = installed.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed wheel")
    prefix, imported = Path(sys.prefix).resolve(), Path(package.__file__).resolve(strict=True)
    expected = Path(installed.locate_file("ragged_pool/__init__.py")).resolve(strict=True)
    if imported != expected or not imported.is_relative_to(prefix) or installed.version != package.__version__:
        raise AssertionError("Imported package differs from this environment's installed distribution")
    source = Path(__file__).resolve().parents[1]
    if (source / "pyproject.toml").is_file() and prefix.is_relative_to(source):
        raise AssertionError("Use an environment outside the checkout")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if len(libraries) != 1 or libraries[0].resolve() not in recorded or libraries[0].resolve() not in loaded:
        raise AssertionError("Expected one recorded and loaded installed native library")
    return {"package_path": str(imported), "package_version": package.__version__,
        "environment_prefix": str(prefix), "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest()}


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    dtype = DTYPES[dtype_name]
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    layouts = make_layouts(dtype_name)
    initial = PaddedRegressor().to(device="cpu", dtype=dtype)
    targets = torch.tensor([-0.3, 0.8, -0.5, 0.25, -0.1], dtype=dtype, device="cpu")
    controls = check_controls(initial, layouts, targets, tolerance)
    histories = {name: trajectory(initial, layout, targets, tolerance) for name, layout in layouts.items()}
    baseline = histories["fixed_holes"]
    invariance = []
    for name, history in histories.items():
        if name == "fixed_holes":
            continue
        comparisons = []
        for number, (actual, expected) in enumerate(zip(history, baseline), 1):
            paths = {}
            for path in ("native", "reference"):
                errors = compare_results(actual[path], expected[path], tolerance, selected_only=True)
                state_errors = {}
                for parameter, value in actual[path + "_state"].items():
                    other = expected[path + "_state"][parameter]
                    torch.testing.assert_close(value, other, **tolerance)
                    state_errors[parameter] = maximum_difference(value, other)
                paths[path] = {**errors, "updated_parameter_max_abs_differences": state_errors}
            comparisons.append({"step": number, "paths": paths})
        invariance.append({"layout": name, "raw_shape": list(layouts[name]["raw"].shape), "steps": comparisons})
    all_empty = check_all_empty(initial, dtype, targets, tolerance)
    all_metrics = [step["metrics"] for history in histories.values() for step in history]
    return {"status": "passed", "purpose": "Synthetic eager second-order integration, not a model-quality, robustness or speed study",
        "seed": SEED, "dtype": dtype_name, "device": "cpu", "normalization": "selected_raw_features",
        "objective": "MSE + 0.5 * sum(selected input_gradient**2) / selected_raw_features",
        "empty_penalty": "Connected zero from the empty selected gradient sum",
        "optimizer": "SGD", "learning_rate": 0.03, "optimizer_steps": 2, "reference_optimizer_steps": 2,
        "selected_items": int(layouts["fixed_holes"]["mask"].sum()),
        "selected_raw_features": int(layouts["fixed_holes"]["mask"].sum()) * 3,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "python_executable": sys.executable, "working_directory": str(Path.cwd()),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), **identity,
        "controls": controls, "all_empty": all_empty,
        "invariance": {"status": "passed", "baseline": "fixed_holes", "batch_size": 5,
            "same_targets_initial_state_and_selected_order": True, "optimizer_steps_per_layout": 2,
            "comparisons": invariance},
        "steps": [step["metrics"] for step in baseline],
        "prediction_max_abs_difference": max(step["predictions_max_abs_difference"] for step in all_metrics),
        "pooled_max_abs_difference": max(step["pooled_max_abs_difference"] for step in all_metrics),
        "first_raw_gradient_max_abs_difference": max(step["first_raw_gradient_max_abs_difference"] for step in all_metrics),
        "total_raw_gradient_max_abs_difference": max(step["total_raw_gradient_max_abs_difference"] for step in all_metrics),
        "parameter_gradient_max_abs_difference": max(max(step["parameter_gradient_max_abs_differences"].values()) for step in all_metrics),
        "updated_parameter_max_abs_difference": max(max(step["updated_parameter_max_abs_differences"].values()) for step in all_metrics),
        "gradients_finite": True, "zero_ignored_raw_gradients": True, "inputs_unchanged": True}


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
