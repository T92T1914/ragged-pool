"""Train with each ragged group's joint parameter gradient clipped before averaging.

The installed CUDA functional bridge supplies serial scalar-loss gradients.
Ordinary full-batch native autograd and primitive GPU math check every row and
two actual SGD updates. There is no noise, privacy accountant, performance or
peak-memory claim. This fixed caller does not add a general training runtime.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import torch


_PATH = Path(__file__).resolve().with_name("train_functional.py")
_SPEC = importlib.util.spec_from_file_location("_group_clipping_functional_helper", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("The adjacent train_functional.py and train_cuda.py helpers are required")
functional = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(functional)
eager = functional.eager
DTYPES, CLIP_NORM = eager.DTYPES, 1.0
BOUNDARIES = ((0, 1, 1, 4, 6, 9), (0, 2, 3, 3, 7, 9, 12))
RESIDUALS = ((0.01, 2.4, -1.7, 0.08, -0.12), (0.04, -1.4, 2.1, -0.8, -0.03, 0.05))
snapshot, unchanged = functional.snapshot, functional.unchanged


def _limit(value):
    if type(value) not in (int, float):
        raise ValueError("The clipping norm must be a finite positive number")
    try:
        limit = float(value)
    except OverflowError as error:
        raise ValueError("The clipping norm must fit a finite positive float") from error
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("The clipping norm must be a finite positive number")
    return limit


def clip_row(gradients, clip_norm):
    """Return new float64 clipped blocks, one joint norm and one scale."""
    limit = _limit(clip_norm)
    if type(gradients) is not dict or not gradients:
        raise ValueError("At least one named gradient block is required")
    first = next(iter(gradients.values()))
    if not isinstance(first, torch.Tensor):
        raise ValueError("Gradient blocks must be CUDA float32/float64 tensors")
    for name, value in gradients.items():
        if (type(name) is not str or not isinstance(value, torch.Tensor)
                or value.device != torch.device("cuda", 0) or value.dtype not in DTYPES.values()
                or value.dtype != first.dtype or value.layout != torch.strided
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Named gradient blocks must be finite matching CUDA0 float32/float64")
    doubles = {name: value.detach().double() for name, value in gradients.items()}
    norm = sum(value.square().sum() for value in doubles.values()).sqrt()
    if not bool(torch.isfinite(norm)):
        raise ValueError("The finite gradient blocks overflowed the float64 joint norm")
    if not bool(norm) and any(bool((value != 0).any()) for value in doubles.values()):
        raise ValueError("The nonzero gradient blocks underflowed the float64 joint norm")
    scale = norm.new_ones(()) if not bool(norm > limit) else norm.new_tensor(limit) / norm
    clipped = {name: value * scale for name, value in doubles.items()}
    if any(not bool(torch.isfinite(value).all()) for value in clipped.values()):
        raise ValueError("The clipped gradient blocks must be finite")
    return clipped, norm, scale


def _admit(model, batch):
    raw, offsets, targets, boundaries = (batch[key] for key in ("raw", "offsets", "targets", "boundaries"))
    if (type(boundaries) is not tuple or not 2 <= len(boundaries) <= 129
            or any(type(value) is not int for value in boundaries)
            or boundaries[0] != 0 or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("Use a bounded declared tuple of ordered group boundaries")
    parameters = dict(model.named_parameters())
    if (len(parameters) != 5 or sum(value.numel() for value in parameters.values()) != 25
            or any(value.device != torch.device("cuda", 0) or value.dtype != raw.dtype
                   or not bool(torch.isfinite(value).all()) for value in parameters.values())):
        raise ValueError("This caller requires the declared finite five-block CUDA0 model")
    for value in (raw, offsets, targets):
        if value.device != torch.device("cuda", 0) or value.layout != torch.strided or not value.is_contiguous():
            raise ValueError("This caller requires contiguous CUDA0 inputs")
    if (raw.dtype not in DTYPES.values() or raw.shape != (boundaries[-1], 3) or boundaries[-1] > 4096
            or offsets.dtype != torch.int64 or offsets.shape != (len(boundaries),)
            or targets.dtype != raw.dtype or targets.shape != (len(boundaries) - 1,)
            or not bool(torch.isfinite(raw).all()) or not bool((raw.abs() <= 6).all())
            or not bool(torch.isfinite(targets).all())
            or not torch.equal(offsets, torch.tensor(boundaries, dtype=torch.int64, device=offsets.device))):
        raise ValueError("Input metadata, raw bounds, targets and declared offsets must agree")
    return parameters


def stream_gradients(model, batch, clip_norm=CLIP_NORM, observer=None):
    """Accumulate one detached functional gradient row at a time, then divide by B.

    The optional acceptance observer receives row, raw named gradients, norm,
    scale, prediction, pooled vector and scalar loss. Only that observer decides
    whether to retain rows. This route retains sums and small output diagnostics.
    """
    _limit(clip_norm)
    parameters = _admit(model, batch)
    tensors = {**parameters, **{key: batch[key] for key in ("raw", "offsets", "targets")}}
    before = snapshot(tensors, gradients=True)
    sums = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in parameters.items()}
    norms, scales, predictions, pooled_rows, losses = [], [], [], [], []
    for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:])):
        offsets = torch.tensor((0, stop - start), dtype=torch.int64, device=batch["offsets"].device)

        def loss(params):
            prediction, pooled = torch.func.functional_call(
                model, params, (batch["raw"][start:stop], offsets), strict=True)
            value = (prediction[0] - batch["targets"][row]).square()
            return value, (prediction, pooled, value)

        gradients, auxiliary = torch.func.grad(loss, has_aux=True)(parameters)
        gradients = {name: value.detach() for name, value in gradients.items()}
        prediction, pooled, value = (item.detach().clone() for item in auxiliary)
        if any(not bool(torch.isfinite(item).all()) for item in (prediction, pooled, value)):
            raise ValueError("Each group's outputs and loss must be finite")
        clipped, norm, scale = clip_row(gradients, clip_norm)
        if observer is not None:
            observer(row, gradients, norm, scale, prediction, pooled, value)
        for name in sums:
            sums[name].add_(clipped[name])
        norms.append(float(norm))
        scales.append(float(scale))
        predictions.append(prediction)
        pooled_rows.append(pooled)
        losses.append(value)
        del gradients, clipped, auxiliary
    count = len(losses)
    means = {name: (value / count).to(parameters[name].dtype) for name, value in sums.items()}
    if any(not bool(torch.isfinite(value).all()) for value in means.values()):
        raise ValueError("The averaged clipped gradients must be finite")
    unchanged(tensors, before, gradients=True)
    return means, {"joint_norms": norms, "scales": scales, "group_count": count,
                   "predictions": torch.cat(predictions), "pooled": torch.cat(pooled_rows),
                   "losses": torch.stack(losses), "inputs_parameters_and_grad_fields_unchanged": True}


def installed_identity():
    identity = functional.installed_identity()
    identity["functional_helper_script_sha256"] = identity.pop("script_sha256")
    identity["cuda_helper_script_sha256"] = identity.pop("helper_script_sha256")
    identity["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


def fixture(dtype_name, device):
    initial, batches, original_hash = eager.fixture(dtype_name, device)
    digest = sha256(original_hash.encode())
    for batch, boundaries, residuals in zip(batches, BOUNDARIES, RESIDUALS):
        batch["boundaries"] = boundaries
        batch["offsets"] = torch.tensor(boundaries, dtype=torch.int64, device=device)
        with torch.no_grad():
            prediction, _ = initial(batch["raw"], batch["offsets"])
            batch["targets"] = (prediction - prediction.new_tensor(residuals)).contiguous()
        for key in ("raw", "offsets", "targets"):
            tensor = batch[key]
            digest.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
            digest.update(functional._bytes(tensor).numpy().tobytes())
    return initial, batches, digest.hexdigest()


def _oracle(model, batch, reference):
    parameters = dict(model.named_parameters())
    tensors = {**parameters, **{key: batch[key] for key in ("raw", "offsets", "targets")}}
    before = snapshot(tensors, gradients=True)
    prediction, pooled = model(batch["raw"], batch["offsets"], batch["boundaries"] if reference else None)
    losses = (prediction - batch["targets"]).square()
    mean = dict(zip(parameters, torch.autograd.grad(losses.mean(), tuple(parameters.values()), retain_graph=True)))
    rows = [dict(zip(parameters, (value.detach().clone() for value in torch.autograd.grad(
            losses[row], tuple(parameters.values()), retain_graph=row + 1 < losses.numel()))))
            for row in range(losses.numel())]
    unchanged(tensors, before, gradients=True)
    return rows, {"predictions": prediction.detach(), "pooled": pooled.detach(), "losses": losses.detach()}, mean


def _aggregate(rows, clip_norm):
    sums = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in rows[0].items()}
    for row in rows:
        clipped, _, _ = clip_row(row, clip_norm)
        for name in sums:
            sums[name].add_(clipped[name])
    return {name: (value / len(rows)).to(rows[0][name].dtype) for name, value in sums.items()}


def analytical_controls(device):
    row = {"a": torch.tensor([3.], dtype=torch.float64, device=device),
           "b": torch.tensor([4.], dtype=torch.float64, device=device)}
    zero = {name: torch.zeros_like(value) for name, value in row.items()}
    clipped, norm, scale = clip_row(row, 1.)
    for name, expected in (("a", .6), ("b", .8)):
        torch.testing.assert_close(clipped[name], row[name].new_tensor([expected]), rtol=0, atol=2e-15)
    zeros, zero_norm, zero_scale = clip_row(zero, 1.)
    if float(norm) != 5 or float(zero_norm) != 0 or float(zero_scale) != 1 or any(bool(x.any()) for x in zeros.values()):
        raise AssertionError("Analytical joint and zero norm controls failed")
    opposite = {name: value * -.1 for name, value in row.items()}
    correct = _aggregate([row, opposite], 1.)
    mean = {name: (row[name] + opposite[name]) / 2 for name in row}
    after, _, _ = clip_row(mean, 1.)
    expected = {"a": row["a"].new_tensor([.15]), "b": row["b"].new_tensor([.2])}
    eager.compare(correct, expected, {"rtol": 0, "atol": 2e-15})
    cancellation_difference = max(eager.difference(correct[name], after[name]) for name in row)
    if cancellation_difference <= .5:
        raise AssertionError("The cancellation control must distinguish clipping after the mean")
    return {"joint_norm": float(norm), "scale": float(scale), "zero_norm": float(zero_norm),
            "zero_scale": float(zero_scale), "cancellation_clipped_means": {name: float(value[0]) for name, value in correct.items()},
            "cancellation_clip_after_mean_max_abs_difference": cancellation_difference}


def _controls(rows, means, mean_gradient, batch, output, tolerance):
    unscaled = {name: torch.stack([row[name].double() for row in rows]).mean(0).to(rows[0][name].dtype) for name in rows[0]}
    normalization_errors = eager.compare(unscaled, mean_gradient, tolerance)
    after, _, _ = clip_row(unscaled, CLIP_NORM)
    blockwise = {name: _aggregate([{name: row[name]} for row in rows], CLIP_NORM)[name] for name in means}
    differences = {"from_unclipped": max(eager.difference(means[name], unscaled[name]) for name in means),
                   "from_clip_after_mean": max(eager.difference(means[name], after[name]) for name in means),
                   "from_blockwise_clip": max(eager.difference(means[name], blockwise[name]) for name in means)}
    if any(value <= tolerance["atol"] for value in differences.values()):
        raise AssertionError("The fixed fixture must distinguish all three clipping controls")
    empty, singleton = [], []
    for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:])):
        if start == stop:
            empty.append(row)
            if any(bool(value.any()) for name, value in rows[row].items() if name != "head.bias"):
                raise AssertionError("Empty groups must have zero non-bias parameter gradients")
            expected = 2 * (output["predictions"][row] - batch["targets"][row])
            torch.testing.assert_close(rows[row]["head.bias"][0], expected, **tolerance)
            if not bool(rows[row]["head.bias"].abs().max() > 0) or bool(output["pooled"][row].any()):
                raise AssertionError("The empty group must keep its head-bias loss contribution")
        if stop - start == 1:
            singleton.append(row)
            if bool(rows[row]["scorer.weight"].any()):
                raise AssertionError("A singleton group's scorer gradient must be zero")
    if not empty or not singleton:
        raise AssertionError("The fixed fixture needs empty and singleton groups")
    return {"mean_loss_parameter_gradient_max_abs_errors": normalization_errors,
            "max_abs_differences": differences, "empty_groups": empty, "singleton_groups": singleton,
            "empty_head_bias_gradients": [float(rows[row]["head.bias"][0]) for row in empty]}


def run(dtype_name, device_index=0, source_revision=None, source_tree=None):
    if dtype_name not in DTYPES or device_index != 0:
        raise ValueError("This fixed consumer supports float32/float64 on CUDA device zero")
    for label in (source_revision, source_tree):
        if label is not None and not re.fullmatch(r"[0-9a-f]{40}", label):
            raise ValueError("Source labels must be complete lowercase SHA1 identifiers")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if sys.version.split()[0] != "3.11.8" or str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise AssertionError("Use the pinned Python 3.11.8 and PyTorch 2.9.1 cu128 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    identity = installed_identity()
    initial, batches, fixture_hash = fixture(dtype_name, device)
    inputs = {f"{index}:{key}": batch[key] for index, batch in enumerate(batches) for key in ("raw", "offsets", "targets")}
    before_inputs, before_initial = snapshot(inputs, gradients=True), snapshot(dict(initial.named_parameters()), gradients=True)
    models = {name: copy.deepcopy(initial) for name in ("native", "primitive")}
    models["functional"] = functional.SetRegressor().to(device=device, dtype=DTYPES[dtype_name])
    models["functional"].load_state_dict(initial.state_dict())
    opts = {name: torch.optim.SGD(model.parameters(), lr=.03, momentum=0, foreach=False, fused=False) for name, model in models.items()}
    ownership = {name: (id(model), id(opts[name]), tuple(id(p) for p in model.parameters()), tuple(p.data_ptr() for p in model.parameters()))
                 for name, model in models.items()}
    if len({pointer for entry in ownership.values() for pointer in entry[3]}) != 15:
        raise AssertionError("Each training route must own all five parameter blocks")
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    steps = []
    for update, batch in enumerate(batches, 1):
        previous = {name: functional._clones(dict(model.named_parameters())) for name, model in models.items()}
        # Nonzero sentinels make accidental .grad accumulation or replacement visible.
        for model in models.values():
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, .125)
        oracles = {name: _oracle(models[name], batch, name == "primitive") for name in ("native", "primitive")}
        row_errors = {name: {key: 0. for key in previous[name]} for name in oracles}

        def observe(row, gradients, norm, scale, prediction, pooled, loss):
            for name, (rows, output, _) in oracles.items():
                errors = eager.compare(gradients, rows[row], tolerance)
                row_errors[name] = {key: max(row_errors[name][key], value) for key, value in errors.items()}
                eager.compare({"prediction": prediction, "pooled": pooled, "loss": loss},
                              {"prediction": output["predictions"][row:row+1], "pooled": output["pooled"][row:row+1],
                               "loss": output["losses"][row]}, tolerance)

        means, diagnostics = stream_gradients(models["functional"], batch, observer=observe)
        if not any(scale < 1 for scale in diagnostics["scales"]) or not any(scale == 1 for scale in diagnostics["scales"]):
            raise AssertionError("The fixed C=1 training fixture needs clipped and unclipped groups")
        expected = {name: _aggregate(oracle[0], CLIP_NORM) for name, oracle in oracles.items()}
        mean_errors = {name: eager.compare(means, values, tolerance) for name, values in expected.items()}
        output_errors = {name: eager.compare({key: diagnostics[key] for key in ("predictions", "pooled", "losses")}, oracle[1], tolerance)
                         for name, oracle in oracles.items()}
        controls = _controls(oracles["native"][0], means, oracles["native"][2], batch,
                             oracles["native"][1], tolerance)
        for name, model in models.items():
            selected = means if name == "functional" else expected[name]
            for key, parameter in model.named_parameters():
                parameter.grad = selected[key].detach().clone()
            opts[name].step()
        states = {name: functional._clones(dict(model.named_parameters())) for name, model in models.items()}
        updates = {name: {key: eager.difference(value, previous[name][key]) for key, value in state.items()} for name, state in states.items()}
        if any(value <= 0 for route in updates.values() for value in route.values()):
            raise AssertionError("Every actual update must change all five parameter blocks")
        for name, model in models.items():
            actual = (id(model), id(opts[name]), tuple(id(p) for p in model.parameters()), tuple(p.data_ptr() for p in model.parameters()))
            if actual != ownership[name] or tuple(id(p) for group in opts[name].param_groups for p in group["params"]) != ownership[name][2]:
                raise AssertionError("Training changed model, parameter or optimizer ownership")
        steps.append({"update": update, "N": batch["raw"].shape[0], "B": diagnostics["group_count"], "offsets": list(batch["boundaries"]),
                      "joint_norms": diagnostics["joint_norms"], "scales": diagnostics["scales"], "per_group_parameter_gradient_max_abs_errors": row_errors,
                      "output_max_abs_errors": output_errors, "clipped_mean_max_abs_errors": mean_errors,
                      "updated_parameter_max_abs_errors": {name: eager.compare(states["functional"], states[name], tolerance) for name in oracles},
                      "parameter_update_max_abs": updates, "controls": controls, "sentinel_grad_fields_preserved_before_handoff": True,
                      "all_parameter_blocks_changed": True, "same_model_parameter_optimizer_objects_and_storage": True})
        unchanged(inputs, before_inputs, gradients=True)
    unchanged(dict(initial.named_parameters()), before_initial, gradients=True)
    analytical = analytical_controls(device)
    if installed_identity() != identity:
        raise AssertionError("Consumer, helper or installed package identity changed")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "eager.serial_torch.func.grad.per_group_joint_clip.cuda_training", "dtype": dtype_name,
            "scope": "Fixed first-order CUDA clipping consumer. No noise, privacy accountant, performance, peak-memory, higher-order, vmap or capture claim.",
            "seed": eager.SEED, "fixture_sha256": fixture_hash, "device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor], "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda, "source_revision": source_revision,
            "source_tree": source_tree, "source_identity_basis": "Caller labels plus separate consumer/helper/module/native byte hashes",
            "clip_norm": CLIP_NORM, "normalization": "global group count including empty groups", "norm": "joint L2 over all five named parameter blocks",
            "arithmetic": "float64 joint norms, clipping and sums; divide once by B and cast once", "optimizer": "SGD", "learning_rate": .03,
            "momentum": 0, "foreach": False, "fused": False, "optimizer_steps": 2, "native_optimizer_steps": 2, "reference_optimizer_steps": 2,
            "parameter_tensors": 5, "scalar_parameters": 25, "steps": steps, "analytical_controls": analytical, "comparison_tolerance": tolerance,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "package_identity_unchanged": True,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    result = run(arguments.dtype, arguments.device, arguments.source_revision, arguments.source_tree)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
