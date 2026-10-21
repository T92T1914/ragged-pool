"""Train a tiny CPU model by clipping each example's joint parameter gradient.

The experimental batched VJP route is checked against serial native and per-row
reference gradients. This example adds no noise or privacy accountant.
"""
import argparse
import copy
from hashlib import sha256
from importlib.metadata import distribution
import json
import math
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
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, mask, pool):
        if raw.shape[0] < 1:
            raise ValueError("This caller requires at least one example")
        # Mask before the encoder can multiply ignored NaN/Inf by zero.
        encoded = torch.tanh(self.encoder(torch.where(mask[..., None], raw, 0)))
        logits = self.scorer(encoded).squeeze(-1).contiguous()
        pooled = pool(encoded, logits, mask)
        return self.head(pooled).squeeze(-1), pooled


def make_layouts(dtype_name):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    raw = torch.randn((5, 6, 3), generator=generator, dtype=DTYPES[dtype_name], device="cpu")
    mask = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0], [0]*6,
                         [1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]], dtype=torch.bool, device="cpu")

    def padded(values, selected, changed=False):
        values = values.clone().contiguous()
        values[~selected] = 1e30 if changed else float("nan")
        positions = (~selected).nonzero()
        first, last = positions[0].tolist(), positions[-1].tolist()
        values[first[0], first[1], 0] = -float("inf") if changed else float("inf")
        values[last[0], last[1], 2] = float("nan") if changed else -float("inf")
        return {"raw": values, "mask": selected.clone().contiguous()}

    moved_mask, moved = torch.roll(mask, 1, dims=1), torch.zeros_like(raw)
    for row in range(5):
        moved[row][moved_mask[row]] = raw[row][mask[row]]
    wider, wider_mask = raw.new_zeros((5, 12, 3)), torch.zeros((5, 12), dtype=torch.bool, device="cpu")
    wider[:, :6], wider_mask[:, :6] = raw, mask
    layouts = {"fixed_holes": padded(raw, mask), "moved_holes": padded(moved, moved_mask),
               "doubled_width": padded(wider, wider_mask), "changed_ignored": padded(raw, mask, True)}
    for layout in layouts.values():
        for row in range(5):
            torch.testing.assert_close(layout["raw"][row][layout["mask"][row]], raw[row][mask[row]], rtol=0, atol=0)
    return layouts


def _snapshots(inputs):
    return [(t.detach().contiguous().view(torch.uint8).clone(), t._version) for t in inputs]


def _unchanged(inputs, snapshots):
    versions = []
    for tensor, (before, version) in zip(inputs, snapshots):
        if tensor._version != version or not torch.equal(tensor.detach().contiguous().view(torch.uint8), before):
            raise AssertionError("The clipped training caller changed input bytes or versions")
        versions.append({"before": version, "after": tensor._version})
    return versions


def per_example_gradients(model, raw, mask, targets, pool, method="batched"):
    """Return detached [B,*parameter.shape] tensors without changing .grad."""
    if method not in ("batched", "serial"):
        raise ValueError("method must be batched or serial")
    if raw.shape[0] < 1 or targets.shape != (raw.shape[0],):
        raise ValueError("This caller needs B>=1 and one scalar target per example")
    inputs = raw, mask, targets
    snapshots = _snapshots(inputs)
    predictions, pooled = model(raw, mask, pool)
    losses = (predictions - targets).square()
    parameters = dict(model.named_parameters())
    if method == "batched":
        identity = torch.eye(raw.shape[0], dtype=losses.dtype, device="cpu")
        gradients = torch.autograd.grad(losses, tuple(parameters.values()), identity, is_grads_batched=True)
    else:
        rows = [torch.autograd.grad(losses[row], tuple(parameters.values()), retain_graph=row + 1 < raw.shape[0])
                for row in range(raw.shape[0])]
        gradients = tuple(torch.stack(block) for block in zip(*rows))
    if any(g.shape != (raw.shape[0], *p.shape) for g, p in zip(gradients, parameters.values())):
        raise AssertionError("Every parameter gradient must retain its example dimension")
    if not all(bool(torch.isfinite(t).all()) for t in (predictions, pooled, losses, *gradients)):
        raise AssertionError("Predictions, losses and every per-example gradient must be finite")
    _unchanged(inputs, snapshots)
    return {name: gradient.detach().clone() for name, gradient in zip(parameters, gradients)}


def clip_per_example(gradients, C):
    """Clip one joint vector per example, aggregate in float64, then cast once."""
    C = float(C)
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    if not gradients:
        raise ValueError("At least one parameter block is required")
    first = next(iter(gradients.values()))
    if first.ndim < 1 or first.shape[0] < 1:
        raise ValueError("Gradient blocks require a nonempty example dimension")
    batch, dtype = first.shape[0], first.dtype
    for gradient in gradients.values():
        if (gradient.ndim < 1 or gradient.shape[0] != batch or gradient.dtype != dtype
                or dtype not in DTYPES.values() or gradient.device.type != "cpu"
                or not bool(torch.isfinite(gradient).all())):
            raise ValueError("Gradient blocks must be finite CPU float32/64 with a common B and dtype")
    norms = sum(g.double().reshape(batch, -1).square().sum(dim=1) for g in gradients.values()).sqrt()
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("The joint gradient norm overflowed float64")
    scales = torch.ones_like(norms)
    active = norms > C
    scales[active] = C / norms[active]  # Zero norms keep scale one, with no division.
    means = {name: (gradient.double() * scales.reshape((batch,) + (1,)*(gradient.ndim - 1))).mean(dim=0).to(dtype)
             for name, gradient in gradients.items()}
    return means, norms, scales


def chunked_clipped_gradients(model, raw, mask, targets, pool, C, chunk_size=2):
    """Return clipped means using bounded selector rows on one retained graph.

    Full-gradient validation oracles are separate. This helper retains the full
    forward graph, input snapshots, and O(B) norm/scale diagnostics.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive Python integer")
    C = float(C)
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    batch = raw.shape[0]
    if batch < 1 or targets.shape != (batch,):
        raise ValueError("This caller needs B>=1 and one scalar target per example")
    effective = min(chunk_size, batch)
    inputs = raw, mask, targets
    snapshots = _snapshots(inputs)
    parameters = dict(model.named_parameters())
    if not parameters:
        raise ValueError("At least one connected parameter block is required")
    predictions, pooled = model(raw, mask, pool)
    losses = (predictions - targets).square()
    if (losses.dtype not in DTYPES.values() or losses.device.type != "cpu"
            or any(p.dtype != losses.dtype or p.device.type != "cpu" for p in parameters.values())):
        raise ValueError("This caller requires CPU float32/64 parameters and losses of one dtype")
    if not all(bool(torch.isfinite(t).all()) for t in (predictions, pooled, losses)):
        raise AssertionError("Predictions, pooled features and losses must be finite")
    sums = {name: torch.zeros_like(parameter, dtype=torch.float64) for name, parameter in parameters.items()}
    norms = torch.empty(batch, dtype=torch.float64, device="cpu")
    scales = torch.empty_like(norms)
    chunking = {"requested_size": chunk_size, "effective_size": effective, "chunk_rows": [],
                "retain_graph": [], "seed_shapes": [], "gradient_shapes": [], "max_rows": 0,
                "parameter_elements": sum(p.numel() for p in parameters.values())}
    for start in range(0, batch, effective):
        stop = min(start + effective, batch)
        rows = stop - start
        # Construct only this chunk's global selectors, never a full eye(B).
        seed = losses.new_zeros((rows, batch))
        seed[torch.arange(rows, device="cpu"), torch.arange(start, stop, device="cpu")] = 1
        retain = stop < batch
        gradients = torch.autograd.grad(losses, tuple(parameters.values()), seed,
                                        retain_graph=retain, create_graph=False,
                                        allow_unused=False, is_grads_batched=True)
        if any(g.shape != (rows, *p.shape) or g.dtype != p.dtype or g.device.type != "cpu"
               for g, p in zip(gradients, parameters.values())):
            raise AssertionError("Chunk gradient blocks must keep their declared example rows")
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise AssertionError("Every chunk parameter gradient must be finite")
        squared_norms = torch.zeros(rows, dtype=torch.float64, device="cpu")
        for gradient in gradients:
            block = gradient.detach().double()
            squared_norms.add_(block.reshape(rows, -1).square().sum(dim=1))
        chunk_norms = squared_norms.sqrt()
        if not bool(torch.isfinite(chunk_norms).all()):
            raise ValueError("The joint gradient norm overflowed float64")
        chunk_scales = torch.ones_like(chunk_norms)
        active = chunk_norms > C
        chunk_scales[active] = C / chunk_norms[active]
        for (name, parameter), gradient in zip(parameters.items(), gradients):
            block = gradient.detach().double()
            sums[name].add_((block * chunk_scales.reshape((rows,) + (1,) * parameter.ndim)).sum(dim=0))
        norms[start:stop], scales[start:stop] = chunk_norms, chunk_scales
        chunking["chunk_rows"].append(rows)
        chunking["retain_graph"].append(retain)
        chunking["seed_shapes"].append(list(seed.shape))
        chunking["gradient_shapes"].append({name: list(g.shape) for name, g in zip(parameters, gradients)})
        chunking["max_rows"] = max(chunking["max_rows"], rows)
        # Assignment evaluates the next grad call before rebinding gradients.
        # Release every prior block alias and seed before that next call.
        del gradients, gradient, block, seed, squared_norms, chunk_norms, chunk_scales, active
    means = {name: (sums[name] / batch).to(parameter.dtype) for name, parameter in parameters.items()}
    _unchanged(inputs, snapshots)
    return means, norms, scales, chunking


def _difference(left, right):
    return float((left.detach() - right.detach()).abs().max()) if left.numel() else 0.0


def _compare(left, right, tolerance):
    errors = {}
    if left.keys() != right.keys():
        raise AssertionError("Parameter blocks differ")
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], **tolerance)
        errors[name] = _difference(tensor, right[name])
    return errors


def _evaluate(model, raw, mask, targets, pool):
    values = raw.detach().clone().requires_grad_()
    predictions, pooled = model(values, mask, pool)
    losses = (predictions - targets).square()
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(losses.mean(), (*named.values(), values))
    if not all(bool(torch.isfinite(t).all()) for t in (predictions, pooled, losses, *gradients)):
        raise AssertionError("Mean-loss evaluation must remain finite")
    if bool(torch.count_nonzero(gradients[-1][~mask])) or bool(torch.count_nonzero(pooled[~mask.any(dim=1)])):
        raise AssertionError("Ignored raw gradients and empty pooled rows must be zero")
    return {"predictions": predictions.detach(), "pooled": pooled.detach(), "losses": losses.detach(),
            "mean_gradients": {name: g.detach() for name, g in zip(named, gradients[:-1])},
            "selected_raw_gradient": gradients[-1][mask].detach()}


def _empty_gradients(per_example, predictions, mask, targets, tolerance):
    empty = ~mask.any(dim=1)
    for name, gradient in per_example.items():
        if name != "head.bias" and bool(torch.count_nonzero(gradient[empty])):
            raise AssertionError("An empty bag may contribute only to the head bias")
    expected = 2 * (predictions[empty] - targets[empty])
    torch.testing.assert_close(per_example["head.bias"][empty, 0], expected, **tolerance)
    if empty.any() and not bool(torch.count_nonzero(expected)):
        raise AssertionError("The empty-bag fixture must have a nonzero head-bias derivative")


def _train(initial, layout, targets, C, tolerance, chunk_size):
    native, reference = copy.deepcopy(initial), copy.deepcopy(initial)
    models = native, reference
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.03) for model in models]
    raw, mask = layout["raw"], layout["mask"]
    inputs = raw, mask, targets
    snapshots = _snapshots(inputs)
    history = []
    for step in (1, 2):
        per = per_example_gradients(native, raw, mask, targets, ragged_pool_padded)
        serial = per_example_gradients(native, raw, mask, targets, ragged_pool_padded, "serial")
        ref = per_example_gradients(reference, raw, mask, targets, ragged_pool_padded_reference, "serial")
        serial_errors, reference_errors = _compare(per, serial, tolerance), _compare(per, ref, tolerance)
        evaluations = [_evaluate(model, raw, mask, targets, pool) for model, pool in
                       zip(models, (ragged_pool_padded, ragged_pool_padded_reference))]
        output_errors = _compare({k: v for k, v in evaluations[0].items() if k != "mean_gradients"},
                                 {k: v for k, v in evaluations[1].items() if k != "mean_gradients"}, tolerance)
        mean_errors = []
        for blocks, evaluation in zip((per, ref), evaluations):
            mean_errors.append(_compare({name: g.mean(dim=0) for name, g in blocks.items()}, evaluation["mean_gradients"], tolerance))
            _empty_gradients(blocks, evaluation["predictions"], mask, targets, tolerance)
        full_means, full_norms, full_scales = clip_per_example(per, C)
        means, norms, scales, chunking = chunked_clipped_gradients(
            native, raw, mask, targets, ragged_pool_padded, C, chunk_size)
        full_errors = _compare(means, full_means, tolerance)
        torch.testing.assert_close(norms, full_norms, **tolerance)
        torch.testing.assert_close(scales, full_scales, **tolerance)
        ref_means, ref_norms, ref_scales = clip_per_example(ref, C)
        clipped_errors = _compare(means, ref_means, tolerance)
        torch.testing.assert_close(norms, ref_norms, **tolerance)
        torch.testing.assert_close(scales, ref_scales, **tolerance)
        unclipped = {name: g.double().mean(dim=0).to(g.dtype) for name, g in per.items()}
        after_mean, _, _ = clip_per_example({name: g.unsqueeze(0) for name, g in unclipped.items()}, C)
        differences = {"from_unclipped": max(_difference(means[n], unclipped[n]) for n in means),
                       "from_clip_after_mean": max(_difference(means[n], after_mean[n]) for n in means)}
        if not bool((scales < 1).any()) or min(differences.values()) == 0:
            raise AssertionError("The fixture must actively clip and distinguish both controls")
        before = [{name: p.detach().clone() for name, p in model.named_parameters()} for model in models]
        for model, optimizer, gradients in zip(models, optimizers, (means, ref_means)):
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                parameter.grad = gradients[name].clone()
            optimizer.step()
        states = [{name: p.detach().clone() for name, p in model.named_parameters()} for model in models]
        state_errors = _compare(states[0], states[1], tolerance)
        updates = [{name: _difference(state[name], old[name]) for name in state} for state, old in zip(states, before)]
        if any(error == 0 or not math.isfinite(error) for update in updates for error in update.values()):
            raise AssertionError("Each actual SGD step must change every parameter in both models")
        metrics = {"step": step, "serial_parameter_errors": serial_errors, "reference_parameter_errors": reference_errors,
                   "output_errors": output_errors, "mean_loss_gradient_errors": mean_errors, "clipped_mean_errors": clipped_errors,
                   "full_clipped_mean_errors": full_errors, "chunking": chunking,
                   "updated_parameter_errors": state_errors, "parameter_update_max_abs": updates,
                   "norms": norms.tolist(), "scales": scales.tolist(), "active_examples": int((scales < 1).sum()),
                   "controls": differences, "input_versions_raw_mask_targets": _unchanged(inputs, snapshots)}
        history.append({"per_example": per, "reference_per_example": ref, "means": means, "reference_means": ref_means,
                        "evaluation": evaluations[0], "reference_evaluation": evaluations[1], "states": states, "metrics": metrics})
    return history


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
    return {"package_path": str(imported), "package_version": package.__version__, "environment_prefix": str(prefix),
            "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest()}


def run(dtype_name, chunk_size=2):
    identity = installed_identity()
    torch.manual_seed(SEED)
    dtype = DTYPES[dtype_name]
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    initial = PaddedRegressor().to(device="cpu", dtype=dtype)
    layouts = make_layouts(dtype_name)
    targets = torch.tensor([-0.3, 8., -0.5, 0.25, -4.], dtype=dtype, device="cpu")
    histories = {name: _train(initial, layout, targets, 1., tolerance, chunk_size) for name, layout in layouts.items()}
    baseline, comparisons = histories["fixed_holes"], []
    for name, history in histories.items():
        if name == "fixed_holes":
            continue
        errors = []
        for actual, expected in zip(history, baseline):
            blocks = {key: _compare(actual[key], expected[key], tolerance) for key in
                      ("per_example", "reference_per_example", "means", "reference_means")}
            for key in ("evaluation", "reference_evaluation"):
                blocks[key] = _compare({k: v for k, v in actual[key].items() if k != "mean_gradients"},
                                       {k: v for k, v in expected[key].items() if k != "mean_gradients"}, tolerance)
            blocks["updated_parameters"] = [_compare(a, b, tolerance) for a, b in zip(actual["states"], expected["states"])]
            errors.append(blocks)
        comparisons.append({"layout": name, "shape": list(layouts[name]["raw"].shape), "steps": errors})
    empty_mask = torch.zeros_like(layouts["fixed_holes"]["mask"])
    empty_raw = torch.full_like(layouts["fixed_holes"]["raw"], float("nan"))
    empty = per_example_gradients(initial, empty_raw, empty_mask, targets, ragged_pool_padded)
    empty_ref = per_example_gradients(initial, empty_raw, empty_mask, targets, ragged_pool_padded_reference, "serial")
    predictions, _ = initial(empty_raw, empty_mask, ragged_pool_padded)
    _empty_gradients(empty, predictions, empty_mask, targets, tolerance)
    empty_errors = _compare(empty, empty_ref, tolerance)
    return {"status": "passed", "dtype": dtype_name, "method": "autograd.grad.is_grads_batched",
        "aggregation_method": "chunked_per_example_joint_clip", "chunk_size_requested": chunk_size,
        "method_scope": "Experimental batched VJP. Batching fallbacks may execute serial work. No torch.func or speed claim.",
        "purpose": "Synthetic optimizer integration. No noise, privacy accounting, model quality or robustness result.",
        "seed": SEED, "examples": 5, "clip_threshold": 1., "norm": "joint L2 over every named parameter",
        "arithmetic": "Joint norms, scales and clipped sums in float64, divide once by global B and cast means once",
        "resource_scope": "Only the chunk helper bounds seed and returned gradient rows. The full forward graph, input snapshots, O(B) diagnostics and separate full-gradient validation oracles remain. No peak-memory or speed result.",
        "zero_norm_scale": 1., "aggregation": "mean over every example, including empty bags",
        "optimizer": "SGD", "learning_rate": 0.03, "optimizer_steps": 2, "reference_optimizer_steps": 2,
        "controls": {"status": "passed", "steps": [step["metrics"]["controls"] for step in baseline]},
        "invariance": {"status": "passed", "same_targets_initial_state_and_selected_order": True, "comparisons": comparisons},
        "empty": {"status": "passed", "encoder_scorer_head_weight_zero": True, "head_bias_nonzero": True,
                  "parameter_errors": empty_errors, "optimizer_steps": 0},
        "steps": [step["metrics"] for step in baseline], "inputs_unchanged": True, "gradients_finite": True,
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), **identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--chunk-size", type=int, default=2, help="Positive selector rows per VJP call, capped at B")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt file")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    text = json.dumps(run(arguments.dtype, arguments.chunk_size), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
