"""Check non-reentrant activation recomputation in two packed CPU updates.

The encoder, dropout, scorer and native pool form one pure checkpointed
segment. The head and loss remain outside it. Matched direct native and
primitive routes check gradients and SGD momentum. Saved-tensor hook events
describe logical forward payload, not allocated memory or performance.
"""
import argparse
import copy
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


def _adjacent():
    path = Path(__file__).resolve().with_name("train_dynamic.py")
    spec = importlib.util.spec_from_file_location("checkpoint_dynamic_fixture", path)
    if spec is None or spec.loader is None:
        raise ImportError("The adjacent train_dynamic.py fixture is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dynamic = _adjacent()
DROPOUT_PROBABILITY, DROPOUT_SEED, TOTAL_STEPS = 0.25, 2211, 2
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0, "weight_decay": 0,
           "nesterov": False, "maximize": False, "foreach": False,
           "differentiable": False, "fused": False}


def installed_identity():
    if sys.version.split()[0] != "3.11.8" or str(torch.__version__) != "2.9.1+cpu":
        raise AssertionError("Use the qualified Python 3.11.8 and Torch 2.9.1+cpu environment")
    identity = dynamic.installed_identity()
    identity["train_dynamic_sha256"] = identity["script_sha256"]
    identity["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


@contextmanager
def _scope(events, name):
    events.append(name + "_enter")
    try:
        yield
    finally:
        events.append(name + "_exit")


def forward(model, raw, offsets, *, checkpointed, preserve_rng_state=True,
            probability=DROPOUT_PROBABILITY, pool=dynamic.ragged_pool, events=None):
    """Run a pure segment with optional, explicitly non-reentrant recomputation."""
    if not model.training:
        raise ValueError("The fixed training consumer requires training mode")
    if probability not in (0.0, DROPOUT_PROBABILITY):
        raise ValueError("Only the declared dropout and disabled-dropout control are supported")
    events = [] if events is None else events

    def segment(features, boundaries):
        encoded = model.encoder(features).tanh().contiguous()
        values = F.dropout(encoded, p=probability, training=True).contiguous()
        logits = model.scorer(values).squeeze(-1).contiguous()
        return pool(values, logits, boundaries)

    if checkpointed:
        # Full replay is requested for this witness. It is not the default
        # early-stop behavior, and no global setting persists after this scope.
        with set_checkpoint_early_stop(False):
            pooled = checkpoint(segment, raw, offsets, use_reentrant=False,
                                preserve_rng_state=preserve_rng_state, determinism_check="default",
                                context_fn=lambda: (_scope(events, "forward"), _scope(events, "recompute")))
    else:
        pooled = segment(raw, offsets)
    return model.head(pooled).squeeze(-1), pooled


def saved_payload(events):
    return {"events": events, "count": len(events),
            "logical_bytes": sum(row["logical_bytes"] for row in events)}


@contextmanager
def _observe_saved(events):
    def pack(tensor):
        # Only metadata enters the observation log. The detached tensor below
        # is the graph's packed value, not an extra retained observation copy.
        events.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "numel": tensor.numel(), "element_size": tensor.element_size(),
                       "logical_bytes": tensor.numel() * tensor.element_size()})
        return tensor.detach(), tensor._version

    def unpack(saved):
        tensor, version = saved
        if tensor._version != version:
            raise RuntimeError("A tensor saved for backward was modified by an inplace operation")
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def _copy(named):
    return {name: value.detach().clone() for name, value in named.items()}


def step(model, opt, batch, index, *, checkpointed, preserve_rng_state=True,
         probability=DROPOUT_PROBABILITY, reference=False):
    names = dict(model.named_parameters())
    parameter_ids = tuple(id(value) for value in names.values())
    if tuple(id(value) for group in opt.param_groups for value in group["params"]) != parameter_ids:
        raise AssertionError("The optimizer must own the original model parameters in order")
    before = _copy(names)
    momentum_before = {name: (None if "momentum_buffer" not in opt.state[value]
                              else opt.state[value]["momentum_buffer"].clone())
                       for name, value in names.items()}
    opt.zero_grad(set_to_none=True)
    raw = batch["raw"].detach().clone().requires_grad_()
    inputs = {"raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    inputs_before = dynamic.snapshot(inputs)
    rng_before = torch.get_rng_state().clone()
    scopes, saves = [], []
    pool = dynamic.primitive_pool(batch["python_boundaries"]) if reference else dynamic.ragged_pool
    # Observation covers original-forward saves only. The context has exited
    # before backward and cannot count replay events as original-forward saves.
    with _observe_saved(saves):
        predictions, pooled = forward(model, raw, batch["offsets"], checkpointed=checkpointed,
                                      preserve_rng_state=preserve_rng_state, probability=probability,
                                      pool=pool, events=scopes)
        loss = (predictions - batch["targets"]).square().mean()
    rng_after_forward = torch.get_rng_state().clone()
    loss.backward()
    rng_after_backward = torch.get_rng_state().clone()
    if raw.grad is None or any(value.grad is None for value in names.values()):
        raise AssertionError("Every raw feature and parameter block needs a first gradient")
    gradients = {name: value.grad.detach().clone() for name, value in names.items()}
    if any(float(value.abs().max()) <= 0 for value in (*gradients.values(), raw.grad)):
        raise AssertionError("The declared updates need informative gradients in every block")
    expected_buffers = {name: (gradient.clone() if momentum_before[name] is None
                               else momentum_before[name].mul(0.8).add(gradient))
                        for name, gradient in gradients.items()}
    expected_parameters = {name: before[name].add(expected_buffers[name], alpha=-0.03) for name in names}
    opt.step()
    parameters = _copy(names)
    momentum = {name: opt.state[value]["momentum_buffer"].detach().clone() for name, value in names.items()}
    dynamic.compare(momentum, expected_buffers, {"rtol": 0, "atol": 0})
    dynamic.compare(parameters, expected_parameters, {"rtol": 0, "atol": 0})
    if any(dynamic.difference(parameters[name], before[name]) <= 0 for name in names):
        raise AssertionError("Each declared update must change every parameter block")
    if tuple(id(value) for value in model.parameters()) != parameter_ids:
        raise AssertionError("An update replaced model parameter objects")
    empty = [row for row, (left, right) in enumerate(zip(batch["python_boundaries"], batch["python_boundaries"][1:]))
             if left == right]
    torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
    output = {"predictions": predictions.detach().clone(), "pooled": pooled.detach().clone(),
              "loss": loss.detach().clone(), "raw_gradient": raw.grad.detach().clone()}
    dynamic.finite((*output.values(), *gradients.values(), *parameters.values(), *momentum.values()))
    versions = dynamic.unchanged(inputs, inputs_before)
    if checkpointed and scopes != ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"]:
        raise AssertionError("The full-replay witness needs one forward and one completed recomputation scope")
    opt.zero_grad(set_to_none=True)
    return {"update": index + 1, "N": raw.shape[0], "B": batch["targets"].shape[0],
            "offsets": batch["offsets"].tolist(), "output": output,
            "parameter_gradients": gradients, "parameters": parameters, "momentum": momentum,
            "rng_before": rng_before, "rng_after_forward": rng_after_forward,
            "rng_after_backward": rng_after_backward, "saved_forward": saved_payload(saves),
            "checkpoint_scope_events": scopes, "input_versions": versions}


def trajectory(initial, batches, *, checkpointed, preserve_rng_state=True,
               probability=DROPOUT_PROBABILITY, reference=False):
    model = copy.deepcopy(initial).train(True)
    opt = torch.optim.SGD(model.parameters(), **OPTIONS)
    torch.manual_seed(DROPOUT_SEED)
    return [step(model, opt, batch, index, checkpointed=checkpointed,
                 preserve_rng_state=preserve_rng_state, probability=probability, reference=reference)
            for index, batch in enumerate(batches[:TOTAL_STEPS])]


def compare_records(actual, expected, tolerance, *, compare_rng=True):
    if len(actual) != len(expected):
        raise AssertionError("Trajectory lengths differ")
    errors = []
    for left, right in zip(actual, expected):
        for field in ("output", "parameter_gradients", "parameters", "momentum"):
            errors.extend(dynamic.compare(left[field], right[field], tolerance).values())
        if compare_rng:
            for field in ("rng_before", "rng_after_forward", "rng_after_backward"):
                if not torch.equal(left[field], right[field]):
                    raise AssertionError("Matched routes must preserve the exact default CPU RNG state")
    return max(errors, default=0.0)


def _rng_digest(value):
    return sha256(value.numpy().tobytes()).hexdigest()


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    initial, batches, tolerance = dynamic.fixture(dtype_name)
    sources = {f"batch{index}:{name}": batch[name] for index, batch in enumerate(batches[:TOTAL_STEPS])
               for name in ("raw", "offsets", "targets")}
    sources.update({"initial:" + name: value for name, value in initial.named_parameters()})
    snapshots = dynamic.snapshot(sources)
    direct = trajectory(initial, batches, checkpointed=False)
    replay = trajectory(initial, batches, checkpointed=True)
    primitive = trajectory(initial, batches, checkpointed=False, reference=True)
    exact_error = compare_records(replay, direct, {"rtol": 0, "atol": 0})
    primitive_error = compare_records(replay, primitive, tolerance)
    without_rng = trajectory(initial, batches, checkpointed=True, preserve_rng_state=False)
    first_forward = {key: without_rng[0]["output"][key] for key in ("predictions", "pooled", "loss")}
    direct_forward = {key: direct[0]["output"][key] for key in first_forward}
    dynamic.compare(first_forward, direct_forward, {"rtol": 0, "atol": 0})
    gradient_error = max(dynamic.difference(without_rng[0]["parameter_gradients"][name],
                                           direct[0]["parameter_gradients"][name])
                         for name in direct[0]["parameter_gradients"])
    parameter_error = max(dynamic.difference(without_rng[-1]["parameters"][name], direct[-1]["parameters"][name])
                          for name in direct[-1]["parameters"])
    threshold = 1e-7 if dtype_name == "float32" else 1e-10
    rng_diverged = not torch.equal(without_rng[-1]["rng_after_backward"], direct[-1]["rng_after_backward"])
    if gradient_error <= threshold or parameter_error <= threshold or not rng_diverged:
        raise AssertionError("Omitting dropout RNG preservation must expose gradient, update and RNG drift")
    disabled_direct = trajectory(initial, batches, checkpointed=False, probability=0.0)
    disabled_replay = trajectory(initial, batches, checkpointed=True, preserve_rng_state=False, probability=0.0)
    disabled_error = compare_records(disabled_replay, disabled_direct, {"rtol": 0, "atol": 0})
    for record in disabled_direct + disabled_replay:
        if not torch.equal(record["rng_before"], record["rng_after_backward"]):
            raise AssertionError("Disabled dropout must not consume the CPU RNG stream")
    steps = []
    for index, (plain, checked) in enumerate(zip(direct, replay)):
        if checked["saved_forward"]["logical_bytes"] >= plain["saved_forward"]["logical_bytes"]:
            raise AssertionError("The fixed checkpointed forward must save less logical hook payload")
        if not torch.equal(checked["rng_after_forward"], checked["rng_after_backward"]):
            raise AssertionError("Preserved recomputation must leave RNG at its post-forward state")
        steps.append({"update": checked["update"], "N": checked["N"], "B": checked["B"],
                      "offsets": checked["offsets"], "loss": float(checked["output"]["loss"]),
                      "saved_forward": {"direct": plain["saved_forward"], "checkpointed": checked["saved_forward"]},
                      "checkpoint_scope_events": checked["checkpoint_scope_events"],
                      "default_rng_before_sha256": _rng_digest(checked["rng_before"]),
                      "default_rng_after_forward_sha256": _rng_digest(checked["rng_after_forward"]),
                      "default_rng_after_backward_sha256": _rng_digest(checked["rng_after_backward"]),
                      "native_exact_max_abs_error": compare_records([checked], [plain], {"rtol": 0, "atol": 0}),
                      "primitive_max_abs_error": compare_records([checked], [primitive[index]], tolerance)})
    dynamic.unchanged(sources, snapshots)
    return {"status": "passed", "method": "checkpoint.non_reentrant.packed_training",
            "scope": "Fixed eager CPU activation recomputation. Saved-forward logical payload is not unique allocated storage, peak RSS, GPU memory or a speed result. No security, model-quality or general checkpoint equivalence claim.",
            "dtype": dtype_name, "device": "cpu", "use_reentrant": False,
            "preserve_rng_state": True, "early_stop": False, "determinism_check": "default",
            "dropout_probability": DROPOUT_PROBABILITY, "dropout_seed": DROPOUT_SEED,
            "optimizer": "SGD", "learning_rate": 0.03, "momentum": 0.8, "optimizer_steps": TOTAL_STEPS,
            "scalar_parameters": sum(value.numel() for value in initial.parameters()),
            "parameter_tensors": len(tuple(initial.parameters())), "comparison_tolerance": tolerance,
            "independent_momentum_recurrence": True, "same_native_exact": True,
            "same_native_max_abs_error": exact_error, "primitive_comparisons_passed": True,
            "primitive_max_abs_error": primitive_error, "inputs_unchanged": True,
            "parameter_ownership_unchanged": True, "saved_accounting": "original_forward_pack_hook_events",
            "memory_performance_claim": False,
            "controls": {"without_rng_preservation": {"first_forward_exact": True,
                "gradient_max_abs_error": gradient_error, "final_parameter_max_abs_error": parameter_error,
                "divergence_threshold": threshold, "diverged": True, "rng_diverged": rng_diverged},
                "dropout_disabled": {"dropout_probability": 0.0, "same_native_exact": True,
                                     "max_abs_error": disabled_error, "rng_unchanged": True}},
            "steps": steps, "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "python_executable": sys.executable, "working_directory": str(Path.cwd()),
            "installed_package": True, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    text = json.dumps(run(arguments.dtype), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
