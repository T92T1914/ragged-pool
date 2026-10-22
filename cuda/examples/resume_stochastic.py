"""Resume one fixed CUDA dropout and shuffled-batch trajectory in fresh processes.

Copy this file, resume_training.py, train_compiled.py and train_cuda.py outside
the checkout. Seven eager CUDA SGD updates use actual dropout and a dedicated
CPU batch-order generator. Only this bounded local checkpoint is admitted.
This does not provide general replay, hostile-file safety or crash recovery.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = SCRIPT.with_name("resume_training.py")
_spec = importlib.util.spec_from_file_location("_cuda_stochastic_resume_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError("Copy resume_training.py, train_compiled.py and train_cuda.py beside this consumer")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
dynamic, eager = base.dynamic, base.eager
DTYPES, OPTIONS, EXACT = base.DTYPES, base.OPTIONS, base.EXACT
optimizer = base.optimizer
_clone, _tensors = base._clone, base._tensors
CURSOR, TOTAL_STEPS = 2, 7
DROPOUT_PROBABILITY, DROPOUT_SEED, ORDER_SEED = 0.25, 2011, 2111
MAX_CHECKPOINT_BYTES, MAX_RESULT_BYTES = 128 * 1024, 512 * 1024
MAX_RNG_STATE_BYTES, CUDA_RNG_STATE_BYTES = 8192, 16


def _ordinary_bytes(value, expected, name):
    if (type(value) is not torch.Tensor or value.device != torch.device("cpu")
            or value.layout != torch.strided or value.dtype != torch.uint8
            or value.shape != expected.shape or not value.is_contiguous()
            or value.storage_offset() != 0 or value.untyped_storage().nbytes() != value.numel()
            or value.requires_grad or value.is_conj() or value.is_neg()
            or not 0 < value.numel() <= MAX_RNG_STATE_BYTES):
        raise ValueError(f"Invalid owned ordinary uint8 CPU tensor for {name}")


def _compare_tree(actual, expected, tolerance):
    # Opaque RNG and dropout-mask bytes do not use the floating tensor envelope.
    if type(expected) is torch.Tensor and expected.dtype == torch.uint8:
        _ordinary_bytes(actual, expected, "byte-state comparison")
        if not torch.equal(actual, expected):
            raise AssertionError("Opaque byte state differs")
        return 0.0
    if type(expected) is torch.Tensor:
        return base._compare_tree(actual, expected, tolerance)
    if type(actual) is not type(expected):
        raise AssertionError("Result structure types differ")
    if type(expected) is dict:
        if list(actual) != list(expected):
            raise AssertionError("Result dictionary keys or order differ")
        return max((_compare_tree(actual[key], expected[key], tolerance) for key in expected), default=0.0)
    if type(expected) in (list, tuple):
        if len(actual) != len(expected):
            raise AssertionError("Result sequence lengths differ")
        return max((_compare_tree(a, b, tolerance) for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Result scalar metadata differs")
    return 0.0


class StochasticRegressor(eager.SetRegressor):
    def forward(self, raw, offsets, boundaries=None):
        if not self.training:
            raise ValueError("This fixed consumer requires explicit training mode")
        encoded = self.encoder(raw).tanh().contiguous()
        dropped = F.dropout(encoded, p=DROPOUT_PROBABILITY, training=True).contiguous()
        logits = self.scorer(dropped).squeeze(-1).contiguous()
        pooled = (eager.ragged_pool(dropped, logits, offsets) if boundaries is None
                  else eager.primitive_pool(dropped, logits, boundaries))
        # Observations own CPU storage and are not model parameters or buffers.
        self.last_dropout = {"encoded": _clone(encoded), "dropped": _clone(dropped),
                             "logits": _clone(logits), "keep": _clone((dropped != 0).to(torch.uint8))}
        return self.head(pooled).squeeze(-1), pooled


def installed_identity():
    """Identify the installed package without creating parent CUDA tensors."""
    identity = base.installed_identity()
    identity["resume_script_sha256"] = identity["script_sha256"]
    identity["script_sha256"] = sha256(SCRIPT.read_bytes()).hexdigest()
    return identity


def fixture(dtype_name):
    initial, batches, tolerance = base.fixture(dtype_name)
    model = StochasticRegressor().to(device="cuda:0", dtype=DTYPES[dtype_name])
    model.load_state_dict(initial.state_dict(), strict=True)
    model.train(True)
    return model, batches, tolerance


class BatchOrder:
    """Own five fixture selections and the CPU stream used at each rollover."""
    def __init__(self):
        self.generator = torch.Generator(device="cpu").manual_seed(ORDER_SEED)
        self.permutation = torch.randperm(5, generator=self.generator).tolist()
        self.cursor, self.epoch = 0, 0

    def state_dict(self):
        return {"generator": self.generator.get_state().clone(),
                "permutation": list(self.permutation), "cursor": self.cursor, "epoch": self.epoch}

    def next_batch(self):
        if self.cursor == 5:
            self.permutation = torch.randperm(5, generator=self.generator).tolist()
            self.cursor, self.epoch = 0, self.epoch + 1
        index = self.permutation[self.cursor]
        self.cursor += 1
        return index

    def apply_validated(self, state):
        self.generator.set_state(state["generator"].clone())
        self.permutation = list(state["permutation"])
        self.cursor, self.epoch = state["cursor"], state["epoch"]


def metadata(dtype_name, initial, batches, identity=None, source_revision=None, source_tree=None):
    identity = installed_identity() if identity is None else identity
    result = base.metadata(dtype_name, initial, batches, identity, source_revision, source_tree)
    result.update({"format": "ragged-pool-cuda-fixed-stochastic-training-state", "schema": 1,
                   "total_updates": TOTAL_STEPS, "resume_script_sha256": identity["resume_script_sha256"],
                   "stochastic_forward": True, "live_rng_continuation": True,
                   "dropout_probability": DROPOUT_PROBABILITY, "dropout_seed": DROPOUT_SEED,
                   "dropout_rng_device": "cuda:0", "cuda_rng_state_bytes": CUDA_RNG_STATE_BYTES,
                   "order_seed": ORDER_SEED, "order_length": 5, "checkpoint_epoch": 0,
                   "checkpoint_order_cursor": CURSOR, "training_mode": True})
    return result


def _rng_state(value, name):
    scratch = torch.Generator(device="cpu")
    expected = scratch.get_state()
    _ordinary_bytes(value, expected, name)
    try:
        scratch.set_state(value.clone())
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid scratch CPU generator state for {name}") from error


def _cuda_rng_state(value):
    # In the pinned release set_state can set the seed before rejecting an
    # invalid offset. Validate the complete state on a scratch CUDA generator.
    scratch = torch.Generator(device="cuda:0")
    expected = scratch.get_state()
    if expected.numel() != CUDA_RNG_STATE_BYTES:
        raise ValueError("The qualified CUDA RNG state must contain exactly 16 bytes")
    _ordinary_bytes(value, expected, "CUDA0 dropout RNG")
    try:
        scratch.set_state(value.clone())
    except (RuntimeError, ValueError) as error:
        raise ValueError("Invalid scratch CUDA0 generator state") from error


def validate_checkpoint(checkpoint, model, opt, order, expected_metadata):
    """Admit saved fields for an existing training target without setting streams."""
    if type(checkpoint) is not dict or set(checkpoint) != {
            "metadata", "model", "optimizer", "cuda_rng", "order", "training"}:
        raise ValueError("The fixed checkpoint fields must match exactly")
    base.validate_checkpoint({key: checkpoint[key] for key in ("metadata", "model", "optimizer")},
                             model, opt, expected_metadata)
    if type(checkpoint["training"]) is not bool or checkpoint["training"] is not True:
        raise ValueError("The fixed checkpoint must preserve training mode")
    if type(order) is not BatchOrder or order.generator.device.type != "cpu":
        raise ValueError("The target needs this example's dedicated CPU batch generator")
    state = checkpoint["order"]
    if type(state) is not dict or set(state) != {"generator", "permutation", "cursor", "epoch"}:
        raise ValueError("The saved batch-order fields must match exactly")
    if (type(state["cursor"]) is not int or state["cursor"] != CURSOR
            or type(state["epoch"]) is not int or state["epoch"] != 0):
        raise ValueError("Only epoch zero after two completed selections is admitted")
    permutation = state["permutation"]
    if (type(permutation) is not list or len(permutation) != 5
            or any(type(value) is not int for value in permutation)
            or sorted(permutation) != list(range(5))):
        raise ValueError("The saved order must be an exact permutation of the five fixtures")
    _cuda_rng_state(checkpoint["cuda_rng"])
    _rng_state(state["generator"], "batch-order RNG")
    independently_seeded = torch.Generator(device="cpu").manual_seed(ORDER_SEED)
    first_order = torch.randperm(5, generator=independently_seeded).tolist()
    if permutation != first_order or not torch.equal(state["generator"], independently_seeded.get_state()):
        raise ValueError("Saved order and generator must match the first seeded permutation")
    tensors = list(_tensors(checkpoint))
    if len({base._storage(value) for value in tensors}) != len(tensors):
        raise ValueError("Every checkpoint tensor must own independent storage")


def make_checkpoint(model, opt, order, expected_metadata):
    checkpoint = base.make_checkpoint(model, opt, expected_metadata)
    checkpoint.update({"cuda_rng": torch.cuda.get_rng_state(0).clone(), "order": order.state_dict(),
                       "training": model.training})
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    return checkpoint


def restore(checkpoint, model, opt, order, expected_metadata):
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    before, contents = base._tree_snapshot(checkpoint), _clone(checkpoint)
    binding = base._bindings(model, opt)
    base.restore({key: checkpoint[key] for key in ("metadata", "model", "optimizer")},
                 model, opt, expected_metadata)
    model.train(checkpoint["training"])
    order.apply_validated(checkpoint["order"])
    # Restore this stream last, after construction, parameter loads and order setup.
    torch.cuda.set_rng_state(checkpoint["cuda_rng"].clone(), device=0)
    if base._bindings(model, opt) != binding:
        raise AssertionError("Restoration changed model or optimizer ownership")
    base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict(),
                               "cuda_rng": torch.cuda.get_rng_state(0), "order": order.state_dict()})
    _compare_tree(checkpoint, contents, EXACT)
    base._tree_unchanged(checkpoint, before)


def load_checkpoint(path, expected_sha256, model, opt, order, expected_metadata):
    if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A complete lowercase expected SHA256 is required")
    payload = base._read_bytes(path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the trusted local artifact")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    return checkpoint


def step(model, opt, batches, order, index, reference=False):
    order_before, rng_before = order.state_dict(), torch.cuda.get_rng_state(0).clone()
    batch_id = order.next_batch()
    record = base.step(model, opt, batches[batch_id], index, reference=reference)
    evidence = model.last_dropout
    encoded, dropped, keep = evidence["encoded"], evidence["dropped"], evidence["keep"].bool()
    if bool((encoded == 0).any()) or not bool(keep.any()) or bool(keep.all()):
        raise AssertionError("The fixed dropout witness needs nonzero values, drops and survivors")
    torch.testing.assert_close(dropped, torch.where(keep, encoded * (1 / (1 - DROPOUT_PROBABILITY)),
                                                  torch.zeros_like(encoded)), rtol=0, atol=0)
    record.update({"batch_id": batch_id, "dropout": _clone(evidence), "rng_before": rng_before,
                   "rng_after": torch.cuda.get_rng_state(0).clone(), "order_before": order_before,
                   "order_after": order.state_dict(), "training": model.training})
    return record


def run_steps(model, opt, batches, order, start=0, stop=TOTAL_STEPS, reference=False):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= TOTAL_STEPS:
        raise ValueError("Use a nonempty subset of the fixed seven updates")
    binding = base._bindings(model, opt)
    records = [step(model, opt, batches, order, index, reference) for index in range(start, stop)]
    if base._bindings(model, opt) != binding:
        raise AssertionError("Continuation replaced its model, optimizer or parameters")
    return records


def _phase(dtype_name, directory, phase, expected_sha256, source_revision, source_tree):
    base._threads()
    identity = installed_identity()
    initial, batches, _ = fixture(dtype_name)
    initial_before = base.snapshot(dict(initial.named_parameters()))
    expected = metadata(dtype_name, initial, batches, identity, source_revision, source_tree)
    model, order = copy.deepcopy(initial), BatchOrder()
    opt = optimizer(model)
    checkpoint_path = directory / "checkpoint.pt"
    if phase == "produce":
        torch.cuda.manual_seed(DROPOUT_SEED)
        records = run_steps(model, opt, batches, order, stop=CURSOR)
        checkpoint = make_checkpoint(model, opt, order, expected)
        digest = base._write_tensor_file(checkpoint_path, checkpoint, MAX_CHECKPOINT_BYTES)
        result_path = directory / "producer-results.pt"
    else:
        checkpoint = load_checkpoint(checkpoint_path, expected_sha256, model, opt, order, expected)
        before, contents = base._tree_snapshot(checkpoint), _clone(checkpoint)
        restore(checkpoint, model, opt, order, expected)
        records = run_steps(model, opt, batches, order, start=CURSOR)
        _compare_tree(checkpoint, contents, EXACT)
        base._tree_unchanged(checkpoint, before)
        base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict(),
                                   "order": order.state_dict(), "cuda_rng": torch.cuda.get_rng_state(0)})
        digest = sha256(base._read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)).hexdigest()
        if digest != expected_sha256:
            raise AssertionError("Checkpoint bytes changed during resumed training")
        result_path = directory / "resume-results.pt"
    base.unchanged(dict(initial.named_parameters()), initial_before)
    if installed_identity() != identity:
        raise AssertionError("Installed dependency or consumer source changed during the child")
    result = {"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(),
              "metadata": expected, "identity": identity,
              "checkpoint_sha256": digest, "records": records}
    result_digest = base._write_tensor_file(result_path, result, MAX_RESULT_BYTES)
    print(json.dumps({"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(),
                      "checkpoint_sha256": digest, "result_sha256": result_digest}, allow_nan=False))


def _child(dtype_name, directory, phase, expected_sha256=None, source_revision=None, source_tree=None):
    command = [sys.executable, "-I", str(SCRIPT), "--dtype", dtype_name,
               "--directory", str(directory), "--phase", phase]
    for flag, value in (("--expected-sha256", expected_sha256), ("--source-revision", source_revision),
                        ("--source-tree", source_tree)):
        if value is not None:
            command.extend((flag, value))
    # The one bounded child exits before either the next child or parent GPU work.
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) as child:
        try:
            stdout, stderr = child.communicate(timeout=120)
        except subprocess.TimeoutExpired as error:
            child.kill()
            stdout, stderr = child.communicate(timeout=15)
            (directory / (phase + ".stdout.log")).write_bytes(stdout)
            (directory / (phase + ".stderr.log")).write_bytes(stderr)
            raise RuntimeError(f"The {phase} child exceeded 120 seconds") from error
    (directory / (phase + ".stdout.log")).write_bytes(stdout)
    (directory / (phase + ".stderr.log")).write_bytes(stderr)
    if child.returncode != 0:
        raise RuntimeError(f"The {phase} child failed with exit code {child.returncode}. Inspect its retained logs.")
    status = json.loads(stdout.decode("utf-8"))
    if (type(status) is not dict
            or set(status) != {"phase", "pid", "parent_pid", "checkpoint_sha256", "result_sha256"}
            or any(type(status[name]) is not int or status[name] <= 0 for name in ("pid", "parent_pid"))
            or status["phase"] != phase):
        raise AssertionError("The child's completion report differs from its actual process")
    if status["pid"] == child.pid:
        relation = "direct"
    elif os.name == "nt" and status["parent_pid"] == child.pid:
        relation = "windows-venv-redirector"
    else:
        raise AssertionError("The Python runtime does not belong to the completed launcher")
    result_path = directory / ("producer-results.pt" if phase == "produce" else "resume-results.pt")
    payload = base._read_bytes(result_path, MAX_RESULT_BYTES)
    if sha256(payload).hexdigest() != status["result_sha256"]:
        raise AssertionError("Child result bytes differ from its completion report")
    result = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    if (type(result) is not dict
            or set(result) != {"phase", "pid", "parent_pid", "metadata", "identity", "checkpoint_sha256", "records"}
            or result["phase"] != phase or type(result["pid"]) is not int or result["pid"] != status["pid"]
            or type(result["parent_pid"]) is not int or result["parent_pid"] != status["parent_pid"]
            or result["checkpoint_sha256"] != status["checkpoint_sha256"]):
        raise AssertionError("Child result metadata differs from its completion report")
    if expected_sha256 is not None and result["checkpoint_sha256"] != expected_sha256:
        raise AssertionError("The resumer used a different checkpoint")
    result.update({"launcher_pid": child.pid, "launcher_relation": relation})
    return result


def controls(initial, batches, expected, checkpoint, native_records):
    """Break one live dependency after valid restore and check its causal witness."""
    results = {}
    threshold = 1e-7 if expected["dtype"] == "float32" else 1e-10
    suffix = native_records[CURSOR:]
    for case in ("reset_dropout_rng", "omit_dropout_rng", "reset_order_generator",
                 "reset_optimizer", "repeat_cursor", "skip_cursor"):
        model, order = copy.deepcopy(initial), BatchOrder()
        opt = optimizer(model)
        torch.cuda.manual_seed(DROPOUT_SEED + 1)
        omitted_state = torch.cuda.get_rng_state(0).clone()
        restore(checkpoint, model, opt, order, expected)
        if case == "reset_dropout_rng":
            torch.cuda.manual_seed(DROPOUT_SEED)
        elif case == "omit_dropout_rng":
            torch.cuda.set_rng_state(omitted_state, device=0)
        elif case == "reset_order_generator":
            order.generator.manual_seed(ORDER_SEED)
        elif case == "reset_optimizer":
            opt = optimizer(model)
        elif case == "repeat_cursor":
            order.cursor = CURSOR - 1
        elif case == "skip_cursor":
            order.cursor = CURSOR + 1
        records = run_steps(model, opt, batches, order, start=CURSOR)
        drift = max(eager.difference(records[-1]["parameters"][name], native_records[-1]["parameters"][name])
                    for name in base.STATE_SHAPES)
        if drift <= threshold:
            raise AssertionError(f"The deliberate {case} control must visibly diverge")
        witness = {}
        if case in ("reset_dropout_rng", "omit_dropout_rng"):
            if records[0]["batch_id"] != suffix[0]["batch_id"]:
                raise AssertionError("Changing only dropout RNG must preserve the next batch")
            mask_changed = not torch.equal(records[0]["dropout"]["keep"], suffix[0]["dropout"]["keep"])
            forward_error = eager.difference(records[0]["output"]["predictions"], suffix[0]["output"]["predictions"])
            if not mask_changed or forward_error <= threshold:
                raise AssertionError("A dropout RNG control needs an immediate mask and forward witness")
            witness = {"next_batch_unchanged": True, "next_dropout_mask_changed": True,
                       "next_forward_max_abs_error": forward_error}
        elif case == "reset_order_generator":
            for actual, expected_record in zip(records[:3], suffix[:3]):
                for field in ("update", "inputs", "output", "parameter_gradients", "parameters",
                              "momentum", "batch_id", "dropout", "rng_before", "rng_after", "training"):
                    _compare_tree(actual[field], expected_record[field], EXACT)
                for boundary in ("order_before", "order_after"):
                    for field in ("permutation", "cursor", "epoch"):
                        _compare_tree(actual[boundary][field], expected_record[boundary][field], EXACT)
            if [row["batch_id"] for row in records[3:]] == [row["batch_id"] for row in suffix[3:]]:
                raise AssertionError("Order RNG reset must be exposed after permutation rollover")
            witness = {"current_permutation_suffix_exact": True, "post_rollover_order_changed": True}
        elif case == "reset_optimizer":
            for field in ("inputs", "output", "parameter_gradients", "dropout", "rng_before", "rng_after"):
                _compare_tree(records[0][field], suffix[0][field], EXACT)
            first_drift = max(eager.difference(records[0]["parameters"][name], suffix[0]["parameters"][name])
                              for name in base.STATE_SHAPES)
            if first_drift <= threshold:
                raise AssertionError("Missing momentum must change the first update after an exact forward")
            witness = {"next_forward_and_gradients_exact": True,
                       "first_update_parameter_max_abs_error": first_drift}
        else:
            position = CURSOR - 1 if case == "repeat_cursor" else CURSOR + 1
            if (records[0]["batch_id"] != checkpoint["order"]["permutation"][position]
                    or records[0]["batch_id"] == suffix[0]["batch_id"]):
                raise AssertionError("A cursor control must select the declared wrong immediate batch")
            witness = {"next_batch_changed": True, "selected_saved_permutation_position": position}
        results[case] = {"final_parameter_max_abs_error": drift, "divergence_threshold": threshold,
                         "diverged": True, "batch_ids": [row["batch_id"] for row in records],
                         "causal_witness": witness, "live_state_bypass_after_valid_restore": True}
    return results


def _rng_digest(value):
    return sha256(value.numpy().tobytes()).hexdigest()


def run(dtype_name, directory, source_revision=None, source_tree=None):
    """Finish fresh children before constructing either parent GPU baseline."""
    base._threads()
    base._source_labels(source_revision, source_tree)
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    identity = installed_identity()
    produced = _child(dtype_name, directory, "produce", source_revision=source_revision, source_tree=source_tree)
    digest = produced["checkpoint_sha256"]
    resumed = _child(dtype_name, directory, "resume", digest, source_revision, source_tree)
    pids = {"parent": os.getpid(), "producer": produced["pid"], "resumer": resumed["pid"]}
    if len(set(pids.values())) != 3:
        raise AssertionError("Parent, producer and fresh resumer must be distinct processes")
    # Child results are CPU tensors. Parent CUDA construction starts only here.
    initial, batches, tolerance = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity, source_revision, source_tree)
    for result in (produced, resumed):
        if not base._same_plain(result["metadata"], expected) or result["identity"] != identity:
            raise AssertionError("Child source, fixture or installed identity differs from the parent")
    sources = {f"batch{index}.{name}": batch[name] for index, batch in enumerate(batches)
               for name in ("raw", "offsets", "targets")}
    sources.update({"initial." + name: value for name, value in initial.named_parameters()})
    snapshots = base.snapshot(sources)
    native, primitive = copy.deepcopy(initial), copy.deepcopy(initial)
    all_parameters = [*initial.parameters(), *native.parameters(), *primitive.parameters()]
    if len({base._storage(value) for value in all_parameters}) != len(all_parameters):
        raise AssertionError("Initial and uninterrupted models must own separate parameter storage")
    torch.cuda.manual_seed(DROPOUT_SEED)
    native_records = run_steps(native, optimizer(native), batches, BatchOrder())
    torch.cuda.manual_seed(DROPOUT_SEED)
    primitive_records = run_steps(primitive, optimizer(primitive), batches, BatchOrder(), reference=True)
    primitive_error = _compare_tree(native_records, primitive_records, tolerance)
    combined = produced["records"] + resumed["records"]
    exact_error = _compare_tree(combined, native_records, EXACT)
    target, order = copy.deepcopy(initial), BatchOrder()
    opt = optimizer(target)
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint = load_checkpoint(checkpoint_path, digest, target, opt, order, expected)
    before, contents = base._tree_snapshot(checkpoint), _clone(checkpoint)
    control_results = controls(initial, batches, expected, checkpoint, native_records)
    _compare_tree(checkpoint, contents, EXACT)
    base._tree_unchanged(checkpoint, before)
    base.unchanged(sources, snapshots)
    payload = base._read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != digest or installed_identity() != identity:
        raise AssertionError("The checkpoint, source or installed dependency changed")
    if native_records[-1]["order_after"]["epoch"] != 1:
        raise AssertionError("The resumed suffix must cross one actual permutation rollover")
    torch.cuda.synchronize(0)
    properties = torch.cuda.get_device_properties(0)
    return {"status": "passed", "method": "state_dict.stochastic_packed_cuda_training_resume",
            "purpose": "Fixed CUDA dropout and shuffled-batch continuation in fresh processes",
            "scope": "Known local eager CUDA0 state only. No general replay, workers, Python/NumPy RNG, distributed, hostile-file safety, crash consistency, performance or model-quality claim.",
            "dtype": dtype_name, "device": "cuda:0", "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor], "training_mode": True,
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller labels plus hashed consumer/helper/package/native and fixture bytes",
            "dropout_probability": DROPOUT_PROBABILITY, "dropout_seed": DROPOUT_SEED,
            "dropout_rng_device": "cuda:0", "cuda_rng_state_bytes": CUDA_RNG_STATE_BYTES,
            "order_seed": ORDER_SEED, "order_regenerations": 1,
            "optimizer": "SGD", "learning_rate": OPTIONS["lr"], "momentum": OPTIONS["momentum"],
            "optimizer_options": dict(OPTIONS), "optimizer_steps": TOTAL_STEPS,
            "producer_steps": CURSOR, "resumed_steps": TOTAL_STEPS - CURSOR,
            "checkpoint_cursor": CURSOR, "checkpoint_epoch": 0,
            "process_ids": pids, "distinct_processes": True,
            "child_launchers": {phase: {name: child[name] for name in
                                ("launcher_pid", "launcher_relation", "parent_pid")}
                                for phase, child in (("producer", produced), ("resumer", resumed))},
            "parent_cuda_fixture_after_children_exit": True,
            "same_native_exact": True, "same_native_max_abs_error": exact_error,
            "primitive_comparisons_passed": True, "primitive_max_abs_error": primitive_error,
            "comparison_tolerance": tolerance, "independent_momentum_recurrence": True,
            "controls": control_results, "checkpoint_sha256": digest, "checkpoint_bytes": len(payload),
            "checkpoint_byte_limit": MAX_CHECKPOINT_BYTES, "checkpoint_unchanged": True,
            "input_sources_unchanged": True, "initial_parameters_unchanged": True,
            "loaded_state_storage_independent": True, "same_parameter_and_optimizer_objects": True,
            "all_parameter_blocks_changed_each_step": True, "checkpoint_metadata": expected,
            "steps": [{"update": row["update"], "batch_id": row["batch_id"],
                       "epoch": row["order_after"]["epoch"], "cursor_before": row["order_before"]["cursor"],
                       "cursor_after": row["order_after"]["cursor"],
                       "cuda_rng_before_sha256": _rng_digest(row["rng_before"]),
                       "cuda_rng_after_sha256": _rng_digest(row["rng_after"]),
                       "order_rng_sha256": _rng_digest(row["order_after"]["generator"]),
                       "dropout_mask_sha256": _rng_digest(row["dropout"]["keep"]),
                       "N": row["inputs"]["raw"].shape[0], "B": row["inputs"]["targets"].shape[0],
                       "offsets": row["inputs"]["offsets"].tolist(), "native_loss": float(row["output"]["loss"]),
                       "primitive_loss": float(primitive_records[index]["output"]["loss"])}
                      for index, row in enumerate(native_records)], "child_timeout_seconds": 120,
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), **identity,
            "child_installed_identities": {"producer": produced["identity"], "resumer": resumed["identity"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--directory", required=True, type=Path, help="New local artifact directory")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--phase", choices=("produce", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        base._source_labels(args.source_revision, args.source_tree)
    except ValueError as error:
        parser.error(str(error))
    if args.phase is not None:
        if args.output is not None or not args.directory.is_dir():
            parser.error("Private phases require an existing artifact directory and no receipt path")
        if (args.phase == "resume") != (args.expected_sha256 is not None):
            parser.error("Only the resume phase requires an expected checkpoint digest")
        _phase(args.dtype, args.directory.resolve(), args.phase, args.expected_sha256,
               args.source_revision, args.source_tree)
        return
    if args.expected_sha256 is not None:
        parser.error("The expected digest belongs only to the private resume phase")
    receipt = {"status": "running", "parent_pid": os.getpid(), "script_sha256": sha256(SCRIPT.read_bytes()).hexdigest()}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write("{}\n")
    try:
        receipt = run(args.dtype, args.directory, args.source_revision, args.source_tree)
    except BaseException as error:
        receipt.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    finally:
        if args.output is not None:
            args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
