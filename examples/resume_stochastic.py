"""Resume a fixed CPU dropout and shuffled-batch consumer in a fresh process.

This example extends the adjacent deterministic training example with two
explicit random streams. It preserves CPU dropout RNG and a dedicated batch
order generator at a completed update boundary. It accepts only its own bounded
local checkpoint, not arbitrary models, workers or checkpoint files.
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


def _adjacent():
    path = Path(__file__).resolve().with_name("resume_training.py")
    spec = importlib.util.spec_from_file_location("stochastic_resume_base", path)
    if spec is None or spec.loader is None:
        raise ImportError("The adjacent resume_training.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _adjacent()
dynamic = base.dynamic
CURSOR, TOTAL_STEPS = 2, 7
DROPOUT_PROBABILITY, DROPOUT_SEED, ORDER_SEED = 0.25, 2011, 2111
MAX_CHECKPOINT_BYTES, MAX_RESULT_BYTES = 128 * 1024, 512 * 1024
MAX_RNG_STATE_BYTES = 8192
OPTIONS = base.OPTIONS
optimizer = base.optimizer
_clone, _tensors = base._clone, base._tensors


def _ordinary_bytes(value, expected, name):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.layout != torch.strided or value.dtype != torch.uint8
            or value.shape != expected.shape or not value.is_contiguous()
            or value.storage_offset() != 0 or value.untyped_storage().nbytes() != value.numel()
            or value.requires_grad or value.is_conj() or value.is_neg()):
        raise ValueError(f"Invalid owned ordinary uint8 CPU tensor for {name}")


def _compare_tree(actual, expected, tolerance):
    # N19's floating tensor envelope does not describe opaque RNG bytes.
    if type(expected) is torch.Tensor and expected.dtype == torch.uint8:
        _ordinary_bytes(actual, expected, "byte-state comparison")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return 0.0
    if type(expected) is torch.Tensor:
        return base._compare_tree(actual, expected, tolerance)
    if type(actual) is not type(expected):
        raise AssertionError("Result structure types differ")
    if type(expected) is dict:
        if list(actual) != list(expected):
            raise AssertionError("Result dictionary keys or order differ")
        return max((_compare_tree(actual[key], expected[key], tolerance) for key in expected), default=0.0)
    if type(expected) is list:
        if len(actual) != len(expected):
            raise AssertionError("Result list lengths differ")
        return max((_compare_tree(a, b, tolerance) for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Result scalar metadata differs")
    return 0.0


class StochasticRegressor(dynamic.SetRegressor):
    def forward(self, raw, offsets, pool=dynamic.ragged_pool):
        if not self.training:
            raise ValueError("This fixed consumer requires explicit training mode")
        encoded = self.encoder(raw).tanh().contiguous()
        dropped = F.dropout(encoded, p=DROPOUT_PROBABILITY, training=self.training).contiguous()
        logits = self.scorer(dropped).squeeze(-1).contiguous()
        pooled = pool(dropped, logits, offsets)
        # These detached observations are not model parameters or buffers.
        self.last_dropout = {"encoded": _clone(encoded), "dropped": _clone(dropped),
                             "logits": _clone(logits), "keep": (dropped != 0).to(torch.uint8)}
        return self.head(pooled).squeeze(-1), pooled


def installed_identity():
    if sys.version.split()[0] != "3.11.8" or str(torch.__version__) != "2.9.1+cpu":
        raise AssertionError("Use the qualified Python 3.11.8 and Torch 2.9.1+cpu environment")
    identity = base.installed_identity()
    identity["resume_script_sha256"] = identity["script_sha256"]
    identity["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


def fixture(dtype_name):
    initial, batches, tolerance = dynamic.fixture(dtype_name)
    model = StochasticRegressor().to(device="cpu", dtype=dynamic.DTYPES[dtype_name])
    model.load_state_dict(initial.state_dict(), strict=True)
    model.train(True)
    return model, batches, tolerance


class BatchOrder:
    """Own the next selection from five fixtures and its separate CPU RNG."""
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


def metadata(dtype_name, initial, batches, identity=None):
    identity = installed_identity() if identity is None else identity
    result = base.metadata(dtype_name, initial, batches, identity)
    result.update({"format": "ragged-pool-fixed-stochastic-training-state", "schema": 1,
                   "total_updates": TOTAL_STEPS, "script_sha256": identity["script_sha256"],
                   "resume_script_sha256": sha256(Path(base.__file__).read_bytes()).hexdigest(),
                   "stochastic_forward": True, "live_rng_continuation": True,
                   "dropout_probability": DROPOUT_PROBABILITY, "dropout_seed": DROPOUT_SEED,
                   "order_seed": ORDER_SEED, "order_length": 5, "checkpoint_epoch": 0,
                   "checkpoint_order_cursor": CURSOR, "training_mode": True})
    return result


def _rng_state(value, name):
    expected = torch.get_rng_state()
    if expected.numel() > MAX_RNG_STATE_BYTES:
        raise ValueError("The qualified CPU RNG state exceeds its explicit bound")
    _ordinary_bytes(value, expected, name)
    # A correctly shaped uint8 tensor can still contain an invalid MT state.
    # Scratch validation must never set either live random stream.
    torch.Generator(device="cpu").set_state(value.clone())


def validate_checkpoint(checkpoint, model, opt, order, expected_metadata):
    """Admit every saved field on scratch state before applying any target."""
    if type(checkpoint) is not dict or set(checkpoint) != {
            "metadata", "model", "optimizer", "cpu_rng", "order", "training"}:
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
    _rng_state(checkpoint["cpu_rng"], "CPU dropout RNG")
    _rng_state(state["generator"], "batch-order RNG")
    independently_seeded = torch.Generator(device="cpu").manual_seed(ORDER_SEED)
    first_order = torch.randperm(5, generator=independently_seeded).tolist()
    if permutation != first_order or not torch.equal(state["generator"], independently_seeded.get_state()):
        raise ValueError("Saved order and generator must match the declared first seeded permutation")
    tensors = list(_tensors(checkpoint))
    if len({base._storage(value) for value in tensors}) != len(tensors):
        raise ValueError("Every checkpoint tensor must own independent storage")


def make_checkpoint(model, opt, order, expected_metadata):
    checkpoint = base.make_checkpoint(model, opt, expected_metadata)
    checkpoint.update({"cpu_rng": torch.get_rng_state().clone(), "order": order.state_dict(),
                       "training": model.training})
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    return checkpoint


def restore(checkpoint, model, opt, order, expected_metadata):
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    original = _clone(checkpoint)
    versions = [value._version for value in _tensors(checkpoint)]
    # The existing model/optimizer admission has already run before any load.
    base.restore({key: checkpoint[key] for key in ("metadata", "model", "optimizer")},
                 model, opt, expected_metadata)
    model.train(checkpoint["training"])
    order.apply_validated(checkpoint["order"])
    # Restore this stream last, after model construction, loads and order setup.
    torch.set_rng_state(checkpoint["cpu_rng"].clone())
    base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict(),
                               "cpu_rng": torch.get_rng_state(), "order": order.state_dict()})
    _compare_tree(checkpoint, original, {"rtol": 0, "atol": 0})
    if versions != [value._version for value in _tensors(checkpoint)]:
        raise AssertionError("Restoration changed checkpoint tensors")


def load_checkpoint(path, expected_sha256, model, opt, order, expected_metadata):
    if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A complete lowercase expected SHA256 is required")
    payload = base._read_bytes(path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the expected local artifact")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_checkpoint(checkpoint, model, opt, order, expected_metadata)
    return checkpoint


def step(model, opt, batches, order, index, reference=False):
    order_before, rng_before = order.state_dict(), torch.get_rng_state().clone()
    batch_id = order.next_batch()
    record = base.step(model, opt, batches[batch_id], index, reference=reference)
    evidence = model.last_dropout
    encoded, dropped, keep = evidence["encoded"], evidence["dropped"], evidence["keep"].bool()
    if bool((encoded == 0).any()) or not bool(keep.any()) or bool(keep.all()):
        raise AssertionError("The fixed dropout witness needs nonzero encoded values, drops and survivors")
    torch.testing.assert_close(dropped, torch.where(keep, encoded * (1 / (1 - DROPOUT_PROBABILITY)),
                                                  torch.zeros_like(encoded)), rtol=0, atol=0)
    record.update({"batch_id": batch_id, "dropout": _clone(evidence), "rng_before": rng_before,
                   "rng_after": torch.get_rng_state().clone(), "order_before": order_before,
                   "order_after": order.state_dict(), "training": model.training})
    return record


def run_steps(model, opt, batches, order, start=0, stop=TOTAL_STEPS, reference=False):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= TOTAL_STEPS:
        raise ValueError("Use a nonempty subset of the fixed seven updates")
    return [step(model, opt, batches, order, index, reference) for index in range(start, stop)]


def _phase(dtype_name, directory, phase, expected_sha256):
    base._threads()
    identity = installed_identity()
    initial, batches, _ = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity)
    model, order = copy.deepcopy(initial), BatchOrder()
    opt = optimizer(model)
    checkpoint_path = directory / "checkpoint.pt"
    if phase == "produce":
        torch.manual_seed(DROPOUT_SEED)
        records = run_steps(model, opt, batches, order, stop=CURSOR)
        checkpoint = make_checkpoint(model, opt, order, expected)
        digest = base._write_tensor_file(checkpoint_path, checkpoint, MAX_CHECKPOINT_BYTES)
        result_path = directory / "producer-results.pt"
    else:
        checkpoint = load_checkpoint(checkpoint_path, expected_sha256, model, opt, order, expected)
        original, versions = _clone(checkpoint), [value._version for value in _tensors(checkpoint)]
        restore(checkpoint, model, opt, order, expected)
        records = run_steps(model, opt, batches, order, start=CURSOR)
        _compare_tree(checkpoint, original, {"rtol": 0, "atol": 0})
        if versions != [value._version for value in _tensors(checkpoint)]:
            raise AssertionError("Resumed updates changed the loaded checkpoint")
        base._disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict(),
                                   "order": order.state_dict(), "cpu_rng": torch.get_rng_state()})
        digest = sha256(base._read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)).hexdigest()
        if digest != expected_sha256:
            raise AssertionError("Checkpoint bytes changed during resumed training")
        result_path = directory / "resume-results.pt"
    result = {"phase": phase, "pid": os.getpid(), "metadata": expected, "identity": identity,
              "checkpoint_sha256": digest, "records": records}
    result_digest = base._write_tensor_file(result_path, result, MAX_RESULT_BYTES)
    print(json.dumps({"phase": phase, "pid": os.getpid(), "checkpoint_sha256": digest,
                      "result_sha256": result_digest}, allow_nan=False))


def _child(dtype_name, directory, phase, expected_sha256=None):
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--dtype", dtype_name,
               "--directory", str(directory), "--phase", phase]
    if expected_sha256 is not None:
        command.extend(["--expected-sha256", expected_sha256])
    # No numerical work occurs in this parent while the one child is active.
    try:
        child = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=120,
                               check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        stdout, stderr = child.stdout, child.stderr
    except subprocess.TimeoutExpired as error:
        (directory / (phase + ".stdout.log")).write_bytes(error.stdout or b"")
        (directory / (phase + ".stderr.log")).write_bytes(error.stderr or b"")
        raise RuntimeError(f"The {phase} child exceeded 120 seconds") from error
    (directory / (phase + ".stdout.log")).write_bytes(stdout)
    (directory / (phase + ".stderr.log")).write_bytes(stderr)
    if child.returncode != 0:
        raise RuntimeError(f"The {phase} child failed with exit code {child.returncode}. Inspect its retained logs.")
    status = json.loads(stdout.decode("utf-8"))
    result_name = "producer-results.pt" if phase == "produce" else "resume-results.pt"
    payload = base._read_bytes(directory / result_name, MAX_RESULT_BYTES)
    if sha256(payload).hexdigest() != status["result_sha256"]:
        raise AssertionError("Child result bytes differ from its completion report")
    result = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    if (type(result) is not dict
            or set(result) != {"phase", "pid", "metadata", "identity", "checkpoint_sha256", "records"}
            or result["phase"] != phase or type(result["pid"]) is not int
            or result["pid"] != status["pid"]
            or result["checkpoint_sha256"] != status["checkpoint_sha256"]):
        raise AssertionError("Child completion metadata differs from its bounded result")
    return result


def controls(initial, batches, expected, checkpoint, native_records):
    """Deliberately break live state only after a valid complete restoration."""
    results = {}
    threshold = 1e-7 if expected["dtype"] == "float32" else 1e-10
    for case in ("reset_dropout_rng", "omit_dropout_rng", "reset_order_generator",
                 "reset_optimizer", "repeat_cursor", "skip_cursor"):
        model, order = copy.deepcopy(initial), BatchOrder()
        opt = optimizer(model)
        torch.manual_seed(DROPOUT_SEED + 1)
        omitted_state = torch.get_rng_state().clone()
        restore(checkpoint, model, opt, order, expected)
        if case == "reset_dropout_rng": torch.manual_seed(DROPOUT_SEED)
        elif case == "omit_dropout_rng": torch.set_rng_state(omitted_state)
        elif case == "reset_order_generator": order.generator.manual_seed(ORDER_SEED)
        elif case == "reset_optimizer": opt = optimizer(model)
        elif case == "repeat_cursor": order.cursor = CURSOR - 1
        elif case == "skip_cursor": order.cursor = CURSOR + 1
        records = run_steps(model, opt, batches, order, start=CURSOR)
        drift = max(dynamic.difference(records[-1]["parameters"][name], native_records[-1]["parameters"][name])
                    for name in native_records[-1]["parameters"])
        if drift <= threshold:
            raise AssertionError(f"The deliberate {case} control must diverge")
        if case == "reset_order_generator":
            expected_suffix = [row["batch_id"] for row in native_records[CURSOR:]]
            if [row["batch_id"] for row in records][:3] != expected_suffix[:3]:
                raise AssertionError("Resetting only order RNG must preserve the current permutation suffix")
            if [row["batch_id"] for row in records][3:] == expected_suffix[3:]:
                raise AssertionError("Order reset must be exposed after the next permutation is generated")
        results[case] = {"final_parameter_max_abs_error": drift, "divergence_threshold": threshold,
                         "diverged": True, "batch_ids": [row["batch_id"] for row in records],
                         "live_state_bypass_after_valid_restore": True}
    return results


def _rng_digest(value):
    return sha256(value.numpy().tobytes()).hexdigest()


def run(dtype_name, directory):
    base._threads()
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    identity = installed_identity()
    initial, batches, tolerance = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity)
    sources = {f"batch{index}:{name}": batch[name] for index, batch in enumerate(batches)
               for name in ("raw", "offsets", "targets")}
    sources.update({"initial:" + name: value for name, value in initial.named_parameters()})
    snapshots = dynamic.snapshot(sources)
    native, primitive = copy.deepcopy(initial), copy.deepcopy(initial)
    torch.manual_seed(DROPOUT_SEED)
    native_records = run_steps(native, optimizer(native), batches, BatchOrder())
    torch.manual_seed(DROPOUT_SEED)
    primitive_records = run_steps(primitive, optimizer(primitive), batches, BatchOrder(), reference=True)
    primitive_error = _compare_tree(native_records, primitive_records, tolerance)
    produced = _child(dtype_name, directory, "produce")
    digest = produced["checkpoint_sha256"]
    resumed = _child(dtype_name, directory, "resume", digest)
    for result in (produced, resumed):
        if base._json(result["metadata"]) != base._json(expected):
            raise AssertionError("Child metadata differs from the exact parent protocol")
        if result["identity"]["native_library_sha256"] != identity["native_library_sha256"]:
            raise AssertionError("Child native identity differs from parent")
    pids = {"parent": os.getpid(), "producer": produced["pid"], "resumer": resumed["pid"]}
    if len(set(pids.values())) != 3:
        raise AssertionError("Parent, producer and fresh resumer must be distinct processes")
    exact_error = _compare_tree(produced["records"] + resumed["records"], native_records, {"rtol": 0, "atol": 0})
    target, order = copy.deepcopy(initial), BatchOrder()
    opt = optimizer(target)
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint = load_checkpoint(checkpoint_path, digest, target, opt, order, expected)
    original = _clone(checkpoint)
    control_results = controls(initial, batches, expected, checkpoint, native_records)
    _compare_tree(checkpoint, original, {"rtol": 0, "atol": 0})
    dynamic.unchanged(sources, snapshots)
    payload = base._read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != digest:
        raise AssertionError("The original checkpoint bytes changed")
    if native_records[-1]["order_after"]["epoch"] != 1:
        raise AssertionError("The resumed suffix must cross one actual order regeneration")
    return {"status": "passed", "method": "state_dict.stochastic_packed_training_resume",
            "purpose": "Fixed CPU dropout and shuffled-batch continuation in a fresh process",
            "scope": "Known local CPU state only. No general stochastic replay, DataLoader workers, Python/NumPy RNG, CUDA, distributed, sandbox, performance or model-quality claim.",
            "dtype": dtype_name, "device": "cpu", "training_mode": True,
            "dropout_probability": DROPOUT_PROBABILITY, "dropout_seed": DROPOUT_SEED,
            "order_seed": ORDER_SEED, "order_regenerations": 1,
            "optimizer": "SGD", "learning_rate": OPTIONS["lr"], "momentum": OPTIONS["momentum"],
            "optimizer_options": dict(OPTIONS), "optimizer_steps": TOTAL_STEPS,
            "producer_steps": CURSOR, "resumed_steps": TOTAL_STEPS - CURSOR,
            "checkpoint_cursor": CURSOR, "checkpoint_epoch": 0,
            "process_ids": pids, "distinct_processes": True,
            "same_native_exact": True, "same_native_max_abs_error": exact_error,
            "primitive_comparisons_passed": True, "primitive_max_abs_error": primitive_error,
            "comparison_tolerance": tolerance, "independent_momentum_recurrence": True,
            "controls": control_results, "checkpoint_sha256": digest, "checkpoint_bytes": len(payload),
            "checkpoint_byte_limit": MAX_CHECKPOINT_BYTES, "checkpoint_unchanged": True,
            "input_sources_unchanged": True, "initial_parameters_unchanged": True,
            "loaded_state_storage_independent": True, "checkpoint_metadata": expected,
            "steps": [{"update": row["update"], "batch_id": row["batch_id"],
                       "epoch": row["order_after"]["epoch"], "cursor_before": row["order_before"]["cursor"],
                       "cursor_after": row["order_after"]["cursor"],
                       "default_rng_before_sha256": _rng_digest(row["rng_before"]),
                       "default_rng_after_sha256": _rng_digest(row["rng_after"]),
                       "order_rng_sha256": _rng_digest(row["order_after"]["generator"]),
                       "N": row["inputs"]["raw"].shape[0], "B": row["inputs"]["targets"].shape[0],
                       "offsets": row["inputs"]["offsets"].tolist(), "loss": float(row["output"]["loss"])}
                      for row in native_records], "child_timeout_seconds": 120,
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "python_executable": sys.executable, "working_directory": str(Path.cwd()),
            "installed_package": True, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), **identity,
            "child_installed_identities": {"producer": produced["identity"], "resumer": resumed["identity"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--directory", required=True, type=Path, help="New local artifact directory")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    parser.add_argument("--phase", choices=("produce", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.phase is not None:
        if arguments.output is not None or not arguments.directory.is_dir():
            parser.error("Private phases need an existing artifact directory and no receipt path")
        if (arguments.phase == "resume") != (arguments.expected_sha256 is not None):
            parser.error("Only the resume phase needs an expected checkpoint digest")
        _phase(arguments.dtype, arguments.directory.resolve(), arguments.phase, arguments.expected_sha256)
        return
    if arguments.expected_sha256 is not None:
        parser.error("The expected digest belongs only to the private resume phase")
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    text = json.dumps(run(arguments.dtype, arguments.directory), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
