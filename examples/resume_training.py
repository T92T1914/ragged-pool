"""Resume three packed CPU training updates in a fresh process.

The fixed five-batch fixture and model come from train_dynamic.py. This eager
example preserves SGD momentum and a completed-update cursor, then checks the
whole resumed trajectory against uninterrupted native and primitive PyTorch
execution. Only checkpoints produced locally by this example are in scope.
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


def _adjacent():
    path = Path(__file__).resolve().with_name("train_dynamic.py")
    specification = importlib.util.spec_from_file_location("resume_dynamic_fixture", path)
    if specification is None or specification.loader is None:
        raise ImportError("The adjacent train_dynamic.py fixture is required")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


dynamic = _adjacent()
CURSOR = 2
TOTAL_STEPS = 5
MAX_CHECKPOINT_BYTES = 128 * 1024
MAX_RESULT_BYTES = 256 * 1024
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0,
           "weight_decay": 0, "nesterov": False, "maximize": False,
           "foreach": False, "differentiable": False, "fused": False}


def installed_identity():
    identity = dynamic.installed_identity()
    identity["fixture_script_sha256"] = identity["script_sha256"]
    identity["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


def fixture(dtype_name):
    return dynamic.fixture(dtype_name)


def optimizer(model):
    return torch.optim.SGD(model.parameters(), **OPTIONS)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if type(value) is dict:
        return {key: _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    return copy.deepcopy(value)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from _tensors(item)
    elif type(value) is list:
        for item in value:
            yield from _tensors(item)


def _storage(value):
    return value.untyped_storage().data_ptr()


def _disjoint(first, second):
    if {_storage(value) for value in _tensors(first)} & {_storage(value) for value in _tensors(second)}:
        raise ValueError("Restored state must use storage separate from checkpoint tensors")


def _ordinary_tensor(value, expected, name):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.layout != torch.strided or value.dtype != expected.dtype
            or value.shape != expected.shape or not value.is_contiguous()
            or value.storage_offset() != 0
            or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            or value.requires_grad or value.is_conj() or value.is_neg()
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1000).any())):
        raise ValueError(f"Invalid finite ordinary CPU tensor for {name}")


def _same_plain(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_plain(actual[key], expected[key]) for key in expected))
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_plain(a, b) for a, b in zip(actual, expected))
    return type(expected) in (str, int, float, bool, type(None)) and actual == expected


def metadata(dtype_name, initial, batches, identity=None):
    """Describe the exact fixed model, data, scripts and installed native bytes."""
    identity = installed_identity() if identity is None else identity
    data = sha256()
    for batch in batches:
        for name in ("raw", "offsets", "targets"):
            value = batch[name]
            data.update(_json([name, str(value.dtype), list(value.shape)]).encode())
            data.update(value.detach().contiguous().numpy().tobytes())
    return {"format": "ragged-pool-fixed-training-state", "schema": 1,
            "dtype": dtype_name, "device": "cpu", "next_update": CURSOR,
            "total_updates": TOTAL_STEPS, "seed": dynamic.SEED,
            "architecture": "encoder3x4-tanh-scorer4x1-pool-head4x1",
            "parameters": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in initial.named_parameters()],
            "optimizer": {"class": "torch.optim.SGD", "options": dict(OPTIONS)},
            "offsets": [list(value) for value in dynamic.OFFSETS],
            "fixture_sha256": data.hexdigest(),
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture_script_sha256": sha256(Path(dynamic.__file__).read_bytes()).hexdigest(),
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "package_version": identity["package_version"],
            "native_library_sha256": identity["native_library_sha256"],
            "stochastic_forward": False, "live_rng_continuation": False}


def _expected_group(count):
    return dict(OPTIONS, params=list(range(count)))


def validate_checkpoint(checkpoint, model, opt, expected_metadata):
    """Validate every field before either model or optimizer state is applied."""
    if type(checkpoint) is not dict or set(checkpoint) != {"metadata", "model", "optimizer"}:
        raise ValueError("Checkpoint must contain exactly metadata, model and optimizer")
    if not _same_plain(checkpoint["metadata"], expected_metadata):
        raise ValueError("Checkpoint metadata does not match the fixed completed-update-two protocol")
    parameters = dict(model.named_parameters())
    descriptor = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                  for name, value in parameters.items()]
    if not _same_plain(descriptor, expected_metadata["parameters"]):
        raise ValueError("Target model parameter order differs from the fixed descriptor")
    state = checkpoint["model"]
    if type(state) is not dict or list(state) != list(parameters):
        raise ValueError("Checkpoint model parameter keys and order must match exactly")
    for name, parameter in parameters.items():
        _ordinary_tensor(state[name], parameter, name)
    optimizer_state = checkpoint["optimizer"]
    if type(optimizer_state) is not dict or set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer state schema")
    if not _same_plain(optimizer_state["param_groups"], [_expected_group(len(parameters))]):
        raise ValueError("Optimizer group order, ordinal IDs or options differ")
    saved = optimizer_state["state"]
    if (type(saved) is not dict or list(saved) != list(range(len(parameters)))
            or any(type(key) is not int for key in saved)):
        raise ValueError("Optimizer state must contain every exact ordinal parameter ID")
    for index, (name, parameter) in enumerate(parameters.items()):
        entry = saved[index]
        if type(entry) is not dict or set(entry) != {"momentum_buffer"}:
            raise ValueError("Each optimizer state needs exactly one momentum buffer")
        _ordinary_tensor(entry["momentum_buffer"], parameter, name + ".momentum_buffer")
    tensors = list(_tensors(checkpoint))
    if len({_storage(value) for value in tensors}) != len(tensors):
        raise ValueError("Checkpoint tensors must have independent storage")
    if (len(opt.param_groups) != 1
            or tuple(id(value) for value in opt.param_groups[0]["params"])
            != tuple(id(value) for value in parameters.values())):
        raise ValueError("Target optimizer must own the target model in declared parameter order")
    group = {key: value for key, value in opt.param_groups[0].items() if key != "params"}
    if not _same_plain(group, OPTIONS):
        raise ValueError("Target optimizer options differ from the fixed policy")


def make_checkpoint(model, opt, expected_metadata):
    """Snapshot a completed update boundary without retaining live parameter storage."""
    if any(value.grad is not None for value in model.parameters()):
        raise ValueError("Clear gradients before saving a completed update boundary")
    checkpoint = {"metadata": copy.deepcopy(expected_metadata),
                  "model": _clone(dict(model.state_dict())),
                  "optimizer": _clone(opt.state_dict())}
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    _disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    return checkpoint


def restore(checkpoint, model, opt, expected_metadata):
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    before = _clone(checkpoint)
    versions = [value._version for value in _tensors(checkpoint)]
    model.load_state_dict(_clone(checkpoint["model"]), strict=True)
    # A same-device/dtype optimizer load can otherwise retain supplied buffers.
    opt.load_state_dict(_clone(checkpoint["optimizer"]))
    opt.zero_grad(set_to_none=True)
    _disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    _compare_tree(checkpoint, before, {"rtol": 0, "atol": 0})
    if versions != [value._version for value in _tensors(checkpoint)]:
        raise AssertionError("Restoration changed checkpoint tensor versions")


def _read_bytes(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a bounded ordinary local artifact file")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise ValueError("Artifact exceeds its fixed byte bound or is empty")
    return payload


def load_checkpoint(path, expected_sha256, model, opt, expected_metadata):
    """Read and validate the same known local bytes once, without applying state."""
    if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A complete lowercase expected SHA256 is required")
    payload = _read_bytes(path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the expected local artifact")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    return checkpoint


def _write_tensor_file(path, value, maximum):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    if len(payload) > maximum:
        raise ValueError("Produced artifact exceeds its fixed byte bound")
    with Path(path).open("xb") as stream:
        stream.write(payload)
    return sha256(payload).hexdigest()


def _compare_tree(actual, expected, tolerance):
    """Compare the complete known step/state structure and return its worst error."""
    if type(expected) is torch.Tensor:
        _ordinary_tensor(actual, expected, "comparison record")
        torch.testing.assert_close(actual, expected, **tolerance)
        return dynamic.difference(actual, expected)
    if type(actual) is not type(expected):
        raise AssertionError("Result structure types differ")
    if type(expected) is dict:
        if list(actual) != list(expected):
            raise AssertionError("Result dictionary keys or order differ")
        return max((_compare_tree(actual[key], expected[key], tolerance) for key in expected), default=0.0)
    if type(expected) is list:
        if len(actual) != len(expected):
            raise AssertionError("Result list lengths differ")
        return max((_compare_tree(left, right, tolerance) for left, right in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Result scalar metadata differs")
    return 0.0


def step(model, opt, batch, index, reference=False):
    """Take one checked update with an independently evaluated momentum recurrence."""
    names = dict(model.named_parameters())
    parameter_ids = tuple(id(value) for value in names.values())
    if tuple(id(value) for group in opt.param_groups for value in group["params"]) != parameter_ids:
        raise AssertionError("Optimizer/model parameter ownership differs")
    before = _clone(names)
    old_momentum = {name: _clone(opt.state[value].get("momentum_buffer"))
                    for name, value in names.items()}
    opt.zero_grad(set_to_none=True)
    raw = batch["raw"].detach().clone().requires_grad_()
    pool = dynamic.primitive_pool(batch["python_boundaries"]) if reference else dynamic.ragged_pool
    call = lambda features, offsets: model(features, offsets, pool)
    output, gradients, _, _ = dynamic.forward_backward(call, model, raw, batch["offsets"], batch["targets"])
    expected_momentum = {name: (gradient.clone() if old_momentum[name] is None
                                else old_momentum[name].mul(OPTIONS["momentum"]).add(gradient))
                         for name, gradient in gradients.items()}
    expected_parameters = {name: before[name].add(expected_momentum[name], alpha=-OPTIONS["lr"])
                           for name in names}
    opt.step()
    parameters = _clone(dict(model.named_parameters()))
    momentum = {name: _clone(opt.state[value]["momentum_buffer"]) for name, value in names.items()}
    _compare_tree(momentum, expected_momentum, {"rtol": 0, "atol": 0})
    _compare_tree(parameters, expected_parameters, {"rtol": 0, "atol": 0})
    if any(dynamic.difference(parameters[name], before[name]) <= 0 for name in names):
        raise AssertionError("Every declared update must change every parameter block")
    if parameter_ids != tuple(id(value) for value in model.parameters()):
        raise AssertionError("An optimizer update replaced parameter objects")
    empty = [row for row, (left, right) in enumerate(zip(batch["python_boundaries"],
                                                       batch["python_boundaries"][1:])) if left == right]
    torch.testing.assert_close(output["pooled"][empty], torch.zeros_like(output["pooled"][empty]), rtol=0, atol=0)
    opt.zero_grad(set_to_none=True)
    return {"update": index + 1,
            "inputs": {name: _clone(batch[name]) for name in ("raw", "offsets", "targets")},
            "output": output, "parameter_gradients": gradients,
            "parameters": parameters, "momentum": momentum}


def run_steps(model, opt, batches, start=0, stop=TOTAL_STEPS, reference=False):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= TOTAL_STEPS:
        raise ValueError("Use a nonempty subset of the fixed five update positions")
    return [step(model, opt, batches[index], index, reference=reference) for index in range(start, stop)]


def _threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _phase(dtype_name, directory, phase, expected_sha256):
    _threads()
    identity = installed_identity()
    initial, batches, _ = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity)
    model = copy.deepcopy(initial)
    opt = optimizer(model)
    checkpoint_path = directory / "checkpoint.pt"
    if phase == "produce":
        records = run_steps(model, opt, batches, stop=CURSOR)
        checkpoint = make_checkpoint(model, opt, expected)
        digest = _write_tensor_file(checkpoint_path, checkpoint, MAX_CHECKPOINT_BYTES)
        result_path = directory / "producer-results.pt"
    else:
        checkpoint = load_checkpoint(checkpoint_path, expected_sha256, model, opt, expected)
        original = _clone(checkpoint)
        versions = [value._version for value in _tensors(checkpoint)]
        restore(checkpoint, model, opt, expected)
        records = run_steps(model, opt, batches, start=CURSOR)
        _compare_tree(checkpoint, original, {"rtol": 0, "atol": 0})
        if versions != [value._version for value in _tensors(checkpoint)]:
            raise AssertionError("Resumed updates changed the loaded checkpoint")
        _disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
        digest = sha256(_read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)).hexdigest()
        if digest != expected_sha256:
            raise AssertionError("The checkpoint file changed during resumed training")
        result_path = directory / "resume-results.pt"
    result = {"phase": phase, "pid": os.getpid(), "metadata": expected,
              "identity": identity, "checkpoint_sha256": digest, "records": records}
    result_digest = _write_tensor_file(result_path, result, MAX_RESULT_BYTES)
    print(json.dumps({"phase": phase, "pid": os.getpid(), "checkpoint_sha256": digest,
                      "result_sha256": result_digest}, allow_nan=False))


def _child(dtype_name, directory, phase, expected_sha256=None):
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--dtype", dtype_name,
               "--directory", str(directory), "--phase", phase]
    if expected_sha256 is not None:
        command.extend(["--expected-sha256", expected_sha256])
    # The parent performs no Torch work until this bounded child has exited.
    try:
        child = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=120, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
    payload = _read_bytes(directory / result_name, MAX_RESULT_BYTES)
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


def run(dtype_name, directory):
    """Run sequential uninterrupted, producer and fresh resumed trajectories."""
    _threads()
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    identity = installed_identity()
    initial, batches, tolerance = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity)
    inputs = {f"batch{index}:{name}": batch[name] for index, batch in enumerate(batches)
              for name in ("raw", "offsets", "targets")}
    source = dynamic.snapshot(inputs)
    native, primitive = copy.deepcopy(initial), copy.deepcopy(initial)
    native_records = run_steps(native, optimizer(native), batches)
    primitive_records = run_steps(primitive, optimizer(primitive), batches, reference=True)
    primitive_error = _compare_tree(native_records, primitive_records, tolerance)
    produced = _child(dtype_name, directory, "produce")
    digest = produced["checkpoint_sha256"]
    if _json(produced["metadata"]) != _json(expected):
        raise AssertionError("Producer metadata differs from parent protocol")
    resumed = _child(dtype_name, directory, "resume", digest)
    if _json(resumed["metadata"]) != _json(expected):
        raise AssertionError("Resumer metadata differs from parent protocol")
    pids = {"parent": os.getpid(), "producer": produced["pid"], "resumer": resumed["pid"]}
    if len(set(pids.values())) != 3:
        raise AssertionError("Parent, producer and resumer must be distinct processes")
    combined = produced["records"] + resumed["records"]
    exact_error = _compare_tree(combined, native_records, {"rtol": 0, "atol": 0})
    checkpoint_path = directory / "checkpoint.pt"
    control_model = copy.deepcopy(initial)
    control_optimizer = optimizer(control_model)
    checkpoint = load_checkpoint(checkpoint_path, digest, control_model, control_optimizer, expected)
    checkpoint_before = _clone(checkpoint)
    # This deliberate control retains correct model/cursor but discards momentum.
    control_model.load_state_dict(_clone(checkpoint["model"]), strict=True)
    control_records = run_steps(control_model, control_optimizer, batches, start=CURSOR)
    control_error = max(dynamic.difference(control_records[-1]["parameters"][name],
                                          native_records[-1]["parameters"][name])
                        for name in native_records[-1]["parameters"])
    threshold = 1e-7 if dtype_name == "float32" else 1e-10
    if control_error <= threshold:
        raise AssertionError("The reset-optimizer control must visibly diverge")
    _compare_tree(checkpoint, checkpoint_before, {"rtol": 0, "atol": 0})
    dynamic.unchanged(inputs, source)
    payload = _read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != digest:
        raise AssertionError("The original checkpoint bytes changed")
    for result in (produced, resumed):
        if result["identity"]["native_library_sha256"] != identity["native_library_sha256"]:
            raise AssertionError("Child native identity differs from parent")
    return {"status": "passed", "method": "state_dict.packed_training_resume",
            "purpose": "Completed-step eager packed CPU training continuation in a fresh process",
            "scope": "Known locally produced state only. No stochastic replay, sandbox, crash-consistency, portability or performance claim.",
            "dtype": dtype_name, "device": "cpu", "optimizer": "SGD",
            "learning_rate": OPTIONS["lr"], "momentum": OPTIONS["momentum"],
            "optimizer_options": dict(OPTIONS), "optimizer_steps": TOTAL_STEPS,
            "producer_steps": CURSOR, "resumed_steps": TOTAL_STEPS - CURSOR,
            "checkpoint_cursor": CURSOR, "process_ids": pids, "distinct_processes": True,
            "same_native_exact": True, "same_native_max_abs_error": exact_error,
            "primitive_comparisons_passed": True, "primitive_max_abs_error": primitive_error,
            "comparison_tolerance": tolerance, "independent_momentum_recurrence": True,
            "model_only_reset_optimizer_diverged": True,
            "model_only_final_parameter_max_abs_error": control_error,
            "model_only_divergence_threshold": threshold,
            "checkpoint_sha256": digest, "checkpoint_bytes": len(payload),
            "checkpoint_byte_limit": MAX_CHECKPOINT_BYTES, "checkpoint_unchanged": True,
            "input_sources_unchanged": True, "loaded_state_storage_independent": True,
            "checkpoint_metadata": expected,
            "no_live_rng_continuation": True, "steps": [{"update": record["update"],
            "N": record["inputs"]["raw"].shape[0], "B": record["inputs"]["targets"].shape[0],
            "offsets": record["inputs"]["offsets"].tolist(), "loss": float(record["output"]["loss"])}
            for record in native_records], "child_timeout_seconds": 120,
            "python_version": sys.version.split()[0], "torch_version": torch.__version__,
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
            parser.error("Private phases require an existing artifact directory and no receipt path")
        if (arguments.phase == "resume") != (arguments.expected_sha256 is not None):
            parser.error("Only the resume phase requires an expected checkpoint digest")
        _phase(arguments.dtype, arguments.directory.resolve(), arguments.phase, arguments.expected_sha256)
        return
    if arguments.expected_sha256 is not None:
        parser.error("The expected digest belongs only to the private resume phase")
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    result = run(arguments.dtype, arguments.directory)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
