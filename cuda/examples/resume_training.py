"""Resume one known CUDA training trajectory in a fresh process.

Copy this file, train_compiled.py and train_cuda.py outside the checkout. The
producer saves completed update two, then exits before a new resumer restores
model parameters and SGD momentum. Explicit CPU staging belongs to this local
consumer, not to the native pooling API. No stochastic or general recovery
service, hostile-file safety, portability or performance claim is made.
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
import stat
import subprocess
import sys

import torch


SCRIPT = Path(__file__).resolve()
FIXTURE_SCRIPT = SCRIPT.with_name("train_compiled.py")
TRAINING_SCRIPT = SCRIPT.with_name("train_cuda.py")
_spec = importlib.util.spec_from_file_location("_cuda_resume_fixture", FIXTURE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError("Copy train_compiled.py and train_cuda.py beside this consumer")
dynamic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dynamic)
eager = dynamic.eager
DTYPES = eager.DTYPES
CURSOR, TOTAL_STEPS = 2, 5
MAX_CHECKPOINT_BYTES, MAX_RESULT_BYTES = 128 * 1024, 256 * 1024
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0,
           "weight_decay": 0, "nesterov": False, "maximize": False,
           "foreach": False, "differentiable": False, "fused": False}
STATE_SHAPES = {"encoder.weight": (4, 3), "encoder.bias": (4,),
                "scorer.weight": (1, 4), "head.weight": (1, 4), "head.bias": (1,)}
PACKAGE_FILES = ("__init__.py", "_registration.py", "_validation.py", "padded.py", "reference.py")
EXACT = {"rtol": 0, "atol": 0}


def _source_labels(revision, tree):
    for value in (revision, tree):
        if value is not None and (type(value) is not str or not re.fullmatch(r"[0-9a-f]{40}", value)):
            raise ValueError("Source labels must be complete lowercase SHA1 identifiers")


def installed_identity():
    """Check the ordinary installed dependency without creating CUDA tensors."""
    if (sys.version.split()[0], str(torch.__version__), torch.version.cuda) != (
            "3.11.8", "2.9.1+cu128", "12.8"):
        raise RuntimeError("Use the pinned Python3.11.8/PyTorch2.9.1 cu128 environment")
    identity = eager.installed_identity()
    training_hash = sha256(TRAINING_SCRIPT.read_bytes()).hexdigest()
    if identity.pop("script_sha256") != training_hash:
        raise AssertionError("The adjacent helper must identify its own training script")
    directory = Path(identity["package_path"]).parent
    identity.update({"script_sha256": sha256(SCRIPT.read_bytes()).hexdigest(),
                     "fixture_script_sha256": sha256(FIXTURE_SCRIPT.read_bytes()).hexdigest(),
                     "training_script_sha256": training_hash,
                     "package_python_sha256": {name: sha256((directory / name).read_bytes()).hexdigest()
                                               for name in PACKAGE_FILES}})
    return identity


def fixture(dtype_name):
    if dtype_name not in DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA0 device is unavailable")
    torch.cuda.set_device(0)
    initial, batches = dynamic.fixture(dtype_name, torch.device("cuda:0"), dynamic=True)
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    return initial, batches, tolerance


def optimizer(model):
    return torch.optim.SGD(model.parameters(), **OPTIONS)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clone(value):
    """Detach and explicitly stage every retained tensor into owned CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone().contiguous()
    if type(value) is dict:
        return {key: _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from _tensors(item)
    elif type(value) in (list, tuple):
        for item in value:
            yield from _tensors(item)


def _storage(value):
    return str(value.device), value.untyped_storage().data_ptr()


def snapshot(tensors):
    return {name: (value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().clone(),
                   value._version, id(value), _storage(value), tuple(value.stride()),
                   value.storage_offset(), value.dtype, tuple(value.shape))
            for name, value in tensors.items()}


def unchanged(tensors, before):
    if list(tensors) != list(before):
        raise AssertionError("Tensor ownership keys changed")
    for name, value in tensors.items():
        saved = before[name]
        actual = (value._version, id(value), _storage(value), tuple(value.stride()),
                  value.storage_offset(), value.dtype, tuple(value.shape))
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        if actual != saved[1:] or not torch.equal(data, saved[0]):
            raise AssertionError(f"Changed {name} bytes, versions or storage ownership")


def _tree_snapshot(value):
    return snapshot({str(index): tensor for index, tensor in enumerate(_tensors(value))})


def _tree_unchanged(value, before):
    unchanged({str(index): tensor for index, tensor in enumerate(_tensors(value))}, before)


def _disjoint(first, second):
    if {_storage(value) for value in _tensors(first)} & {_storage(value) for value in _tensors(second)}:
        raise ValueError("Retained state must own storage separate from live state")


def _ordinary_tensor(value, expected, name, device="cpu"):
    if (type(value) is not torch.Tensor or value.device != torch.device(device)
            or value.layout != torch.strided or value.dtype != expected.dtype
            or value.shape != expected.shape or not value.is_contiguous()
            or value.storage_offset() != 0 or value.requires_grad
            or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            or value.is_conj() or value.is_neg() or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1000).any())):
        raise ValueError(f"Invalid bounded ordinary {device} tensor for {name}")


def _same_plain(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_plain(actual[key], expected[key]) for key in expected))
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_plain(a, b) for a, b in zip(actual, expected))
    return type(expected) in (str, int, float, bool, type(None)) and actual == expected


def metadata(dtype_name, initial, batches, identity=None, source_revision=None, source_tree=None):
    """Bind the fixed fixture, completed cursor, source labels and installed bytes."""
    _source_labels(source_revision, source_tree)
    identity = installed_identity() if identity is None else identity
    data = sha256()
    for batch in batches:
        for name in ("raw", "offsets", "targets"):
            value = _clone(batch[name])
            data.update(_json([name, str(value.dtype), list(value.shape)]).encode())
            data.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"format": "ragged-pool-cuda-fixed-training-state", "schema": 1,
            "dtype": dtype_name, "device": "cuda:0", "checkpoint_tensor_device": "cpu",
            "next_update": CURSOR, "total_updates": TOTAL_STEPS, "seed": eager.SEED,
            "architecture": "encoder3x4-tanh-scorer4x1-pool-head4x1", "training": True,
            "parameters": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in initial.named_parameters()],
            "optimizer": {"class": "torch.optim.SGD", "options": dict(OPTIONS)},
            "offsets": [list(batch["boundaries"]) for batch in batches],
            "fixture_sha256": data.hexdigest(), "source_revision": source_revision,
            "source_tree": source_tree, "script_sha256": identity["script_sha256"],
            "fixture_script_sha256": identity["fixture_script_sha256"],
            "training_script_sha256": identity["training_script_sha256"],
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda, "package_version": identity["package_version"],
            "native_library_sha256": identity["native_library_sha256"],
            "package_python_sha256": dict(identity["package_python_sha256"]),
            "stochastic_forward": False, "live_rng_continuation": False}


def _bindings(model, opt):
    parameters = dict(model.named_parameters())
    if (list(parameters) != list(STATE_SHAPES) or len(model.state_dict()) != len(parameters)
            or not model.training or type(opt) is not torch.optim.SGD):
        raise ValueError("Use the declared training model and SGD optimizer")
    for name, value in parameters.items():
        if (value.shape != STATE_SHAPES[name] or value.device != torch.device("cuda:0")
                or value.dtype not in DTYPES.values() or value.layout != torch.strided
                or not value.is_contiguous() or value.storage_offset() != 0
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                or value.is_conj() or value.is_neg() or not value.requires_grad):
            raise ValueError("Target parameters must retain their declared CUDA0 representation")
    if len({_storage(value) for value in parameters.values()}) != len(parameters):
        raise ValueError("Target parameters must own independent storage")
    if (len(opt.param_groups) != 1 or tuple(id(value) for value in opt.param_groups[0]["params"])
            != tuple(id(value) for value in parameters.values())
            or not _same_plain({key: value for key, value in opt.param_groups[0].items()
                                if key != "params"}, OPTIONS)):
        raise ValueError("Target optimizer must own the ordered live model with exact options")
    return (id(model), id(opt), tuple(id(value) for value in parameters.values()),
            tuple(_storage(value) for value in parameters.values()))


def validate_checkpoint(checkpoint, model, opt, expected_metadata):
    """Reject every state field before either target load method is called."""
    _bindings(model, opt)
    if type(checkpoint) is not dict or set(checkpoint) != {"metadata", "model", "optimizer"}:
        raise ValueError("Checkpoint must contain exactly metadata, model and optimizer")
    if not _same_plain(checkpoint["metadata"], expected_metadata):
        raise ValueError("Checkpoint metadata differs from the completed-update-two protocol")
    parameters = dict(model.named_parameters())
    descriptor = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                  for name, value in parameters.items()]
    if not _same_plain(descriptor, expected_metadata["parameters"]):
        raise ValueError("Target parameter names, order, shapes or dtype differ")
    state = checkpoint["model"]
    if type(state) is not dict or list(state) != list(parameters):
        raise ValueError("Checkpoint model keys and order must match exactly")
    for name, parameter in parameters.items():
        _ordinary_tensor(state[name], parameter, name)
    saved_optimizer = checkpoint["optimizer"]
    if type(saved_optimizer) is not dict or set(saved_optimizer) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer state schema")
    if not _same_plain(saved_optimizer["param_groups"], [dict(OPTIONS, params=list(range(5)))]):
        raise ValueError("Optimizer group order, ordinal IDs or options differ")
    saved = saved_optimizer["state"]
    if (type(saved) is not dict or list(saved) != list(range(5))
            or any(type(key) is not int for key in saved)):
        raise ValueError("Optimizer state must contain every exact ordinal parameter ID")
    for index, (name, parameter) in enumerate(parameters.items()):
        if type(saved[index]) is not dict or set(saved[index]) != {"momentum_buffer"}:
            raise ValueError("Every saved parameter needs exactly one momentum buffer")
        _ordinary_tensor(saved[index]["momentum_buffer"], parameter, name + ".momentum_buffer")
    values = list(_tensors(checkpoint))
    if len({_storage(value) for value in values}) != len(values):
        raise ValueError("Checkpoint tensors must have independent storage")


def make_checkpoint(model, opt, expected_metadata):
    if any(value.grad is not None for value in model.parameters()):
        raise ValueError("Clear gradients before saving a completed update boundary")
    before = _tree_snapshot({"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    checkpoint = {"metadata": copy.deepcopy(expected_metadata), "model": _clone(dict(model.state_dict())),
                  "optimizer": _clone(opt.state_dict())}
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    _tree_unchanged({"model": dict(model.named_parameters()), "optimizer": opt.state_dict()}, before)
    _disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
    return checkpoint


def restore(checkpoint, model, opt, expected_metadata):
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    before, contents, binding = _tree_snapshot(checkpoint), _clone(checkpoint), _bindings(model, opt)
    model.load_state_dict(_clone(checkpoint["model"]), strict=True)
    # Public loading moves floating momentum to each live parameter's device/dtype.
    opt.load_state_dict(_clone(checkpoint["optimizer"]))
    opt.zero_grad(set_to_none=True)
    if _bindings(model, opt) != binding or set(opt.state) != set(model.parameters()):
        raise AssertionError("Restoration changed ownership or omitted live momentum")
    for index, (name, parameter) in enumerate(model.named_parameters()):
        entry = opt.state[parameter]
        if type(entry) is not dict or set(entry) != {"momentum_buffer"}:
            raise AssertionError("Restored momentum has an unexpected schema")
        _ordinary_tensor(entry["momentum_buffer"], parameter, name + ".momentum_buffer", "cuda:0")
        _compare_tree(_clone(parameter), checkpoint["model"][name], EXACT)
        _compare_tree(_clone(entry["momentum_buffer"]),
                      checkpoint["optimizer"]["state"][index]["momentum_buffer"], EXACT)
    live = {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()}
    if len({_storage(value) for value in _tensors(live)}) != 10:
        raise AssertionError("Restored parameters and momentum must own ten distinct storages")
    _disjoint(checkpoint, live)
    _compare_tree(checkpoint, contents, EXACT)
    _tree_unchanged(checkpoint, before)


def _read_bytes(path, maximum):
    path = Path(path)
    information = path.lstat()
    if (not stat.S_ISREG(information.st_mode) or path.is_symlink()
            or getattr(information, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or not 0 < information.st_size <= maximum):
        raise ValueError("Expected a bounded ordinary local artifact file")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise ValueError("Artifact exceeded its fixed byte bound")
    return payload


def load_checkpoint(path, expected_sha256, model, opt, expected_metadata):
    if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A complete lowercase expected SHA256 is required")
    payload = _read_bytes(path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the trusted local artifact")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    return checkpoint


def _write_tensor_file(path, value, maximum):
    if any(tensor.device.type != "cpu" for tensor in _tensors(value)):
        raise ValueError("Persistent tensors must have been explicitly staged to CPU")
    stream = io.BytesIO()
    torch.save(value, stream)
    payload = stream.getvalue()
    if len(payload) > maximum:
        raise ValueError("Produced artifact exceeds its fixed byte bound")
    with Path(path).open("xb") as target:
        target.write(payload)
    return sha256(payload).hexdigest()


def _compare_tree(actual, expected, tolerance):
    if type(expected) is torch.Tensor:
        _ordinary_tensor(actual, expected, "comparison record")
        torch.testing.assert_close(actual, expected, **tolerance)
        if tolerance == EXACT and not torch.equal(actual.reshape(-1).view(torch.uint8),
                                                  expected.reshape(-1).view(torch.uint8)):
            raise AssertionError("Exact continuation requires identical tensor bytes")
        return eager.difference(actual, expected)
    if type(actual) is not type(expected):
        raise AssertionError("Result structure types differ")
    if type(expected) is dict:
        if list(actual) != list(expected):
            raise AssertionError("Result dictionary keys or order differ")
        return max((_compare_tree(actual[key], expected[key], tolerance) for key in expected), default=0.0)
    if type(expected) in (list, tuple):
        if len(actual) != len(expected):
            raise AssertionError("Result list lengths differ")
        return max((_compare_tree(a, b, tolerance) for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Result scalar metadata differs")
    return 0.0


def step(model, opt, batch, index, reference=False):
    """Take one GPU update and check the single-tensor momentum recurrence."""
    binding = _bindings(model, opt)
    inputs = {name: batch[name] for name in ("raw", "offsets", "targets")}
    inputs_before = snapshot(inputs)
    parameters = dict(model.named_parameters())
    before = {name: value.detach().clone() for name, value in parameters.items()}
    previous = {name: (None if "momentum_buffer" not in opt.state.get(value, {})
                       else opt.state[value]["momentum_buffer"].detach().clone())
                for name, value in parameters.items()}
    opt.zero_grad(set_to_none=True)
    raw = batch["raw"].detach().clone().requires_grad_()
    output, gradients, _ = dynamic.evaluate(model, model, raw, batch, reference)
    momentum = {name: (gradient.clone() if previous[name] is None
                       else previous[name].mul(OPTIONS["momentum"]).add(gradient))
                for name, gradient in gradients.items()}
    expected = {name: before[name].add(momentum[name], alpha=-OPTIONS["lr"]) for name in parameters}
    opt.step()
    for name, parameter in parameters.items():
        _compare_tree(_clone(opt.state[parameter]["momentum_buffer"]), _clone(momentum[name]), EXACT)
        _compare_tree(_clone(parameter), _clone(expected[name]), EXACT)
        if eager.difference(parameter, before[name]) <= 0:
            raise AssertionError("Every update must change every parameter block")
        if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(momentum[name]).all()):
            raise AssertionError("Updated parameters and momentum must remain finite")
    if _bindings(model, opt) != binding:
        raise AssertionError("An update changed model or optimizer ownership")
    unchanged(inputs, inputs_before)
    opt.zero_grad(set_to_none=True)
    return {"update": index + 1, "inputs": _clone(inputs), "output": _clone(output),
            "parameter_gradients": _clone(gradients), "parameters": _clone(parameters),
            "momentum": _clone({name: opt.state[value]["momentum_buffer"]
                                for name, value in parameters.items()})}


def run_steps(model, opt, batches, start=0, stop=TOTAL_STEPS, reference=False):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= TOTAL_STEPS:
        raise ValueError("Use a nonempty subset of the five fixed update positions")
    binding = _bindings(model, opt)
    records = []
    for index in range(start, stop):
        records.append(step(model, opt, batches[index], index, reference))
        if _bindings(model, opt) != binding:
            raise AssertionError("Continuation replaced its model, optimizer or parameters")
    return records


def _threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _phase(dtype_name, directory, phase, expected_sha256, source_revision, source_tree):
    _threads()
    identity = installed_identity()
    initial, batches, _ = fixture(dtype_name)
    initial_before = snapshot(dict(initial.named_parameters()))
    expected = metadata(dtype_name, initial, batches, identity, source_revision, source_tree)
    model, checkpoint_path = copy.deepcopy(initial), directory / "checkpoint.pt"
    opt = optimizer(model)
    if phase == "produce":
        records = run_steps(model, opt, batches, stop=CURSOR)
        checkpoint = make_checkpoint(model, opt, expected)
        digest = _write_tensor_file(checkpoint_path, checkpoint, MAX_CHECKPOINT_BYTES)
        result_path = directory / "producer-results.pt"
    else:
        checkpoint = load_checkpoint(checkpoint_path, expected_sha256, model, opt, expected)
        before, contents = _tree_snapshot(checkpoint), _clone(checkpoint)
        restore(checkpoint, model, opt, expected)
        records = run_steps(model, opt, batches, start=CURSOR)
        _compare_tree(checkpoint, contents, EXACT)
        _tree_unchanged(checkpoint, before)
        _disjoint(checkpoint, {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()})
        digest = sha256(_read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)).hexdigest()
        if digest != expected_sha256:
            raise AssertionError("The checkpoint file changed during continuation")
        result_path = directory / "resume-results.pt"
    unchanged(dict(initial.named_parameters()), initial_before)
    if installed_identity() != identity:
        raise AssertionError("Installed dependency or consumer source changed during the child")
    result = {"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(),
              "metadata": expected, "identity": identity,
              "checkpoint_sha256": digest, "records": records}
    result_digest = _write_tensor_file(result_path, result, MAX_RESULT_BYTES)
    print(json.dumps({"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(),
                      "checkpoint_sha256": digest,
                      "result_sha256": result_digest}, allow_nan=False))


def _child(dtype_name, directory, phase, expected_sha256=None, source_revision=None, source_tree=None):
    command = [sys.executable, "-I", str(SCRIPT), "--dtype", dtype_name,
               "--directory", str(directory), "--phase", phase]
    for flag, value in (("--expected-sha256", expected_sha256), ("--source-revision", source_revision),
                        ("--source-tree", source_tree)):
        if value is not None:
            command.extend((flag, value))
    # No parent CUDA tensors are created until both bounded children have exited.
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
    if (type(status) is not dict or set(status) != {"phase", "pid", "parent_pid", "checkpoint_sha256", "result_sha256"}
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
    payload = _read_bytes(result_path, MAX_RESULT_BYTES)
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


def _step_errors(first, second, tolerance):
    _compare_tree(first["inputs"], second["inputs"], EXACT)
    return {field: eager.compare(first[field], second[field], tolerance)
            for field in ("output", "parameter_gradients", "parameters", "momentum")}


def run(dtype_name, directory, source_revision=None, source_tree=None):
    """Run fresh children first, then the parent's two uninterrupted GPU routes."""
    _threads()
    _source_labels(source_revision, source_tree)
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    identity = installed_identity()
    produced = _child(dtype_name, directory, "produce", source_revision=source_revision, source_tree=source_tree)
    digest = produced["checkpoint_sha256"]
    resumed = _child(dtype_name, directory, "resume", digest, source_revision, source_tree)
    pids = {"parent": os.getpid(), "producer": produced["pid"], "resumer": resumed["pid"]}
    if len(set(pids.values())) != 3:
        raise AssertionError("Parent, producer and resumer must be distinct processes")
    # Child result tensors are on CPU. Only now does the parent create CUDA state.
    initial, batches, tolerance = fixture(dtype_name)
    expected = metadata(dtype_name, initial, batches, identity, source_revision, source_tree)
    for result in (produced, resumed):
        if not _same_plain(result["metadata"], expected) or result["identity"] != identity:
            raise AssertionError("Child source, fixture or installed identity differs from the parent")
    initial_before = snapshot(dict(initial.named_parameters()))
    inputs = {f"batch{index}.{name}": batch[name] for index, batch in enumerate(batches, 1)
              for name in ("raw", "offsets", "targets")}
    inputs_before = snapshot(inputs)
    native, primitive = copy.deepcopy(initial), copy.deepcopy(initial)
    all_parameters = [*initial.parameters(), *native.parameters(), *primitive.parameters()]
    if len({_storage(value) for value in all_parameters}) != len(all_parameters):
        raise AssertionError("Initial and uninterrupted models must own separate parameter storage")
    native_records = run_steps(native, optimizer(native), batches)
    primitive_records = run_steps(primitive, optimizer(primitive), batches, reference=True)
    primitive_error = _compare_tree(native_records, primitive_records, tolerance)
    combined = produced["records"] + resumed["records"]
    exact_error = _compare_tree(combined, native_records, EXACT)
    checkpoint_path = directory / "checkpoint.pt"
    correct, control = copy.deepcopy(initial), copy.deepcopy(initial)
    correct_opt, control_opt = optimizer(correct), optimizer(control)
    checkpoint = load_checkpoint(checkpoint_path, digest, correct, correct_opt, expected)
    before, contents = _tree_snapshot(checkpoint), _clone(checkpoint)
    restore(checkpoint, correct, correct_opt, expected)
    control.load_state_dict(_clone(checkpoint["model"]), strict=True)
    with torch.no_grad():
        next_batch = batches[CURSOR]
        expected_forward = correct(next_batch["raw"], next_batch["offsets"])
        control_forward = control(next_batch["raw"], next_batch["offsets"])
        _compare_tree(_clone(control_forward), _clone(expected_forward), EXACT)
    control_records = run_steps(control, control_opt, batches, start=CURSOR)
    control_error = max(eager.difference(control_records[-1]["parameters"][name], native_records[-1]["parameters"][name])
                        for name in STATE_SHAPES)
    threshold = 1e-7 if dtype_name == "float32" else 1e-10
    if control_error <= threshold:
        raise AssertionError("The reset-optimizer control must visibly diverge")
    _compare_tree(checkpoint, contents, EXACT)
    _tree_unchanged(checkpoint, before)
    unchanged(inputs, inputs_before)
    unchanged(dict(initial.named_parameters()), initial_before)
    payload = _read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != digest or installed_identity() != identity:
        raise AssertionError("Checkpoint, consumer source or installed dependency changed")
    torch.cuda.synchronize(0)
    properties = torch.cuda.get_device_properties(0)
    steps = [{"update": record["update"], "N": record["inputs"]["raw"].shape[0],
              "B": record["inputs"]["targets"].shape[0], "offsets": record["inputs"]["offsets"].tolist(),
              "native_loss": float(record["output"]["loss"]),
              "primitive_loss": float(primitive_records[index]["output"]["loss"]),
              "primitive_max_abs_errors": _step_errors(record, primitive_records[index], tolerance),
              "resumed_max_abs_errors": _step_errors(record, combined[index], EXACT)}
             for index, record in enumerate(native_records)]
    return {"status": "passed", "method": "state_dict.packed_cuda_training_resume",
            "scope": "Known local eager training state only; no stochastic replay, sandbox, crash-consistency or portability claim",
            "dtype": dtype_name, "device": "cuda:0", "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller labels plus hashed consumer/helper/package/native and fixture bytes",
            "optimizer": "SGD", "learning_rate": OPTIONS["lr"], "momentum": OPTIONS["momentum"],
            "optimizer_options": dict(OPTIONS), "optimizer_steps": TOTAL_STEPS,
            "producer_steps": CURSOR, "resumed_steps": TOTAL_STEPS - CURSOR,
            "checkpoint_cursor": CURSOR, "process_ids": pids, "distinct_processes": True,
            "child_launchers": {"producer": {name: produced[name] for name in
                                ("launcher_pid", "launcher_relation", "parent_pid")},
                                "resumer": {name: resumed[name] for name in
                                ("launcher_pid", "launcher_relation", "parent_pid")}},
            "parent_cuda_fixture_after_children_exit": True,
            "same_native_exact": True, "same_native_max_abs_error": exact_error,
            "primitive_comparisons_passed": True, "primitive_max_abs_error": primitive_error,
            "comparison_tolerance": tolerance, "independent_momentum_recurrence": True,
            "model_only_next_forward_exact": True, "model_only_reset_optimizer_diverged": True,
            "model_only_final_parameter_max_abs_error": control_error,
            "model_only_divergence_threshold": threshold, "checkpoint_sha256": digest,
            "checkpoint_bytes": len(payload), "checkpoint_byte_limit": MAX_CHECKPOINT_BYTES,
            "checkpoint_unchanged": True, "input_sources_unchanged": True,
            "initial_parameters_unchanged": True, "loaded_state_storage_independent": True,
            "same_parameter_and_optimizer_objects": True, "all_parameter_blocks_changed_each_step": True,
            "checkpoint_metadata": expected, "no_live_rng_continuation": True, "steps": steps,
            "child_timeout_seconds": 120, "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            **identity, "child_installed_identities": {"producer": produced["identity"], "resumer": resumed["identity"]}}


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
        _source_labels(args.source_revision, args.source_tree)
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
