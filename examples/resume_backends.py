"""Continue two installed training updates on the other device in a fresh process.

The fixed consumer uses the adjacent train_backends.py model and independent
pooling arithmetic. It preserves five parameter tensors and five SGD momentum
buffers at a completed update boundary. Only trusted locally produced artifacts
and the declared CPU/CUDA wheel pair are admitted.
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


CURSOR, TOTAL_STEPS = 2, 4
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0, "weight_decay": 0,
           "nesterov": False, "maximize": False, "foreach": False,
           "differentiable": False, "fused": False}
STATE_SHAPES = {"encoder.weight": (4, 3), "encoder.bias": (4,),
                "scorer.weight": (1, 4), "head.weight": (1, 4), "head.bias": (1,)}
DTYPES = {"float32": torch.float32, "float64": torch.float64}
DIRECTIONS = {"cpu-to-cuda": ("cpu", "cuda:0"), "cuda-to-cpu": ("cuda:0", "cpu")}
BOUNDARIES = ((0, 2, 2, 5, 7, 9), (0, 3, 5, 5, 8, 10, 12))
SEED = 1811
MAX_CHECKPOINT_BYTES = 128 * 1024
MAX_RESULT_BYTES = 384 * 1024
EXACT = {"rtol": 0, "atol": 0}
SCRIPT = Path(__file__).resolve()
PAIR_FIELDS = ("distribution_name", "distribution_version", "wheel_build_tag",
               "wheel_sha256", "native_library_sha256", "metadata_sha256")
METADATA_KEYS = {"format", "schema", "dtype", "direction", "source_device", "target_device",
                 "checkpoint_tensor_device", "next_update", "total_updates", "seed",
                 "architecture", "training", "parameters", "optimizer", "offsets",
                 "fixture_sha256", "source_revision", "source_tree", "script_sha256",
                 "fixture_script_sha256", "python_version", "torch_version",
                 "cuda_runtime_version", "installed_packages", "stochastic_forward"}


def _same_plain(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_same_plain(actual[key], expected[key]) for key in expected)
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(_same_plain(a, b) for a, b in zip(actual, expected))
    return type(expected) in (str, int, float, bool, type(None)) and actual == expected


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone().contiguous()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif type(value) in (list, tuple):
        for item in value:
            yield from _tensors(item)


def _storage(value):
    return str(value.device), value.untyped_storage().data_ptr()


def _tree_snapshot(value):
    return [(item, item.detach().clone(), item._version, _storage(item), item.shape,
             item.stride(), item.storage_offset()) for item in _tensors(value)]


def _tree_unchanged(value, before):
    actual = list(_tensors(value))
    if len(actual) != len(before):
        raise AssertionError("A saved tensor tree changed its membership")
    for item, saved in zip(actual, before):
        original, contents, version, storage, shape, stride, offset = saved
        if (item is not original or item._version != version or _storage(item) != storage
                or item.shape != shape or item.stride() != stride or item.storage_offset() != offset):
            raise AssertionError("A saved tensor changed ownership, version or representation")
        torch.testing.assert_close(item, contents, **EXACT)
        if not torch.equal(item.detach().reshape(-1).view(torch.uint8),
                           contents.reshape(-1).view(torch.uint8)):
            raise AssertionError("A saved tensor changed its bytes")


def _ordinary_tensor(value, shape, dtype, device, name):
    if (type(value) is not torch.Tensor or value.device != torch.device(device)
            or value.layout != torch.strided or value.dtype != dtype or tuple(value.shape) != tuple(shape)
            or not value.is_contiguous() or value.storage_offset() != 0
            or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            or value.requires_grad or torch.is_inference(value) or value.is_conj() or value.is_neg()
            or not bool(torch.isfinite(value).all()) or bool((value.abs() > 1000).any())):
        raise ValueError("Invalid bounded ordinary tensor for " + name)


def _source_labels(revision, tree):
    if any(label is not None and (type(label) is not str or not re.fullmatch(r"[0-9a-f]{40}", label))
           for label in (revision, tree)):
        raise ValueError("Source labels must be full lowercase SHA1 identifiers")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_metadata(value):
    if type(value) is not dict or set(value) != METADATA_KEYS:
        raise ValueError("Invalid handoff metadata schema")
    dtype_name, direction = value["dtype"], value["direction"]
    if type(dtype_name) is not str or dtype_name not in DTYPES or type(direction) is not str or direction not in DIRECTIONS:
        raise ValueError("Unknown handoff dtype or direction")
    source, target = DIRECTIONS[direction]
    fixed = {"format": "ragged-pool-fixed-backend-training-state", "schema": 1,
             "source_device": source, "target_device": target, "checkpoint_tensor_device": "cpu",
             "next_update": CURSOR, "total_updates": TOTAL_STEPS, "seed": SEED,
             "architecture": "encoder3x4-tanh-scorer4x1-pool-head4x1", "training": True,
             "parameters": [{"name": name, "shape": list(shape), "dtype": str(DTYPES[dtype_name])}
                            for name, shape in STATE_SHAPES.items()],
             "optimizer": {"class": "torch.optim.SGD", "options": dict(OPTIONS)},
             "offsets": [list(BOUNDARIES[index % 2]) for index in range(TOTAL_STEPS)],
             "python_version": "3.11.8", "torch_version": "2.9.1+cu128", "cuda_runtime_version": "12.8",
             "stochastic_forward": False}
    if not all(_same_plain(value[key], expected) for key, expected in fixed.items()):
        raise ValueError("Metadata differs from the fixed completed-update-two handoff")
    _source_labels(value["source_revision"], value["source_tree"])
    if not all(_digest(value[key]) for key in ("fixture_sha256", "script_sha256", "fixture_script_sha256")):
        raise ValueError("Metadata needs complete source and fixture digests")
    pair = value["installed_packages"]
    if type(pair) is not dict or set(pair) != {"cpu", "cuda"}:
        raise ValueError("Metadata must bind both installed packages")
    for backend, name, tag in (("cpu", "ragged-pool", "1cu128"), ("cuda", "ragged-pool-cuda", "")):
        item = pair[backend]
        if (type(item) is not dict or set(item) != set(PAIR_FIELDS)
                or not _same_plain({key: item[key] for key in PAIR_FIELDS[:3]},
                                   {"distribution_name": name, "distribution_version": "0.1.0", "wheel_build_tag": tag})
                or not all(_digest(item[key]) for key in PAIR_FIELDS[3:])):
            raise ValueError("Metadata package identities differ from the explicit cu128 pair")


def metadata(dtype_name, direction, initial, batches, identity, source_revision=None, source_tree=None):
    """Describe the fixed protocol without importing either installed package."""
    if dtype_name not in DTYPES or direction not in DIRECTIONS:
        raise ValueError("Unknown handoff dtype or direction")
    _source_labels(source_revision, source_tree)
    parameters = dict(initial.named_parameters())
    if list(parameters) != list(STATE_SHAPES) or any(
            tuple(parameters[name].shape) != shape or parameters[name].dtype != DTYPES[dtype_name]
            for name, shape in STATE_SHAPES.items()):
        raise ValueError("Use the fixed five-block model and declared dtype")
    if type(batches) not in (list, tuple) or len(batches) != TOTAL_STEPS:
        raise ValueError("Metadata requires four fixed update batches")
    data = sha256()
    for index, batch in enumerate(batches):
        boundaries = BOUNDARIES[index % 2]
        if not _same_plain(list(batch["boundaries"]), list(boundaries)):
            raise ValueError("The four packed partitions differ from the fixed alternating sequence")
        for name, shape, dtype in (("raw", (boundaries[-1], 3), DTYPES[dtype_name]),
                                   ("targets", (len(boundaries) - 1,), DTYPES[dtype_name]),
                                   ("offsets", (len(boundaries),), torch.int64)):
            tensor = batch[name]
            _ordinary_tensor(tensor, shape, dtype, "cpu", "fixture." + name)
            if name == "offsets" and tensor.tolist() != list(boundaries):
                raise ValueError("Packed offsets differ from the declared partitions")
            data.update(json.dumps([name, str(dtype), list(shape)], separators=(",", ":")).encode())
            data.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    source, target = DIRECTIONS[direction]
    result = {"format": "ragged-pool-fixed-backend-training-state", "schema": 1,
              "dtype": dtype_name, "direction": direction, "source_device": source, "target_device": target,
              "checkpoint_tensor_device": "cpu", "next_update": CURSOR, "total_updates": TOTAL_STEPS,
              "seed": SEED, "architecture": "encoder3x4-tanh-scorer4x1-pool-head4x1", "training": True,
              "parameters": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                             for name, value in parameters.items()],
              "optimizer": {"class": "torch.optim.SGD", "options": dict(OPTIONS)},
              "offsets": [list(BOUNDARIES[index % 2]) for index in range(TOTAL_STEPS)],
              "fixture_sha256": data.hexdigest(), "source_revision": source_revision, "source_tree": source_tree,
              "script_sha256": identity["script_sha256"], "fixture_script_sha256": identity["fixture_script_sha256"],
              "python_version": "3.11.8", "torch_version": "2.9.1+cu128", "cuda_runtime_version": "12.8",
              "installed_packages": {backend: {key: identity["installed_packages"][backend][key] for key in PAIR_FIELDS}
                                     for backend in ("cpu", "cuda")}, "stochastic_forward": False}
    _validate_metadata(result)
    return result


def optimizer(model):
    return torch.optim.SGD(model.parameters(), **OPTIONS)


def _bindings(model, opt, device, dtype):
    parameters = dict(model.named_parameters())
    if (list(parameters) != list(STATE_SHAPES) or len(model.state_dict()) != len(parameters)
            or not model.training or type(opt) is not torch.optim.SGD):
        raise ValueError("Use the fixed training model and SGD optimizer")
    for name, parameter in parameters.items():
        if (type(parameter) is not torch.nn.Parameter or tuple(parameter.shape) != STATE_SHAPES[name]
                or parameter.device != torch.device(device) or parameter.dtype != dtype
                or parameter.layout != torch.strided or not parameter.is_contiguous()
                or parameter.storage_offset() != 0
                or parameter.untyped_storage().nbytes() != parameter.numel() * parameter.element_size()
                or torch.is_inference(parameter) or parameter.is_conj() or parameter.is_neg()
                or not parameter.requires_grad):
            raise ValueError("Live parameters must keep the declared device, dtype and representation")
    if len({_storage(value) for value in parameters.values()}) != len(parameters):
        raise ValueError("Live parameters must own distinct storage")
    if (len(opt.param_groups) != 1 or tuple(id(value) for value in opt.param_groups[0]["params"])
            != tuple(id(value) for value in parameters.values())
            or not _same_plain({key: value for key, value in opt.param_groups[0].items() if key != "params"}, OPTIONS)):
        raise ValueError("Optimizer must own the ordered live model with exact options")
    return (id(model), id(opt), tuple(id(value) for value in parameters.values()),
            tuple(_storage(value) for value in parameters.values()))


def _validate_checkpoint(checkpoint, model, opt, expected_metadata, device):
    _validate_metadata(expected_metadata)
    dtype = DTYPES[expected_metadata["dtype"]]
    _bindings(model, opt, device, dtype)
    if type(checkpoint) is not dict or set(checkpoint) != {"metadata", "model", "optimizer"}:
        raise ValueError("Checkpoint must contain exactly metadata, model and optimizer")
    if not _same_plain(checkpoint["metadata"], expected_metadata):
        raise ValueError("Checkpoint metadata differs from the expected handoff")
    state = checkpoint["model"]
    if type(state) is not dict or list(state) != list(STATE_SHAPES):
        raise ValueError("Saved model names and order must match the fixed model")
    for name, shape in STATE_SHAPES.items():
        _ordinary_tensor(state[name], shape, dtype, "cpu", name)
    saved_optimizer = checkpoint["optimizer"]
    if type(saved_optimizer) is not dict or set(saved_optimizer) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer state schema")
    if not _same_plain(saved_optimizer["param_groups"], [dict(OPTIONS, params=list(range(5)))]):
        raise ValueError("Saved optimizer options, group order or parameter IDs differ")
    saved = saved_optimizer["state"]
    if (type(saved) is not dict or list(saved) != list(range(5))
            or any(type(key) is not int for key in saved)):
        raise ValueError("Saved optimizer needs every exact ordinal parameter ID")
    for index, (name, shape) in enumerate(STATE_SHAPES.items()):
        if type(saved[index]) is not dict or set(saved[index]) != {"momentum_buffer"}:
            raise ValueError("Every saved parameter needs exactly one momentum buffer")
        _ordinary_tensor(saved[index]["momentum_buffer"], shape, dtype, "cpu", name + ".momentum_buffer")
    values = list(_tensors(checkpoint))
    if len(values) != 10 or len({_storage(value) for value in values}) != 10:
        raise ValueError("Checkpoint parameters and momentum must own ten independent storages")


def validate_checkpoint(checkpoint, model, opt, expected_metadata):
    """Admit every saved field before either target loading method is called."""
    _validate_metadata(expected_metadata)
    _validate_checkpoint(checkpoint, model, opt, expected_metadata, expected_metadata["target_device"])


def make_checkpoint(model, opt, expected_metadata):
    _validate_metadata(expected_metadata)
    _bindings(model, opt, expected_metadata["source_device"], DTYPES[expected_metadata["dtype"]])
    if any(value.grad is not None for value in model.parameters()):
        raise ValueError("Clear gradients before saving a completed update boundary")
    live = {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()}
    before = _tree_snapshot(live)
    checkpoint = {"metadata": copy.deepcopy(expected_metadata), "model": _clone(dict(model.state_dict())),
                  "optimizer": _clone(opt.state_dict())}
    _validate_checkpoint(checkpoint, model, opt, expected_metadata, expected_metadata["source_device"])
    _tree_unchanged(live, before)
    if {_storage(value) for value in _tensors(checkpoint)} & {_storage(value) for value in _tensors(live)}:
        raise AssertionError("Checkpoint tensors must not alias the live source")
    return checkpoint


def restore(checkpoint, model, opt, expected_metadata):
    """Copy validated CPU state into an already placed target model and optimizer."""
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    device, dtype = expected_metadata["target_device"], DTYPES[expected_metadata["dtype"]]
    binding = _bindings(model, opt, device, dtype)
    before, contents = _tree_snapshot(checkpoint), _clone(checkpoint)
    # Independent copies also prevent CPU optimizer.load_state_dict from keeping
    # momentum storage owned by the original checkpoint.
    model.load_state_dict(_clone(checkpoint["model"]), strict=True, assign=False)
    opt.load_state_dict(_clone(checkpoint["optimizer"]))
    opt.zero_grad(set_to_none=True)
    if _bindings(model, opt, device, dtype) != binding or set(opt.state) != set(model.parameters()):
        raise AssertionError("Restore changed ownership or omitted live momentum")
    for index, (name, parameter) in enumerate(model.named_parameters()):
        saved = opt.state[parameter]
        if type(saved) is not dict or set(saved) != {"momentum_buffer"}:
            raise AssertionError("Restored optimizer state has an unexpected schema")
        _ordinary_tensor(saved["momentum_buffer"], STATE_SHAPES[name], dtype, device, name + ".momentum")
        _compare_tree(_clone(parameter), checkpoint["model"][name], EXACT)
        _compare_tree(_clone(saved["momentum_buffer"]),
                      checkpoint["optimizer"]["state"][index]["momentum_buffer"], EXACT)
    live = {"model": dict(model.named_parameters()), "optimizer": opt.state_dict()}
    if len({_storage(value) for value in _tensors(live)}) != 10:
        raise AssertionError("Restored model and momentum must own ten distinct storages")
    if {_storage(value) for value in _tensors(checkpoint)} & {_storage(value) for value in _tensors(live)}:
        raise AssertionError("Restored live state aliases checkpoint storage")
    _tree_unchanged(checkpoint, before)
    _compare_tree(checkpoint, contents, EXACT)
    return {"exact_checkpoint_values": True, "target_device_and_dtype": True,
            "same_model_optimizer_parameter_objects": True, "checkpoint_storage_disjoint": True}


def _read_bytes(path, maximum):
    path = Path(path)
    information = path.lstat()
    if (not stat.S_ISREG(information.st_mode) or path.is_symlink()
            or getattr(information, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            or not 0 < information.st_size <= maximum):
        raise ValueError("Expected a bounded ordinary local artifact file")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise ValueError("Artifact exceeded its fixed byte bound")
    return payload


def load_checkpoint(path, expected_sha256, model, opt, expected_metadata):
    if not _digest(expected_sha256):
        raise ValueError("A complete lowercase expected SHA256 is required")
    payload = _read_bytes(path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the trusted local artifact")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    return checkpoint


def _write_tensor_file(path, value, maximum):
    if any(item.device.type != "cpu" for item in _tensors(value)):
        raise ValueError("Persistent tensors must be explicitly staged to CPU")
    stream = io.BytesIO()
    torch.save(value, stream)
    payload = stream.getvalue()
    if not 0 < len(payload) <= maximum:
        raise ValueError("Produced artifact exceeds its fixed byte bound")
    with Path(path).open("xb") as target:
        target.write(payload)
    return sha256(payload).hexdigest()


def _compare_tree(actual, expected, tolerance):
    if type(expected) is torch.Tensor:
        _ordinary_tensor(actual, expected.shape, expected.dtype, "cpu", "comparison")
        torch.testing.assert_close(actual, expected, **tolerance)
        if tolerance == EXACT and not torch.equal(actual.reshape(-1).view(torch.uint8),
                                                  expected.reshape(-1).view(torch.uint8)):
            raise AssertionError("An exact tensor comparison changed bytes")
        return float((actual - expected).abs().max()) if expected.numel() else 0.0
    if type(actual) is not type(expected):
        raise AssertionError("Comparison structure types differ")
    if type(expected) is dict:
        if list(actual) != list(expected):
            raise AssertionError("Comparison dictionary names or order differ")
        return max((_compare_tree(actual[key], expected[key], tolerance) for key in expected), default=0.0)
    if type(expected) in (list, tuple):
        if len(actual) != len(expected):
            raise AssertionError("Comparison sequence lengths differ")
        return max((_compare_tree(a, b, tolerance) for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Comparison scalar metadata differs")
    return 0.0


def _helper():
    path = SCRIPT.with_name("train_backends.py")
    specification = importlib.util.spec_from_file_location("backend_resume_fixture", path)
    if specification is None or specification.loader is None:
        raise ImportError("The adjacent train_backends.py fixture is required")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _identity(helper, cpu_wheel, cuda_wheel):
    return {"script_sha256": sha256(SCRIPT.read_bytes()).hexdigest(),
            "fixture_script_sha256": sha256(SCRIPT.with_name("train_backends.py").read_bytes()).hexdigest(),
            "installed_packages": {
                "cpu": helper.installed_identity("ragged-pool", helper.cpu_package, cpu_wheel, "1cu128"),
                "cuda": helper.installed_identity("ragged-pool-cuda", helper.cuda_package, cuda_wheel, "")}}


def _fixture(helper, dtype_name):
    initial, original, _ = helper.fixture(dtype_name)
    return initial, [original[index % 2] for index in range(TOTAL_STEPS)]


def _threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _environment():
    _threads()
    if (sys.version_info[:3] != (3, 11, 8) or str(torch.__version__) != "2.9.1+cu128"
            or torch.version.cuda != "12.8"):
        raise AssertionError("Use the declared CPython3.11.8 and PyTorch2.9.1+cu128 environment")
    if any((parent / "pyproject.toml").is_file() for parent in SCRIPT.parents):
        raise AssertionError("Copy both consumers outside every checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA0 device is unavailable")


def _step(helper, model, opt, source_batch, index, device, primitive=False):
    dtype = next(model.parameters()).dtype
    binding = _bindings(model, opt, device, dtype)
    source_before = _tree_snapshot({key: source_batch[key] for key in ("raw", "offsets", "targets")})
    batch = {key: source_batch[key].to(device=device).clone() for key in ("raw", "offsets", "targets")}
    batch["boundaries"] = source_batch["boundaries"]
    raw = batch["raw"].detach().clone().requires_grad_()
    opt.zero_grad(set_to_none=True)
    parameters = dict(model.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    previous = {name: (None if "momentum_buffer" not in opt.state.get(parameter, {})
                       else opt.state[parameter]["momentum_buffer"].detach().clone())
                for name, parameter in parameters.items()}
    route = ("cuda" if torch.device(device).type == "cuda" else "cpu") + ("_primitive" if primitive else "_native")
    output, gradients = helper.evaluate(model, raw, batch, route)
    momentum = {name: (parameter.grad.detach().clone() if previous[name] is None
                       else previous[name].mul(OPTIONS["momentum"]).add(parameter.grad))
                for name, parameter in parameters.items()}
    expected = {name: before[name].add(momentum[name], alpha=-OPTIONS["lr"]) for name in parameters}
    opt.step()
    for name, parameter in parameters.items():
        _compare_tree(_clone(parameter), _clone(expected[name]), EXACT)
        _compare_tree(_clone(opt.state[parameter]["momentum_buffer"]), _clone(momentum[name]), EXACT)
        if float((parameter.detach() - before[name]).abs().max()) <= 0:
            raise AssertionError("Each update must change every parameter block")
    if _bindings(model, opt, device, dtype) != binding:
        raise AssertionError("Training changed model or optimizer ownership")
    _tree_unchanged({key: source_batch[key] for key in ("raw", "offsets", "targets")}, source_before)
    opt.zero_grad(set_to_none=True)
    return {"update": index + 1, "batch_id": index % 2, "device": str(torch.device(device)),
            "inputs": _clone({key: source_batch[key] for key in ("raw", "offsets", "targets")}),
            "output": _clone(output), "parameter_gradients": _clone(gradients),
            "parameters": _clone(parameters),
            "momentum": _clone({name: opt.state[parameter]["momentum_buffer"] for name, parameter in parameters.items()})}


def _steps(helper, model, opt, batches, start, stop, device, primitive=False):
    return [_step(helper, model, opt, batches[index], index, device, primitive) for index in range(start, stop)]


def _new_target(initial, device):
    # Device placement precedes optimizer construction and creates no stale
    # optimizer references to parameters from a pre-transfer model.
    model = copy.deepcopy(initial).to(device=device)
    return model, optimizer(model)


def _manual_copy(model, opt, model_state, momentum, device):
    """Give the oracle named state without using the checkpoint restore helper."""
    parameters = dict(model.named_parameters())
    if list(model_state) != list(parameters) or list(momentum) != list(parameters):
        raise AssertionError("Oracle state names or order differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(model_state[name].to(device=device))
            opt.state[parameter] = {"momentum_buffer": momentum[name].to(device=device).clone()}


def _phase(dtype_name, direction, directory, phase, digest, cpu_wheel, cuda_wheel, revision, tree):
    _environment()
    helper = _helper()
    identity = _identity(helper, cpu_wheel, cuda_wheel)
    initial, batches = _fixture(helper, dtype_name)
    before = _tree_snapshot({"initial": dict(initial.named_parameters()), "batches": batches})
    expected = metadata(dtype_name, direction, initial, batches, identity, revision, tree)
    source, target = DIRECTIONS[direction]
    model, opt = _new_target(initial, source if phase == "produce" else target)
    path = directory / "checkpoint.pt"
    witness = None
    if phase == "produce":
        records = _steps(helper, model, opt, batches, 0, CURSOR, source)
        checkpoint = make_checkpoint(model, opt, expected)
        digest = _write_tensor_file(path, checkpoint, MAX_CHECKPOINT_BYTES)
    else:
        checkpoint = load_checkpoint(path, digest, model, opt, expected)
        checkpoint_before = _tree_snapshot(checkpoint)
        witness = restore(checkpoint, model, opt, expected)
        records = _steps(helper, model, opt, batches, CURSOR, TOTAL_STEPS, target)
        _tree_unchanged(checkpoint, checkpoint_before)
        if sha256(_read_bytes(path, MAX_CHECKPOINT_BYTES)).hexdigest() != digest:
            raise AssertionError("Checkpoint file changed during continuation")
    _tree_unchanged({"initial": dict(initial.named_parameters()), "batches": batches}, before)
    if _identity(helper, cpu_wheel, cuda_wheel) != identity:
        raise AssertionError("Installed packages or source changed during the child")
    result = {"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(), "metadata": expected,
              "identity": identity, "checkpoint_sha256": digest, "records": records,
              "restore_witness": witness}
    result_path = directory / ("producer-results.pt" if phase == "produce" else "resume-results.pt")
    result_digest = _write_tensor_file(result_path, result, MAX_RESULT_BYTES)
    print(json.dumps({"phase": phase, "pid": os.getpid(), "parent_pid": os.getppid(),
                      "checkpoint_sha256": digest, "result_sha256": result_digest}, allow_nan=False))


def _child(dtype_name, direction, directory, phase, digest, cpu_wheel, cuda_wheel, revision, tree):
    if (directory / "timeout.json").exists():
        raise RuntimeError("A prior timeout requires an owned-runtime check before more work")
    command = [sys.executable, "-I", "-B", str(SCRIPT), "--dtype", dtype_name, "--direction", direction,
               "--directory", str(directory), "--phase", phase, "--cpu-wheel", str(cpu_wheel),
               "--cuda-wheel", str(cuda_wheel)]
    for flag, value in (("--expected-sha256", digest), ("--source-revision", revision), ("--source-tree", tree)):
        if value is not None:
            command.extend((flag, value))
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        stdout, stderr = child.communicate(timeout=120)
    except subprocess.TimeoutExpired as error:
        marker = directory / "timeout.json"
        marker.write_text(json.dumps({
            "status": "needs-owned-runtime-check", "launcher_pid": child.pid, "phase": phase,
            "deadline_seconds": 120,
            "explanation": "Cleanup addresses the direct launcher only. A Windows virtual-environment redirector may have a descendant runtime whose completion is unknown. Halt further GPU work until the owned runtime is checked."
        }, indent=2) + "\n", encoding="utf-8")
        (directory / (phase + ".stdout.log")).write_bytes(error.stdout or b"")
        (directory / (phase + ".stderr.log")).write_bytes(error.stderr or b"")
        try:
            child.kill()
            cleanup_stdout, cleanup_stderr = child.communicate(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            if isinstance(cleanup_error, subprocess.TimeoutExpired):
                (directory / (phase + ".cleanup.stdout.log")).write_bytes(cleanup_error.stdout or b"")
                (directory / (phase + ".cleanup.stderr.log")).write_bytes(cleanup_error.stderr or b"")
            raise RuntimeError("Child cleanup did not establish runtime completion. Inspect " + str(marker)) from cleanup_error
        (directory / (phase + ".cleanup.stdout.log")).write_bytes(cleanup_stdout or b"")
        (directory / (phase + ".cleanup.stderr.log")).write_bytes(cleanup_stderr or b"")
        raise RuntimeError("Child timeout addressed only the launcher. Inspect " + str(marker)) from error
    (directory / (phase + ".stdout.log")).write_bytes(stdout)
    (directory / (phase + ".stderr.log")).write_bytes(stderr)
    if child.returncode != 0:
        raise RuntimeError(f"The {phase} child failed with exit code {child.returncode}. Inspect its retained logs.")
    if len(stdout) > 16 * 1024:
        raise AssertionError("Child completion report exceeded its byte bound")
    status = json.loads(stdout.decode("utf-8"))
    if (type(status) is not dict or set(status) != {"phase", "pid", "parent_pid", "checkpoint_sha256", "result_sha256"}
            or status["phase"] != phase
            or any(type(status[key]) is not int or status[key] <= 0 for key in ("pid", "parent_pid"))
            or not _digest(status["checkpoint_sha256"]) or not _digest(status["result_sha256"])):
        raise AssertionError("Child completion report differs from the owned protocol")
    if status["pid"] == child.pid:
        relation = "direct"
    elif os.name == "nt" and status["parent_pid"] == child.pid:
        relation = "windows-venv-redirector"
    else:
        raise AssertionError("The Python runtime does not belong to the completed launcher")
    result_path = directory / ("producer-results.pt" if phase == "produce" else "resume-results.pt")
    payload = _read_bytes(result_path, MAX_RESULT_BYTES)
    if sha256(payload).hexdigest() != status["result_sha256"]:
        raise AssertionError("Child result bytes differ from the completion report")
    result = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    if (type(result) is not dict
            or set(result) != {"phase", "pid", "parent_pid", "metadata", "identity", "checkpoint_sha256", "records", "restore_witness"}
            or any(not _same_plain(result[key], status[key]) for key in ("phase", "pid", "parent_pid", "checkpoint_sha256"))
            or (digest is not None and result["checkpoint_sha256"] != digest)):
        raise AssertionError("Child result metadata differs from its completion report")
    result.update({"launcher_pid": child.pid, "launcher_relation": relation})
    return result


def _errors(actual, expected, tolerance):
    _compare_tree({key: actual[key] for key in ("update", "batch_id", "device", "inputs")},
                  {key: expected[key] for key in ("update", "batch_id", "device", "inputs")}, EXACT)
    return {field: _compare_tree(actual[field], expected[field], tolerance)
            for field in ("output", "parameter_gradients", "parameters", "momentum")}


def _controls(helper, initial, batches, checkpoint, expected, native_suffix):
    target, dtype = expected["target_device"], expected["dtype"]
    threshold = 1e-8 if dtype == "float32" else 1e-11
    results = {}
    for case in ("reset_momentum", "swap_same_shaped_momentum"):
        model, opt = _new_target(initial, target)
        restore(checkpoint, model, opt, expected)
        if case == "reset_momentum":
            opt = optimizer(model)
        else:
            parameters = dict(model.named_parameters())
            first, second = parameters["scorer.weight"], parameters["head.weight"]
            if torch.equal(opt.state[first]["momentum_buffer"], opt.state[second]["momentum_buffer"]):
                raise AssertionError("The swap control needs distinct equal-shaped momentum values")
            opt.state[first]["momentum_buffer"], opt.state[second]["momentum_buffer"] = (
                opt.state[second]["momentum_buffer"], opt.state[first]["momentum_buffer"])
        records = _steps(helper, model, opt, batches, CURSOR, TOTAL_STEPS, target)
        for field in ("inputs", "output", "parameter_gradients"):
            _compare_tree(records[0][field], native_suffix[0][field], EXACT)
        first_drift = max(helper.difference(records[0]["parameters"][name], native_suffix[0]["parameters"][name])
                          for name in STATE_SHAPES)
        final_drift = max(helper.difference(records[-1]["parameters"][name], native_suffix[-1]["parameters"][name])
                          for name in STATE_SHAPES)
        if first_drift <= threshold or final_drift <= threshold:
            raise AssertionError("The momentum control must change both first and final updates")
        results[case] = {"next_forward_and_gradients_exact": True, "first_update_max_abs_error": first_drift,
                         "final_parameter_max_abs_error": final_drift, "divergence_threshold": threshold,
                         "diverged": True, "live_state_bypass_after_valid_restore": True}
    return results


def _plain(value):
    if type(value) is torch.Tensor:
        return value.tolist()
    if type(value) is dict:
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    return value


def run(dtype_name, direction, directory, cpu_wheel, cuda_wheel, source_revision=None, source_tree=None):
    """Finish both fresh children before constructing parent CUDA models."""
    _environment()
    _source_labels(source_revision, source_tree)
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    cpu_wheel, cuda_wheel = Path(cpu_wheel).resolve(strict=True), Path(cuda_wheel).resolve(strict=True)
    helper = _helper()
    identity = _identity(helper, cpu_wheel, cuda_wheel)
    produced = _child(dtype_name, direction, directory, "produce", None,
                      cpu_wheel, cuda_wheel, source_revision, source_tree)
    digest = produced["checkpoint_sha256"]
    resumed = _child(dtype_name, direction, directory, "resume", digest,
                     cpu_wheel, cuda_wheel, source_revision, source_tree)
    pids = {"parent": os.getpid(), "producer": produced["pid"], "resumer": resumed["pid"]}
    if len(set(pids.values())) != 3:
        raise AssertionError("Parent, producer and fresh resumer must be distinct runtimes")
    initial, batches = _fixture(helper, dtype_name)
    original = {"initial": dict(initial.named_parameters()), "batches": batches}
    original_before = _tree_snapshot(original)
    expected = metadata(dtype_name, direction, initial, batches, identity, source_revision, source_tree)
    for result in (produced, resumed):
        if not _same_plain(result["metadata"], expected) or not _same_plain(result["identity"], identity):
            raise AssertionError("Child source, fixture or installed wheel identity changed")
    if resumed["restore_witness"] != {"exact_checkpoint_values": True, "target_device_and_dtype": True,
                                      "same_model_optimizer_parameter_objects": True, "checkpoint_storage_disjoint": True}:
        raise AssertionError("The resumer did not establish all restoration postconditions")
    source, target = DIRECTIONS[direction]
    dtype = DTYPES[dtype_name]
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    target_model, target_opt = _new_target(initial, target)
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint = load_checkpoint(checkpoint_path, digest, target_model, target_opt, expected)
    checkpoint_before = _tree_snapshot(checkpoint)
    # The native target oracle receives the exact source checkpoint by named
    # copies and explicit momentum assignment, without restore/load_state_dict.
    _manual_copy(target_model, target_opt, checkpoint["model"],
                 {name: checkpoint["optimizer"]["state"][index]["momentum_buffer"]
                  for index, name in enumerate(STATE_SHAPES)}, target)
    native_suffix = _steps(helper, target_model, target_opt, batches, CURSOR, TOTAL_STEPS, target)
    source_model, source_opt = _new_target(initial, source)
    native_prefix = _steps(helper, source_model, source_opt, batches, 0, CURSOR, source)
    _compare_tree(_clone(dict(source_model.named_parameters())), checkpoint["model"], tolerance)
    _compare_tree(_clone(source_opt.state_dict()), checkpoint["optimizer"], tolerance)
    primitive_model, primitive_opt = _new_target(initial, source)
    primitive_prefix = _steps(helper, primitive_model, primitive_opt, batches, 0, CURSOR, source, primitive=True)
    primitive_target, primitive_target_opt = _new_target(initial, target)
    _manual_copy(primitive_target, primitive_target_opt, dict(primitive_model.named_parameters()),
                 {name: primitive_opt.state[parameter]["momentum_buffer"]
                  for name, parameter in primitive_model.named_parameters()}, target)
    primitive_suffix = _steps(helper, primitive_target, primitive_target_opt, batches, CURSOR, TOTAL_STEPS,
                              target, primitive=True)
    actual, native, primitive = produced["records"] + resumed["records"], native_prefix + native_suffix, primitive_prefix + primitive_suffix
    if len(actual) != TOTAL_STEPS:
        raise AssertionError("The two processes did not complete exactly four updates")
    native_errors = [_errors(a, b, tolerance) for a, b in zip(actual, native)]
    primitive_errors = [_errors(a, b, tolerance) for a, b in zip(actual, primitive)]
    controls = _controls(helper, initial, batches, checkpoint, expected, native_suffix)
    _tree_unchanged(checkpoint, checkpoint_before)
    _tree_unchanged(original, original_before)
    payload = _read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)
    if sha256(payload).hexdigest() != digest or _identity(helper, cpu_wheel, cuda_wheel) != identity:
        raise AssertionError("Checkpoint, source or installed wheel bytes changed")
    torch.cuda.synchronize(0)
    properties = torch.cuda.get_device_properties(0)
    return {"status": "passed", "method": "state_dict.coinstalled_backend_training_handoff",
            "scope": "Four deterministic synthetic updates on the declared installed CPU/CUDA pair. No general optimizer, stochastic replay, distributed recovery, hostile-file safety, exact CPU/GPU equivalence, performance or model-quality claim.",
            "dtype": dtype_name, "direction": direction, "source_device": source, "target_device": target,
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller labels plus hashed consumer/helper, wheels, native and fixture bytes",
            "optimizer_options": dict(OPTIONS), "optimizer_steps": TOTAL_STEPS, "producer_steps": CURSOR,
            "resumed_steps": TOTAL_STEPS - CURSOR, "checkpoint_cursor": CURSOR,
            "process_ids": pids, "distinct_processes": True,
            "child_launchers": {phase: {key: child[key] for key in ("launcher_pid", "launcher_relation", "parent_pid")}
                                for phase, child in (("producer", produced), ("resumer", resumed))},
            "parent_cuda_models_after_children_exit": True, "exact_restore": resumed["restore_witness"],
            "native_target_oracle_uses_named_copies": True, "primitive_handoff_uses_named_copies": True,
            "native_comparison_max_abs_error": max(max(row.values()) for row in native_errors),
            "primitive_comparison_max_abs_error": max(max(row.values()) for row in primitive_errors),
            "native_step_errors": native_errors, "primitive_step_errors": primitive_errors,
            "comparison_tolerance": tolerance, "controls": controls, "checkpoint_sha256": digest,
            "checkpoint_bytes": len(payload), "checkpoint_byte_limit": MAX_CHECKPOINT_BYTES,
            "checkpoint_unchanged": True, "input_sources_unchanged": True, "initial_parameters_unchanged": True,
            "independent_momentum_recurrence": True, "same_parameter_and_optimizer_objects": True,
            "all_parameter_blocks_changed_each_step": True, "checkpoint_metadata": expected,
            "explicit_device_transfers": True, "checkpoint_tensor_device": "cpu",
            "stochastic_forward": False, "live_rng_continuation": False,
            "installed_identity": identity, "steps": _plain(actual), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor], "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
            "python_executable": sys.executable, "environment_prefix": str(Path(sys.prefix).resolve()),
            "working_directory": str(Path.cwd()), "script_path": str(SCRIPT), "directory": str(directory),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), default="cpu-to-cuda")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cpu-wheel", type=Path, required=True)
    parser.add_argument("--cuda-wheel", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("produce", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    _source_labels(arguments.source_revision, arguments.source_tree)
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    if arguments.phase:
        if (arguments.phase == "resume") != (arguments.expected_sha256 is not None):
            parser.error("Only the resume phase requires an expected checkpoint digest")
        _phase(arguments.dtype, arguments.direction, arguments.directory.resolve(strict=True), arguments.phase,
               arguments.expected_sha256, arguments.cpu_wheel.resolve(strict=True), arguments.cuda_wheel.resolve(strict=True),
               arguments.source_revision, arguments.source_tree)
        return
    if arguments.expected_sha256 is not None:
        parser.error("An expected checkpoint digest belongs to the private resume phase")
    result = run(arguments.dtype, arguments.direction, arguments.directory, arguments.cpu_wheel, arguments.cuda_wheel,
                 arguments.source_revision, arguments.source_tree)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
