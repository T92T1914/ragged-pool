"""Continue one completed Linux CPU DDP accumulation window in fresh workers.

Copy this file and train_distributed.py outside the checkout and use the ordinary
installed CPU wheel. The fixed protocol saves five parameter tensors and five
momentum buffers as bounded JSON after an optimizer update and gradient clear.
It checks deliberate stop and continuation, not crash recovery or elastic training.
"""
import argparse
import copy
from datetime import timedelta
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import struct
import subprocess
import sys
import time
import traceback

import torch


STATE_SHAPES = {"encoder.weight": (4, 3), "encoder.bias": (4,),
                "scorer.weight": (1, 4), "head.weight": (1, 4), "head.bias": (1,)}
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0,
           "weight_decay": 0, "nesterov": False, "maximize": False,
           "foreach": False, "differentiable": False, "fused": False}
DTYPES = {"float32": torch.float32, "float64": torch.float64}
MAX_CHECKPOINT_BYTES = 64 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
GROUP_TIMEOUT_SECONDS = 30
PARENT_TIMEOUT_SECONDS = 120
ADMISSION_TIMEOUT_SECONDS = 45
FORMAT = "ragged-pool.ddp-completed-window.v1"
_TRAINING = None


class ReadinessMismatch(ValueError):
    def __init__(self, rank, mode, field, changed_names):
        self.details = {"rank": rank, "status": "rejected", "mode": mode,
                        "field": field, "changed_tensor_names": changed_names}
        super().__init__(f"Parent rejected rank {rank} pre-process-group {mode} {field}: "
                         + ", ".join(changed_names))


def _helper():
    global _TRAINING
    if _TRAINING is None:
        path = Path(__file__).resolve().with_name("train_distributed.py")
        spec = importlib.util.spec_from_file_location("ragged_ddp_training_helper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINING = module
    return _TRAINING


def _same_plain(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_same_plain(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(_same_plain(a, b) for a, b in zip(first, second))
    return first == second


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def metadata(dtype_name, fixture_sha256, identity):
    if dtype_name not in DTYPES or not _digest(fixture_sha256):
        raise ValueError("Expected a supported dtype and complete fixture digest")
    path = Path(__file__).resolve()
    result = {"format": FORMAT, "dtype": dtype_name, "python": identity["python"],
              "torch": identity["torch"], "device": "cpu", "world_size": 2,
              "microbatches_per_window": 2, "next_window": 1, "next_microbatch": 0,
              "completed_windows": 1, "phase": "completed_optimizer_window", "total_windows": 2,
              "seed": 1811, "fixture_sha256": fixture_sha256,
              "consumer_sha256": sha256(path.read_bytes()).hexdigest(),
              "helper_sha256": sha256(path.with_name("train_distributed.py").read_bytes()).hexdigest(),
              "parameter_names": list(STATE_SHAPES),
              "parameter_shapes": [list(shape) for shape in STATE_SHAPES.values()],
              "optimizer_options": copy.deepcopy(OPTIONS),
              "installed": {"package_version": identity["package_version"],
                            "native_library_sha256": identity["native_library_sha256"],
                            "wheel_sha256": identity.get("wheel_sha256"),
                            "build_config": copy.deepcopy(identity["build_config"])}}
    _validate_metadata(result)
    return result


def _validate_metadata(value):
    fixed = {"format": FORMAT, "python": "3.11.8", "torch": "2.9.1+cpu", "device": "cpu",
             "world_size": 2, "microbatches_per_window": 2, "next_window": 1,
             "next_microbatch": 0, "completed_windows": 1, "phase": "completed_optimizer_window",
             "total_windows": 2, "seed": 1811, "parameter_names": list(STATE_SHAPES),
             "parameter_shapes": [list(shape) for shape in STATE_SHAPES.values()],
             "optimizer_options": OPTIONS}
    expected_keys = set(fixed) | {"dtype", "fixture_sha256", "consumer_sha256", "helper_sha256", "installed"}
    if (type(value) is not dict or set(value) != expected_keys
            or type(value["dtype"]) is not str or value["dtype"] not in DTYPES
            or any(not _same_plain(value[name], expected) for name, expected in fixed.items())
            or any(not _digest(value[name]) for name in ("fixture_sha256", "consumer_sha256", "helper_sha256"))):
        raise ValueError("Checkpoint metadata differs from the fixed completed-window protocol")
    installed = value["installed"]
    if (type(installed) is not dict or set(installed) != {
            "package_version", "native_library_sha256", "wheel_sha256", "build_config"}
            or type(installed["package_version"]) is not str or not installed["package_version"]
            or not _digest(installed["native_library_sha256"])
            or installed["wheel_sha256"] is not None and not _digest(installed["wheel_sha256"])
            or type(installed["build_config"]) is not dict):
        raise ValueError("Checkpoint installed-byte identity is incomplete")
    _encode_json(value, MAX_CHECKPOINT_BYTES)


def _encode_json(value, maximum):
    payload = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if not 0 < len(payload) <= maximum:
        raise ValueError("The fixed JSON artifact exceeded its byte bound")
    return payload


def _unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON object keys are refused")
        result[name] = value
    return result


def _nonfinite_constant(value):
    raise ValueError("Nonfinite JSON constants are refused")


def _decode_json(payload, maximum):
    if type(payload) is not bytes or not 0 < len(payload) <= maximum:
        raise ValueError("Expected bounded JSON bytes")
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=_nonfinite_constant)


def _validate_plain(checkpoint, expected_metadata):
    _validate_metadata(expected_metadata)
    if (type(checkpoint) is not dict or set(checkpoint) != {"metadata", "model", "momentum"}
            or not _same_plain(checkpoint["metadata"], expected_metadata)):
        raise ValueError("Checkpoint fields or metadata differ from the expected completed window")
    dtype_name = expected_metadata["dtype"]
    maximum = 3.4028234663852886e38 if dtype_name == "float32" else sys.float_info.max
    for field in ("model", "momentum"):
        records = checkpoint[field]
        if type(records) is not list or len(records) != len(STATE_SHAPES):
            raise ValueError("Checkpoint needs exactly five ordered tensor records per state kind")
        for record, (name, shape) in zip(records, STATE_SHAPES.items()):
            if (type(record) is not dict or set(record) != {"name", "shape", "dtype", "values"}
                    or not _same_plain(record["name"], name)
                    or not _same_plain(record["shape"], list(shape))
                    or not _same_plain(record["dtype"], dtype_name)
                    or type(record["values"]) is not list
                    or len(record["values"]) != math.prod(shape)
                    or any(type(value) is not float or not math.isfinite(value)
                           or abs(value) > maximum for value in record["values"])):
                raise ValueError("Checkpoint tensor names, shapes, dtype, scalar types or finite values differ")
            if dtype_name == "float32" and any(
                    struct.unpack("!f", struct.pack("!f", value))[0] != value
                    for value in record["values"]):
                raise ValueError("Checkpoint scalars must exactly represent the declared float32 values")
    _encode_json(checkpoint, MAX_CHECKPOINT_BYTES)


def encode_checkpoint(checkpoint):
    if type(checkpoint) is not dict or "metadata" not in checkpoint:
        raise ValueError("Expected the declared checkpoint object")
    _validate_plain(checkpoint, checkpoint["metadata"])
    return _encode_json(checkpoint, MAX_CHECKPOINT_BYTES)


def decode_checkpoint(payload, expected_sha256, expected_metadata):
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Expected bounded checkpoint bytes")
    if not _digest(expected_sha256) or sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint digest differs from the trusted local artifact")
    result = _decode_json(payload, MAX_CHECKPOINT_BYTES)
    _validate_plain(result, expected_metadata)
    return result


def _read_bytes(path, maximum):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not 0 < info.st_size <= maximum):
        raise ValueError("Expected a bounded ordinary local JSON file")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if not 0 < len(payload) <= maximum:
        raise ValueError("The local JSON file exceeded its bound during reading")
    return payload


def load_checkpoint(path, expected_sha256, expected_metadata):
    return decode_checkpoint(_read_bytes(path, MAX_CHECKPOINT_BYTES), expected_sha256, expected_metadata)


def _storage(value):
    return value.untyped_storage().data_ptr()


def _bindings(model, opt, dtype_name):
    if dtype_name not in DTYPES or type(opt) is not torch.optim.SGD:
        raise ValueError("Use the fixed CPU dtype and ordinary SGD optimizer")
    named = dict(model.named_parameters())
    if (tuple(named) != tuple(STATE_SHAPES) or tuple(model.state_dict()) != tuple(STATE_SHAPES)
            or not model.training or len(opt.param_groups) != 1
            or tuple(id(value) for value in opt.param_groups[0]["params"])
            != tuple(id(value) for value in named.values())
            or not _same_plain({name: value for name, value in opt.param_groups[0].items()
                                if name != "params"}, OPTIONS)):
        raise ValueError("Model and optimizer must retain the declared ordered live bindings and options")
    for name, value in named.items():
        if (type(value) is not torch.nn.Parameter or tuple(value.shape) != STATE_SHAPES[name]
                or value.dtype != DTYPES[dtype_name] or value.device != torch.device("cpu")
                or value.layout != torch.strided or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                or torch.is_inference(value) or value.is_conj() or value.is_neg()
                or not value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError("Live model tensors must retain their finite fixed CPU representation")
    if len({_storage(value) for value in named.values()}) != len(named):
        raise ValueError("The five live parameters must own independent storage")
    return named


def _tensor_records(named, dtype_name):
    return [{"name": name, "shape": list(STATE_SHAPES[name]), "dtype": dtype_name,
             "values": value.detach().reshape(-1).tolist()} for name, value in named.items()]


def _state(model, opt, expected_metadata):
    named = _bindings(model, opt, expected_metadata["dtype"])
    if set(opt.state) != set(named.values()):
        raise ValueError("Every live parameter must have its saved momentum buffer")
    momentum = {}
    for name, value in named.items():
        state = opt.state[value]
        if type(state) is not dict or set(state) != {"momentum_buffer"}:
            raise ValueError("Every live parameter needs exactly one momentum buffer")
        buffer = state["momentum_buffer"]
        if (type(buffer) is not torch.Tensor or buffer.shape != value.shape
                or buffer.dtype != value.dtype or buffer.device != value.device
                or buffer.layout != torch.strided or not buffer.is_contiguous()
                or buffer.storage_offset() != 0
                or buffer.untyped_storage().nbytes() != buffer.numel() * buffer.element_size()
                or buffer.requires_grad or torch.is_inference(buffer) or buffer.is_conj()
                or buffer.is_neg() or not bool(torch.isfinite(buffer).all())):
            raise ValueError("Live momentum must retain its finite fixed CPU representation")
        momentum[name] = buffer
    if len({_storage(value) for value in (*named.values(), *momentum.values())}) != 10:
        raise ValueError("Model parameters and momentum must own ten independent storages")
    result = {"metadata": copy.deepcopy(expected_metadata),
              "model": _tensor_records(named, expected_metadata["dtype"]),
              "momentum": _tensor_records(momentum, expected_metadata["dtype"])}
    _validate_plain(result, expected_metadata)
    return result


def make_checkpoint(model, opt, expected_metadata):
    _validate_metadata(expected_metadata)
    if any(value.grad is not None for value in model.parameters()):
        raise ValueError("Clear every gradient before saving a completed accumulation window")
    return _state(model, opt, expected_metadata)


def validate_checkpoint(checkpoint, model, opt, expected_metadata):
    """Admit all saved fields and live bindings before either target is loaded."""
    _validate_plain(checkpoint, expected_metadata)
    _bindings(model, opt, expected_metadata["dtype"])
    if any(value.grad is not None for value in model.parameters()):
        raise ValueError("Restore targets must start with every gradient cleared")
    if opt.state:
        _state(model, opt, expected_metadata)


def restore(checkpoint, model, opt, expected_metadata):
    validate_checkpoint(checkpoint, model, opt, expected_metadata)
    dtype = DTYPES[expected_metadata["dtype"]]
    named = _bindings(model, opt, expected_metadata["dtype"])
    binding = tuple((id(value), _storage(value)) for value in named.values())
    original = copy.deepcopy(checkpoint)
    model_state = {record["name"]: torch.tensor(record["values"], dtype=dtype).reshape(record["shape"])
                   for record in checkpoint["model"]}
    saved_momentum = {record["name"]: torch.tensor(record["values"], dtype=dtype).reshape(record["shape"])
                      for record in checkpoint["momentum"]}
    # assign=False preserves the parameter objects already owned by the optimizer.
    # Independent clones also prevent same-device optimizer loading from retaining
    # storage owned by the decoded checkpoint tensors.
    model.load_state_dict({name: value.clone() for name, value in model_state.items()},
                          strict=True, assign=False)
    opt.load_state_dict({"state": {index: {"momentum_buffer": saved_momentum[name].clone()}
                                  for index, name in enumerate(STATE_SHAPES)},
                         "param_groups": [dict(copy.deepcopy(OPTIONS), params=list(range(5)))]})
    opt.zero_grad(set_to_none=True)
    restored = make_checkpoint(model, opt, expected_metadata)
    if (not _same_plain(restored, checkpoint) or not _same_plain(checkpoint, original)
            or tuple((id(value), _storage(value)) for value in model.parameters()) != binding):
        raise AssertionError("Restore changed bindings, checkpoint contents or exact state values")
    decoded_storage = {_storage(value) for value in (*model_state.values(), *saved_momentum.values())}
    live_storage = {_storage(value) for value in model.parameters()} | {
        _storage(opt.state[value]["momentum_buffer"]) for value in model.parameters()}
    if decoded_storage & live_storage:
        raise AssertionError("Restored live state aliases decoded checkpoint storage")
    return {"exact_checkpoint_values": True, "cpu_device_and_dtype": True,
            "same_model_optimizer_parameter_objects": True, "independent_live_storage": True,
            "all_parameter_gradients_none": True}


def _write_plain(path, value, maximum=MAX_RECEIPT_BYTES):
    payload = _encode_json(value, maximum)
    path = Path(path)
    if path.exists():
        raise FileExistsError("The task-owned JSON output must be new")
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        stream.write(payload)
    # Names belong to this fresh scenario. Publish only the closed JSON bytes,
    # so a peer cannot observe a partially written ready or release receipt.
    temporary.rename(path)
    return sha256(payload).hexdigest()


def _write_receipt(path, value):
    return _write_plain(path, _helper()._json_value(value))


def _read_plain(path):
    return _decode_json(_read_bytes(path, MAX_RECEIPT_BYTES), MAX_RECEIPT_BYTES)


def _wait_for_parent(directory):
    deadline = time.monotonic() + ADMISSION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if (directory / "abort.json").exists():
            raise RuntimeError("Parent refused checkpoint readiness before process-group initialization")
        go = directory / "go.json"
        if go.exists():
            if not _same_plain(_read_plain(go), {"release": True}):
                raise ValueError("Parent admission release differs from the fixed protocol")
            return
        time.sleep(0.05)
    raise RuntimeError("Parent admission gate exceeded its finite deadline")


def _worker(rank, dtype_name, directory, phase, expected_metadata, wheel,
            checkpoint_path=None, checkpoint_sha256=None, reject_rank1=False):
    directory = Path(directory)
    result = {"status": "error", "phase": phase, "rank": rank,
              "pid": os.getpid(), "parent_pid": os.getppid(),
              "process_group_initialization_attempted": False,
              "ddp_construction_attempted": False}
    initialized = False
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (directory / f"worker-{rank}.log").open("x", encoding="utf-8", buffering=1) as stream:
        sys.stdout = sys.stderr = stream
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            helper = _helper()
            identity = helper.installed_identity(wheel)
            initial, batches, fixture_digest = helper.accumulation_fixture(dtype_name)
            actual_metadata = metadata(dtype_name, fixture_digest, identity)
            if not _same_plain(actual_metadata, expected_metadata):
                raise ValueError("Worker source, installed bytes or fixtures differ before admission")
            template_before = helper.snapshot(dict(initial.named_parameters()))
            fixture_inputs = helper._accumulation_fixture_inputs(batches)
            fixture_before = helper.snapshot(fixture_inputs)
            models, optimizers, witnesses, states = {}, {}, {}, {}
            modes = ("correct", "reset_momentum_control") if phase == "resumed" else ("correct",)
            for mode in modes:
                model = copy.deepcopy(initial)
                opt = torch.optim.SGD(model.parameters(), **OPTIONS)
                if phase == "resumed":
                    checkpoint = load_checkpoint(checkpoint_path, checkpoint_sha256, expected_metadata)
                    witnesses[mode] = restore(checkpoint, model, opt, expected_metadata)
                    if reject_rank1 and rank == 1 and mode == "correct":
                        # This changes actual live state. The parent must notice
                        # the mismatch before DDP can broadcast any model values.
                        first = next(model.parameters())
                        opt.state[first]["momentum_buffer"].view(-1)[0].add_(1)
                    states[mode] = make_checkpoint(model, opt, expected_metadata)
                else:
                    states[mode] = {"model": _tensor_records(dict(model.named_parameters()), dtype_name),
                                    "momentum": None}
                models[mode], optimizers[mode] = model, opt
            # metadata describes the expected checkpoint protocol. model_cursor
            # separately identifies where these admitted live models will begin.
            cursor = 1 if phase == "resumed" else 0
            ready = {"phase": phase, "rank": rank, "pid": os.getpid(),
                     "parent_pid": os.getppid(), "metadata": actual_metadata,
                     "model_cursor": {"completed_windows": cursor, "next_window": cursor,
                                      "next_microbatch": 0},
                     "identity": identity, "checkpoint_sha256": checkpoint_sha256,
                     "restored_states": states, "restore_witnesses": witnesses,
                     "process_group_initialization_attempted": False,
                     "ddp_construction_attempted": False}
            ready["deliberate_rejection"] = ({"rank": 1, "field": "momentum",
                                                "tensor_name": next(iter(STATE_SHAPES))}
                                               if phase == "resumed" and reject_rank1 and rank == 1 else None)
            _write_plain(directory / f"ready-{rank}.json", ready)
            result["ready_written"] = True
            _wait_for_parent(directory)
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel

            def loopback_gloo(store, process_rank, size, timeout):
                options = dist.ProcessGroupGloo._Options()
                options._devices = [dist.ProcessGroupGloo.create_device(
                    hostname="127.0.0.1", lazy_init=False)]
                options._threads = 1
                options._timeout = timeout
                return dist.ProcessGroupGloo(store, process_rank, size, options)

            if not dist.is_available() or not dist.is_gloo_available():
                raise RuntimeError("This pinned environment does not provide Gloo")
            dist.Backend.register_backend("ragged_gloo_loopback", loopback_gloo, devices=["cpu"])
            _write_plain(directory / f"pg-init-attempt-{rank}.json", {"rank": rank})
            result["process_group_initialization_attempted"] = True
            dist.init_process_group("ragged_gloo_loopback",
                                    init_method=(directory / "gloo-rendezvous").resolve().as_uri(),
                                    rank=rank, world_size=2,
                                    timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS))
            initialized = True
            records, communication, completed_states = {}, {}, {}
            windows = (0, 1) if phase == "continuous" else ((0,) if phase == "producer" else (1,))
            for mode in modes:
                model, opt = models[mode], optimizers[mode]
                if mode == "reset_momentum_control":
                    for parameter in model.parameters():
                        opt.state[parameter]["momentum_buffer"].zero_()
                _write_plain(directory / f"ddp-attempt-{rank}-{mode}.json", {"rank": rank, "mode": mode})
                result["ddp_construction_attempted"] = True
                ddp = DistributedDataParallel(model, device_ids=None, find_unused_parameters=False)
                if phase == "resumed" and mode == "correct":
                    if not _same_plain(make_checkpoint(model, opt, expected_metadata), states[mode]):
                        raise AssertionError("DDP construction changed an already attested restored state")
                hook_state = {"group": dist.group.WORLD, "events": [], "window": None, "microbatch": None}
                ddp.register_comm_hook(hook_state, helper._counted_allreduce)
                records[mode] = []
                for window in windows:
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise AssertionError("A completed-window continuation began with pending gradients")
                    batch = [helper.accumulation_rank_batch(batches, window, micro, rank) for micro in range(2)]
                    step = helper._accumulation_step(ddp, opt, batch, window=window,
                                                     mode="global_window_mean", hook_state=hook_state,
                                                     distributed=True)
                    if any(parameter.grad is None for parameter in model.parameters()):
                        raise AssertionError("The completed update lost a parameter gradient")
                    opt.zero_grad(set_to_none=True)
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise AssertionError("The completed update did not clear all gradients")
                    step.update({"window": window, "entry_gradients_none": True,
                                 "post_step_gradients_were_present": True,
                                 "completed_boundary_gradients_none": True})
                    records[mode].append(step)
                communication[mode] = hook_state["events"]
                current = _state(model, opt, expected_metadata)
                if phase == "producer":
                    completed_states[mode] = make_checkpoint(model, opt, expected_metadata)
                else:
                    completed_states[mode] = {"completed_windows": 2, "next_window": 2,
                                              "next_microbatch": 0,
                                              "all_parameter_gradients_none": True,
                                              "model": current["model"], "momentum": current["momentum"]}
                if phase == "producer":
                    payload = encode_checkpoint(completed_states[mode])
                    digest = sha256(payload).hexdigest()
                    if rank == 0:
                        with Path(checkpoint_path).open("xb") as target:
                            target.write(payload)
                    result["checkpoint_sha256"] = digest
                del ddp
            helper.unchanged(dict(initial.named_parameters()), template_before)
            helper.unchanged(fixture_inputs, fixture_before)
            result.update({"status": "passed", "identity": identity, "metadata": actual_metadata,
                           "routes": records, "communication_hooks": communication,
                           "completed_states": completed_states,
                           "template_unchanged": True, "fixtures_unchanged": True})
        except BaseException as error:
            traceback.print_exc()
            result.update({"error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            try:
                if initialized:
                    dist.destroy_process_group()
                    result["process_group_destroyed"] = True
            finally:
                try:
                    _write_receipt(directory / f"worker-{rank}.json", result)
                finally:
                    sys.stdout, sys.stderr = original_stdout, original_stderr


def _check_ready(ready, rank, process, phase, expected_metadata, expected_identity,
                 expected_initial, checkpoint, checkpoint_sha256, reject_rank1):
    fields = {"phase", "rank", "pid", "parent_pid", "metadata", "identity", "checkpoint_sha256",
              "restored_states", "restore_witnesses", "process_group_initialization_attempted",
              "ddp_construction_attempted", "deliberate_rejection", "model_cursor"}
    if (type(ready) is not dict or set(ready) != fields
            or not _same_plain(ready["phase"], phase) or not _same_plain(ready["rank"], rank)
            or not _same_plain(ready["pid"], process.pid)
            or not _same_plain(ready["parent_pid"], os.getpid())
            or not _same_plain(ready["metadata"], expected_metadata)
            or not _same_plain(ready["identity"], expected_identity)
            or not _same_plain(ready["checkpoint_sha256"], checkpoint_sha256)
            or ready["process_group_initialization_attempted"] is not False
            or ready["ddp_construction_attempted"] is not False):
        raise ValueError("Worker ready receipt differs from its owned pre-process-group identity")
    cursor = 1 if phase == "resumed" else 0
    if not _same_plain(ready["model_cursor"], {"completed_windows": cursor, "next_window": cursor,
                                             "next_microbatch": 0}):
        raise ValueError("Worker ready model cursor differs from its cohort")
    expected_control = ({"rank": 1, "field": "momentum", "tensor_name": next(iter(STATE_SHAPES))}
                        if phase == "resumed" and reject_rank1 and rank == 1 else None)
    if not _same_plain(ready["deliberate_rejection"], expected_control):
        raise ValueError("Worker ready receipt has the wrong deliberate-control attribution")
    if phase == "resumed":
        expected_states = {"correct": checkpoint, "reset_momentum_control": checkpoint}
        expected_witness = {"exact_checkpoint_values": True, "cpu_device_and_dtype": True,
                            "same_model_optimizer_parameter_objects": True,
                            "independent_live_storage": True, "all_parameter_gradients_none": True}
        expected_witnesses = {mode: expected_witness for mode in expected_states}
    else:
        expected_states = {"correct": {"model": expected_initial, "momentum": None}}
        expected_witnesses = {}
    if not _same_plain(ready["restore_witnesses"], expected_witnesses):
        raise ValueError("Worker restoration witness differs from the declared pre-process-group contract")
    if type(ready["restored_states"]) is not dict or ready["restored_states"].keys() != expected_states.keys():
        raise ValueError("Worker ready state modes differ from the fixed cohort")
    for mode, expected in expected_states.items():
        actual = ready["restored_states"][mode]
        if phase == "resumed":
            _validate_plain(actual, expected_metadata)
        if not _same_plain(actual, expected):
            for field in ("model", "momentum"):
                if not _same_plain(actual[field], expected[field]):
                    changed = [name for index, name in enumerate(STATE_SHAPES)
                               if not _same_plain(actual[field][index], expected[field][index])]
                    raise ReadinessMismatch(rank, mode, field, changed)
            raise ValueError("Worker pre-process-group state metadata differs")
    return {"rank": rank, "status": "accepted", "canonical_state_exact": True}


def _launch_cohort(dtype_name, directory, phase, expected_metadata, identity, expected_initial,
                   wheel, checkpoint_path=None, checkpoint_sha256=None, checkpoint=None,
                   reject_rank1=False):
    directory.mkdir(parents=True, exist_ok=False)
    rendezvous = directory / "gloo-rendezvous"
    if rendezvous.exists():
        raise ValueError("The task-owned rendezvous path must be absent")
    environment = dict(os.environ, GLOO_DEVICE_TRANSPORT="TCP", TORCH_GLOO_LAZY_INIT="0")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        environment[name] = "1"
    started, ready, state_checks = [], {}, []
    failure, cleanup, released, verified = None, [], False, False
    start_time = time.monotonic()
    try:
        for rank in range(2):
            arguments = (rank, dtype_name, str(directory), phase, expected_metadata, wheel,
                         str(checkpoint_path) if checkpoint_path is not None else None,
                         checkpoint_sha256, reject_rank1)
            code = ("import runpy; runpy.run_path(" + repr(str(Path(__file__).resolve()))
                    + ")['_worker'](*" + repr(arguments) + ")")
            with (directory / f"bootstrap-{rank}.log").open("xb") as stream:
                process = subprocess.Popen([sys.executable, "-I", "-B", "-c", code],
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment, shell=False)
            started.append(process)
        while True:
            for rank, process in enumerate(started):
                path = directory / f"ready-{rank}.json"
                if rank not in ready and path.exists():
                    value = _read_plain(path)
                    ready[rank] = value
                if process.poll() is not None and process.returncode != 0:
                    raise RuntimeError(f"Owned worker {process.pid} exited with code {process.returncode}")
            if not released and len(ready) == 2:
                # This gate compares only bounded plain JSON while workers live.
                # DDP construction has not yet had a chance to broadcast rank 0.
                for rank, process in enumerate(started):
                    try:
                        state_checks.append(_check_ready(
                            ready[rank], rank, process, phase, expected_metadata, identity,
                            expected_initial, checkpoint, checkpoint_sha256, reject_rank1))
                    except ReadinessMismatch as error:
                        state_checks.append(error.details)
                        raise
                verified = True
                _write_plain(directory / "go.json", {"release": True})
                released = True
            if all(process.poll() is not None for process in started):
                break
            elapsed = time.monotonic() - start_time
            if elapsed > PARENT_TIMEOUT_SECONDS:
                raise RuntimeError("Owned cohort exceeded the finite parent deadline")
            if not released and elapsed > ADMISSION_TIMEOUT_SECONDS:
                raise RuntimeError("Owned cohort exceeded the finite readiness deadline")
            time.sleep(0.05)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        try:
            if not (directory / "abort.json").exists():
                _write_plain(directory / "abort.json", {"abort": True, "reason": failure})
        except BaseException as abort_error:
            failure += f"; abort publication failed: {type(abort_error).__name__}: {abort_error}"
        finally:
            # Give both fresh workers a short chance to observe the noncollective
            # abort. Owned cleanup still runs if publishing that file failed.
            deadline = time.monotonic() + 2
            try:
                for process in started:
                    try:
                        process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                cleanup = _helper()._cleanup_owned(started)
    finally:
        if not failure:
            for process in started:
                process.wait(timeout=1)
    owned = [{"rank": rank, "pid": process.pid, "exitcode": process.poll(),
              "alive": process.poll() is None} for rank, process in enumerate(started)]
    lifecycle = {"owned_processes": owned, "all_owned_workers_reaped": len(owned) == 2
                 and all(not item["alive"] and item["exitcode"] is not None for item in owned),
                 "cleanup_actions": cleanup, "failure": failure, "shell": False,
                 "parent_deadline_seconds": PARENT_TIMEOUT_SECONDS,
                 "admission_deadline_seconds": ADMISSION_TIMEOUT_SECONDS,
                 "process_group_timeout_seconds": GROUP_TIMEOUT_SECONDS}
    receipts = []
    for rank, process in enumerate(started):
        path = directory / f"worker-{rank}.json"
        if path.exists():
            value = _read_plain(path)
            if (value.get("pid") != process.pid or value.get("parent_pid") != os.getpid()
                    or value.get("rank") != rank or value.get("phase") != phase):
                raise AssertionError("Worker result does not match its directly owned process")
            receipts.append(value)
    result = {"lifecycle": lifecycle,
              "admission": {"released": released, "pre_process_group_states_verified": verified,
                            "ready_receipts": [ready[key] for key in sorted(ready)],
                            "state_checks": state_checks, "reason": failure},
              "process_group_initialization_attempted": [
                  (directory / f"pg-init-attempt-{rank}.json").exists() for rank in range(2)],
              "ddp_construction_attempted": [any(directory.glob(f"ddp-attempt-{rank}-*.json"))
                                              for rank in range(2)],
              "workers": receipts}
    if not lifecycle["all_owned_workers_reaped"]:
        raise RuntimeError("A directly owned cohort process could not be reaped")
    if not failure and (len(receipts) != 2 or any(value["status"] != "passed"
                                                or not value.get("process_group_destroyed")
                                                for value in receipts)):
        raise AssertionError("A successful cohort lacks complete destroyed-process-group receipts")
    _write_plain(directory / "parent-cohort.json", result)
    return result


def _tolerance(dtype_name):
    return ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
            else {"rtol": 2e-10, "atol": 2e-12})


def _checkpoint_tensors(checkpoint, field, dtype_name):
    return {record["name"]: torch.tensor(record["values"], dtype=DTYPES[dtype_name]).reshape(record["shape"])
            for record in checkpoint[field]}


def _handoff_oracle(helper, initial, batches, checkpoint, dtype_name, route):
    """Copy named state directly, independently of the checkpoint restore helper."""
    model = copy.deepcopy(initial)
    opt = torch.optim.SGD(model.parameters(), **OPTIONS)
    parameters = _checkpoint_tensors(checkpoint, "model", dtype_name)
    momentum = _checkpoint_tensors(checkpoint, "momentum", dtype_name)
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(parameters[name])
            opt.state[value]["momentum_buffer"] = momentum[name].clone()
    return helper._step(model, opt, helper.accumulation_global_batch(batches, 1),
                        route, "global_group_mean")


def _compare_records(helper, first, second, tolerance):
    result = {field: helper.compare(first[field], second[field], tolerance)
              for field in ("parameter_gradients", "parameters", "momentum")}
    result["microbatches"] = [helper.compare(
        {name: first["microbatches"][micro][name]
         for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
        {name: second["microbatches"][micro][name]
         for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
        tolerance) for micro in range(2)]
    result["metric"] = helper.compare({"metric": first["metric"]}, {"metric": second["metric"]}, tolerance)
    return result


def _compare_results(helper, cohorts, checkpoint, initial, batches, dtype_name):
    workers = {phase: [helper._tensor_value(value) for value in cohort["workers"]]
               for phase, cohort in cohorts.items()}
    native, native_first = helper._accumulation_oracle(initial, batches, "native")
    primitive, primitive_first = helper._accumulation_oracle(initial, batches, "primitive")
    handoff = {route: _handoff_oracle(helper, initial, batches, checkpoint, dtype_name, route)
               for route in ("native", "primitive")}
    tolerance, exact = _tolerance(dtype_name), {"rtol": 0, "atol": 0}
    rank_errors, replay_errors, oracle_errors, local_errors, worker_errors = [], [], [], [], []
    for phase, phase_workers in workers.items():
        modes = ("correct", "reset_momentum_control") if phase == "resumed" else ("correct",)
        for mode in modes:
            for index, record in enumerate(phase_workers[0]["routes"][mode]):
                rank_errors.append({"phase": phase, "mode": mode, "window": record["window"],
                                    "comparison": {field: helper.compare(
                                        record[field], phase_workers[1]["routes"][mode][index][field], exact)
                                        for field in ("parameter_gradients", "parameters", "momentum")}})
    for window, phase in ((0, "producer"), (1, "resumed")):
        for rank in range(2):
            continuous = workers["continuous"][rank]["routes"]["correct"][window]
            replay = workers[phase][rank]["routes"]["correct"][0]
            replay_errors.append({"window": window, "rank": rank,
                                  "comparison": _compare_records(helper, continuous, replay, exact)})
        oracle_record = {field: helper.compare(native[window][field], primitive[window][field], tolerance)
                         for field in ("parameter_gradients", "parameters", "momentum")}
        oracle_record["outputs"] = helper.compare(
            {name: native[window][name] for name in ("predictions", "pooled", "raw_gradients", "metric")},
            {name: primitive[window][name] for name in ("predictions", "pooled", "raw_gradients", "metric")}, tolerance)
        oracle_errors.append(oracle_record)
        raw_start = group_start = 0
        for micro in range(2):
            for rank in range(2):
                batch = helper.accumulation_rank_batch(batches, window, micro, rank)
                raw_stop = raw_start + batch["raw"].shape[0]
                group_stop = group_start + batch["targets"].numel()
                replay = workers[phase][rank]["routes"]["correct"][0]
                observed = replay["microbatches"][micro]
                record = {"window": window, "microbatch": micro, "rank": rank}
                choices = [("native", native[window]), ("primitive", primitive[window])]
                if window == 1:
                    choices += [("handoff_native", handoff["native"]), ("handoff_primitive", handoff["primitive"])]
                for label, oracle in choices:
                    expected = {"predictions": oracle["predictions"][group_start:group_stop],
                                "pooled": oracle["pooled"][group_start:group_stop],
                                "raw_gradients": oracle["raw_gradients"][raw_start:raw_stop] * 2}
                    record[label] = {"outputs": helper.compare({name: observed[name] for name in expected},
                                                                expected, tolerance),
                                     "metric": helper.compare({"metric": replay["metric"]},
                                                              {"metric": oracle["metric"]}, tolerance)}
                    for field in ("parameter_gradients", "parameters", "momentum"):
                        record[label][field] = helper.compare(replay[field], oracle[field], tolerance)
                worker_errors.append(record)
                if micro == 0:
                    local_errors.append({"window": window, "rank": rank,
                                         "parameter_gradients": helper.compare(
                                             replay["first_micro_parameter_gradients"],
                                             primitive_first[window][rank]["parameter_gradients"], tolerance)})
                raw_start, group_start = raw_stop, group_stop
    prefix = workers["producer"][0]["routes"]["correct"][0]
    checkpoint_errors = {field: helper.compare(_checkpoint_tensors(checkpoint, saved, dtype_name),
                                               prefix[field], exact)
                         for field, saved in (("parameters", "model"), ("momentum", "momentum"))}
    controls, threshold = [], 1e-8 if dtype_name == "float32" else 1e-11
    for rank in range(2):
        correct = workers["resumed"][rank]["routes"]["correct"][0]
        wrong = workers["resumed"][rank]["routes"]["reset_momentum_control"][0]
        same_forward = [helper.compare(
            {name: correct["microbatches"][micro][name]
             for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
            {name: wrong["microbatches"][micro][name]
             for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
            exact) for micro in range(2)]
        same_gradients = helper.compare(correct["parameter_gradients"], wrong["parameter_gradients"], exact)
        drift = {field: max(helper.maximum_difference(correct[field][name], wrong[field][name])
                            for name in STATE_SHAPES) for field in ("parameters", "momentum")}
        if not all(value > threshold for value in drift.values()):
            raise AssertionError("Resetting momentum did not change the continued optimizer update")
        controls.append({"rank": rank, "forward_and_raw_gradients_exact": same_forward,
                         "parameter_gradients_exact": same_gradients,
                         "max_absolute_drift": drift, "required_drift_greater_than": threshold})
    return ({"tolerance": tolerance, "replay_tolerance": exact, "raw_input_gradient_scale": 2,
             "rank_equality": rank_errors, "continuous_vs_restarted": replay_errors,
             "native_vs_primitive": oracle_errors, "worker_vs_oracle": worker_errors,
             "first_micro_vs_local_primitive": local_errors,
             "checkpoint_vs_producer": checkpoint_errors, "reset_momentum_controls": controls},
            {"native": native, "primitive": primitive, "local_first_micro_at_native_state": native_first,
             "local_first_micro_at_primitive_state": primitive_first, "manual_handoff": handoff})


def run(dtype_name, directory, wheel=None, reject_resume_rank1=False):
    if sys.platform != "linux":
        raise RuntimeError(
            "The two-rank Gloo consumer is qualified on Linux only; the tested "
            "Windows PyTorch 2.9.1+cpu wheel did not provide a usable loopback transport")
    if dtype_name not in DTYPES or type(reject_resume_rank1) is not bool:
        raise ValueError("Expected a supported dtype and explicit boolean admission control")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    helper = _helper()
    identity = helper.installed_identity(wheel)
    initial, batches, fixture_digest = helper.accumulation_fixture(dtype_name)
    expected_metadata = metadata(dtype_name, fixture_digest, identity)
    expected_initial = _tensor_records(dict(initial.named_parameters()), dtype_name)
    initial_before = helper.snapshot(dict(initial.named_parameters()))
    inputs = helper._accumulation_fixture_inputs(batches)
    inputs_before = helper.snapshot(inputs)
    checkpoint_path = directory / "completed-window.json"
    cohorts = {}
    for phase in ("continuous", "producer"):
        cohorts[phase] = _launch_cohort(dtype_name, directory / phase, phase, expected_metadata,
                                        identity, expected_initial, wheel,
                                        checkpoint_path=checkpoint_path if phase == "producer" else None)
        if cohorts[phase]["lifecycle"]["failure"]:
            raise RuntimeError(f"The {phase} cohort failed: " + cohorts[phase]["lifecycle"]["failure"])
    # Every producer process has exited before the parent reads the checkpoint
    # or launches either fresh restored model. No in-process group is reused.
    producer = cohorts["producer"]["workers"]
    checkpoint_sha256 = producer[0]["checkpoint_sha256"]
    if any(value["checkpoint_sha256"] != checkpoint_sha256 for value in producer):
        raise AssertionError("Producer ranks disagree on the completed checkpoint bytes")
    checkpoint = load_checkpoint(checkpoint_path, checkpoint_sha256, expected_metadata)
    if any(not _same_plain(value["completed_states"]["correct"], checkpoint) for value in producer):
        raise AssertionError("Producer ranks differ from the actual completed checkpoint")
    cohorts["resumed"] = _launch_cohort(
        dtype_name, directory / "resumed", "resumed", expected_metadata, identity,
        expected_initial, wheel, checkpoint_path, checkpoint_sha256, checkpoint, reject_resume_rank1)
    result = {"status": "passed", "dtype": dtype_name, "identity": identity,
              "metadata": expected_metadata, "checkpoint_sha256": checkpoint_sha256,
              "checkpoint_path": str(checkpoint_path), "cohorts": cohorts,
              "deliberate_rejection_rank": 1 if reject_resume_rank1 else None,
              "transport_scope": "single host; TCP Gloo device 127.0.0.1; unique file rendezvous per cohort",
              "numerical_threads_per_worker": 1, "gloo_communication_threads_per_worker": 1}
    resumed = cohorts["resumed"]
    if reject_resume_rank1:
        checks = resumed["admission"]["state_checks"]
        expected_checks = [{"rank": 0, "status": "accepted", "canonical_state_exact": True},
                           {"rank": 1, "status": "rejected", "mode": "correct", "field": "momentum",
                            "changed_tensor_names": [next(iter(STATE_SHAPES))]}]
        if (not _same_plain(checks, expected_checks) or len(resumed["admission"]["ready_receipts"]) != 2
                or resumed["admission"]["released"] or resumed["process_group_initialization_attempted"] != [False, False]
                or resumed["ddp_construction_attempted"] != [False, False]
                or not resumed["lifecycle"]["failure"]):
            raise AssertionError("The injected rank-one control did not cause the intended pre-process-group state refusal")
        result["status"] = "failed"
        result["admission_control"] = {"parent_rejected_actual_rank1_momentum": True,
                                       "rank0_canonical_state_accepted": True,
                                       "both_owned_ready_receipts_observed": True,
                                       "changed_tensor_name": next(iter(STATE_SHAPES)),
                                       "no_process_group_or_ddp_attempted": True}
    elif resumed["lifecycle"]["failure"]:
        raise RuntimeError("The resumed cohort failed: " + resumed["lifecycle"]["failure"])
    else:
        comparison, oracles = _compare_results(helper, cohorts, checkpoint, initial, batches, dtype_name)
        result.update({"comparison": comparison, "oracles": oracles})
    helper.unchanged(dict(initial.named_parameters()), initial_before)
    helper.unchanged(inputs, inputs_before)
    if sha256(_read_bytes(checkpoint_path, MAX_CHECKPOINT_BYTES)).hexdigest() != checkpoint_sha256:
        raise AssertionError("A cohort or comparison changed the checkpoint bytes")
    result.update({"checkpoint_unchanged": True, "template_unchanged": True, "fixtures_unchanged": True})
    _write_receipt(directory / "parent.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--reject-resume-rank1", action="store_true",
                        help="Require the parent to refuse a deliberately changed rank-one momentum witness before Gloo")
    arguments = parser.parse_args()
    for name in ("source_revision", "source_tree"):
        value = getattr(arguments, name)
        if value is not None and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a complete lowercase Git object ID")
    if (arguments.source_revision is None) != (arguments.source_tree is None):
        parser.error("Supply both source identity labels together")
    if arguments.output.exists():
        parser.error("The output path must be new")
    wheel = arguments.wheel or os.environ.get("RAGGED_WHEEL")
    wheel = str(Path(wheel).resolve(strict=True)) if wheel else None
    result = run(arguments.dtype, arguments.directory, wheel, arguments.reject_resume_rank1)
    result["source_labels"] = {"revision": arguments.source_revision, "tree": arguments.source_tree,
                                "meaning": "caller-supplied labels; consumer and helper hashes are independently recorded"}
    _write_receipt(arguments.output, result)
    print(json.dumps({"method": "completed-window CPU DDP continuation", "status": result["status"],
                      "dtype": result["dtype"], "checkpoint_sha256": result["checkpoint_sha256"],
                      "all_owned_workers_reaped": all(value["lifecycle"]["all_owned_workers_reaped"]
                                                       for value in result["cohorts"].values()),
                      "output": str(arguments.output.resolve())}, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
