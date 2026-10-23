"""CPU admission, ownership and mathematics for the fixed DDP checkpoint.

These fixtures do not create process groups or workers. They distinguish a
validated completed-window state transfer from Linux distributed acceptance,
which belongs to the separately executed installed consumer.
"""
import builtins
import copy
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "resume_distributed.py"
NAMES = ("encoder.weight", "encoder.bias", "scorer.weight", "head.weight", "head.bias")
SHAPES = ((4, 3), (4,), (1, 4), (1, 4), (1,))
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0, "weight_decay": 0,
           "nesterov": False, "maximize": False, "foreach": False,
           "differentiable": False, "fused": False}


def example():
    specification = importlib.util.spec_from_file_location("distributed_resume_admission_tests", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)


def make_model(dtype_name="float64", offset=0.0):
    with torch.random.fork_rng(devices=[]):
        model = FixtureModel().to(dtype=getattr(torch, dtype_name))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            sequence = torch.linspace(-0.25, 0.35, parameter.numel(), dtype=parameter.dtype)
            parameter.copy_(sequence.reshape(parameter.shape) + 0.04 * index + offset)
    return model


def synthetic_identity():
    """Declared test metadata, not a claimed installed-package observation."""
    return {"python": "3.11.8", "torch": "2.9.1+cpu", "package_version": "0.1.0",
            "native_library_sha256": "1" * 64, "wheel_sha256": "2" * 64,
            "build_config": {"schema": 1, "variant": "cpu", "torch_version": "2.9.1+cpu",
                             "cuda_version": None, "wheel_build_tag": ""}}


def populate_momentum(model, optimizer, scale=0.025):
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, scale * (index + 1))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def prepared(dtype_name="float64"):
    module = example()
    model = make_model(dtype_name)
    optimizer = torch.optim.SGD(model.parameters(), **OPTIONS)
    populate_momentum(model, optimizer)
    metadata = module.metadata(dtype_name, sha256(b"CPU state-admission fixture").hexdigest(),
                               synthetic_identity())
    checkpoint = module.make_checkpoint(model, optimizer, metadata)
    return module, model, optimizer, metadata, checkpoint


def tensor_identity(value):
    return (id(value), value._version, value.data_ptr(), value.untyped_storage().data_ptr(),
            value.untyped_storage().nbytes(), tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.dtype, value.device, value.requires_grad)


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def exact(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.device == second.device and first.shape == second.shape
        assert torch.equal(first.detach().contiguous().view(torch.uint8),
                           second.detach().contiguous().view(torch.uint8))
    elif isinstance(first, dict):
        # JSON object key order is incidental. Tensor-record list order and
        # optimizer parameter bindings have their own explicit checks below.
        assert first.keys() == second.keys()
        for name in first:
            exact(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            exact(a, b)
    else:
        assert first == second


def live_snapshot(model, optimizer):
    state = optimizer.state_dict()
    return {"model": clone_tree(dict(model.state_dict())), "optimizer": clone_tree(state),
            "model_id": id(model), "optimizer_id": id(optimizer), "training": model.training,
            "parameters": tuple(tensor_identity(value) for value in model.parameters()),
            "optimizer_tensors": tuple(tensor_identity(value) for value in tensors(state)),
            "bindings": tuple(tuple(id(value) for value in group["params"])
                              for group in optimizer.param_groups),
            "gradients": tuple(None if value.grad is None else
                               (tensor_identity(value.grad), value.grad.detach().clone())
                               for value in model.parameters())}


def test_import_does_not_load_training_helper_native_packages_or_distributed_runtime(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    original_spec = importlib.util.spec_from_file_location

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"} or name.startswith("torch.distributed"):
            raise AssertionError("Checkpoint helper import must remain CPU-local and lazy")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"} or name.startswith("torch.distributed"):
            raise AssertionError("Checkpoint helper import must remain CPU-local and lazy")
        return original_import_module(name, *args, **kwargs)

    def guarded_spec(name, location, *args, **kwargs):
        if Path(location).name == "train_distributed.py":
            raise AssertionError("Import must not load the adjacent training consumer")
        return original_spec(name, location, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not query devices or launch distributed work")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", guarded_spec)
    with (patch.object(torch.cuda, "is_available", side_effect=forbidden),
          patch.object(torch.cuda, "device_count", side_effect=forbidden),
          patch.object(torch.cuda, "_lazy_init", side_effect=forbidden),
          patch.object(torch.distributed, "init_process_group", side_effect=forbidden),
          patch.object(torch.distributed, "new_group", side_effect=forbidden),
          patch.object(torch.distributed, "all_reduce", side_effect=forbidden),
          patch.object(torch.multiprocessing, "spawn", side_effect=forbidden),
          patch.object(subprocess, "Popen", side_effect=forbidden)):
        module = example()
        assert module._TRAINING is None


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_checkpoint_roundtrip_preserves_all_weights_momentum_and_live_parameter_bindings(dtype_name, tmp_path):
    module, trained, trained_optimizer, metadata, checkpoint = prepared(dtype_name)
    assert tuple(module.STATE_SHAPES) == NAMES and tuple(module.STATE_SHAPES.values()) == SHAPES
    assert metadata["parameter_names"] == list(NAMES)
    assert metadata["parameter_shapes"] == [list(shape) for shape in SHAPES]
    assert metadata["next_window"] == 1 and metadata["total_windows"] == 2
    assert metadata["optimizer_options"] == OPTIONS
    assert [record["name"] for record in checkpoint["model"]] == list(NAMES)
    assert [record["name"] for record in checkpoint["momentum"]] == list(NAMES)
    original = copy.deepcopy(checkpoint)
    payload = module.encode_checkpoint(checkpoint)
    assert type(payload) is bytes and len(payload) < module.MAX_CHECKPOINT_BYTES == 64 * 1024
    path = tmp_path / "completed-window.json"
    path.write_bytes(payload)
    with patch.object(torch, "load", side_effect=AssertionError("This checkpoint is plain JSON")):
        decoded = module.load_checkpoint(path, sha256(payload).hexdigest(), metadata)
        with pytest.raises((TypeError, ValueError)):
            module.load_checkpoint(path, "0" * 64, metadata)
        # Whitespace leaves JSON meaning intact but changes the trusted bytes.
        path.write_bytes(payload + b" ")
        with pytest.raises((TypeError, ValueError)):
            module.load_checkpoint(path, sha256(payload).hexdigest(), metadata)
        path.write_bytes(payload)
    exact(decoded, original)
    target = make_model(dtype_name, offset=0.7)
    optimizer = torch.optim.SGD(target.parameters(), **OPTIONS)
    parameter_bindings = tuple((id(value), value.data_ptr(), value.untyped_storage().data_ptr())
                               for value in target.parameters())
    group_bindings = tuple(id(value) for value in optimizer.param_groups[0]["params"])
    module.restore(decoded, target, optimizer, metadata)
    exact(dict(target.state_dict()), dict(trained.state_dict()))
    exact(optimizer.state_dict(), trained_optimizer.state_dict())
    assert tuple((id(value), value.data_ptr(), value.untyped_storage().data_ptr())
                 for value in target.parameters()) == parameter_bindings
    assert tuple(id(value) for value in optimizer.param_groups[0]["params"]) == group_bindings
    assert all(value.grad is None for value in target.parameters())
    live = [*target.parameters(), *(optimizer.state[value]["momentum_buffer"] for value in target.parameters())]
    source = [*trained.parameters(), *(trained_optimizer.state[value]["momentum_buffer"]
                                      for value in trained.parameters())]
    assert len({value.untyped_storage().data_ptr() for value in live}) == 10
    assert not ({value.untyped_storage().data_ptr() for value in live}
                & {value.untyped_storage().data_ptr() for value in source})
    before = live_snapshot(target, optimizer)
    decoded["model"][0]["values"][0] += 0.5
    decoded["momentum"][0]["values"][0] -= 0.5
    exact(live_snapshot(target, optimizer), before)
    exact(checkpoint, original)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("gradient", [0.0, 0.25])
def test_checkpoint_boundary_rejects_even_zero_pending_gradients_without_mutation(gradient):
    module, model, optimizer, metadata, _ = prepared()
    next(model.parameters()).grad = torch.full_like(next(model.parameters()), gradient)
    before = live_snapshot(model, optimizer)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        module.make_checkpoint(model, optimizer, metadata)
    exact(live_snapshot(model, optimizer), before)


@pytest.mark.parametrize("gradient", [0.0, 0.125])
def test_valid_saved_state_refuses_a_pending_target_gradient_before_loading(gradient):
    module, _, _, metadata, checkpoint = prepared()
    target = make_model(offset=0.7)
    optimizer = torch.optim.SGD(target.parameters(), **OPTIONS)
    populate_momentum(target, optimizer, scale=0.013)
    parameter = next(target.parameters())
    parameter.grad = torch.full_like(parameter, gradient)
    before = live_snapshot(target, optimizer)
    with (patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load,
          patch.object(optimizer, "load_state_dict", wraps=optimizer.load_state_dict) as optimizer_load):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            module.restore(checkpoint, target, optimizer, metadata)
        assert model_load.call_count == optimizer_load.call_count == 0
    exact(live_snapshot(target, optimizer), before)


BAD_CHECKPOINTS = (
    "unknown_field", "metadata_missing", "cursor_boolean", "cursor_repeat", "cursor_skip",
    "world_size_boolean", "optimizer_lr_boolean", "optimizer_momentum_changed", "optimizer_unknown",
    "fixture_mismatch", "metadata_names_reordered", "parameter_order", "parameter_duplicate",
    "momentum_order", "momentum_missing", "parameter_shape", "momentum_shape", "parameter_dtype",
    "parameter_nonfinite", "momentum_nonfinite", "scalar_boolean", "scalar_string", "scalar_integer",
    "values_nested", "values_short", "record_unknown",
)


@pytest.mark.parametrize("case", BAD_CHECKPOINTS)
def test_bad_saved_state_refuses_before_either_target_loader_and_preserves_live_state(case):
    module, _, _, metadata, checkpoint = prepared()
    bad = copy.deepcopy(checkpoint)
    if case == "unknown_field":
        bad["unknown"] = 0
    elif case == "metadata_missing":
        bad.pop("metadata")
    elif case.startswith("cursor_"):
        bad["metadata"]["next_window"] = {"cursor_boolean": True, "cursor_repeat": 0, "cursor_skip": 2}[case]
    elif case == "world_size_boolean":
        bad["metadata"]["world_size"] = True
    elif case == "optimizer_lr_boolean":
        bad["metadata"]["optimizer_options"]["lr"] = True
    elif case == "optimizer_momentum_changed":
        bad["metadata"]["optimizer_options"]["momentum"] = 0.0
    elif case == "optimizer_unknown":
        bad["metadata"]["optimizer_options"]["unknown"] = 0
    elif case == "fixture_mismatch":
        bad["metadata"]["fixture_sha256"] = "0" * 64
    elif case == "metadata_names_reordered":
        bad["metadata"]["parameter_names"][2:4] = bad["metadata"]["parameter_names"][2:4][::-1]
    elif case == "parameter_order":
        bad["model"][2:4] = bad["model"][2:4][::-1]
    elif case == "parameter_duplicate":
        bad["model"][3]["name"] = bad["model"][2]["name"]
    elif case == "momentum_order":
        bad["momentum"][2:4] = bad["momentum"][2:4][::-1]
    elif case == "momentum_missing":
        bad["momentum"].pop()
    elif case == "parameter_shape":
        bad["model"][0]["shape"] = [12]
    elif case == "momentum_shape":
        bad["momentum"][0]["shape"] = [3, 4]
    elif case == "parameter_dtype":
        bad["model"][0]["dtype"] = "float32"
    elif case == "parameter_nonfinite":
        bad["model"][0]["values"][0] = float("inf")
    elif case == "momentum_nonfinite":
        bad["momentum"][0]["values"][0] = float("nan")
    elif case.startswith("scalar_"):
        bad["model"][0]["values"][0] = {
            "scalar_boolean": True, "scalar_string": "0.25", "scalar_integer": 1
        }[case]
    elif case == "values_nested":
        bad["model"][0]["values"] = [bad["model"][0]["values"]]
    elif case == "values_short":
        bad["model"][0]["values"].pop()
    elif case == "record_unknown":
        bad["momentum"][0]["unknown"] = 0
    else:
        raise AssertionError("Unknown test fixture")
    target = make_model(offset=0.7)
    optimizer = torch.optim.SGD(target.parameters(), **OPTIONS)
    populate_momentum(target, optimizer, scale=0.013)
    # A rejected late momentum/schema field must preserve already-owned target
    # buffers. Keep this target otherwise admissible so its own boundary checks
    # cannot hide a missing saved-state validation.
    assert all(value.grad is None for value in target.parameters())
    before = live_snapshot(target, optimizer)
    with (patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load,
          patch.object(optimizer, "load_state_dict", wraps=optimizer.load_state_dict) as optimizer_load):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            module.restore(bad, target, optimizer, metadata)
        assert model_load.call_count == optimizer_load.call_count == 0
    exact(live_snapshot(target, optimizer), before)


@pytest.mark.parametrize("case", ["foreign_optimizer", "parameter_order", "option_changed",
                                  "extra_group", "shared_parameter_storage", "model_not_training"])
def test_invalid_live_target_refuses_before_loading_and_leaves_its_state_unchanged(case):
    module, _, _, metadata, checkpoint = prepared()
    target = make_model(offset=0.7)
    optimizer_model = make_model(offset=0.9) if case == "foreign_optimizer" else target
    optimizer = torch.optim.SGD(optimizer_model.parameters(), **OPTIONS)
    populate_momentum(optimizer_model, optimizer)
    if case == "parameter_order":
        optimizer.param_groups[0]["params"][2:4] = optimizer.param_groups[0]["params"][2:4][::-1]
    elif case == "option_changed":
        optimizer.param_groups[0]["lr"] = 0.2
    elif case == "extra_group":
        optimizer.param_groups.append(dict(OPTIONS, params=[]))
    elif case == "shared_parameter_storage":
        target.head.weight.data = target.scorer.weight.data
    elif case == "model_not_training":
        target.eval()
    before = live_snapshot(target, optimizer)
    with (patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load,
          patch.object(optimizer, "load_state_dict", wraps=optimizer.load_state_dict) as optimizer_load):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            module.restore(checkpoint, target, optimizer, metadata)
        assert model_load.call_count == optimizer_load.call_count == 0
    exact(live_snapshot(target, optimizer), before)


@pytest.mark.parametrize("case", ["cursor_boolean", "cursor_repeat", "cursor_skip", "mid_window", "lr_boolean",
                                  "momentum_changed", "names_reordered", "world_size_boolean"])
def test_fixed_protocol_metadata_is_required_even_when_saved_and_expected_metadata_agree(case):
    module, _, _, metadata, checkpoint = prepared()
    expected = copy.deepcopy(metadata)
    if case.startswith("cursor_"):
        expected["next_window"] = {"cursor_boolean": True, "cursor_repeat": 0, "cursor_skip": 2}[case]
    elif case == "mid_window":
        expected["next_microbatch"] = 1
    elif case == "lr_boolean":
        expected["optimizer_options"]["lr"] = True
    elif case == "momentum_changed":
        expected["optimizer_options"]["momentum"] = 0.0
    elif case == "names_reordered":
        expected["parameter_names"][2:4] = expected["parameter_names"][2:4][::-1]
    else:
        expected["world_size"] = True
    checkpoint["metadata"] = copy.deepcopy(expected)
    target = make_model(offset=0.7)
    optimizer = torch.optim.SGD(target.parameters(), **OPTIONS)
    before = live_snapshot(target, optimizer)
    with (patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load,
          patch.object(optimizer, "load_state_dict", wraps=optimizer.load_state_dict) as optimizer_load):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            module.restore(checkpoint, target, optimizer, expected)
        assert model_load.call_count == optimizer_load.call_count == 0
    exact(live_snapshot(target, optimizer), before)


@pytest.mark.parametrize("field", ["model", "momentum"])
def test_float32_state_rejects_finite_json_scalars_that_would_round_during_restore(field):
    module, _, _, metadata, checkpoint = prepared("float32")
    # Python's 0.1 is finite but is not exactly representable in float32. A
    # loader must not apply it and discover a changed checkpoint only afterward.
    checkpoint[field][0]["values"][0] = 0.1
    target = make_model("float32", offset=0.7)
    optimizer = torch.optim.SGD(target.parameters(), **OPTIONS)
    before = live_snapshot(target, optimizer)
    with (patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load,
          patch.object(optimizer, "load_state_dict", wraps=optimizer.load_state_dict) as optimizer_load):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            module.restore(checkpoint, target, optimizer, metadata)
        assert model_load.call_count == optimizer_load.call_count == 0
    exact(live_snapshot(target, optimizer), before)


@pytest.mark.parametrize("case", ["duplicate_root", "duplicate_nested", "nan_constant", "inf_constant",
                                  "overflow_number", "invalid_utf8", "empty", "oversize"])
def test_json_admission_refuses_duplicate_nonfinite_invalid_or_oversize_bytes(case):
    module, _, _, metadata, checkpoint = prepared()
    payload = module.encode_checkpoint(checkpoint)
    if case == "duplicate_root":
        encoded_metadata = json.dumps(metadata, separators=(",", ":")).encode()
        payload = b'{"metadata":' + encoded_metadata + b"," + payload[1:]
    elif case == "duplicate_nested":
        changed = payload.replace(b'"lr":0.03', b'"lr":0.03,"lr":0.03', 1)
        assert changed != payload
        payload = changed
    elif case in ("nan_constant", "inf_constant", "overflow_number"):
        start = payload.index(b'"values":[') + len(b'"values":[')
        stop = payload.index(b",", start)
        replacement = {"nan_constant": b"NaN", "inf_constant": b"Infinity", "overflow_number": b"1e400"}[case]
        payload = payload[:start] + replacement + payload[stop:]
    elif case == "invalid_utf8":
        payload = b"\xff"
    elif case == "empty":
        payload = b""
    elif case == "oversize":
        payload = b" " * (module.MAX_CHECKPOINT_BYTES + 1)
    before = copy.deepcopy(metadata)
    with patch.object(torch, "load", side_effect=AssertionError("No pickle loader belongs in this JSON path")):
        with pytest.raises((TypeError, ValueError)):
            module.decode_checkpoint(payload, sha256(payload).hexdigest(), metadata)
    exact(metadata, before)


@pytest.mark.parametrize("case", ["digest_mismatch", "digest_boolean", "digest_incomplete", "nonbytes"])
def test_checkpoint_requires_literal_bytes_and_complete_trusted_digest(case):
    module, _, _, metadata, checkpoint = prepared()
    payload = module.encode_checkpoint(checkpoint)
    digest = sha256(payload).hexdigest()
    if case == "digest_mismatch":
        digest = "0" * 64
    elif case == "digest_boolean":
        digest = True
    elif case == "digest_incomplete":
        digest = digest[:-1]
    else:
        payload = bytearray(payload)
    with pytest.raises((TypeError, ValueError)):
        module.decode_checkpoint(payload, digest, metadata)


@pytest.mark.parametrize("case", ["directory", "empty", "oversize"])
def test_file_admission_checks_type_and_size_before_json_decode(case, tmp_path):
    module, _, _, metadata, checkpoint = prepared()
    path = tmp_path / "owned-state.json"
    if case == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"" if case == "empty" else b" " * (module.MAX_CHECKPOINT_BYTES + 1))
    with patch.object(module, "decode_checkpoint", side_effect=AssertionError("File admission must precede decode")):
        with pytest.raises((TypeError, ValueError)):
            module.load_checkpoint(path, "0" * 64, metadata)


def test_abort_publication_error_still_reaps_exactly_the_owned_fake_children(monkeypatch, tmp_path):
    module = example()

    class FakeOwnedProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake owned worker", timeout)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

    owned = [FakeOwnedProcess(10001), FakeOwnedProcess(10002)]
    launches, cleanup_calls = [], []

    def fake_launch(arguments, **options):
        assert options["shell"] is False
        process = owned[len(launches)]
        launches.append(process)
        return process

    def cleanup(processes):
        assert processes == owned
        cleanup_calls.append(tuple(processes))
        for process in processes:
            process.terminate()
            process.wait(timeout=0)
        return [{"pid": process.pid, "action": "terminate"} for process in processes]

    original_write = module._write_plain

    def fail_abort_only(path, value, maximum=module.MAX_RECEIPT_BYTES):
        if Path(path).name == "abort.json":
            raise OSError("Injected abort publication failure")
        return original_write(path, value, maximum)

    ticks = iter((0.0, float(module.PARENT_TIMEOUT_SECONDS + 1)))
    last_tick = float(module.PARENT_TIMEOUT_SECONDS + 1)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, last_tick))
    monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("The forced deadline must not sleep"))
    monkeypatch.setattr(module.subprocess, "Popen", fake_launch)
    monkeypatch.setattr(module, "_write_plain", fail_abort_only)
    monkeypatch.setattr(module, "_helper", lambda: SimpleNamespace(_cleanup_owned=cleanup))
    # No readiness receipt is produced. Placeholder state is never admitted and
    # no helper, actual process group, subprocess or numerical model is started.
    result = module._launch_cohort("float64", tmp_path / "owned-fake-cohort", "producer", {}, {}, [], None)
    assert launches == owned and cleanup_calls == [tuple(owned)]
    assert all(process.terminated and process.poll() == -15 for process in owned)
    assert result["lifecycle"]["all_owned_workers_reaped"] is True
    assert [item["pid"] for item in result["lifecycle"]["owned_processes"]] == [10001, 10002]
    assert "Owned cohort exceeded the finite parent deadline" in result["lifecycle"]["failure"]
    assert "abort publication failed: OSError: Injected abort publication failure" in result["lifecycle"]["failure"]
    assert result["admission"]["released"] is False
    assert result["process_group_initialization_attempted"] == [False, False]


def functional_prediction(model, raw, offsets):
    values = F.linear(raw, model.encoder.weight, model.encoder.bias).tanh()
    logits = F.linear(values, model.scorer.weight).squeeze(-1)
    rows = []
    for start, stop in zip(offsets, offsets[1:]):
        features, scores = values[start:stop].double(), logits[start:stop].double()
        row = (features.sum(0) * 0 + scores.sum() * 0 if start == stop else
               (torch.softmax(scores, 0).unsqueeze(1) * features).sum(0))
        rows.append(row.to(values.dtype))
    pooled = torch.stack(rows)
    return F.linear(pooled, model.head.weight, model.head.bias).squeeze(-1), pooled


def serial_batches(dtype_name):
    dtype = getattr(torch, dtype_name)
    boundaries = ((0, 2, 2, 3, 5, 6, 6, 7, 9, 12), (0, 2, 2, 3, 5, 7))
    batches = []
    for window, offsets in enumerate(boundaries):
        raw = torch.linspace(-0.4, 0.6, offsets[-1] * 3, dtype=dtype).reshape(-1, 3) + window * 0.03
        targets = torch.linspace(-0.2, 0.55, len(offsets) - 1, dtype=dtype) + window * 0.09
        batches.append({"raw": raw, "targets": targets, "offsets": offsets})
    return batches


def backward(model, batch):
    for parameter in model.parameters():
        parameter.grad = None
    predictions, pooled = functional_prediction(model, batch["raw"], batch["offsets"])
    loss = (predictions - batch["targets"]).square().mean()
    loss.backward()
    return {"predictions": predictions.detach().clone(), "pooled": pooled.detach().clone(),
            "gradients": {name: value.grad.detach().clone() for name, value in model.named_parameters()}}


def manual_update(model, momentum):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            buffer = 0.8 * momentum.get(name, torch.zeros_like(parameter)) + parameter.grad
            parameter.add_(buffer, alpha=-0.03)
            momentum[name] = buffer.detach().clone()


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_serial_resume_matches_uninterrupted_updates_and_reset_momentum_changes_only_the_update(dtype_name):
    module = example()
    initial = make_model(dtype_name)
    batches = serial_batches(dtype_name)
    assert [batch["targets"].numel() for batch in batches] == [9, 5]
    fixture_bytes = b"".join(batch[name].numpy().tobytes() for batch in batches for name in ("raw", "targets"))
    metadata = module.metadata(dtype_name, sha256(fixture_bytes).hexdigest(), synthetic_identity())
    prefix = copy.deepcopy(initial)
    prefix_optimizer = torch.optim.SGD(prefix.parameters(), **OPTIONS)
    backward(prefix, batches[0])
    prefix_optimizer.step()
    prefix_optimizer.zero_grad(set_to_none=True)
    checkpoint = module.make_checkpoint(prefix, prefix_optimizer, metadata)
    payload = module.encode_checkpoint(checkpoint)
    decoded = module.decode_checkpoint(payload, sha256(payload).hexdigest(), metadata)

    resumed = make_model(dtype_name, offset=0.7)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), **OPTIONS)
    module.restore(decoded, resumed, resumed_optimizer, metadata)
    reset = copy.deepcopy(prefix)
    reset_optimizer = torch.optim.SGD(reset.parameters(), **OPTIONS)
    expected = copy.deepcopy(initial)
    expected_momentum = {}
    backward(expected, batches[0])
    manual_update(expected, expected_momentum)
    torch.testing.assert_close(dict(prefix.state_dict()), dict(expected.state_dict()), rtol=0, atol=0)

    resumed_forward = backward(resumed, batches[1])
    reset_forward = backward(reset, batches[1])
    exact(resumed_forward, reset_forward)
    backward(expected, batches[1])
    manual_update(expected, expected_momentum)
    resumed_optimizer.step()
    reset_optimizer.step()
    torch.testing.assert_close(dict(resumed.state_dict()), dict(expected.state_dict()), rtol=0, atol=0)
    for name, parameter in resumed.named_parameters():
        torch.testing.assert_close(resumed_optimizer.state[parameter]["momentum_buffer"],
                                   expected_momentum[name], rtol=0, atol=0)
    drift = max(float((a - b).abs().max().detach()) for a, b in zip(resumed.parameters(), reset.parameters()))
    assert drift > (1e-7 if dtype_name == "float32" else 1e-10)
    exact(checkpoint, decoded)
