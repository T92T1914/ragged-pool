"""Check CUDA training-state association, admission and continuation."""
import copy
from hashlib import sha256
import importlib.util
import io
from pathlib import Path

import pytest
import torch


def example():
    path = Path(__file__).resolve().parents[1] / "examples/resume_training.py"
    spec = importlib.util.spec_from_file_location("cuda_resume_test_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert type(actual) is torch.Tensor and actual.dtype == expected.dtype
        assert actual.shape == expected.shape and actual.device == expected.device
        assert torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                           expected.detach().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected)
        for name in expected:
            exact(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            exact(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def snapshot(value):
    return [(tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone(), tensor._version,
             tensor.untyped_storage()._cdata, tensor.storage_offset(), tensor.stride())
            for tensor in tensors(value)]


def unchanged(value, before):
    after = list(tensors(value))
    assert len(after) == len(before)
    for tensor, (data, version, storage, offset, stride) in zip(after, before):
        assert torch.equal(tensor.detach().contiguous().reshape(-1).view(torch.uint8), data)
        assert (tensor._version, tensor.untyped_storage()._cdata, tensor.storage_offset(), tensor.stride()) == (
            version, storage, offset, stride)


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128" and torch.version.cuda == "12.8"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)


@pytest.fixture(scope="module")
def prepared():
    module = example()
    initial, batches, tolerance = module.fixture("float64")
    model = copy.deepcopy(initial)
    opt = module.optimizer(model)
    metadata = module.metadata("float64", initial, batches,
                               source_revision="0" * 40, source_tree="1" * 40)
    prefix = module.run_steps(model, opt, batches, stop=2)
    checkpoint = module.make_checkpoint(model, opt, metadata)
    return module, initial, batches, tolerance, model, opt, metadata, checkpoint, prefix


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_checkpoint_is_cpu_snapshot_and_native_suffix_is_exact(dtype):
    module = example()
    initial, batches, _ = module.fixture(dtype)
    source = {"initial": initial.state_dict(), "batches": [
        {name: batch[name] for name in ("raw", "offsets", "targets")} for batch in batches]}
    source_before = snapshot(source)
    full, split = copy.deepcopy(initial), copy.deepcopy(initial)
    full_opt, split_opt = module.optimizer(full), module.optimizer(split)
    metadata = module.metadata(dtype, initial, batches, source_revision="0" * 40, source_tree="1" * 40)
    full_records = module.run_steps(full, full_opt, batches)
    prefix = module.run_steps(split, split_opt, batches, stop=2)
    checkpoint = module.make_checkpoint(split, split_opt, metadata)
    assert len(list(tensors(checkpoint))) == 10
    assert all(value.device.type == "cpu" and not value.requires_grad for value in tensors(checkpoint))
    assert len({value.untyped_storage()._cdata for value in tensors(checkpoint)}) == 10
    checkpoint_before = snapshot(checkpoint)
    target = copy.deepcopy(initial)
    target_opt = module.optimizer(target)
    parameter_ids = tuple(id(value) for value in target.parameters())
    module.restore(checkpoint, target, target_opt, metadata)
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    assert tuple(id(value) for value in target_opt.param_groups[0]["params"]) == parameter_ids
    exact(target.state_dict(), split.state_dict())
    exact(target_opt.state_dict(), split_opt.state_dict())
    restored = [*target.parameters(), *[state["momentum_buffer"] for state in target_opt.state.values()]]
    expected_dtype = next(target.parameters()).dtype
    assert all(value.device == torch.device("cuda:0") and value.dtype == expected_dtype
               for value in restored)
    assert len({value.untyped_storage()._cdata for value in restored}) == 10
    assert not ({value.untyped_storage()._cdata for value in restored}
                & {value.untyped_storage()._cdata for value in tensors(checkpoint)})
    suffix = module.run_steps(target, target_opt, batches, start=2)
    exact(prefix + suffix, full_records)
    exact(target.state_dict(), full.state_dict())
    exact(target_opt.state_dict(), full_opt.state_dict())
    unchanged(checkpoint, checkpoint_before)
    unchanged(source, source_before)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_momentum_recurrence_is_checked_without_reading_optimizer_result(dtype):
    module = example()
    initial, batches, _ = module.fixture(dtype)
    actual, manual = copy.deepcopy(initial), copy.deepcopy(initial)
    actual_opt = module.optimizer(actual)
    records = module.run_steps(actual, actual_opt, batches, stop=2)
    momentum = {}
    for batch in batches[:2]:
        raw = batch["raw"].detach().clone().requires_grad_()
        predictions, _ = manual(raw, batch["offsets"])
        loss = (predictions - batch["targets"]).square().mean()
        named = dict(manual.named_parameters())
        gradients = torch.autograd.grad(loss, tuple(named.values()))
        with torch.no_grad():
            for (name, parameter), gradient in zip(named.items(), gradients):
                momentum[name] = gradient.detach().clone() if name not in momentum else momentum[name].mul(0.8).add(gradient)
                parameter.add_(momentum[name], alpha=-0.03)
    exact(actual.state_dict(), manual.state_dict())
    for name, parameter in actual.named_parameters():
        exact(actual_opt.state[parameter]["momentum_buffer"], momentum[name])
    assert len(records) == 2


def test_weights_only_reset_matches_next_prediction_but_changes_next_update(prepared):
    module, initial, batches, _, _, _, metadata, checkpoint, _ = prepared
    resumed, reset = copy.deepcopy(initial), copy.deepcopy(initial)
    resumed_opt, reset_opt = module.optimizer(resumed), module.optimizer(reset)
    module.restore(checkpoint, resumed, resumed_opt, metadata)
    reset.load_state_dict(copy.deepcopy(checkpoint["model"]), strict=True)
    assert not reset_opt.state
    with torch.no_grad():
        for left, right in zip(resumed(batches[2]["raw"], batches[2]["offsets"]),
                               reset(batches[2]["raw"], batches[2]["offsets"])):
            exact(left, right)
    module.run_steps(resumed, resumed_opt, batches, start=2, stop=3)
    module.run_steps(reset, reset_opt, batches, start=2, stop=3)
    assert any(not torch.equal(left, right) for left, right in zip(resumed.parameters(), reset.parameters()))


@pytest.mark.parametrize("fault", [
    "cursor_bool", "cursor_repeat", "cursor_skip", "metadata_extra", "metadata_missing",
    "model_missing", "model_extra", "model_order", "model_shape", "model_dtype", "model_device",
    "model_nonfinite", "model_grad", "model_view", "model_stride", "model_alias",
    "optimizer_missing", "optimizer_extra", "missing_buffer", "extra_buffer", "momentum_shape",
    "momentum_dtype", "momentum_device", "momentum_nan", "momentum_view", "momentum_grad", "momentum_alias",
    "id_bool", "id_reordered", "id_duplicate", "id_missing", "state_key_bool", "extra_group",
    "option_momentum", "option_lr", "option_extra"])
def test_bad_checkpoint_stops_before_either_target_application_and_recovers(prepared, fault, monkeypatch):
    module, initial, _, _, _, _, metadata, checkpoint, _ = prepared
    bad = copy.deepcopy(checkpoint)
    names = list(bad["model"])
    name = names[0]
    group = bad["optimizer"]["param_groups"][0]
    buffer = bad["optimizer"]["state"][0]["momentum_buffer"]
    if fault.startswith("cursor_"):
        bad["metadata"]["next_update"] = {"cursor_bool": True, "cursor_repeat": 1, "cursor_skip": 3}[fault]
    elif fault == "metadata_extra": bad["metadata"]["unknown"] = 0
    elif fault == "metadata_missing": bad["metadata"].pop(next(iter(bad["metadata"])))
    elif fault == "model_missing": bad["model"].pop(name)
    elif fault == "model_extra": bad["model"]["unknown"] = torch.zeros(1)
    elif fault == "model_order": bad["model"] = dict(reversed(list(bad["model"].items())))
    elif fault == "model_shape": bad["model"][name] = bad["model"][name].reshape(-1)
    elif fault == "model_dtype": bad["model"][name] = bad["model"][name].float()
    elif fault == "model_device": bad["model"][name] = bad["model"][name].to("cuda:0")
    elif fault == "model_nonfinite": bad["model"][name].view(-1)[0] = float("inf")
    elif fault == "model_grad": bad["model"][name].requires_grad_()
    elif fault == "model_view":
        value = bad["model"][name]
        bad["model"][name] = torch.zeros(value.numel() + 1, dtype=value.dtype)[1:].reshape(value.shape)
    elif fault == "model_stride": bad["model"][name] = bad["model"][name].t().contiguous().t()
    elif fault == "model_alias": bad["model"]["head.weight"] = bad["model"]["scorer.weight"]
    elif fault == "optimizer_missing": bad["optimizer"].pop("state")
    elif fault == "optimizer_extra": bad["optimizer"]["unknown"] = 0
    elif fault == "missing_buffer": bad["optimizer"]["state"][0].pop("momentum_buffer")
    elif fault == "extra_buffer": bad["optimizer"]["state"][0]["unknown"] = torch.zeros(1)
    elif fault == "momentum_shape": bad["optimizer"]["state"][0]["momentum_buffer"] = buffer.reshape(-1)
    elif fault == "momentum_dtype": bad["optimizer"]["state"][0]["momentum_buffer"] = buffer.float()
    elif fault == "momentum_device": bad["optimizer"]["state"][0]["momentum_buffer"] = buffer.to("cuda:0")
    elif fault == "momentum_nan": buffer.view(-1)[0] = float("nan")
    elif fault == "momentum_view":
        bad["optimizer"]["state"][0]["momentum_buffer"] = torch.zeros(buffer.numel() + 1, dtype=buffer.dtype)[:buffer.numel()].reshape(buffer.shape)
    elif fault == "momentum_grad": buffer.requires_grad_()
    elif fault == "momentum_alias": bad["optimizer"]["state"][3]["momentum_buffer"] = bad["optimizer"]["state"][2]["momentum_buffer"]
    elif fault == "id_bool": group["params"][0] = False
    elif fault == "id_reordered": group["params"][2:4] = group["params"][2:4][::-1]
    elif fault == "id_duplicate": group["params"][3] = group["params"][2]
    elif fault == "id_missing": group["params"].pop()
    elif fault == "state_key_bool": bad["optimizer"]["state"][False] = bad["optimizer"]["state"].pop(0)
    elif fault == "extra_group": bad["optimizer"]["param_groups"].append(copy.deepcopy(group))
    elif fault == "option_momentum": group["momentum"] = 0.0
    elif fault == "option_lr": group["lr"] = 0.1
    elif fault == "option_extra": group["unknown"] = 0
    target = copy.deepcopy(initial)
    opt = module.optimizer(target)
    before = snapshot(target.state_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid checkpoint reached a target load_state_dict")

    with monkeypatch.context() as patch:
        patch.setattr(target, "load_state_dict", forbidden)
        patch.setattr(opt, "load_state_dict", forbidden)
        with pytest.raises(ValueError):
            module.restore(bad, target, opt, metadata)
    unchanged(target.state_dict(), before)
    assert not opt.state
    module.restore(checkpoint, target, opt, metadata)
    assert len(opt.state) == 5


def test_same_shape_live_optimizer_reordering_is_rejected_before_load(prepared, monkeypatch):
    module, initial, _, _, _, _, metadata, checkpoint, _ = prepared
    target = copy.deepcopy(initial)
    opt = module.optimizer(target)
    assert tuple(target.scorer.weight.shape) == tuple(target.head.weight.shape) == (1, 4)
    opt.param_groups[0]["params"][2:4] = opt.param_groups[0]["params"][2:4][::-1]
    before = snapshot(target.state_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid live association reached target application")

    monkeypatch.setattr(target, "load_state_dict", forbidden)
    monkeypatch.setattr(opt, "load_state_dict", forbidden)
    with pytest.raises(ValueError):
        module.restore(checkpoint, target, opt, metadata)
    unchanged(target.state_dict(), before)
    assert not opt.state


def test_checkpoint_requires_a_completed_update_without_live_gradients(prepared):
    module, _, _, _, trained, opt, metadata, _, _ = prepared
    parameter = next(trained.parameters())
    parameter.grad = torch.ones_like(parameter)
    try:
        with pytest.raises(ValueError):
            module.make_checkpoint(trained, opt, metadata)
    finally:
        opt.zero_grad(set_to_none=True)


def checkpoint_file(tmp_path, checkpoint):
    path = tmp_path / "checkpoint.pt"
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    data = buffer.getvalue()
    path.write_bytes(data)
    return path, data, sha256(data).hexdigest()


@pytest.mark.parametrize("fault", ["digest", "oversize", "empty", "directory"])
def test_file_admission_precedes_restricted_deserialization(prepared, tmp_path, monkeypatch, fault):
    module, initial, _, _, _, _, metadata, checkpoint, _ = prepared
    path, _, digest = checkpoint_file(tmp_path, checkpoint)
    if fault == "digest": path.write_bytes(b"changed")
    elif fault == "oversize": path.write_bytes(b"x" * (module.MAX_CHECKPOINT_BYTES + 1))
    elif fault == "empty": path.write_bytes(b"")
    else: path = tmp_path
    target = copy.deepcopy(initial)
    opt = module.optimizer(target)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected bytes reached torch.load")

    monkeypatch.setattr(module.torch, "load", forbidden)
    with pytest.raises((ValueError, OSError)):
        module.load_checkpoint(path, digest, target, opt, metadata)


def test_exact_verified_cpu_buffer_is_loaded_once(prepared, tmp_path, monkeypatch):
    module, initial, _, _, _, _, metadata, checkpoint, _ = prepared
    path, data, digest = checkpoint_file(tmp_path, checkpoint)
    original, calls = torch.load, []

    def observe(buffer, *args, **kwargs):
        assert isinstance(buffer, io.BytesIO) and buffer.getvalue() == data
        assert kwargs.get("weights_only") is True and kwargs.get("map_location") == "cpu"
        calls.append(buffer.getvalue())
        path.write_bytes(b"substitution after verification")
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", observe)
    target = copy.deepcopy(initial)
    loaded = module.load_checkpoint(path, digest, target, module.optimizer(target), metadata)
    assert calls == [data]
    exact(loaded, checkpoint)
