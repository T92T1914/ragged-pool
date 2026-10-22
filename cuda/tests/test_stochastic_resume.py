"""Admit one fixed CUDA stochastic checkpoint without changing live state."""
import copy
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "resume_stochastic.py"


def example():
    spec = importlib.util.spec_from_file_location("cuda_stochastic_resume_tests", SOURCE)
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
    return [(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().clone(),
             tensor._version, tensor.untyped_storage()._cdata,
             tensor.storage_offset(), tuple(tensor.stride())) for tensor in tensors(value)]


def unchanged(value, before):
    after = list(tensors(value))
    assert len(after) == len(before)
    for tensor, (data, version, storage, offset, stride) in zip(after, before):
        actual = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        assert torch.equal(actual, data)
        assert (tensor._version, tensor.untyped_storage()._cdata, tensor.storage_offset(),
                tuple(tensor.stride())) == (version, storage, offset, stride)


def default_rng():
    return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128" and torch.version.cuda == "12.8"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    previous = default_rng()
    yield
    torch.cuda.set_rng_state(previous["cuda"], 0)
    torch.set_rng_state(previous["cpu"])


@pytest.fixture(autouse=True)
def preserve_default_random_streams():
    previous = default_rng()
    yield
    torch.cuda.set_rng_state(previous["cuda"], 0)
    torch.set_rng_state(previous["cpu"])


def prepare(dtype_name="float64"):
    module = example()
    initial, batches, tolerance = module.fixture(dtype_name)
    metadata = module.metadata(dtype_name, initial, batches,
                               source_revision="0" * 40, source_tree="1" * 40)
    model, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(model)
    torch.cuda.manual_seed(module.DROPOUT_SEED)
    module.run_steps(model, opt, batches, order, stop=2)
    checkpoint = module.make_checkpoint(model, opt, order, metadata)
    return module, initial, batches, tolerance, metadata, checkpoint


@pytest.fixture(scope="module")
def prepared():
    return prepare()


def live_target(module, initial, metadata, checkpoint):
    target, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(target)
    module.restore(checkpoint, target, opt, order, metadata)
    # Rejection must preserve populated momentum, existing gradients and a
    # sampler position that differs from the saved checkpoint.
    with torch.no_grad():
        next(target.parameters()).add_(0.25)
    for parameter in target.parameters():
        parameter.grad = torch.full_like(parameter, 0.5)
        opt.state[parameter]["momentum_buffer"].add_(0.125)
    order.next_batch()
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(7333).get_state())
    torch.cuda.manual_seed(7333)
    return target, opt, order


def assert_rejected_without_application(module, checkpoint, target, opt, order, metadata):
    parameters = tuple(target.parameters())
    bindings = (id(target), id(opt), tuple(id(value) for value in parameters),
                tuple(id(value) for value in opt.param_groups[0]["params"]))
    order_binding = (id(order), id(order.generator), id(order.permutation))
    live = {"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
            "gradients": [value.grad for value in parameters]}
    before, order_before, rng_before = snapshot(live), copy.deepcopy(order.state_dict()), default_rng()
    training_before = target.training
    stream_before = torch.cuda.current_stream(0).cuda_stream
    with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
        with patch.object(opt, "load_state_dict", wraps=opt.load_state_dict) as opt_load:
            with patch.object(order, "apply_validated", wraps=order.apply_validated) as order_load:
                with patch.object(torch, "set_rng_state", wraps=torch.set_rng_state) as cpu_load:
                    with patch.object(torch.cuda, "set_rng_state", wraps=torch.cuda.set_rng_state) as cuda_load:
                        with pytest.raises((ValueError, RuntimeError, TypeError)):
                            module.restore(checkpoint, target, opt, order, metadata)
                        assert (model_load.call_count, opt_load.call_count, order_load.call_count,
                                cpu_load.call_count, cuda_load.call_count) == (0, 0, 0, 0, 0)
    unchanged({"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
               "gradients": [value.grad for value in target.parameters()]}, before)
    exact(order.state_dict(), order_before)
    exact(default_rng(), rng_before)
    assert target.training is training_before
    assert (id(order), id(order.generator), id(order.permutation)) == order_binding
    assert torch.cuda.current_stream(0).cuda_stream == stream_before
    assert (id(target), id(opt), tuple(id(value) for value in target.parameters()),
            tuple(id(value) for value in opt.param_groups[0]["params"])) == bindings


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_exact_native_restore_preserves_owned_state_and_crosses_epoch(dtype_name):
    module, initial, batches, _, metadata, checkpoint = prepare(dtype_name)
    sources = {"initial": dict(initial.named_parameters()), "batches": [
        {name: batch[name] for name in ("raw", "offsets", "targets")} for batch in batches]}
    sources_before, checkpoint_before = snapshot(sources), snapshot(checkpoint)
    checkpoint_contents = copy.deepcopy(checkpoint)
    assert set(checkpoint) == {"metadata", "model", "optimizer", "cuda_rng", "order", "training"}
    saved_tensors = list(tensors(checkpoint))
    assert len(saved_tensors) == 12
    assert all(value.device.type == "cpu" and not value.requires_grad for value in saved_tensors)
    assert len({value.untyped_storage()._cdata for value in saved_tensors}) == 12
    assert checkpoint["cuda_rng"].dtype == torch.uint8 and checkpoint["cuda_rng"].shape == (16,)

    full = copy.deepcopy(initial)
    torch.cuda.manual_seed(module.DROPOUT_SEED)
    full_records = module.run_steps(full, module.optimizer(full), batches, module.BatchOrder())
    target, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(target)
    binding = (id(target), id(opt), tuple(id(value) for value in target.parameters()))
    torch.cuda.manual_seed(77)
    cpu_before = torch.get_rng_state().clone()
    module.restore(checkpoint, target, opt, order, metadata)
    exact(torch.get_rng_state(), cpu_before)
    exact(torch.cuda.get_rng_state(0), checkpoint["cuda_rng"])
    exact(order.state_dict(), checkpoint["order"])
    assert target.training is True
    assert tuple(id(value) for value in opt.param_groups[0]["params"]) == binding[2]
    restored = {"model": dict(target.named_parameters()), "optimizer": opt.state_dict()}
    assert all(value.device == torch.device("cuda:0") for value in tensors(restored))
    assert len({value.untyped_storage()._cdata for value in tensors(restored)}) == 10
    suffix = module.run_steps(target, opt, batches, order, start=2)
    exact(suffix, full_records[2:])
    exact(torch.get_rng_state(), cpu_before)
    exact(torch.cuda.get_rng_state(0), full_records[-1]["rng_after"])
    exact(order.state_dict(), full_records[-1]["order_after"])
    assert [row["batch_id"] for row in full_records[:5]] == checkpoint["order"]["permutation"]
    assert [row["order_after"]["epoch"] for row in suffix] == [0, 0, 0, 1, 1]
    assert [row["order_after"]["cursor"] for row in suffix] == [3, 4, 5, 1, 2]
    assert (id(target), id(opt), tuple(id(value) for value in target.parameters())) == binding
    assert not ({value.untyped_storage()._cdata for value in tensors(restored)}
                & {value.untyped_storage()._cdata for value in saved_tensors})
    exact(checkpoint, checkpoint_contents)
    unchanged(checkpoint, checkpoint_before)
    unchanged(sources, sources_before)


@pytest.mark.parametrize("fault", [
    "cuda_type", "cuda_dtype", "cuda_device", "cuda_layout", "cuda_rank", "cuda_length",
    "cuda_legacy_eight", "cuda_oversize", "cuda_offset", "cuda_stride", "cuda_storage",
    "cuda_invalid_offset", "order_rng_type", "order_rng_dtype", "order_rng_device",
    "order_rng_layout", "order_rng_length", "order_rng_offset", "order_rng_stride",
    "order_rng_storage", "order_rng_semantics", "order_rng_wrong_valid",
    "permutation_type", "permutation_duplicate", "permutation_bounds", "permutation_bool",
    "permutation_wrong_valid", "cursor_bool", "cursor_repeat", "cursor_skip", "epoch_bool",
    "epoch_wrong", "step_bool", "step_repeat", "step_skip", "training_false",
    "training_integer", "metadata_extra", "metadata_missing", "field_extra", "field_missing",
])
def test_new_checkpoint_fields_refuse_before_any_target_application(prepared, fault):
    module, initial, batches, _, metadata, checkpoint = prepared
    bad = copy.deepcopy(checkpoint)
    if fault.startswith("cuda_") or fault.startswith("order_rng_"):
        container = bad if fault.startswith("cuda_") else bad["order"]
        key = "cuda_rng" if fault.startswith("cuda_") else "generator"
        kind = fault.removeprefix("cuda_").removeprefix("order_rng_")
        value = container[key]
        if kind == "type": container[key] = value.tolist()
        elif kind == "dtype": container[key] = value.to(torch.int64)
        elif kind == "device": container[key] = value.to("cuda:0")
        elif kind == "layout": container[key] = value.to_sparse()
        elif kind == "rank": container[key] = value.reshape(2, 8)
        elif kind == "length": container[key] = value[:-1].clone()
        elif kind == "legacy_eight": container[key] = value[:8].clone()
        elif kind == "oversize": container[key] = torch.zeros(module.MAX_RNG_STATE_BYTES + 1, dtype=torch.uint8)
        elif kind == "offset":
            container[key] = torch.cat((torch.zeros(1, dtype=torch.uint8), value))[1:]
        elif kind == "stride":
            expanded = torch.zeros(value.numel() * 2, dtype=torch.uint8)
            expanded[::2] = value
            container[key] = expanded[::2]
        elif kind == "storage": container[key] = torch.cat((value, torch.zeros(1, dtype=torch.uint8)))[:-1]
        elif kind == "invalid_offset": container[key].view(torch.int64)[1] = 1
        elif kind == "semantics": container[key].zero_()
        else: container[key] = torch.Generator(device="cpu").manual_seed(999).get_state()
    elif fault == "permutation_type": bad["order"]["permutation"] = tuple(bad["order"]["permutation"])
    elif fault == "permutation_duplicate": bad["order"]["permutation"][0] = bad["order"]["permutation"][1]
    elif fault == "permutation_bounds": bad["order"]["permutation"][0] = 5
    elif fault == "permutation_bool": bad["order"]["permutation"][0] = True
    elif fault == "permutation_wrong_valid": bad["order"]["permutation"].reverse()
    elif fault.startswith("cursor_"):
        bad["order"]["cursor"] = {"cursor_bool": True, "cursor_repeat": 1, "cursor_skip": 3}[fault]
    elif fault == "epoch_bool": bad["order"]["epoch"] = False
    elif fault == "epoch_wrong": bad["order"]["epoch"] = 1
    elif fault.startswith("step_"):
        bad["metadata"]["next_update"] = {"step_bool": True, "step_repeat": 1, "step_skip": 3}[fault]
    elif fault == "training_false": bad["training"] = False
    elif fault == "training_integer": bad["training"] = 1
    elif fault == "metadata_extra": bad["metadata"]["unknown"] = 0
    elif fault == "metadata_missing": bad["metadata"].pop("dropout_seed")
    elif fault == "field_extra": bad["unknown"] = None
    else: bad.pop("cuda_rng")
    # This otherwise admissible weight change makes a late RNG failure useful:
    # the target must not be partially updated before the refusal.
    next(iter(bad["model"].values())).add_(0.125)
    source_before = snapshot({"initial": dict(initial.named_parameters()), "batches": batches})
    target, opt, order = live_target(module, initial, metadata, checkpoint)
    assert_rejected_without_application(module, bad, target, opt, order, metadata)
    module.restore(checkpoint, target, opt, order, metadata)
    assert target.training and all(value.grad is None for value in target.parameters())
    exact(order.state_dict(), checkpoint["order"])
    exact(torch.cuda.get_rng_state(0), checkpoint["cuda_rng"])
    unchanged({"initial": dict(initial.named_parameters()), "batches": batches}, source_before)


def test_invalid_cuda_offset_mutates_only_a_scratch_generator(prepared):
    module, _, _, _, _, checkpoint = prepared
    malformed = checkpoint["cuda_rng"].clone()
    malformed.view(torch.int64)[1] = 1
    torch.cuda.manual_seed(7333)
    before = default_rng()
    scratch = torch.Generator(device="cuda:0").manual_seed(module.DROPOUT_SEED + 177)
    seed_before = scratch.initial_seed()
    with pytest.raises((ValueError, RuntimeError)):
        scratch.set_state(malformed)
    # PyTorch 2.9.1 sets a CUDA generator's seed before rejecting its unaligned
    # offset. This is why probing the live default generator is insufficient.
    assert scratch.initial_seed() != seed_before
    assert torch.equal(scratch.get_state()[:8], malformed[:8])
    exact(default_rng(), before)
    with pytest.raises(ValueError):
        module._cuda_rng_state(malformed)
    exact(default_rng(), before)


def test_cross_field_rng_model_storage_alias_is_rejected_before_load():
    module, initial, _, _, metadata, checkpoint = prepare("float32")
    bad = copy.deepcopy(checkpoint)
    bad["model"]["head.weight"] = bad["cuda_rng"].view(torch.float32).reshape(1, 4)
    assert bad["model"]["head.weight"].untyped_storage()._cdata == bad["cuda_rng"].untyped_storage()._cdata
    assert bool(torch.isfinite(bad["model"]["head.weight"]).all())
    target, opt, order = live_target(module, initial, metadata, checkpoint)
    assert_rejected_without_application(module, bad, target, opt, order, metadata)
    module.restore(checkpoint, target, opt, order, metadata)


@pytest.mark.parametrize("fault", ["eval", "optimizer_order", "optimizer_options", "order_device"])
def test_live_target_binding_refusal_preserves_state_then_valid_restore(prepared, fault):
    module, initial, _, _, metadata, checkpoint = prepared
    target, opt, order = live_target(module, initial, metadata, checkpoint)
    if fault == "eval": target.eval()
    elif fault == "optimizer_order":
        opt.param_groups[0]["params"][2:4] = opt.param_groups[0]["params"][2:4][::-1]
    elif fault == "optimizer_options": opt.param_groups[0]["momentum"] = 0.0
    else: order.generator = torch.Generator(device="cuda:0").manual_seed(91)
    assert_rejected_without_application(module, checkpoint, target, opt, order, metadata)
    if fault == "eval": target.train(True)
    elif fault == "optimizer_order":
        opt.param_groups[0]["params"][2:4] = opt.param_groups[0]["params"][2:4][::-1]
    elif fault == "optimizer_options": opt.param_groups[0]["momentum"] = module.OPTIONS["momentum"]
    else: order.generator = torch.Generator(device="cpu").manual_seed(module.ORDER_SEED)
    module.restore(checkpoint, target, opt, order, metadata)
    assert target.training and all(value.grad is None for value in target.parameters())
    exact(order.state_dict(), checkpoint["order"])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_stochastic_updates_match_independent_momentum_recurrence(dtype_name):
    module = example()
    model, batches, tolerance = module.fixture(dtype_name)
    opt, order, previous = module.optimizer(model), module.BatchOrder(), {}
    torch.cuda.manual_seed(module.DROPOUT_SEED)
    for index in range(7):
        before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        record = module.step(model, opt, batches, order, index)
        for name, parameter in model.named_parameters():
            gradient = record["parameter_gradients"][name]
            momentum = gradient.clone() if index == 0 else 0.8 * previous[name] + gradient
            torch.testing.assert_close(opt.state[parameter]["momentum_buffer"].cpu(), momentum, **tolerance)
            torch.testing.assert_close(parameter.detach().cpu(), before[name] - 0.03 * momentum, **tolerance)
            previous[name] = momentum.clone()


@pytest.mark.parametrize("fault", ["digest", "corrupt", "oversize", "empty", "directory"])
def test_file_identity_is_checked_before_deserialization(prepared, tmp_path, fault):
    module, initial, _, _, metadata, checkpoint = prepared
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    payload = stream.getvalue()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    if fault == "digest": digest = "0" * 64
    elif fault == "corrupt": path.write_bytes(payload[:-9])
    elif fault == "oversize": path.write_bytes(b"x" * (module.MAX_CHECKPOINT_BYTES + 1))
    elif fault == "empty": path.write_bytes(b"")
    else: path = tmp_path
    target, opt, order = live_target(module, initial, metadata, checkpoint)
    live = {"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
            "gradients": [value.grad for value in target.parameters()]}
    before, rng_before, order_before = snapshot(live), default_rng(), order.state_dict()
    with patch.object(torch, "load", wraps=torch.load) as load:
        with pytest.raises((ValueError, RuntimeError, TypeError, OSError)):
            module.load_checkpoint(path, digest, target, opt, order, metadata)
        assert load.call_count == 0
    unchanged({"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
               "gradients": [value.grad for value in target.parameters()]}, before)
    exact(default_rng(), rng_before)
    exact(order.state_dict(), order_before)


def test_verified_buffer_load_is_restricted_and_does_not_restore_live_state(prepared, tmp_path, monkeypatch):
    module, initial, _, _, metadata, checkpoint = prepared
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    payload = stream.getvalue()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)
    target, opt, order = live_target(module, initial, metadata, checkpoint)
    live = {"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
            "gradients": [value.grad for value in target.parameters()]}
    before, rng_before, order_before = snapshot(live), default_rng(), order.state_dict()
    original, calls = torch.load, []

    def observe(buffer, *args, **kwargs):
        assert isinstance(buffer, io.BytesIO) and buffer.getvalue() == payload
        assert kwargs.get("weights_only") is True and kwargs.get("map_location") == "cpu"
        calls.append(buffer.getvalue())
        path.write_bytes(b"substitution after verification")
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", observe)
    loaded = module.load_checkpoint(path, sha256(payload).hexdigest(), target, opt, order, metadata)
    assert calls == [payload]
    exact(loaded, checkpoint)
    unchanged({"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
               "gradients": [value.grad for value in target.parameters()]}, before)
    exact(default_rng(), rng_before)
    exact(order.state_dict(), order_before)
    module.restore(loaded, target, opt, order, metadata)
    assert all(value.grad is None for value in target.parameters())
