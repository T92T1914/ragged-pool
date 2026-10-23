"""CPU fixture checks for the cross-device example's state admission helpers.

These tests exercise checkpoint structure, ownership and refusal behavior. They
do not execute a CUDA operator or establish cross-device training acceptance.
"""
import builtins
import copy
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "resume_backends.py"
BOUNDARIES = ((0, 2, 2, 5, 7, 9), (0, 3, 5, 5, 8, 10, 12)) * 2


def example():
    specification = importlib.util.spec_from_file_location("backend_resume_state_tests", SOURCE)
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
    # Preserve the calling test's CPU random stream while constructing modules.
    with torch.random.fork_rng(devices=[]):
        model = FixtureModel().to(dtype=getattr(torch, dtype_name))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(-0.2, 0.3, parameter.numel(), dtype=parameter.dtype)
            parameter.copy_(values.reshape(parameter.shape) + 0.04 * index + offset)
    return model


def fixture_batches(dtype_name):
    dtype = getattr(torch, dtype_name)
    batches = []
    for index, boundaries in enumerate(BOUNDARIES):
        raw = torch.linspace(-0.4, 0.6, boundaries[-1] * 3, dtype=dtype).reshape(-1, 3)
        batches.append({"raw": raw + index * 0.01,
                        "targets": torch.linspace(-0.1, 0.5, len(boundaries) - 1, dtype=dtype),
                        "offsets": torch.tensor(boundaries, dtype=torch.int64), "boundaries": boundaries})
    return batches


def fixture_identity():
    packages = {}
    for route, name, build in (("cpu", "ragged-pool", "1cu128"), ("cuda", "ragged-pool-cuda", "")):
        packages[route] = {"distribution_name": name, "distribution_version": "0.1.0",
                           "wheel_build_tag": build, "wheel_sha256": "1" * 64,
                           "native_library_sha256": "2" * 64, "metadata_sha256": "3" * 64}
    return {"script_sha256": "4" * 64, "fixture_script_sha256": "5" * 64,
            "installed_packages": packages}


def populate_cpu_momentum(model, opt, count=2):
    """Create nonempty SGD state using fixed synthetic CPU gradients."""
    for step in range(count):
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, 0.015 * (index + 1) * (step + 1))
        opt.step()
        opt.zero_grad(set_to_none=True)


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if type(value) is dict:
        return {key: clone_tree(item) for key, item in value.items()}
    if type(value) is list:
        return [clone_tree(item) for item in value]
    if type(value) is tuple:
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def exact(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.device == second.device and first.shape == second.shape
        assert torch.equal(first.detach().contiguous().view(torch.uint8),
                           second.detach().contiguous().view(torch.uint8))
    elif isinstance(first, dict):
        assert list(first) == list(second)
        for key in first:
            exact(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            exact(a, b)
    else:
        assert first == second


def tensor_identity(value):
    return (id(value), value._version, value.data_ptr(), value.untyped_storage().data_ptr(),
            value.untyped_storage().nbytes(), tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.dtype, value.device, value.requires_grad)


def live_snapshot(model, opt):
    return {"model": clone_tree(dict(model.state_dict())), "optimizer": clone_tree(opt.state_dict()),
            "identities": tuple(tensor_identity(value) for value in
                                list(model.parameters()) + list(tensors(opt.state_dict()))),
            "parameter_ids": tuple(id(value) for value in model.parameters()),
            "group_bindings": tuple(tuple(id(value) for value in group["params"]) for group in opt.param_groups),
            "gradients": tuple(None if value.grad is None else
                               (tensor_identity(value.grad), value.grad.detach().clone()) for value in model.parameters()),
            "model_id": id(model), "optimizer_id": id(opt), "training": model.training}


def assert_live_unchanged(model, opt, before):
    exact(live_snapshot(model, opt), before)


def prepared(dtype_name="float64"):
    module = example()
    initial = make_model(dtype_name)
    batches = fixture_batches(dtype_name)
    metadata = module.metadata(dtype_name, "cuda-to-cpu", initial, batches, fixture_identity(),
                               source_revision="a" * 40, source_tree="b" * 40)
    trained = copy.deepcopy(initial)
    trained_opt = module.optimizer(trained)
    populate_cpu_momentum(trained, trained_opt)
    # Saved tensors are explicit CPU fixtures under the declared target protocol.
    # No source CUDA training is performed by this test.
    checkpoint = {"metadata": copy.deepcopy(metadata), "model": clone_tree(dict(trained.state_dict())),
                  "optimizer": clone_tree(trained_opt.state_dict())}
    target = make_model(dtype_name, offset=0.2)
    target_opt = module.optimizer(target)
    populate_cpu_momentum(target, target_opt, count=1)
    return module, metadata, checkpoint, trained, trained_opt, target, target_opt


def test_importing_state_helpers_does_not_import_operator_packages_or_query_cuda(monkeypatch):
    before = set(sys.modules)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("State helper import must not import either operator package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with patch.object(torch.cuda, "is_available", side_effect=AssertionError("No CUDA query during helper import")):
        example()
    added = set(sys.modules) - before
    assert not any(name == "ragged_pool_cuda" or name.startswith("ragged_pool_cuda.") for name in added)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_valid_cpu_fixture_restore_copies_state_and_preserves_target_bindings(dtype_name):
    module, metadata, checkpoint, trained, trained_opt, target, target_opt = prepared(dtype_name)
    before = live_snapshot(target, target_opt)
    saved = clone_tree(checkpoint)
    identities = tuple(tensor_identity(value) for value in tensors(checkpoint))
    assert module.validate_checkpoint(checkpoint, target, target_opt, metadata) is None
    assert_live_unchanged(target, target_opt, before)
    module.restore(checkpoint, target, target_opt, metadata)
    exact(dict(target.state_dict()), dict(trained.state_dict()))
    exact(target_opt.state_dict(), trained_opt.state_dict())
    assert tuple(id(value) for value in target.parameters()) == before["parameter_ids"]
    assert tuple(tuple(id(value) for value in group["params"]) for group in target_opt.param_groups) == before["group_bindings"]
    assert id(target) == before["model_id"] and id(target_opt) == before["optimizer_id"]
    assert tuple(value.data_ptr() for value in target.parameters()) == tuple(item[2] for item in before["identities"][:5])
    assert all(value.grad is None for value in target.parameters())
    saved_storage = {value.untyped_storage().data_ptr() for value in tensors(checkpoint)}
    restored = list(target.parameters()) + list(tensors(target_opt.state_dict()))
    assert all(value.untyped_storage().data_ptr() not in saved_storage for value in restored)
    assert len(saved_storage) == 10
    exact(checkpoint, saved)
    assert tuple(tensor_identity(value) for value in tensors(checkpoint)) == identities
    with torch.no_grad():
        next(target.parameters()).add_(0.125)
        next(iter(target_opt.state.values()))["momentum_buffer"].add_(0.25)
    exact(checkpoint, saved)
    restored_before = live_snapshot(target, target_opt)
    checkpoint["model"]["encoder.weight"].add_(0.5)
    checkpoint["optimizer"]["state"][0]["momentum_buffer"].add_(0.75)
    assert_live_unchanged(target, target_opt, restored_before)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_cpu_source_checkpoint_creation_owns_all_saved_state(dtype_name):
    module = example()
    model = make_model(dtype_name)
    opt = module.optimizer(model)
    populate_cpu_momentum(model, opt)
    metadata = module.metadata(dtype_name, "cpu-to-cuda", model, fixture_batches(dtype_name), fixture_identity())
    before = live_snapshot(model, opt)
    checkpoint = module.make_checkpoint(model, opt, metadata)
    assert set(checkpoint) == {"metadata", "model", "optimizer"}
    assert all(value.device.type == "cpu" and not value.requires_grad for value in tensors(checkpoint))
    pointers = [value.untyped_storage().data_ptr() for value in tensors(checkpoint)]
    assert len(pointers) == len(set(pointers)) == 10
    live_pointers = {value.untyped_storage().data_ptr() for value in
                     list(model.parameters()) + list(tensors(opt.state_dict()))}
    assert not set(pointers) & live_pointers
    exact(checkpoint["model"], dict(model.state_dict()))
    exact(checkpoint["optimizer"], opt.state_dict())
    assert_live_unchanged(model, opt, before)
    checkpoint["model"]["encoder.weight"].add_(0.1)
    checkpoint["optimizer"]["state"][0]["momentum_buffer"].add_(0.2)
    assert_live_unchanged(model, opt, before)


@pytest.mark.parametrize("fault", ["extra", "schema_bool", "cursor_repeat", "cursor_skip", "direction",
                                  "target_device", "parameter_order", "parameter_dtype", "optimizer_policy",
                                  "package_pin", "digest"])
def test_metadata_protocol_is_validated_even_when_expected_record_matches(fault):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    bad = copy.deepcopy(checkpoint)
    record = bad["metadata"]
    if fault == "extra":
        record["unknown"] = 1
    elif fault == "schema_bool":
        record["schema"] = True
    elif fault == "cursor_repeat":
        record["next_update"] = 1
    elif fault == "cursor_skip":
        record["next_update"] = 3
    elif fault == "direction":
        record["direction"] = "cuda-to-cuda"
    elif fault == "target_device":
        record["target_device"] = "cuda:0"
    elif fault == "parameter_order":
        record["parameters"][2:4] = record["parameters"][2:4][::-1]
    elif fault == "parameter_dtype":
        record["parameters"][0]["dtype"] = "torch.float32"
    elif fault == "optimizer_policy":
        record["optimizer"]["options"]["momentum"] = 0.0
    elif fault == "package_pin":
        record["installed_packages"]["cpu"]["wheel_build_tag"] = ""
    else:
        record["fixture_sha256"] = "G" * 64
    before = live_snapshot(target, target_opt)
    with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
        with patch.object(target_opt, "load_state_dict", wraps=target_opt.load_state_dict) as opt_load:
            with pytest.raises(ValueError):
                module.restore(bad, target, target_opt, copy.deepcopy(record))
            assert model_load.call_count == opt_load.call_count == 0
    assert_live_unchanged(target, target_opt, before)
    module.restore(checkpoint, target, target_opt, metadata)


def damage(checkpoint, fault):
    state, saved = checkpoint["model"], checkpoint["optimizer"]["state"]
    parameter = state["encoder.weight"]
    buffer = saved[0]["momentum_buffer"]
    if fault == "missing_parameter":
        state.pop("head.bias")
    elif fault == "extra_parameter":
        state["unknown"] = torch.zeros(1, dtype=parameter.dtype)
    elif fault == "parameter_order":
        items = list(state.items())
        items[2:4] = items[2:4][::-1]
        checkpoint["model"] = dict(items)
    elif fault == "parameter_shape":
        state["encoder.weight"] = parameter.reshape(-1).clone()
    elif fault == "parameter_dtype":
        state["encoder.weight"] = parameter.float()
    elif fault == "nonfinite":
        parameter[0, 0] = float("inf")
    elif fault == "sparse":
        state["encoder.weight"] = parameter.to_sparse()
    elif fault == "noncontiguous":
        state["encoder.weight"] = parameter.t().contiguous().t()
    elif fault == "oversized_storage":
        state["encoder.weight"] = torch.zeros(parameter.numel() + 1, dtype=parameter.dtype)[:parameter.numel()].reshape(parameter.shape)
    elif fault == "offset_storage":
        state["encoder.weight"] = torch.zeros(parameter.numel() + 1, dtype=parameter.dtype)[1:].reshape(parameter.shape)
    elif fault == "alias":
        saved[0]["momentum_buffer"] = parameter
    elif fault == "requires_grad":
        parameter.requires_grad_()
    elif fault == "inference_tensor":
        with torch.inference_mode():
            state["encoder.weight"] = parameter.clone()
    elif fault == "meta_tensor":
        state["encoder.weight"] = torch.empty(parameter.shape, dtype=parameter.dtype, device="meta")
    elif fault == "momentum_dtype":
        saved[0]["momentum_buffer"] = buffer.float()
    elif fault == "missing_momentum":
        saved[0].pop("momentum_buffer")
    elif fault == "extra_momentum":
        saved[0]["unknown"] = torch.zeros(1)
    elif fault == "ordinal_bool":
        saved[False] = saved.pop(0)
    elif fault == "ordinal_reorder":
        checkpoint["optimizer"]["param_groups"][0]["params"][2:4] = [3, 2]
    elif fault == "optimizer_policy":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = True
    elif fault == "extra_group":
        checkpoint["optimizer"]["param_groups"].append(copy.deepcopy(checkpoint["optimizer"]["param_groups"][0]))
    else:
        raise AssertionError("Unknown fixture fault")


@pytest.mark.parametrize("fault", ["missing_parameter", "extra_parameter", "parameter_order", "parameter_shape",
                                  "parameter_dtype", "nonfinite", "sparse", "noncontiguous", "oversized_storage",
                                  "offset_storage", "alias", "requires_grad", "inference_tensor", "meta_tensor",
                                  "momentum_dtype", "missing_momentum",
                                  "extra_momentum", "ordinal_bool", "ordinal_reorder", "optimizer_policy", "extra_group"])
def test_bad_saved_state_refuses_before_either_live_loader(fault):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    bad = clone_tree(checkpoint)
    damage(bad, fault)
    before = live_snapshot(target, target_opt)
    with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
        with patch.object(target_opt, "load_state_dict", wraps=target_opt.load_state_dict) as opt_load:
            with pytest.raises(ValueError):
                module.restore(bad, target, target_opt, metadata)
            assert model_load.call_count == opt_load.call_count == 0
    assert_live_unchanged(target, target_opt, before)


@pytest.mark.parametrize("fault", ["same_shape_reordered_optimizer", "foreign_optimizer", "target_options",
                                  "target_dtype", "evaluation_mode", "source_device", "uncleared_gradient"])
def test_live_target_and_source_admission_preserves_existing_state(fault):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    if fault == "same_shape_reordered_optimizer":
        parameters = target_opt.param_groups[0]["params"]
        assert parameters[2].shape == parameters[3].shape
        parameters[2:4] = parameters[2:4][::-1]
    elif fault == "foreign_optimizer":
        target_opt = module.optimizer(make_model())
    elif fault == "target_options":
        target_opt.param_groups[0]["momentum"] = 0.0
    elif fault == "target_dtype":
        target.float()
    elif fault == "evaluation_mode":
        target.eval()
    elif fault == "uncleared_gradient":
        metadata = module.metadata("float64", "cpu-to-cuda", target, fixture_batches("float64"), fixture_identity())
        next(target.parameters()).grad = torch.ones_like(next(target.parameters()))
    before = live_snapshot(target, target_opt)
    with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
        with patch.object(target_opt, "load_state_dict", wraps=target_opt.load_state_dict) as opt_load:
            with pytest.raises(ValueError):
                if fault in ("source_device", "uncleared_gradient"):
                    # CPU cannot produce a CUDA-source checkpoint, and a CPU
                    # source must clear pending gradients before a boundary.
                    module.make_checkpoint(target, target_opt, metadata)
                else:
                    module.restore(checkpoint, target, target_opt, metadata)
            assert model_load.call_count == opt_load.call_count == 0
    assert_live_unchanged(target, target_opt, before)


def serialized(checkpoint):
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


@pytest.mark.parametrize("fault", ["wrong_digest", "malformed_digest", "truncated", "empty", "oversize", "missing", "directory"])
def test_file_admission_refuses_before_framework_loader_or_target_application(tmp_path, fault):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    data = serialized(checkpoint)
    path = tmp_path / "state.pt"
    path.write_bytes(data)
    digest = sha256(data).hexdigest()
    if fault == "wrong_digest":
        digest = "0" * 64
    elif fault == "malformed_digest":
        digest = "A" + digest[1:]
    elif fault == "truncated":
        path.write_bytes(data[:-8])
    elif fault == "empty":
        path.write_bytes(b"")
    elif fault == "oversize":
        path.write_bytes(b"x" * (module.MAX_CHECKPOINT_BYTES + 1))
    elif fault == "missing":
        path.unlink()
    elif fault == "directory":
        path = tmp_path
    before = live_snapshot(target, target_opt)
    with patch.object(torch, "load", wraps=torch.load) as loader:
        with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
            with patch.object(target_opt, "load_state_dict", wraps=target_opt.load_state_dict) as opt_load:
                with pytest.raises((ValueError, OSError)):
                    module.load_checkpoint(path, digest, target, target_opt, metadata)
                assert loader.call_count == model_load.call_count == opt_load.call_count == 0
    assert_live_unchanged(target, target_opt, before)


def test_symlink_file_is_refused_before_framework_loading_when_available(tmp_path):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    data = serialized(checkpoint)
    source, link = tmp_path / "state.pt", tmp_path / "linked.pt"
    source.write_bytes(data)
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError) as error:
        pytest.skip("Symlink creation unavailable in this environment: " + str(error))
    before = live_snapshot(target, target_opt)
    with patch.object(torch, "load", wraps=torch.load) as loader:
        with pytest.raises(ValueError):
            module.load_checkpoint(link, sha256(data).hexdigest(), target, target_opt, metadata)
        assert loader.call_count == 0
    assert_live_unchanged(target, target_opt, before)


def test_valid_file_is_read_once_loaded_from_verified_bytes_and_never_applied(tmp_path):
    module, metadata, checkpoint, _, _, target, target_opt = prepared()
    data = serialized(checkpoint)
    path = tmp_path / "state.pt"
    path.write_bytes(data)
    before = live_snapshot(target, target_opt)
    real_load = torch.load

    def load_verified(buffer, **kwargs):
        assert type(buffer) is io.BytesIO
        assert buffer.getvalue() == data and buffer.tell() == 0
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        # The loader must consume its verified buffer even if the path changes.
        path.write_bytes(b"changed after the one permitted read")
        return real_load(buffer, **kwargs)

    with patch.object(module, "_read_bytes", wraps=module._read_bytes) as read:
        with patch.object(torch, "load", side_effect=load_verified) as loader:
            with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
                with patch.object(target_opt, "load_state_dict", wraps=target_opt.load_state_dict) as opt_load:
                    loaded = module.load_checkpoint(path, sha256(data).hexdigest(), target, target_opt, metadata)
                    assert read.call_count == loader.call_count == 1
                    assert model_load.call_count == opt_load.call_count == 0
    exact(loaded, checkpoint)
    assert_live_unchanged(target, target_opt, before)
