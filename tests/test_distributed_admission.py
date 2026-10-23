"""Serial numerical admission and state preservation before a DDP update.

The reduction spy operates on one CPU status tensor. No test starts a worker or
process group. The installed Linux consumer supplies the distributed evidence.
"""
import builtins
import copy
import importlib
import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NAMES = ("encoder.weight", "encoder.bias", "scorer.weight", "head.weight", "head.bias")
OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0, "weight_decay": 0,
           "nesterov": False, "maximize": False, "foreach": False,
           "differentiable": False, "fused": False}


def example(filename="train_distributed.py"):
    spec = importlib.util.spec_from_file_location("distributed_admission_tests_" + filename[:-3], EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)


def warmed_target(dtype_name="float64"):
    """Use an actual optimizer update to establish every momentum buffer."""
    with torch.random.fork_rng(devices=[]):
        model = FixtureModel().to(dtype=getattr(torch, dtype_name))
    with torch.no_grad():
        for index, value in enumerate(model.parameters()):
            value.fill_(0.05 * (index + 1))
    optimizer = torch.optim.SGD(model.parameters(), **OPTIONS)
    sum(value.square().sum() for value in model.parameters()).backward()
    optimizer.step()
    assert all(bool((optimizer.state[value]["momentum_buffer"] != 0).all())
               for value in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    for index, value in enumerate(model.parameters()):
        value.grad = torch.full_like(value, 0.02 * (index + 1))
    return model, optimizer


def tensor_state(value):
    """Record bytes independently of floating point NaN equality."""
    return {"object": id(value), "version": value._version,
            "shape": tuple(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "layout": str(value.layout),
            "data_pointer": value.data_ptr(),
            "storage_pointer": value.untyped_storage().data_ptr(),
            "storage_bytes": value.untyped_storage().nbytes(),
            "stride": tuple(value.stride()), "offset": value.storage_offset(),
            "bytes": bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())}


def snapshot(model, optimizer, gradients=True):
    result = {"parameters": {name: tensor_state(value) for name, value in model.named_parameters()},
              "bindings": tuple(tuple(id(value) for value in group["params"])
                                for group in optimizer.param_groups),
              "options": [{name: copy.deepcopy(value) for name, value in group.items() if name != "params"}
                          for group in optimizer.param_groups],
              "optimizer": {id(value): {name: tensor_state(item) for name, item in state.items()}
                            for value, state in optimizer.state.items()}}
    if gradients:
        result["gradients"] = {name: None if value.grad is None else tensor_state(value.grad)
                               for name, value in model.named_parameters()}
    return result


def gradients(model):
    return {name: value.grad.detach().clone() for name, value in model.named_parameters()}


def inject(model, case):
    if case == "finite_norm_overflow":
        for value in model.parameters():
            value.grad.fill_(torch.finfo(value.dtype).max)
        assert all(bool(torch.isfinite(value.grad).all()) for value in model.parameters())
    else:
        value = {"nan": float("nan"), "posinf": float("inf"), "neginf": -float("inf")}[case]
        dict(model.named_parameters())["head.bias"].grad.fill_(value)


def independent_clip(named_gradients, cap):
    flat = torch.cat([value.reshape(-1) for value in named_gradients.values()])
    norm = torch.linalg.vector_norm(flat, ord=2)
    epsilon = torch.tensor(1e-6, dtype=flat.dtype)
    coefficient = torch.minimum(torch.tensor(1.0, dtype=flat.dtype), float(cap) / (norm + epsilon))
    return norm, coefficient, {name: value * coefficient for name, value in named_gradients.items()}


@pytest.mark.parametrize("filename", ["train_distributed.py", "admit_distributed.py"])
def test_import_leaves_native_packages_helper_devices_and_processes_unopened(filename, monkeypatch):
    original_import, original_import_module = builtins.__import__, importlib.import_module
    original_spec = importlib.util.spec_from_file_location

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Import must leave native pooling packages unloaded")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Import must leave native pooling packages unloaded")
        return original_import_module(name, *args, **kwargs)

    def guarded_spec(name, location, *args, **kwargs):
        if filename == "admit_distributed.py" and Path(location).name == "train_distributed.py":
            raise AssertionError("Import must leave the adjacent helper unloaded")
        return original_spec(name, location, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not query CUDA or start distributed work")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", guarded_spec)
    with (patch.object(torch.cuda, "is_available", side_effect=forbidden),
          patch.object(torch.cuda, "device_count", side_effect=forbidden),
          patch.object(torch.cuda, "_lazy_init", side_effect=forbidden),
          patch.object(torch.distributed, "init_process_group", side_effect=forbidden),
          patch.object(torch.distributed, "all_reduce", side_effect=forbidden),
          patch.object(torch.multiprocessing, "spawn", side_effect=forbidden),
          patch.object(subprocess, "Popen", side_effect=forbidden)):
        example(filename)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("vector", [(0.0, 0.0), (3.0, 4.0)])
def test_finite_probe_and_gate_are_observational_even_for_zero_norm(dtype_name, vector):
    module = example()
    model, optimizer = warmed_target(dtype_name)
    for value in model.parameters():
        value.grad.zero_()
    named = dict(model.named_parameters())
    named["encoder.weight"].grad[0, 0], named["head.bias"].grad[0] = vector
    before = snapshot(model, optimizer)
    result = module.probe_parameter_gradients(model, optimizer)
    assert result["finite_elements"] is result["joint_norm_finite"] is True
    assert result["local_bad"] == 0 and type(result["local_bad"]) is int
    assert result["joint_norm_value"] == (0.0 if vector == (0.0, 0.0) else 5.0)
    assert result["joint_norm_kind"] == "finite" and result["nonfinite_elements"] == []
    admitted = module.admit_parameter_gradients(model, optimizer)
    assert admitted["admitted"] is True and admitted["bad_count"] == 0
    assert admitted["admission_scalar_collectives"] == 0
    assert snapshot(model, optimizer) == before


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("case", ["nan", "posinf", "neginf", "finite_norm_overflow"])
def test_numeric_probe_does_not_throw_or_change_post_injection_bytes(dtype_name, case, monkeypatch):
    module = example()
    model, optimizer = warmed_target(dtype_name)
    inject(model, case)
    before, calls = snapshot(model, optimizer), []
    original = torch.nn.utils.get_total_norm

    def observed_norm(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["error_if_nonfinite"] is False and kwargs["foreach"] is False
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", observed_norm)
    result = module.probe_parameter_gradients(model, optimizer)
    assert len(calls) == 1
    assert result["local_bad"] == 1 and type(result["local_bad"]) is int
    assert result["joint_norm_finite"] is False and result["joint_norm_value"] is None
    assert result["joint_norm_kind"] == ("nan" if case == "nan" else "posinf")
    if case == "finite_norm_overflow":
        assert result["finite_elements"] is True and result["nonfinite_elements"] == []
    else:
        assert result["finite_elements"] is False
        assert result["nonfinite_elements"] == [{"name": "head.bias", "flat_index": 0, "kind": case}]
    assert snapshot(model, optimizer) == before


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("local_bad,peer_bad", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("coordinated", [False, True])
def test_local_peer_truth_table_distinguishes_shared_admission_from_local_policy(dtype_name, local_bad, peer_bad, coordinated):
    module = example()
    model, optimizer = warmed_target(dtype_name)
    if local_bad:
        inject(model, "nan")
    before = snapshot(model, optimizer)
    result = module.admit_parameter_gradients(model, optimizer, coordinated=coordinated, peer_bad=peer_bad)
    expected_bad = local_bad + peer_bad if coordinated else local_bad
    assert result["local_bad"] == local_bad and result["bad_count"] == expected_bad
    assert result["admitted"] is (expected_bad == 0)
    assert result["coordinated"] is coordinated
    assert result["admission_scalar_collectives"] == 0
    assert snapshot(model, optimizer) == before


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("case", ["finite", "nan", "posinf", "neginf", "finite_norm_overflow"])
def test_every_rank_participates_in_one_cpu_int64_sum_before_rejection(dtype_name, case, monkeypatch):
    module = example()
    model, optimizer = warmed_target(dtype_name)
    if case != "finite":
        inject(model, case)
    before, calls = snapshot(model, optimizer), []
    local_bad = int(case != "finite")

    def status_sum(value, op, group=None, async_op=False):
        assert value.device.type == "cpu" and value.dtype == torch.int64 and value.numel() == 1
        assert op == torch.distributed.ReduceOp.SUM and async_op is False
        assert int(value) == local_bad
        calls.append(int(value))
        value.add_(1)  # A bad peer. This is a status spy, not distributed execution.

    monkeypatch.setattr(torch.distributed, "all_reduce", status_sum)
    result = module.admit_parameter_gradients(model, optimizer, distributed=True)
    assert calls == [local_bad]
    assert result["bad_count"] == local_bad + 1 and result["admitted"] is False
    assert result["admission_scalar_collectives"] == 1
    assert snapshot(model, optimizer) == before


@pytest.mark.parametrize("peer_bad", [-1, 2, True])
def test_serial_peer_status_accepts_only_the_declared_binary_integer(peer_bad):
    module = example()
    model, optimizer = warmed_target()
    before = snapshot(model, optimizer)
    with pytest.raises((TypeError, ValueError)):
        module.admit_parameter_gradients(model, optimizer, peer_bad=peer_bad)
    assert snapshot(model, optimizer) == before


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_peer_rejection_preserves_healthy_bytes_while_local_only_policy_can_clip_them(dtype_name):
    module = example()
    model, optimizer = warmed_target(dtype_name)
    for value in model.parameters():
        value.grad.zero_()
    named = dict(model.named_parameters())
    named["encoder.weight"].grad[0, 0], named["head.bias"].grad[0] = 3.0, 4.0
    before = snapshot(model, optimizer)
    shared = module.admit_parameter_gradients(model, optimizer, coordinated=True, peer_bad=1)
    assert shared["admitted"] is False and shared["bad_count"] == 1
    assert snapshot(model, optimizer) == before
    local = module.admit_parameter_gradients(model, optimizer, coordinated=False, peer_bad=1)
    assert local["admitted"] is True and local["bad_count"] == 0
    assert snapshot(model, optimizer) == before
    parameters_and_momentum = snapshot(model, optimizer, gradients=False)
    clipped = module.clip_parameter_gradients(model, optimizer, 1.0)
    assert float(clipped["coefficient"]) < 1.0
    assert tensor_state(named["encoder.weight"].grad)["bytes"] != before["gradients"]["encoder.weight"]["bytes"]
    assert snapshot(model, optimizer, gradients=False) == parameters_and_momentum


def pack(chunks):
    offsets, item_count = [0], 0
    for batch in chunks:
        offsets.extend(item_count + value for value in batch["offsets"].tolist()[1:])
        item_count += batch["raw"].shape[0]
    return {"raw": torch.cat([batch["raw"] for batch in chunks]),
            "targets": torch.cat([batch["targets"] for batch in chunks]),
            "offsets": torch.tensor(offsets, dtype=torch.int64), "boundaries": tuple(offsets)}


def functional_prediction(model, raw, batch):
    values = F.linear(raw, model.encoder.weight, model.encoder.bias).tanh()
    logits = F.linear(values, model.scorer.weight).squeeze(-1)
    rows = []
    for start, stop in zip(batch["boundaries"], batch["boundaries"][1:]):
        features, scores = values[start:stop].double(), logits[start:stop].double()
        row = (features.sum(0) * 0 + scores.sum() * 0 if start == stop else
               (torch.softmax(scores, 0).unsqueeze(1) * features).sum(0))
        rows.append(row.to(values.dtype))
    return F.linear(torch.stack(rows), model.head.weight, model.head.bias).squeeze(-1)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_admitted_windows_match_full_window_manual_clipping_and_sgd(dtype_name):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    actual, expected = copy.deepcopy(initial), copy.deepcopy(initial)
    optimizer = torch.optim.SGD(actual.parameters(), **OPTIONS)
    momentum = {}
    tol = {"rtol": 2e-5, "atol": 2e-7} if dtype_name == "float32" else {"rtol": 2e-12, "atol": 2e-14}
    for window, cap in enumerate((0.25, 0.1)):
        micros = [pack(ranks) for ranks in windows[window]]
        full = pack([batch for ranks in windows[window] for batch in ranks])
        for value in expected.parameters():
            value.grad = None
        raw = full["raw"].detach().clone().requires_grad_(True)
        predictions = functional_prediction(expected, raw, full)
        (predictions - full["targets"]).square().mean().backward()
        pre = gradients(expected)
        norm, coefficient, clipped = independent_clip(pre, cap)
        assert float(coefficient) < 1.0
        with torch.no_grad():
            for name, value in expected.named_parameters():
                momentum[name] = 0.8 * momentum.get(name, torch.zeros_like(value)) + clipped[name]
                value.add_(momentum[name], alpha=-0.03)
        result = module._accumulation_step(actual, optimizer, micros, window=window, route="primitive",
                                           distributed=False, clipping="global", admission="normal")
        assert result["admission"]["admitted"] is True and result["admission"]["bad_count"] == 0
        assert result["admission"]["admission_scalar_collectives"] == 0
        torch.testing.assert_close(result["clipping"]["pre_clip_norm"], norm, **tol)
        torch.testing.assert_close(result["clipping"]["pre_clip_gradients"], pre, **tol)
        torch.testing.assert_close(result["parameter_gradients"], clipped, **tol)
        torch.testing.assert_close(result["parameters"], dict(expected.named_parameters()), **tol)
        torch.testing.assert_close(result["momentum"], momentum, **tol)
        start = 0
        for micro, batch in enumerate(micros):
            stop = start + batch["raw"].shape[0]
            torch.testing.assert_close(result["microbatches"][micro]["raw_gradients"], raw.grad[start:stop], **tol)
            start = stop
        assert start == raw.shape[0]


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_rejected_second_window_reaches_neither_clipping_nor_sgd_and_preserves_injected_state(dtype_name, monkeypatch):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), **OPTIONS)
    module._accumulation_step(model, optimizer, [pack(ranks) for ranks in windows[0]],
                              window=0, route="primitive", distributed=False,
                              clipping="global", admission="normal")
    assert all(bool((optimizer.state[value]["momentum_buffer"] != 0).any()) for value in model.parameters())
    original, observed = module.admit_parameter_gradients, []

    def observed_gate(*args, **kwargs):
        observed.append(snapshot(model, optimizer))
        result = original(*args, **kwargs)
        assert result["local_bad"] == 1 and result["admitted"] is False
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected admission must return before clipping or SGD")

    monkeypatch.setattr(module, "admit_parameter_gradients", observed_gate)
    monkeypatch.setattr(module, "clip_parameter_gradients", forbidden)
    monkeypatch.setattr(optimizer, "step", forbidden)
    result = module._accumulation_step(model, optimizer, [pack(ranks) for ranks in windows[1]],
                                       window=1, route="primitive", distributed=False,
                                       clipping="global", admission="coordinated", inject_nonfinite=True)
    assert len(observed) == 1 and snapshot(model, optimizer) == observed[0]
    assert result["admission"]["admitted"] is False
    assert result["optimizer_steps"] == 0
