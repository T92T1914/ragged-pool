"""Serial mathematics and admission for clipping a complete DDP update.

No test creates a process group or worker. The installed Linux consumer supplies
the actual distributed execution and collective-count evidence separately.
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
    spec = importlib.util.spec_from_file_location("distributed_clipping_tests_" + filename[:-3], EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)


def prepared(dtype_name="float64"):
    with torch.random.fork_rng(devices=[]):
        model = FixtureModel().to(dtype=getattr(torch, dtype_name))
    with torch.no_grad():
        for index, value in enumerate(model.parameters()):
            value.fill_(0.05 * (index + 1))
    optimizer = torch.optim.SGD(model.parameters(), **OPTIONS)
    for index, value in enumerate(model.parameters()):
        optimizer.state[value]["momentum_buffer"] = torch.full_like(value, 0.01 * (index + 1))
        value.grad = torch.full_like(value, 0.02 * (index + 1))
    return model, optimizer


def copies(named):
    return {name: value.detach().clone() for name, value in named.items()}


def signature(value):
    common = (id(value), value._version, tuple(value.shape), value.dtype, value.device, value.layout)
    if value.layout == torch.strided:
        return common + (value.data_ptr(), value.untyped_storage().data_ptr(),
                         value.untyped_storage().nbytes(), tuple(value.stride()), value.storage_offset())
    return common


def snapshot(model, optimizer, gradients=True):
    result = {"parameters": copies(dict(model.named_parameters())),
              "parameter_identity": tuple(signature(value) for value in model.parameters()),
              "groups": [{name: copy.deepcopy(value) for name, value in group.items() if name != "params"}
                         for group in optimizer.param_groups],
              "bindings": tuple(tuple(id(value) for value in group["params"]) for group in optimizer.param_groups),
              "momentum": {id(value): {name: (signature(item), item.detach().clone())
                                       for name, item in state.items()}
                           for value, state in optimizer.state.items()}}
    if gradients:
        result["gradients"] = {name: None if value.grad is None else
                               (signature(value.grad), value.grad.detach().clone())
                               for name, value in model.named_parameters()}
    return result


def exact(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            exact(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            exact(a, b)
    else:
        assert first == second


def independent_clip(gradients, cap, replica_factor=1):
    """One flat gradient vector and explicitly gradient-dtype coefficient math."""
    flat = torch.cat([value.reshape(-1) for value in gradients.values()])
    norm = torch.linalg.vector_norm(flat, ord=2)
    coefficient_norm = (norm.square() * replica_factor).sqrt() if replica_factor != 1 else norm
    epsilon = torch.tensor(1e-6, dtype=flat.dtype)
    coefficient = torch.minimum(torch.tensor(1.0, dtype=flat.dtype),
                                float(cap) / (coefficient_norm + epsilon))
    return norm, coefficient_norm, coefficient, {name: value * coefficient for name, value in gradients.items()}


def set_vector(model, first, second):
    for value in model.parameters():
        value.grad = torch.zeros_like(value)
    parameters = dict(model.named_parameters())
    parameters["encoder.weight"].grad[0, 0] = first
    parameters["head.bias"].grad[0] = second


@pytest.mark.parametrize("filename", ["train_distributed.py", "clip_distributed.py"])
def test_import_is_lazy_and_never_queries_cuda_or_starts_distributed_work(filename, monkeypatch):
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
        if filename == "clip_distributed.py" and Path(location).name == "train_distributed.py":
            raise AssertionError("Import must leave the adjacent training helper unloaded")
        return original_spec(name, location, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not query devices or start distributed work")

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
@pytest.mark.parametrize("vector,cap", [((0.0, 0.0), 1.0), ((0.3, 0.4), 1.0),
                                       ((3.0, 4.0), 5.0), ((3.0, 4.0), 1.0)])
def test_joint_clipping_returns_preclip_norm_and_uses_one_dtype_coefficient(dtype_name, vector, cap):
    module = example()
    model, optimizer = prepared(dtype_name)
    set_vector(model, *vector)
    gradients = copies({name: value.grad for name, value in model.named_parameters()})
    norm, _, coefficient, expected = independent_clip(gradients, cap)
    before = snapshot(model, optimizer, gradients=False)
    gradient_bindings = tuple((id(value.grad), value.grad.data_ptr()) for value in model.parameters())
    result = module.clip_parameter_gradients(model, optimizer, cap)
    assert result["pre_clip_norm"].dtype == getattr(torch, dtype_name)
    assert result["coefficient"].dtype == getattr(torch, dtype_name)
    torch.testing.assert_close(result["pre_clip_norm"], norm, rtol=0, atol=0)
    torch.testing.assert_close(result["coefficient"], coefficient, rtol=0, atol=0)
    exact(result["pre_clip_gradients"], gradients)
    exact(result["post_clip_gradients"], expected)
    exact({name: value.grad for name, value in model.named_parameters()}, expected)
    assert tuple((id(value.grad), value.grad.data_ptr()) for value in model.parameters()) == gradient_bindings
    exact(snapshot(model, optimizer, gradients=False), before)
    if vector == (0.0, 0.0) or vector == (0.3, 0.4):
        assert float(coefficient) == 1.0
    elif cap == 5.0:
        assert float(coefficient) < 1.0  # The declared epsilon also applies at the boundary.
    assert result["norm_scalar_collectives"] == 0


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_replica_norm_control_counts_identical_replicas_twice_and_changes_the_clip(dtype_name):
    module = example()
    correct, correct_optimizer = prepared(dtype_name)
    wrong, wrong_optimizer = prepared(dtype_name)
    for model in (correct, wrong):
        set_vector(model, 3.0, 4.0)
    gradients = copies({name: value.grad for name, value in correct.named_parameters()})
    norm, double_counted, coefficient, expected = independent_clip(gradients, 1.0, replica_factor=2)
    reference = module.clip_parameter_gradients(correct, correct_optimizer, 1.0)
    control = module.clip_parameter_gradients(wrong, wrong_optimizer, 1.0, rule="replica_norm_control")
    torch.testing.assert_close(control["pre_clip_norm"], norm, rtol=0, atol=0)
    torch.testing.assert_close(control["coefficient_norm"], double_counted, rtol=0, atol=0)
    torch.testing.assert_close(control["coefficient"], coefficient, rtol=0, atol=0)
    exact(control["post_clip_gradients"], expected)
    assert control["norm_scalar_collectives"] == 0  # Serial factor-two witness, not Gloo execution.
    assert float((reference["post_clip_gradients"]["encoder.weight"]
                  - control["post_clip_gradients"]["encoder.weight"]).abs().max()) > 0.1


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_joint_clipping_differs_from_block_clipping_and_rank_local_clipping(dtype_name):
    module = example()
    global_model, optimizer = prepared(dtype_name)
    set_vector(global_model, 2.0, 0.5)  # Mean of rank vectors (4,0) and (0,1).
    result = module.clip_parameter_gradients(global_model, optimizer, 1.0)
    local_outputs = []
    for vector in ((4.0, 0.0), (0.0, 1.0)):
        local_model, local_optimizer = prepared(dtype_name)
        set_vector(local_model, *vector)
        local_outputs.append(module.clip_parameter_gradients(local_model, local_optimizer, 1.0)["post_clip_gradients"])
    local_mean = {name: (local_outputs[0][name] + local_outputs[1][name]) / 2 for name in NAMES}
    assert max(float((result["post_clip_gradients"][name] - local_mean[name]).abs().max())
               for name in NAMES) > 0.2
    before = result["pre_clip_gradients"]
    blockwise = {name: independent_clip({name: value}, 1.0)[3][name] for name, value in before.items()}
    assert max(float((result["post_clip_gradients"][name] - blockwise[name]).abs().max())
               for name in NAMES) > 0.02


@pytest.mark.parametrize("cap", [0, -1.0, True, float("nan"), float("inf"), "1", None])
def test_invalid_cap_refuses_before_any_gradient_parameter_or_optimizer_mutation(cap):
    module = example()
    model, optimizer = prepared()
    before = snapshot(model, optimizer)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        module.clip_parameter_gradients(model, optimizer, cap)
    exact(snapshot(model, optimizer), before)


@pytest.mark.parametrize("case", ["missing_last", "noncontiguous", "sparse", "shared_gradient",
                                  "parameter_alias", "momentum_alias", "mixed_dtype", "parameter_shape",
                                  "optimizer_order", "foreign_optimizer", "optimizer_option"])
def test_invalid_gradient_or_binding_contract_refuses_with_an_otherwise_valid_target(case):
    module = example()
    model, optimizer = prepared()
    named = dict(model.named_parameters())
    if case == "missing_last":
        named["head.bias"].grad = None
    elif case == "noncontiguous":
        named["encoder.weight"].grad = torch.ones((3, 4), dtype=torch.float64).t()
    elif case == "sparse":
        named["encoder.weight"].grad = torch.sparse_coo_tensor([[0], [0]], [1.0], (4, 3), dtype=torch.float64)
    elif case == "shared_gradient":
        named["head.weight"].grad = named["scorer.weight"].grad
    elif case == "parameter_alias":
        named["head.weight"].grad = named["head.weight"].detach()
    elif case == "momentum_alias":
        parameter = named["head.weight"]
        parameter.grad = optimizer.state[parameter]["momentum_buffer"]
    elif case == "mixed_dtype":
        parameter = named["encoder.weight"]
        parameter.grad = None
        parameter.data = parameter.data.float()
        parameter.grad = torch.full_like(parameter, 0.02)
        optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 0.01)
    elif case == "parameter_shape":
        parameter = named["encoder.weight"]
        parameter.grad = None
        parameter.data = parameter.data.reshape(3, 4).clone()
        parameter.grad = torch.full_like(parameter, 0.02)
        optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 0.01)
    elif case == "optimizer_order":
        optimizer.param_groups[0]["params"][2:4] = optimizer.param_groups[0]["params"][2:4][::-1]
    elif case == "foreign_optimizer":
        _, optimizer = prepared()
    else:
        optimizer.param_groups[0]["lr"] = 0.2
    before = snapshot(model, optimizer)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        module.clip_parameter_gradients(model, optimizer, 0.25)
    exact(snapshot(model, optimizer), before)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("case", ["late_nan", "late_infinity", "finite_norm_overflow"])
def test_nonfinite_or_overflow_refuses_before_mutation_then_accepts_repaired_gradients(dtype_name, case):
    module = example()
    model, optimizer = prepared(dtype_name)
    valid = copies({name: value.grad for name, value in model.named_parameters()})
    if case == "finite_norm_overflow":
        for value in model.parameters():
            value.grad.fill_(torch.finfo(value.dtype).max)
        assert all(bool(torch.isfinite(value.grad).all()) for value in model.parameters())
    else:
        list(model.parameters())[-1].grad.fill_(float("nan") if case == "late_nan" else float("inf"))
    before = snapshot(model, optimizer)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        module.clip_parameter_gradients(model, optimizer, 0.25)
    exact(snapshot(model, optimizer), before)
    for name, value in model.named_parameters():
        value.grad = valid[name].clone()
    parameters_and_optimizer = snapshot(model, optimizer, gradients=False)
    expected = independent_clip(valid, 0.25)[3]
    result = module.clip_parameter_gradients(model, optimizer, 0.25)
    torch.testing.assert_close(result["post_clip_gradients"], expected)
    exact(snapshot(model, optimizer, gradients=False), parameters_and_optimizer)


def test_parameter_gradient_clipping_leaves_the_raw_input_gradient_untouched():
    module = example()
    model, optimizer = prepared()
    optimizer.zero_grad(set_to_none=True)
    raw = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float64, requires_grad=True)
    loss = raw.square().sum() + sum(value.square().sum() for value in model.parameters())
    loss.backward()
    before, raw_identity = raw.grad.detach().clone(), signature(raw.grad)
    result = module.clip_parameter_gradients(model, optimizer, 0.01)
    assert float(result["coefficient"]) < 1.0
    exact(raw.grad, before)
    assert signature(raw.grad) == raw_identity


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_momentum_accumulates_clipped_gradients_and_can_exceed_the_gradient_cap(dtype_name):
    module = example()
    model, _ = prepared(dtype_name)
    optimizer = torch.optim.SGD(model.parameters(), **OPTIONS)
    momentum = {}
    cap = 0.25
    tol = {"rtol": 2e-6, "atol": 2e-8} if dtype_name == "float32" else {"rtol": 2e-13, "atol": 2e-15}
    for _ in range(2):
        for value in model.parameters():
            value.grad = torch.ones_like(value)
        before = copies(dict(model.named_parameters()))
        gradients = copies({name: value.grad for name, value in model.named_parameters()})
        clipped = independent_clip(gradients, cap)[3]
        module.clip_parameter_gradients(model, optimizer, cap)
        optimizer.step()
        for name, value in model.named_parameters():
            momentum[name] = 0.8 * momentum.get(name, torch.zeros_like(value)) + clipped[name]
            torch.testing.assert_close(optimizer.state[value]["momentum_buffer"], momentum[name], **tol)
            expected = before[name].clone()
            expected.add_(momentum[name], alpha=-0.03)
            torch.testing.assert_close(value, expected, **tol)
    momentum_norm = torch.linalg.vector_norm(torch.cat([
        optimizer.state[value]["momentum_buffer"].reshape(-1) for value in model.parameters()]))
    assert float(momentum_norm) > 0.4 > cap


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
    pooled = torch.stack(rows)
    return F.linear(pooled, model.head.weight, model.head.bias).squeeze(-1)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_two_clipped_accumulation_updates_match_independent_full_window_manual_sgd(dtype_name):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    actual, expected = copy.deepcopy(initial), copy.deepcopy(initial)
    optimizer = torch.optim.SGD(actual.parameters(), **OPTIONS)
    momentum = {}
    tol = {"rtol": 2e-5, "atol": 2e-7} if dtype_name == "float32" else {"rtol": 2e-12, "atol": 2e-14}
    for window, cap in enumerate((0.25, 2.0)):
        micros = [pack(ranks) for ranks in windows[window]]
        full = pack([batch for ranks in windows[window] for batch in ranks])
        for value in expected.parameters():
            value.grad = None
        raw = full["raw"].detach().clone().requires_grad_(True)
        predictions = functional_prediction(expected, raw, full)
        (predictions - full["targets"]).square().mean().backward()
        pre = copies({name: value.grad for name, value in expected.named_parameters()})
        norm, _, coefficient, clipped = independent_clip(pre, cap)
        assert (float(coefficient) < 1.0) == (window == 0)
        with torch.no_grad():
            for name, value in expected.named_parameters():
                momentum[name] = 0.8 * momentum.get(name, torch.zeros_like(value)) + clipped[name]
                value.add_(momentum[name], alpha=-0.03)
        result = module._accumulation_step(actual, optimizer, micros, window=window, route="primitive",
                                           distributed=False, clipping="global")
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
