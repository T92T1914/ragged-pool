"""CPU mathematics and admission fixtures for the two-rank training example.

These tests emulate gradient averaging with serial PyTorch arithmetic. They do
not initialize a process group, run Gloo, or establish distributed acceptance.
The separate installed consumer exercises the actual two-process path.
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


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "train_distributed.py"
PARAMETER_NAMES = ("encoder.weight", "encoder.bias", "scorer.weight", "head.weight", "head.bias")
BOUNDARIES = (((0, 2, 2), (0, 1, 3, 4)), ((0,), (0, 0, 1, 3, 6)))


def example():
    specification = importlib.util.spec_from_file_location("distributed_consumer_math_tests", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def snapshot(batch):
    return {name: (value.detach().clone(), id(value), value._version, value.data_ptr(),
                   value.storage_offset(), tuple(value.stride()))
            for name, value in batch.items() if isinstance(value, torch.Tensor)}


def unchanged(batch, before):
    for name, (saved, identity, version, pointer, storage_offset, stride) in before.items():
        actual = batch[name]
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        assert (id(actual), actual._version, actual.data_ptr(), actual.storage_offset(), tuple(actual.stride())) == (
            identity, version, pointer, storage_offset, stride)


def test_import_does_not_load_native_packages_query_cuda_or_start_distributed_work(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Helper import must leave native operator packages unloaded")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Helper import must leave native operator packages unloaded")
        return original_import_module(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Helper import must not query devices or start distributed work")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    with (patch.object(torch.cuda, "is_available", side_effect=forbidden),
          patch.object(torch.cuda, "device_count", side_effect=forbidden),
          patch.object(torch.cuda, "_lazy_init", side_effect=forbidden),
          patch.object(torch.distributed, "init_process_group", side_effect=forbidden),
          patch.object(torch.distributed, "new_group", side_effect=forbidden),
          patch.object(torch.distributed, "all_reduce", side_effect=forbidden),
          patch.object(torch.multiprocessing, "spawn", side_effect=forbidden),
          patch.object(torch.multiprocessing, "start_processes", side_effect=forbidden),
          patch.object(subprocess, "Popen", side_effect=forbidden)):
        example()


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_fixed_batches_rebase_offsets_and_preserve_rank_inputs(dtype_name):
    module = example()
    initial, batches, digest = module.fixture(dtype_name)
    assert tuple(name for name, _ in initial.named_parameters()) == PARAMETER_NAMES
    assert len(digest) == 64 and all(value in "0123456789abcdef" for value in digest)
    assert len(batches) == 2
    for step, expected in enumerate(BOUNDARIES):
        rank_batches = [module.rank_batch(batches, step, rank) for rank in (0, 1)]
        before = [snapshot(batch) for batch in rank_batches]
        for rank, batch in enumerate(rank_batches):
            assert tuple(batch["boundaries"]) == expected[rank]
            assert batch["raw"].shape == (expected[rank][-1], 3)
            assert batch["targets"].shape == (len(expected[rank]) - 1,)
            assert batch["raw"].device.type == "cpu"
            assert batch["raw"].dtype == getattr(torch, dtype_name)
            assert batch["offsets"].dtype == torch.int64
            assert batch["offsets"].tolist() == list(expected[rank])
        combined = module.global_batch(batches, step)
        # Rebase the second packed array by the first rank's number of items.
        expected_offsets = expected[0] + tuple(value + expected[0][-1] for value in expected[1][1:])
        assert tuple(combined["boundaries"]) == expected_offsets
        assert combined["offsets"].tolist() == list(expected_offsets)
        torch.testing.assert_close(combined["raw"], torch.cat([batch["raw"] for batch in rank_batches]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(combined["targets"], torch.cat([batch["targets"] for batch in rank_batches]),
                                   rtol=0, atol=0)
        for batch, saved in zip(rank_batches, before):
            unchanged(batch, saved)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_global_group_normalization_matches_an_independent_scalar_derivative(dtype_name):
    module = example()
    dtype = getattr(torch, dtype_name)
    parameter = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    # Rank zero owns x=[1,2], rank one owns x=[3]. With target zero, the global
    # mean squared loss is w^2*(1+4+9)/3, whose derivative at w=1 is 28/3.
    rank_sums = [((parameter * torch.tensor(values, dtype=dtype)).square()).sum()
                 for values in ([1.0, 2.0], [3.0])]
    correctly_scaled = sum(module.normalized_local_loss(value, 3) for value in rank_sums) / 2
    actual = torch.autograd.grad(correctly_scaled, parameter, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.tensor(28.0 / 3.0, dtype=dtype))
    # Averaging local means gives each rank equal weight despite unequal B.
    # Its derivative is (5+18)/2 = 11.5, an observable change in the objective.
    mean_of_local_means = (rank_sums[0] / 2 + rank_sums[1]) / 2
    wrong = torch.autograd.grad(mean_of_local_means, parameter)[0]
    torch.testing.assert_close(wrong, torch.tensor(11.5, dtype=dtype))
    assert float((wrong - actual).abs()) > 2.0


@pytest.mark.parametrize("argument", ["global_groups", "world_size"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2", None])
def test_normalization_rejects_invalid_count_metadata_without_changing_loss(argument, bad):
    module = example()
    local_sse = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    before = (local_sse.detach().clone(), local_sse._version, local_sse.data_ptr(), local_sse.grad)
    options = {"global_groups": 3, "world_size": 2, argument: bad}
    with pytest.raises((TypeError, ValueError)):
        module.normalized_local_loss(local_sse, **options)
    torch.testing.assert_close(local_sse, before[0], rtol=0, atol=0)
    assert (local_sse._version, local_sse.data_ptr(), local_sse.grad) == before[1:]


@pytest.mark.parametrize("case", ["python-number", "vector", "nan", "infinity", "negative", "integer", "boolean", "complex", "meta"])
def test_normalization_rejects_values_outside_its_scalar_cpu_loss_contract(case):
    module = example()
    losses = {
        "python-number": 2.5,
        "vector": torch.tensor([2.5], dtype=torch.float64),
        "nan": torch.tensor(float("nan"), dtype=torch.float64),
        "infinity": torch.tensor(float("inf"), dtype=torch.float64),
        "negative": torch.tensor(-0.25, dtype=torch.float64),
        "integer": torch.tensor(2, dtype=torch.int64),
        "boolean": torch.tensor(True),
        "complex": torch.tensor(2.5 + 0j, dtype=torch.complex128),
        "meta": torch.empty((), device="meta", dtype=torch.float64),
    }
    with pytest.raises((TypeError, ValueError)):
        module.normalized_local_loss(losses[case], global_groups=3)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_empty_group_has_zero_pool_gradients_and_contributes_head_bias_loss(dtype_name):
    module = example()
    dtype = getattr(torch, dtype_name)
    values = torch.empty((0, 4), dtype=dtype, requires_grad=True)
    logits = torch.empty((0,), dtype=dtype, requires_grad=True)
    pooled = module.primitive_pool(values, logits, (0, 0))
    assert pooled.shape == (1, 4)
    torch.testing.assert_close(pooled, torch.zeros_like(pooled), rtol=0, atol=0)
    input_gradients = torch.autograd.grad(pooled.sum(), (values, logits))
    for gradient, source in zip(input_gradients, (values, logits)):
        assert gradient.shape == source.shape
        assert gradient.numel() == 0

    model, _, _ = module.fixture(dtype_name)
    with torch.no_grad():
        model.head.bias.fill_(0.25)
    raw = torch.empty((0, 3), dtype=dtype, requires_grad=True)
    offsets = torch.tensor([0, 0], dtype=torch.int64)
    predictions, model_pooled = model(raw, offsets, (0, 0), "primitive")
    torch.testing.assert_close(model_pooled, torch.zeros((1, 4), dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(predictions, torch.tensor([0.25], dtype=dtype), rtol=0, atol=0)
    # An empty group is still a target-bearing example. At target -0.75 its
    # squared error is one and d(loss)/d(head.bias) is exactly two.
    loss = (predictions + 0.75).square().sum()
    torch.testing.assert_close(loss, torch.tensor(1.0, dtype=dtype), rtol=0, atol=0)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        expected = torch.full_like(parameter, 2.0) if name == "head.bias" else torch.zeros_like(parameter)
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    assert raw.grad is not None and raw.grad.shape == raw.shape


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_no_group_rank_retains_zero_gradients_for_every_parameter_block(dtype_name):
    module = example()
    model, batches, _ = module.fixture(dtype_name)
    batch = module.rank_batch(batches, 1, 0)
    before = snapshot(batch)
    raw = batch["raw"].detach().clone().requires_grad_(True)
    predictions, pooled = model(raw, batch["offsets"], batch["boundaries"], "primitive")
    assert predictions.shape == (0,) and pooled.shape == (0, 4)
    local_sse = (predictions - batch["targets"]).square().sum()
    torch.testing.assert_close(local_sse, torch.zeros_like(local_sse), rtol=0, atol=0)
    module.normalized_local_loss(local_sse, global_groups=4).backward()
    assert tuple(name for name, _ in model.named_parameters()) == PARAMETER_NAMES
    for _, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
    assert raw.grad is not None and raw.grad.shape == (0, 3)
    unchanged(batch, before)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("step", [0, 1])
def test_serial_average_of_rank_gradients_matches_concatenated_global_objective(dtype_name, step):
    module = example()
    initial, batches, _ = module.fixture(dtype_name)
    global_batch = module.global_batch(batches, step)
    global_model = copy.deepcopy(initial)
    global_raw = global_batch["raw"].detach().clone().requires_grad_(True)
    predictions, _ = global_model(global_raw, global_batch["offsets"], global_batch["boundaries"], "primitive")
    global_loss = (predictions - global_batch["targets"]).square().mean()
    global_loss.backward()
    groups = global_batch["targets"].numel()
    rank_gradients, rank_input_gradients = [], []
    for rank in (0, 1):
        batch = module.rank_batch(batches, step, rank)
        before = snapshot(batch)
        rank_model = copy.deepcopy(initial)
        rank_raw = batch["raw"].detach().clone().requires_grad_(True)
        rank_predictions, _ = rank_model(rank_raw, batch["offsets"], batch["boundaries"], "primitive")
        local_loss = module.normalized_local_loss((rank_predictions - batch["targets"]).square().sum(), groups)
        local_loss.backward()
        rank_gradients.append({name: parameter.grad.detach().clone()
                               for name, parameter in rank_model.named_parameters()})
        rank_input_gradients.append(rank_raw.grad.detach().clone())
        unchanged(batch, before)
    tolerance = {"rtol": 2e-5, "atol": 2e-7} if dtype_name == "float32" else {"rtol": 2e-12, "atol": 2e-14}
    for name, parameter in global_model.named_parameters():
        serial_average = (rank_gradients[0][name] + rank_gradients[1][name]) / 2
        torch.testing.assert_close(serial_average, parameter.grad, **tolerance)
    # DDP averages parameter gradients, while each rank's raw input gradient is
    # still local. Undo the declared world-size factor for this comparison.
    torch.testing.assert_close(torch.cat(rank_input_gradients) / 2, global_raw.grad, **tolerance)
