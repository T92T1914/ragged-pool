"""Serial CPU checks for the distributed accumulation example.

These checks use independent scalar derivatives and primitive tensor arithmetic.
They never create a process group or a worker. Actual synchronization and owned
process cleanup are qualified separately by the installed Linux consumer.
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
from torch.nn import functional as F


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "train_distributed.py"
PARAMETER_NAMES = ("encoder.weight", "encoder.bias", "scorer.weight", "head.weight", "head.bias")
BOUNDARIES = (
    (((0, 2, 2), (0, 1, 3, 4)), ((0,), (0, 0, 1, 3, 6))),
    (((0,), (0, 2)), ((0, 0, 1, 3, 5), (0,))),
)
SGD_OPTIONS = {
    "lr": 0.03, "momentum": 0.8, "dampening": 0, "weight_decay": 0,
    "nesterov": False, "maximize": False, "foreach": False,
    "differentiable": False, "fused": False,
}


def example():
    specification = importlib.util.spec_from_file_location("distributed_accumulation_math_tests", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def snapshot(tensors):
    return {name: (value.detach().clone(), id(value), value._version, value.data_ptr(),
                   value.storage_offset(), tuple(value.stride()))
            for name, value in tensors.items() if isinstance(value, torch.Tensor)}


def unchanged(tensors, before):
    for name, (saved, identity, version, pointer, offset, stride) in before.items():
        actual = tensors[name]
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        assert (id(actual), actual._version, actual.data_ptr(), actual.storage_offset(), tuple(actual.stride())) == (
            identity, version, pointer, offset, stride)


def pack(chunks):
    """Independently concatenate packed groups in the supplied chunk order."""
    offsets, raw_parts, targets = [0], [], []
    consumed_items = 0
    for batch in chunks:
        raw_parts.append(batch["raw"])
        targets.append(batch["targets"])
        # Read each local offset, rather than trusting the example's rebasing.
        local_offsets = batch["offsets"].tolist()
        offsets.extend(consumed_items + value for value in local_offsets[1:])
        consumed_items += batch["raw"].shape[0]
    return {
        "raw": torch.cat(raw_parts), "targets": torch.cat(targets),
        "offsets": torch.tensor(offsets, dtype=torch.int64), "boundaries": tuple(offsets),
    }


def functional_prediction(model, raw, batch):
    """Build the small model without the consumer's forward or pool helpers."""
    parameters = dict(model.named_parameters())
    values = F.linear(raw, parameters["encoder.weight"], parameters["encoder.bias"]).tanh()
    logits = F.linear(values, parameters["scorer.weight"]).squeeze(-1)
    rows = []
    offsets = batch["offsets"].tolist()
    for first, last in zip(offsets, offsets[1:]):
        features = values[first:last].double()
        scores = logits[first:last].double()
        if first == last:
            row = features.sum(0) * 0 + scores.sum() * 0
        else:
            row = (torch.softmax(scores, 0).unsqueeze(1) * features).sum(0)
        rows.append(row.to(values.dtype))
    pooled = torch.stack(rows) if rows else values[:0] + logits.sum() * 0
    prediction = F.linear(pooled, parameters["head.weight"], parameters["head.bias"]).squeeze(-1)
    return prediction, pooled


def parameter_gradients(model):
    return {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}


def manual_sgd(model, momentum):
    """Apply the declared recurrence without sharing torch.optim's update code."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            previous = momentum.get(name, torch.zeros_like(parameter))
            buffer = previous * 0.8 + parameter.grad
            momentum[name] = buffer.detach().clone()
            parameter.add_(buffer, alpha=-0.03)


def clear_gradients(model):
    for parameter in model.parameters():
        parameter.grad = None


def close_parameters(actual, expected, tolerance):
    assert tuple(actual) == PARAMETER_NAMES and tuple(expected) == PARAMETER_NAMES
    for name in PARAMETER_NAMES:
        torch.testing.assert_close(actual[name], expected[name], **tolerance)


def tolerance(dtype_name):
    return ({"rtol": 2e-5, "atol": 2e-7} if dtype_name == "float32"
            else {"rtol": 2e-12, "atol": 2e-14})


def test_import_does_not_load_native_packages_query_cuda_or_start_workers(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Import must leave native operator packages unloaded")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name.split(".")[0] in {"ragged_pool", "ragged_pool_cuda"}:
            raise AssertionError("Import must leave native operator packages unloaded")
        return original_import_module(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not query devices, create a group, or launch a worker")

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


def test_accumulation_cli_refuses_windows_before_creating_files_or_workers(monkeypatch, tmp_path):
    module = example()
    destination = tmp_path / "unused-accumulation"
    output = tmp_path / "unused-accumulation.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("An unqualified platform must not launch workers")

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "_launch_workers", forbidden)
    monkeypatch.setattr(module.sys, "argv", [str(SOURCE), "--accumulate", "--dtype", "float32",
                                           "--directory", str(destination), "--output", str(output)])
    with pytest.raises(RuntimeError, match="qualified on Linux only"):
        module.main()
    assert not destination.exists() and not output.exists()


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_fixed_microbatches_pack_independently_and_preserve_inputs(dtype_name):
    module = example()
    state_before = torch.random.get_rng_state().clone()
    initial, windows, digest = module.accumulation_fixture(dtype_name)
    torch.testing.assert_close(torch.random.get_rng_state(), state_before, rtol=0, atol=0)
    assert tuple(name for name, _ in initial.named_parameters()) == PARAMETER_NAMES
    assert sum(value.numel() for value in initial.parameters()) == 25
    assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    assert len(windows) == 2
    assert module.ACCUM_BOUNDARIES == BOUNDARIES
    for window in (0, 1):
        assert len(windows[window]) == 2
        flattened, saved_batches = [], []
        for micro in (0, 1):
            assert len(windows[window][micro]) == 2
            for rank in (0, 1):
                batch = module.accumulation_rank_batch(windows, window, micro, rank)
                assert batch is windows[window][micro][rank]
                expected = BOUNDARIES[window][micro][rank]
                assert tuple(batch["boundaries"]) == expected
                assert batch["offsets"].tolist() == list(expected)
                assert batch["raw"].shape == (expected[-1], 3)
                assert batch["targets"].shape == (len(expected) - 1,)
                for name in ("raw", "targets"):
                    assert batch[name].device.type == "cpu"
                    assert batch[name].dtype == getattr(torch, dtype_name)
                    assert batch[name].is_contiguous() and bool(torch.isfinite(batch[name]).all())
                assert batch["offsets"].dtype == torch.int64 and not batch["offsets"].requires_grad
                flattened.append(batch)
                saved_batches.append((batch, snapshot(batch)))
        independent = pack(flattened)
        combined = module.accumulation_global_batch(windows, window)
        assert tuple(combined["boundaries"]) == independent["boundaries"]
        for name in ("raw", "targets", "offsets"):
            torch.testing.assert_close(combined[name], independent[name], rtol=0, atol=0)
        group_counts = tuple(sum(len(batch["offsets"]) - 1 for batch in micro)
                             for micro in windows[window])
        assert group_counts == ((5, 4) if window == 0 else (1, 4))
        assert module.accumulation_global_counts(windows, window) == (sum(group_counts), group_counts)
        assert combined["targets"].numel() == (9 if window == 0 else 5)
        for batch, before in saved_batches:
            unchanged(batch, before)


@pytest.mark.parametrize("index", ["window", "micro", "rank"])
@pytest.mark.parametrize("bad", [True, -1, 2, 0.0])
def test_rank_microbatch_admission_rejects_invalid_indices(index, bad):
    module = example()
    _, windows, _ = module.accumulation_fixture("float64")
    options = {"window": 0, "micro": 0, "rank": 0, index: bad}
    with pytest.raises((TypeError, ValueError)):
        module.accumulation_rank_batch(windows, **options)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_window_normalization_matches_scalar_derivative_and_exposes_wrong_micro_mean(dtype_name):
    module = example()
    dtype = getattr(torch, dtype_name)
    parameter = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    # Micro zero has x=[1,2] on rank zero and [3] on rank one. Micro one
    # has no groups on rank zero and x=[4] on rank one. The four-group loss
    # is w^2*(1+4+9+16)/4, with derivative 15 at w=1.
    rank_inputs = (([1.0, 2.0], [3.0]), ([], [4.0]))
    sums = [[(parameter * torch.tensor(values, dtype=dtype)).square().sum()
             for values in micro] for micro in rank_inputs]
    correct = sum(module.accumulation_loss(value, 4, (3, 1)[micro], "global_window_mean")
                  for micro in (0, 1) for value in sums[micro]) / 2
    actual = torch.autograd.grad(correct, parameter, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.tensor(15.0, dtype=dtype))
    first = sum(module.accumulation_loss(value, 4, 3, "global_window_mean")
                for value in sums[0]) / 2
    torch.testing.assert_close(torch.autograd.grad(first, parameter, retain_graph=True)[0],
                               torch.tensor(7.0, dtype=dtype))
    wrong = sum(module.accumulation_loss(value, 4, (3, 1)[micro], "microbatch_mean_control")
                for micro in (0, 1) for value in sums[micro]) / 2
    wrong_derivative = torch.autograd.grad(wrong, parameter)[0]
    # Equal microbatch weighting changes the derivative to (28/3 + 32)/2.
    torch.testing.assert_close(wrong_derivative, torch.tensor(62.0 / 3.0, dtype=dtype))
    assert float((wrong_derivative - actual).abs()) > 5.0


@pytest.mark.parametrize("mode", ["global_window_mean", "eager_sync", "microbatch_mean_control"])
def test_globally_empty_window_refuses_finite_zero_loss(mode):
    module = example()
    local_sse = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    before = snapshot({"loss": local_sse})
    with pytest.raises((TypeError, ValueError)):
        module.accumulation_loss(local_sse, 0, 0, mode)
    unchanged({"loss": local_sse}, before)
    assert local_sse.grad is None


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_no_group_microbatches_retain_zero_gradients_for_every_parameter(dtype_name):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    for window, micro, rank in ((0, 1, 0), (1, 0, 0), (1, 1, 1)):
        batch = windows[window][micro][rank]
        before = snapshot(batch)
        model = copy.deepcopy(initial)
        raw = batch["raw"].detach().clone().requires_grad_(True)
        predictions, pooled = model(raw, batch["offsets"], batch["boundaries"], "primitive")
        assert predictions.shape == (0,) and pooled.shape == (0, 4)
        loss = (predictions - batch["targets"]).square().sum() / (9 if window == 0 else 5)
        gradients = torch.autograd.grad(loss, (raw, *model.parameters()))
        assert gradients[0].shape == (0, 3)
        for gradient, parameter in zip(gradients[1:], model.parameters()):
            torch.testing.assert_close(gradient, torch.zeros_like(parameter), rtol=0, atol=0)
        unchanged(batch, before)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["global_window_mean", "eager_sync"])
def test_serial_accumulation_matches_two_independent_full_window_momentum_updates(dtype_name, mode):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    template_before = snapshot(dict(initial.named_parameters()))
    actual_model, oracle_model = copy.deepcopy(initial), copy.deepcopy(initial)
    optimizer = torch.optim.SGD(actual_model.parameters(), **SGD_OPTIONS)
    oracle_momentum = {}
    tol = tolerance(dtype_name)
    for window in (0, 1):
        micros = [pack(windows[window][micro]) for micro in (0, 1)]
        full = pack([batch for micro in windows[window] for batch in micro])
        before = [(batch, snapshot(batch)) for batch in (*micros, full)]
        group_count = full["targets"].numel()

        prefix_model = copy.deepcopy(oracle_model)
        prefix_raw = micros[0]["raw"].detach().clone().requires_grad_(True)
        prefix_prediction, _ = functional_prediction(prefix_model, prefix_raw, micros[0])
        ((prefix_prediction - micros[0]["targets"]).square().sum() / group_count).backward()
        prefix_gradients = parameter_gradients(prefix_model)
        assert any(bool(value.ne(0).any()) for value in prefix_gradients.values())

        clear_gradients(oracle_model)
        full_raw = full["raw"].detach().clone().requires_grad_(True)
        predictions, pooled = functional_prediction(oracle_model, full_raw, full)
        expected_loss = (predictions - full["targets"]).square().mean()
        expected_loss.backward()
        gradients = parameter_gradients(oracle_model)
        manual_sgd(oracle_model, oracle_momentum)

        result = module._accumulation_step(actual_model, optimizer, micros, window=window,
                                           mode=mode, route="primitive", distributed=False)
        close_parameters(result["first_micro_parameter_gradients"], prefix_gradients, tol)
        close_parameters(result["parameter_gradients"], gradients, tol)
        close_parameters(result["parameters"], dict(oracle_model.named_parameters()), tol)
        close_parameters(result["momentum"], oracle_momentum, tol)
        torch.testing.assert_close(result["metric"], expected_loss.detach(), **tol)
        item_start = group_start = 0
        assert len(result["microbatches"]) == 2
        for micro, batch in enumerate(micros):
            item_stop = item_start + batch["raw"].shape[0]
            group_stop = group_start + batch["targets"].shape[0]
            saved = result["microbatches"][micro]
            torch.testing.assert_close(saved["predictions"], predictions[group_start:group_stop], **tol)
            torch.testing.assert_close(saved["pooled"], pooled[group_start:group_stop], **tol)
            torch.testing.assert_close(saved["raw_gradients"], full_raw.grad[item_start:item_stop], **tol)
            item_start, group_start = item_stop, group_stop
        assert item_start == full_raw.shape[0] and group_start == full["targets"].shape[0]
        for batch, saved in before:
            unchanged(batch, saved)
    unchanged(dict(initial.named_parameters()), template_before)


def manual_control(initial, windows, mode):
    """Independent erroneous schedules used only to establish test sensitivity."""
    model, momentum, records = copy.deepcopy(initial), {}, []
    for window in windows:
        micros = [pack(ranks) for ranks in window]
        window_groups = sum(batch["targets"].numel() for batch in micros)
        clear_gradients(model)
        for index, batch in enumerate(micros):
            if mode == "drop_first_gradient" and index == 1:
                clear_gradients(model)
            predictions, _ = functional_prediction(model, batch["raw"], batch)
            denominator = (2 * batch["targets"].numel()
                           if mode == "microbatch_mean_control" else window_groups)
            ((predictions - batch["targets"]).square().sum() / denominator).backward()
            if mode == "step_every_micro":
                manual_sgd(model, momentum)
                clear_gradients(model)
        if mode != "step_every_micro":
            manual_sgd(model, momentum)
        records.append({"parameters": {name: value.detach().clone() for name, value in model.named_parameters()},
                        "momentum": {name: value.detach().clone() for name, value in momentum.items()}})
    return records


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_dropped_first_gradients_per_micro_updates_and_equal_micro_means_change_training(dtype_name):
    module = example()
    initial, windows, _ = module.accumulation_fixture(dtype_name)
    correct = manual_control(initial, windows, "global_window_mean")
    threshold = 1e-7 if dtype_name == "float32" else 1e-10
    for mode in ("drop_first_gradient", "step_every_micro", "microbatch_mean_control"):
        control = manual_control(initial, windows, mode)
        for window in (0, 1):
            for field in ("parameters", "momentum"):
                drift = max(float((control[window][field][name] - correct[window][field][name]).abs().max())
                            for name in PARAMETER_NAMES)
                assert drift > threshold, (mode, window, field, drift)

    # The consumer's declared wrong-objective route must reproduce the same
    # independently constructed error, rather than a control that merely fails.
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), **SGD_OPTIONS)
    expected_control = manual_control(initial, windows, "microbatch_mean_control")
    for window in (0, 1):
        result = module._accumulation_step(model, optimizer,
                                           [pack(windows[window][micro]) for micro in (0, 1)],
                                           window=window, mode="microbatch_mean_control",
                                           route="primitive", distributed=False)
        for field in ("parameters", "momentum"):
            close_parameters(result[field], expected_control[window][field], tolerance(dtype_name))
