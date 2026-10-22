"""Recomputation, RNG and saved-forward-payload controls for the CPU consumer."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "train_checkpointed.py"


def example():
    spec = importlib.util.spec_from_file_location("activation_checkpoint_tests", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(module, model, raw, offsets, targets, *, checkpointed, preserve=True, probability=0.25):
    events = []
    predictions, pooled = module.forward(model, raw, offsets, checkpointed=checkpointed,
                                         preserve_rng_state=preserve, probability=probability, events=events)
    after_forward = torch.get_rng_state().clone()
    loss = (predictions - targets).square().mean()
    leaves = (raw, *model.parameters()) if raw.requires_grad else tuple(model.parameters())
    gradients = torch.autograd.grad(loss, leaves)
    return predictions, pooled, gradients, after_forward, torch.get_rng_state().clone(), events


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_actual_segment_is_recomputed_in_full_with_dropout_and_native_pool(dtype_name):
    module = example()
    model, batches, _ = module.dynamic.fixture(dtype_name)
    raw = batches[0]["raw"].clone().requires_grad_()
    events = []
    torch.manual_seed(module.DROPOUT_SEED)
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder:
        with patch.object(module.F, "dropout", wraps=module.F.dropout) as dropout:
            with patch.object(module, "checkpoint", wraps=module.checkpoint) as checkpoint:
                predictions, pooled = module.forward(model, raw, batches[0]["offsets"],
                                                     checkpointed=True, events=events)
                assert encoder.call_count == dropout.call_count == 1
                (predictions.square().mean() + pooled.square().mean()).backward()
    assert encoder.call_count == dropout.call_count == 2
    assert checkpoint.call_count == 1
    assert checkpoint.call_args.kwargs["use_reentrant"] is False
    assert checkpoint.call_args.kwargs["preserve_rng_state"] is True
    assert checkpoint.call_args.kwargs["determinism_check"] == "default"
    assert all(call.kwargs == {"p": 0.25, "training": True} for call in dropout.call_args_list)
    assert events == ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"]
    assert not hasattr(model, "last_dropout")
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert raw.grad is not None


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("raw_requires_grad", [True, False])
def test_non_reentrant_autograd_grad_matches_direct_and_keeps_grad_attributes_empty(dtype_name, raw_requires_grad):
    module = example()
    initial, batches, _ = module.dynamic.fixture(dtype_name)
    evaluations = []
    for checkpointed in (False, True):
        model = copy.deepcopy(initial)
        raw = batches[0]["raw"].clone().requires_grad_(raw_requires_grad)
        torch.manual_seed(3211)
        result = evaluate(module, model, raw, batches[0]["offsets"], batches[0]["targets"],
                          checkpointed=checkpointed)
        assert raw.grad is None and all(parameter.grad is None for parameter in model.parameters())
        assert all(bool(torch.isfinite(gradient).all()) for gradient in result[2])
        assert len(result[2]) == (6 if raw_requires_grad else 5)
        evaluations.append(result)
    direct, actual = evaluations
    for expected, observed in zip(direct[:2], actual[:2]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for expected, observed in zip(direct[2], actual[2]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert torch.equal(direct[4], actual[4]) and torch.equal(actual[3], actual[4])
    assert actual[5] == ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"]


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_unhooked_rng_control_changes_gradients_and_disabled_dropout_agrees(dtype_name):
    module = example()
    initial, batches, _ = module.dynamic.fixture(dtype_name)
    threshold = 1e-7 if dtype_name == "float32" else 1e-10
    for probability in (0.25, 0.0):
        results, before_states = [], []
        for checkpointed in (False, True):
            model = copy.deepcopy(initial)
            raw = batches[0]["raw"].clone().requires_grad_()
            torch.manual_seed(3211)
            before_states.append(torch.get_rng_state().clone())
            # evaluate uses ordinary autograd.grad without observation hooks.
            results.append(evaluate(module, model, raw, batches[0]["offsets"], batches[0]["targets"],
                                    checkpointed=checkpointed, preserve=False, probability=probability))
        direct, recomputed = results
        for expected, actual in zip(direct[:2], recomputed[:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if probability == 0.25:
            gradient_error = max(float((actual - expected).abs().max())
                                 for actual, expected in zip(recomputed[2], direct[2]))
            assert gradient_error > threshold
            assert not torch.equal(recomputed[4], direct[4])
            assert torch.equal(direct[3], recomputed[3])
        else:
            for expected, actual in zip(direct[2], recomputed[2]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for result, before in zip(results, before_states):
                assert torch.equal(before, result[3]) and torch.equal(before, result[4])


@pytest.mark.parametrize("boundaries", [(0, 0, 1, 1, 3), (0, 0, 0), (0, 1)])
def test_singleton_and_empty_groups_match_direct_gradients(boundaries):
    module = example()
    initial, _, _ = module.dynamic.fixture("float64")
    generator = torch.Generator(device="cpu").manual_seed(42)
    raw_source = torch.randn(boundaries[-1], 3, generator=generator, dtype=torch.float64) * 0.1
    offsets = torch.tensor(boundaries, dtype=torch.int64)
    targets = torch.linspace(-0.1, 0.1, len(boundaries) - 1, dtype=torch.float64)
    results = []
    for checkpointed in (False, True):
        model, raw = copy.deepcopy(initial), raw_source.clone().requires_grad_()
        torch.manual_seed(module.DROPOUT_SEED)
        results.append(evaluate(module, model, raw, offsets, targets, checkpointed=checkpointed))
    for expected, observed in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for expected, observed in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    empty = [index for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])) if left == right]
    torch.testing.assert_close(results[1][1][empty], torch.zeros_like(results[1][1][empty]), rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["raw", "offsets"])
def test_saved_original_input_mutation_is_refused_before_backward(changed):
    module = example()
    model, batches, _ = module.dynamic.fixture("float64")
    raw, offsets = batches[0]["raw"].clone().requires_grad_(), batches[0]["offsets"].clone()
    predictions, _ = module.forward(model, raw, offsets, checkpointed=True)
    with torch.no_grad():
        if changed == "raw": raw[0, 0].add_(0.125)
        else: offsets[1] = 1  # Still valid boundaries, but not the saved original.
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        predictions.square().mean().backward()


def test_original_native_admission_failure_propagates_and_clean_retry_works():
    module = example()
    initial, batches, _ = module.dynamic.fixture("float64")
    invalid = batches[0]["offsets"].clone()
    invalid[0] = 1
    messages = []
    for checkpointed in (False, True):
        model = copy.deepcopy(initial)
        torch.manual_seed(5111)
        with pytest.raises(RuntimeError) as failure:
            module.forward(model, batches[0]["raw"], invalid, checkpointed=checkpointed)
        messages.append(str(failure.value))
    assert messages[0] == messages[1] and "offset" in messages[0].lower()
    results = []
    for checkpointed in (False, True):
        # A failed dropout forward can consume RNG. Retry alignment is explicit.
        torch.manual_seed(5111)
        results.append(evaluate(module, copy.deepcopy(initial), batches[0]["raw"].clone().requires_grad_(),
                                batches[0]["offsets"], batches[0]["targets"], checkpointed=checkpointed))
    for expected, observed in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert torch.equal(results[0][4], results[1][4])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_each_update_matches_independent_momentum_math_and_preserves_sources(dtype_name):
    module = example()
    model, batches, tolerance = module.dynamic.fixture(dtype_name)
    opt = torch.optim.SGD(model.parameters(), **module.OPTIONS)
    snapshots = module.dynamic.snapshot({"raw": batches[0]["raw"], "offsets": batches[0]["offsets"],
                                         "targets": batches[0]["targets"]})
    previous = {}
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    torch.manual_seed(module.DROPOUT_SEED)
    for index, batch in enumerate(batches[:2]):
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        record = module.step(model, opt, batch, index, checkpointed=True)
        for name, parameter in model.named_parameters():
            gradient = record["parameter_gradients"][name]
            buffer = gradient.clone() if index == 0 else 0.8 * previous[name] + gradient
            torch.testing.assert_close(opt.state[parameter]["momentum_buffer"], buffer, **tolerance)
            torch.testing.assert_close(parameter, before[name] - 0.03 * buffer, **tolerance)
            previous[name] = buffer.clone()
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    module.dynamic.unchanged({"raw": batches[0]["raw"], "offsets": batches[0]["offsets"],
                              "targets": batches[0]["targets"]}, snapshots)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_two_updates_controls_and_saved_payload_receipt(dtype_name):
    module = example()
    receipt = module.run(dtype_name)
    assert receipt["status"] == "passed" and receipt["optimizer_steps"] == 2
    assert receipt["same_native_exact"] and receipt["same_native_max_abs_error"] == 0
    assert receipt["primitive_comparisons_passed"]
    assert receipt["saved_accounting"] == "original_forward_pack_hook_events"
    assert receipt["memory_performance_claim"] is False
    assert [row["N"] for row in receipt["steps"]] == [9, 12]
    assert [row["B"] for row in receipt["steps"]] == [5, 6]
    for index, row in enumerate(receipt["steps"]):
        assert row["checkpoint_scope_events"] == ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"]
        assert row["default_rng_after_forward_sha256"] == row["default_rng_after_backward_sha256"]
        if index:
            assert row["default_rng_before_sha256"] == receipt["steps"][index - 1]["default_rng_after_backward_sha256"]
        for payload in row["saved_forward"].values():
            assert payload["count"] == len(payload["events"])
            independent_bytes = 0
            for event in payload["events"]:
                assert event["numel"] == math.prod(event["shape"])
                assert event["logical_bytes"] == event["numel"] * event["element_size"]
                assert all(type(value) is int for value in event["shape"])
                assert all(not isinstance(value, torch.Tensor) for value in event.values())
                independent_bytes += event["numel"] * event["element_size"]
            assert payload["logical_bytes"] == independent_bytes
        assert row["saved_forward"]["checkpointed"]["logical_bytes"] < row["saved_forward"]["direct"]["logical_bytes"]
    bad = receipt["controls"]["without_rng_preservation"]
    assert bad["first_forward_exact"] and bad["rng_diverged"] and bad["diverged"]
    assert bad["gradient_max_abs_error"] > bad["divergence_threshold"]
    assert bad["final_parameter_max_abs_error"] > bad["divergence_threshold"]
    good = receipt["controls"]["dropout_disabled"]
    assert good["dropout_probability"] == 0 and good["same_native_exact"]
    assert good["max_abs_error"] == 0 and good["rng_unchanged"]


def test_saved_hook_metadata_does_not_hide_late_input_mutation():
    module = example()
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    events = []
    with module._observe_saved(events):
        loss = value.square().sum()
    with torch.no_grad():
        value.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()
    assert events and all(type(event) is dict for event in events)


def test_cli_refuses_existing_receipt_before_numerical_work(tmp_path):
    module = example()
    output = tmp_path / "receipt.json"
    output.write_text("existing\n", encoding="utf-8")
    with patch.object(sys, "argv", [str(SOURCE), "--dtype", "float64", "--output", str(output)]):
        with patch.object(module, "run") as run:
            with pytest.raises(SystemExit) as failure:
                module.main()
    assert failure.value.code == 2 and not run.called
    assert output.read_text(encoding="utf-8") == "existing\n"
