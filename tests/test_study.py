"""Check the study's data isolation, model comparison and numerical baseline."""
import hashlib
import importlib.util
from itertools import combinations
from pathlib import Path
import sys

import pytest
import torch

from ragged_pool import ragged_pool_reference


def load_example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ragged_pool_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = load_example("learn_sets")
baselines = load_example("pool_baselines")


def tensor_digest(tensor):
    # Hash the actual generated values rather than a seed or assigned identifier.
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def declared_splits():
    return [study.make_split(seed, count, lower, upper) for seed, count, lower, upper in (
        (71001, 512, 1, 16), (71002, 128, 1, 16),
        (71003, 256, 1, 16), (71004, 256, 17, 32),
    )]


def test_generated_splits_match_class_lengths_and_have_disjoint_actual_data(declared_splits):
    bag_hashes = []
    instance_hashes = []
    for split, count, lower, upper in zip(
        declared_splits, (512, 128, 256, 256), (1, 1, 1, 17), (16, 16, 16, 32),
    ):
        bags, labels, lengths = split["bags"], split["labels"], split["lengths"]
        assert len(bags) == len(labels) == len(lengths) == count
        assert int((labels == 0).sum()) == int((labels == 1).sum()) == count // 2
        histograms = [{length: 0 for length in range(lower, upper + 1)} for _ in range(2)]
        for bag, label, length in zip(bags, labels.tolist(), lengths):
            assert bag.dtype == torch.float32 and bag.device.type == "cpu"
            assert bag.shape == (length, 4) and lower <= length <= upper
            assert bool(torch.isfinite(bag).all()) and float(bag.abs().max()) <= 6
            histograms[int(label)][length] += 1
        assert histograms[0] == histograms[1]
        hashes = {tensor_digest(bag) for bag in bags}
        rows = {tensor_digest(row) for bag in bags for row in bag}
        assert len(hashes) == count
        assert len(rows) == sum(lengths)
        bag_hashes.append(hashes)
        instance_hashes.append(rows)
    for first, second in combinations(range(4), 2):
        assert bag_hashes[first].isdisjoint(bag_hashes[second])
        assert instance_hashes[first].isdisjoint(instance_hashes[second])


def test_generation_and_packing_are_deterministic_without_relabeling_bags(declared_splits):
    again = study.make_split(71002, 128, 1, 16)
    original = declared_splits[1]
    assert again["lengths"] == original["lengths"]
    assert again["digest"] == original["digest"]
    torch.testing.assert_close(again["labels"], original["labels"], rtol=0, atol=0)
    assert [tensor_digest(bag) for bag in again["bags"]] == [tensor_digest(bag) for bag in original["bags"]]
    indices = [7, 2, 11, 0]
    values, offsets, labels = study.pack_batch(original, indices)
    assert values.is_contiguous() and offsets.is_contiguous()
    assert offsets.dtype == torch.int64
    assert offsets.tolist() == [0] + list(torch.tensor([original["lengths"][i] for i in indices]).cumsum(0).tolist())
    torch.testing.assert_close(values, torch.cat([original["bags"][i] for i in indices]), rtol=0, atol=0)
    torch.testing.assert_close(labels, original["labels"][indices], rtol=0, atol=0)


def test_full_training_schedule_is_shared_and_deterministic():
    first = list(study.training_batches(11))
    second = list(study.training_batches(11))
    other_seed = list(study.training_batches(29))
    assert first == second and first != other_seed
    assert len(first) == 200
    for batch in first:
        assert len(batch) == len(set(batch)) == 16
        assert min(batch) >= 0 and max(batch) < 512
    # Each full pass through the training split must use every bag exactly once.
    for start in range(0, 200 - 31, 32):
        assert sorted(index for batch in first[start:start + 32] for index in batch) == list(range(512))


@pytest.mark.parametrize("pooling", ["native", "reference", "mean", "max", "sum"])
def test_models_use_declared_parameters_and_are_permutation_invariant(pooling, declared_splits):
    model = study.make_model(11, pooling)
    expected_parameters = 113 if pooling in ("native", "reference") else 97
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    split = declared_splits[1]
    indices = [0, 1, 2, 3]
    values, offsets, labels = study.pack_batch(split, indices)
    reordered = torch.cat([split["bags"][index].flip(0) for index in indices]).contiguous()
    with torch.no_grad():
        actual = model(values, offsets)
        permuted = model(reordered, offsets)
    assert actual.shape == labels.shape and bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, permuted, rtol=2e-5, atol=2e-6)


def test_native_and_reference_models_share_initialization_and_optimizer_steps(declared_splits):
    native = study.make_model(11, "native")
    reference = study.make_model(11, "reference")
    assert native.state_dict().keys() == reference.state_dict().keys()
    initial = {name: parameter.detach().clone() for name, parameter in native.named_parameters()}
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.003) for model in (native, reference)]
    for indices in list(study.training_batches(11))[:3]:
        values, offsets, labels = study.pack_batch(declared_splits[0], indices)
        inputs = [values.detach().clone().requires_grad_() for _ in range(2)]
        predictions = []
        for model, optimizer, features in zip((native, reference), optimizers, inputs):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, offsets)
            torch.nn.functional.binary_cross_entropy_with_logits(prediction, labels).backward()
            predictions.append(prediction)
        torch.testing.assert_close(predictions[0], predictions[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=2e-5, atol=2e-6)
        for (name, parameter), (_, expected) in zip(native.named_parameters(), reference.named_parameters()):
            assert parameter.grad is not None and expected.grad is not None, name
            assert bool(torch.isfinite(parameter.grad).all()), name
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-5, atol=2e-6)
        for optimizer in optimizers:
            optimizer.step()
        for parameter, expected in zip(native.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=2e-5, atol=2e-6)
    changed = [name for name, parameter in native.named_parameters() if not torch.equal(initial[name], parameter)]
    assert any("scorer" in name for name in changed)
    assert any("encoder" in name for name in changed)
    assert any("head" in name for name in changed)


def test_training_selects_the_earliest_best_validation_state(monkeypatch):
    training = study.make_split(822, 16, 1, 1)
    validation = study.make_split(823, 16, 1, 1)
    evaluated_states = []
    validation_losses = iter((0.8, 0.4, 0.4, 0.4))

    def controlled_validation(model, split, reverse=False):
        assert split is validation and not reverse
        evaluated_states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        return {"bce": next(validation_losses), "accuracy": 0.5, "bags": 16}, torch.zeros(16)

    class RecordingBudget:
        def __init__(self):
            self.locations = []

        def check(self, location):
            self.locations.append(location)

    monkeypatch.setattr(study, "_evaluate", controlled_validation)
    budget = RecordingBudget()
    selected, final_state, result = study._fit(training, validation, 11, "mean", 60, budget)
    assert result["updates_completed"] == 60 and len(result["training_losses"]) == 60
    assert [row["update"] for row in result["validation_history"]] == [20, 40, 60]
    assert result["selected_update"] == 40
    assert len(evaluated_states) == 4 and len(budget.locations) == 3
    for name, value in selected.state_dict().items():
        torch.testing.assert_close(value, evaluated_states[1][name], rtol=0, atol=0)
        torch.testing.assert_close(value, evaluated_states[3][name], rtol=0, atol=0)
        torch.testing.assert_close(final_state[name], evaluated_states[2][name], rtol=0, atol=0)
    assert any(not torch.equal(final_state[name], value) for name, value in selected.state_dict().items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lengths", [[], [0, 0, 0], [1, 0, 2, 4], [6]])
@pytest.mark.parametrize("upstream_layout", ["strided", "expanded"])
def test_builtin_baseline_outputs_and_both_gradients(dtype, lengths, upstream_layout):
    generator = torch.Generator().manual_seed(812)
    count, width = sum(lengths), 3
    offsets = torch.tensor([0] + list(torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()))
    offsets = offsets.to(torch.int64)
    values = torch.randn(count, width, generator=generator, dtype=dtype)
    logits = torch.randn(count, generator=generator, dtype=dtype) * 4
    snapshots = [tensor.clone() for tensor in (values, logits, offsets)]
    versions = [tensor._version for tensor in (values, logits, offsets)]
    if upstream_layout == "strided":
        upstream = torch.randn(len(lengths), width * 2, generator=generator, dtype=dtype)[:, ::2]
    else:
        upstream = torch.tensor([0.25, -0.5, 1.0], dtype=dtype).expand(len(lengths), -1)
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (values, logits)]
    expected = ragged_pool_reference(*expected_inputs, offsets)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, upstream)
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    for pool in (baselines.ragged_pool_builtin, baselines.ragged_pool_builtin_known_valid):
        inputs = [tensor.detach().clone().requires_grad_() for tensor in (values, logits)]
        input_snapshots = [tensor.detach().clone() for tensor in inputs]
        input_versions = [tensor._version for tensor in inputs]
        actual = pool(*inputs, offsets)
        assert actual.shape == (len(lengths), width)
        torch.testing.assert_close(actual, expected, **tolerance)
        gradients = torch.autograd.grad(actual, inputs, upstream)
        for gradient, expected_gradient in zip(gradients, expected_gradients):
            assert bool(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, expected_gradient, **tolerance)
        for tensor, snapshot, version in zip(inputs, input_snapshots, input_versions):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
            assert tensor._version == version
    for tensor, snapshot, version in zip((values, logits, offsets), snapshots, versions):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        assert tensor._version == version


@pytest.mark.parametrize("negative_input", [0, 1, 2])
def test_checked_builtin_preserves_stability_and_contiguous_negative_support(negative_input):
    inputs = [torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=torch.float64),
              torch.tensor([1000.0, 999.0, -1000.0], dtype=torch.float64),
              torch.tensor([0, 0, 3])]
    inputs[negative_input] = torch._neg_view(-inputs[negative_input])
    assert inputs[negative_input].is_neg() and inputs[negative_input].is_contiguous()
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()
    expected_inputs = [tensor.resolve_neg().detach().clone() for tensor in inputs]
    expected_inputs[0].requires_grad_()
    expected_inputs[1].requires_grad_()
    actual = baselines.ragged_pool_builtin(*inputs)
    expected = ragged_pool_reference(*expected_inputs)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    upstream = torch.tensor([[0.5, -1.0], [1.5, 2.0]], dtype=torch.float64)
    actual_gradients = torch.autograd.grad(actual, inputs[:2], upstream)
    expected_gradients = torch.autograd.grad(expected, expected_inputs[:2], upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", ["decreasing", "wrong_end", "nonfinite", "out_of_range", "values_layout", "logits_layout", "offsets_layout"])
def test_checked_builtin_rejects_before_segment_reduction(case, monkeypatch):
    values = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    logits = torch.zeros(4, dtype=torch.float64)
    offsets = torch.tensor([0, 2, 4])
    if case == "decreasing":
        offsets = torch.tensor([0, 3, 2, 4])
    elif case == "wrong_end":
        offsets = torch.tensor([0, 2, 3])
    elif case == "nonfinite":
        logits[0] = float("nan")
    elif case == "out_of_range":
        values[0, 0] = 1001
    elif case == "values_layout":
        values = torch._neg_view(-torch.arange(24, dtype=torch.float64).reshape(8, 3))[::2]
    elif case == "logits_layout":
        logits = torch._neg_view(-torch.zeros(8, dtype=torch.float64))[::2]
    elif case == "offsets_layout":
        offsets = torch._neg_view(-torch.tensor([0, 0, 2, 0, 4, 0]))[::2]

    def must_not_reduce(*args, **kwargs):
        raise AssertionError("Malformed inputs reached segment_reduce")

    monkeypatch.setattr(torch, "segment_reduce", must_not_reduce)
    with pytest.raises(RuntimeError):
        baselines.ragged_pool_builtin(values, logits, offsets)
