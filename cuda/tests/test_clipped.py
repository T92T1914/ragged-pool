"""Check the chosen update against full-batch loss rows and analytic vectors."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def example():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128" and torch.version.cuda == "12.8"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    path = Path(__file__).resolve().parents[1] / "examples/train_clipped.py"
    spec = importlib.util.spec_from_file_location("cuda_clipped_test_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(dtype, *values):
    return {str(index): torch.tensor([value], dtype=dtype, device="cuda:0")
            for index, value in enumerate(values)}


def stamp(tensors):
    return {key: (id(value), value._version, value.untyped_storage()._cdata,
                  value.detach().contiguous().reshape(-1).view(torch.uint8).clone(),
                  value.grad, None if value.grad is None else value.grad.clone())
            for key, value in tensors.items()}


def unchanged(tensors, before):
    for key, value in tensors.items():
        identity, version, storage, data, grad, grad_data = before[key]
        assert (id(value), value._version, value.untyped_storage()._cdata) == (identity, version, storage)
        assert torch.equal(value.detach().contiguous().reshape(-1).view(torch.uint8), data)
        assert value.grad is grad
        if grad is not None:
            assert torch.equal(grad, grad_data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_norm_and_zero_have_independent_analytic_answers(example, dtype):
    source = row(dtype, 1.2, 1.6)
    before = stamp(source)
    clipped, norm, scale = example.clip_row(source, 1.0)
    torch.testing.assert_close(torch.stack(list(clipped.values())).flatten(),
                               torch.tensor([0.6, 0.8], dtype=torch.float64, device="cuda:0"))
    assert float(norm) == pytest.approx(2.0) and float(scale) == pytest.approx(0.5)
    assert all(value.dtype == torch.float64 for value in clipped.values())
    unchanged(source, before)
    zero, norm, scale = example.clip_row(row(dtype, 0, 0), 1.0)
    assert float(norm) == 0 and float(scale) == 1
    assert all(torch.equal(value, torch.zeros_like(value)) for value in zero.values())


def test_opposing_nonzero_rows_do_not_use_the_norm_of_the_mean(example):
    first, n1, s1 = example.clip_row(row(torch.float64, 2), 1.0)
    second, n2, s2 = example.clip_row(row(torch.float64, -3), 1.0)
    assert (float(n1), float(n2), float(s1), float(s2)) == (2, 3, 0.5, 1 / 3)
    assert float((first["0"] + second["0"]) / 2) == 0
    wrong, _, _ = example.clip_row(row(torch.float64, -0.5), 1.0)
    assert float(wrong["0"]) == -0.5


@pytest.mark.parametrize("threshold", [True, False, None, "1", 0, -1, float("nan"), float("inf")])
def test_invalid_threshold_does_not_change_gradient_source(example, threshold):
    source = row(torch.float64, 0.2, -0.7)
    before = stamp(source)
    with pytest.raises((TypeError, ValueError)):
        example.clip_row(source, threshold)
    unchanged(source, before)
    # A refusal must not poison a subsequent valid call.
    _, norm, scale = example.clip_row(source, 2.0)
    assert float(norm) < 2 and float(scale) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_nonfinite_or_overflowing_joint_norm_is_refused(example, value):
    source = row(torch.float64, value, 0.1)
    before = stamp(source)
    with pytest.raises((ValueError, RuntimeError)):
        example.clip_row(source, 1.0)
    unchanged(source, before)


def test_nonzero_squared_norm_underflow_is_refused_instead_of_skipping_clipping(example):
    source = row(torch.float64, 1e-200)
    before = stamp(source)
    with pytest.raises(ValueError, match="underflow"):
        example.clip_row(source, 1e-250)
    unchanged(source, before)


def batch_for(module, dtype, boundaries):
    initial, batches, _ = module.fixture(dtype, torch.device("cuda:0"))
    model = module.functional.SetRegressor().to(device="cuda:0", dtype=module.DTYPES[dtype])
    model.load_state_dict(initial.state_dict())
    source = batches[0]
    count = boundaries[-1]
    batch = {"raw": source["raw"][:count].detach().clone().requires_grad_(),
             "offsets": torch.tensor(boundaries, dtype=torch.int64, device="cuda:0"),
             "boundaries": tuple(boundaries),
             "targets": torch.linspace(-0.6, 0.8, len(boundaries) - 1,
                                        dtype=module.DTYPES[dtype], device="cuda:0")}
    return model, batch


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("boundaries", [(0, 0, 0), (0, 1), (0, 1, 1, 3), (0, 2, 2, 5, 6)])
def test_stream_matches_original_full_batch_loss_rows_with_grad_sentinels(example, dtype, boundaries):
    model, batch = batch_for(example, dtype, boundaries)
    native = example.eager.SetRegressor().to(device="cuda:0", dtype=example.DTYPES[dtype])
    native.load_state_dict(model.state_dict())
    predictions, pools = native(batch["raw"], batch["offsets"])
    losses = (predictions - batch["targets"]).square()
    names = tuple(dict(native.named_parameters()))
    expected = []
    for index in range(len(losses)):
        gradients = torch.autograd.grad(losses[index], tuple(native.parameters()),
                                        retain_graph=index + 1 < len(losses))
        expected.append(dict(zip(names, gradients)))
    parameters = dict(model.named_parameters())
    for parameter in parameters.values():
        parameter.grad = torch.full_like(parameter, 0.37)
    batch["raw"].grad = torch.full_like(batch["raw"], -0.21)
    tensors = {**parameters, **{key: batch[key] for key in ("raw", "offsets", "targets")}}
    before = stamp(tensors)
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    observed = []

    def observer(index, gradients, norm, scale, prediction, pooled, loss):
        assert index == len(observed)
        assert tuple(gradients) == names
        for name in names:
            torch.testing.assert_close(gradients[name], expected[index][name], **tolerance)
        torch.testing.assert_close(prediction.reshape(()), predictions[index], **tolerance)
        torch.testing.assert_close(pooled.reshape(-1), pools[index], **tolerance)
        torch.testing.assert_close(loss.reshape(()), losses[index], **tolerance)
        observed.append(index)

    means, diagnostics = example.stream_gradients(model, batch, 0.7, observer=observer)
    scales = []
    for gradients in expected:
        flat = torch.cat([gradients[name].detach().double().flatten() for name in names])
        norm = torch.linalg.vector_norm(flat)
        scales.append(1.0 if float(norm) == 0 else min(1.0, 0.7 / float(norm)))
    for name in names:
        wanted = torch.stack([gradients[name].double() * scale
                              for gradients, scale in zip(expected, scales)]).mean(0).to(parameters[name].dtype)
        torch.testing.assert_close(means[name], wanted, **tolerance)
        assert means[name].grad_fn is None
    assert observed == list(range(len(losses)))
    assert diagnostics is not None
    for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if start == stop:
            assert torch.equal(pools[index], torch.zeros_like(pools[index]))
            for name in names:
                if name != "head.bias":
                    assert torch.equal(expected[index][name], torch.zeros_like(expected[index][name]))
            assert float(expected[index]["head.bias"].abs().max()) > 0
    unchanged(tensors, before)


@pytest.mark.parametrize("fault", ["zero_groups", "offset_disagreement", "nan_target", "backward_offset"])
def test_invalid_batch_refuses_before_parameter_or_grad_mutation_then_recovers(example, fault):
    model, batch = batch_for(example, "float64", (0, 1, 1, 3))
    valid = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    if fault == "zero_groups":
        batch = {"raw": batch["raw"][:0], "offsets": batch["offsets"][:1],
                 "boundaries": (0,), "targets": batch["targets"][:0]}
    elif fault == "offset_disagreement":
        batch["offsets"][-1] = 2
    elif fault == "nan_target":
        batch["targets"][0] = float("nan")
    else:
        batch["boundaries"] = (0, 2, 1, 3)
        batch["offsets"] = torch.tensor(batch["boundaries"], dtype=torch.int64, device="cuda:0")
    tensors = dict(model.named_parameters())
    for value in tensors.values():
        value.grad = torch.ones_like(value)
    before = stamp(tensors)
    with pytest.raises((ValueError, RuntimeError)):
        example.stream_gradients(model, batch, 1.0)
    unchanged(tensors, before)
    result, _ = example.stream_gradients(model, valid, 1.0)
    assert all(bool(torch.isfinite(value).all()) for value in result.values())
    unchanged(tensors, before)
