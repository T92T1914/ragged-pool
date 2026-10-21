"""Per-example derivatives and independent clipping/reduction witnesses."""
import importlib.util
from pathlib import Path

import pytest
import torch

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


_spec = importlib.util.spec_from_file_location(
    'ragged_pool_clipped_example', Path(__file__).resolve().parents[1] / 'examples/train_clipped.py',
)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_clipping_uses_one_norm_across_parameter_blocks_before_averaging(dtype):
    # First example's joint vector is (3,4), second is zero. For C=1 the
    # clipped mean is (0.3,0.4). Clipping the mean gives (0.6,0.8), and
    # clipping each parameter separately gives (0.5,0.5).
    gradients = {'left': torch.tensor([[3.], [0.]], dtype=dtype),
                 'right': torch.tensor([[4.], [0.]], dtype=dtype)}
    before = {key: value.clone() for key, value in gradients.items()}
    mean, norms, scales = _example.clip_per_example(gradients, 1.)
    torch.testing.assert_close(norms, torch.tensor([5., 0.], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(scales, torch.tensor([.2, 1.], dtype=torch.float64), rtol=0, atol=0)
    for name, expected, wrong_aggregate in (('left', .3, .6), ('right', .4, .8)):
        torch.testing.assert_close(mean[name], torch.tensor([expected], dtype=dtype))
        assert not torch.allclose(mean[name], torch.tensor([wrong_aggregate], dtype=dtype))
        assert not torch.allclose(mean[name], torch.tensor([.5], dtype=dtype))
        assert torch.equal(gradients[name], before[name])
        assert bool(torch.isfinite(mean[name]).all())


@pytest.mark.parametrize('threshold', [0., -1., float('inf'), float('nan')])
def test_clipping_rejects_a_nonpositive_or_nonfinite_threshold(threshold):
    with pytest.raises(ValueError):
        _example.clip_per_example({'weight': torch.ones((2, 3), dtype=torch.float64)}, threshold)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_batched_parameter_rows_match_serial_reference_mean_loss_and_empty_head_bias(dtype_name):
    dtype = getattr(torch, dtype_name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = _example.PaddedRegressor().to(dtype)
    layout = _example.make_layouts(dtype_name)['fixed_holes']
    raw, mask = layout['raw'], layout['mask']
    targets = raw.new_tensor([-.3, 8., -.5, .25, -4.])
    originals = [(tensor.view(torch.uint8).clone(), tensor._version) for tensor in (raw, mask)]
    parameters = dict(model.named_parameters())
    # A caller may still have a previous optimizer step's .grad fields. The
    # VJP helper returns gradients and must preserve both existing and None.
    first_parameter = next(iter(parameters.values()))
    first_parameter.grad = torch.full_like(first_parameter, .125)
    grad_before = {name: None if value.grad is None else value.grad.clone() for name, value in parameters.items()}
    parameter_before = {name: value.detach().clone() for name, value in parameters.items()}
    actual = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded)
    serial = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded, method='serial')
    reference = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded_reference, method='serial')
    predictions, pooled = model(raw, mask, ragged_pool_padded)
    mean_gradients = torch.autograd.grad((predictions-targets).square().mean(), tuple(parameters.values()))
    tolerance = dict(rtol=4e-5, atol=4e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-11)
    assert actual.keys() == serial.keys() == reference.keys() == parameters.keys()
    empty, = (~mask.any(dim=1)).nonzero().flatten().tolist()
    assert torch.count_nonzero(pooled[empty]) == 0
    for (name, parameter), mean_gradient in zip(parameters.items(), mean_gradients):
        assert actual[name].shape == (len(targets), *parameter.shape)
        torch.testing.assert_close(actual[name], serial[name], **tolerance)
        torch.testing.assert_close(actual[name], reference[name], **tolerance)
        torch.testing.assert_close(actual[name].mean(dim=0), mean_gradient, **tolerance)
        if grad_before[name] is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, grad_before[name])
        assert torch.equal(parameter.detach(), parameter_before[name])
        if name != 'head.bias':
            assert torch.count_nonzero(actual[name][empty]) == 0
    torch.testing.assert_close(actual['head.bias'][empty],
                               (2 * (model.head.bias.detach()-targets[empty])), **tolerance)
    assert bool(actual['head.bias'][empty].abs().sum() > 0)
    for tensor, (before, version) in zip((raw, mask), originals):
        assert tensor._version == version and torch.equal(tensor.view(torch.uint8), before)


def test_consumer_rejects_zero_examples_without_narrowing_the_pooling_api():
    model = _example.PaddedRegressor().double()
    raw = torch.empty((0, 3, 3), dtype=torch.float64)
    mask = torch.empty((0, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        _example.per_example_gradients(model, raw, mask, torch.empty((0,), dtype=raw.dtype), ragged_pool_padded)
    assert ragged_pool_padded(raw, torch.empty((0, 3), dtype=raw.dtype), mask).shape == (0, 3)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_installed_consumer_takes_two_clipped_updates_with_controls_and_padding_invariance(dtype_name):
    receipt = _example.run(dtype_name)
    assert receipt['status'] == 'passed' and receipt['dtype'] == dtype_name
    assert receipt['method'] == 'autograd.grad.is_grads_batched'
    assert receipt['optimizer_steps'] == receipt['reference_optimizer_steps'] == 2
    assert receipt['controls']['status'] == receipt['invariance']['status'] == receipt['empty']['status'] == 'passed'
