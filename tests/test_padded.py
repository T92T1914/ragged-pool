"""Padded admission, independent derivatives and ignored-padding behavior."""
import math

import pytest
import torch

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


POOLS = (ragged_pool_padded, ragged_pool_padded_reference)


def tolerance(dtype):
    return {'rtol': 2e-5, 'atol': 2e-6} if dtype == torch.float32 else {'rtol': 1e-10, 'atol': 1e-12}


def inputs(dtype=torch.float64, case='holes'):
    generator = torch.Generator().manual_seed(529)
    if case == 'empty_batch':
        return torch.empty(0, 4, 3, dtype=dtype), torch.empty(0, 4, dtype=dtype), torch.empty(0, 4, dtype=torch.bool)
    if case == 'zero_steps':
        return torch.empty(4, 0, 3, dtype=dtype), torch.empty(4, 0, dtype=dtype), torch.empty(4, 0, dtype=torch.bool)
    values = torch.randn(4, 5, 3, generator=generator, dtype=dtype)
    logits = torch.randn(4, 5, generator=generator, dtype=dtype)
    mask = torch.tensor([[True, False, True, False, True], [False] * 5,
                         [False, True, False, False, False], [True, True, False, True, False]])
    if case == 'all':
        mask.fill_(True)
    elif case == 'none':
        mask.fill_(False)
    elif case == 'singletons':
        mask.fill_(False)
        mask[:, 2] = True
    return values, logits, mask


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['holes', 'all', 'none', 'singletons', 'empty_batch', 'zero_steps'])
def test_padded_outputs_and_both_derivatives_match_independent_rows(dtype, case):
    values, logits, mask = inputs(dtype, case)
    actual_inputs = [values.clone().requires_grad_(), logits.clone().requires_grad_()]
    reference_inputs = [values.clone().requires_grad_(), logits.clone().requires_grad_()]
    actual = ragged_pool_padded(*actual_inputs, mask)
    expected = ragged_pool_padded_reference(*reference_inputs, mask)
    upstream = torch.randn(actual.shape, generator=torch.Generator().manual_seed(719), dtype=dtype)
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)
    reference_grads = torch.autograd.grad(expected, reference_inputs, upstream)
    assert actual.shape == (values.shape[0], values.shape[2])
    assert actual.dtype == dtype and actual.device.type == 'cpu'
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    for gradient, reference_gradient in zip(actual_grads, reference_grads):
        torch.testing.assert_close(gradient, reference_gradient, **tolerance(dtype))
        assert bool(torch.isfinite(gradient).all())
        assert torch.count_nonzero(gradient[~mask]) == 0
    assert torch.count_nonzero(actual[~mask.any(dim=1)]) == 0
    if case == 'singletons':
        torch.testing.assert_close(actual, values[:, 2], **tolerance(dtype))
        assert torch.count_nonzero(actual_grads[1]) == 0
    assert mask.grad is None


@pytest.mark.parametrize('pool', POOLS)
def test_known_weights_holes_and_empty_row_have_hand_calculated_gradients(pool):
    values = torch.tensor([[[1., 2.], [99., 99.], [3., -2.]],
                           [[17., 29.], [31., 43.], [47., 53.]],
                           [[7., 5.], [99., 99.], [99., 99.]]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[0., 999., math.log(3)], [2., 3., 4.], [0., 9., 8.]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False], [True, False, False]])
    pooled = pool(values, logits, mask)
    gradients = torch.autograd.grad(pooled, (values, logits), torch.tensor([[2., -3.], [4., 8.], [-1., 2.]], dtype=torch.float64))
    torch.testing.assert_close(pooled, torch.tensor([[2.5, -1.], [0., 0.], [7., 5.]], dtype=torch.float64))
    expected_values = torch.zeros_like(values)
    expected_values[0, 0] = torch.tensor([0.5, -0.75])
    expected_values[0, 2] = torch.tensor([1.5, -2.25])
    expected_values[2, 0] = torch.tensor([-1., 2.])
    expected_logits = torch.zeros_like(logits)
    expected_logits[0, 0], expected_logits[0, 2] = -3., 3.
    torch.testing.assert_close(gradients[0], expected_values)
    torch.testing.assert_close(gradients[1], expected_logits)


@pytest.mark.parametrize('pool', POOLS)
@pytest.mark.parametrize('case', ['holes', 'none'])
def test_nonfinite_padding_is_ignored_before_math_and_inputs_are_unchanged(pool, case):
    values, logits, mask = inputs(case=case)
    values[~mask] = float('nan')
    logits[~mask] = float('inf')
    values.requires_grad_()
    logits.requires_grad_()
    snapshots = (values.detach().clone(), logits.detach().clone(), mask.clone())
    versions = (values._version, logits._version, mask._version)
    output = pool(values, logits, mask)
    gradients = torch.autograd.grad(output.sum(), (values, logits))
    assert bool(torch.isfinite(output).all())
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert torch.count_nonzero(gradient[~mask]) == 0
    for original, snapshot, version in zip((values, logits, mask), snapshots, versions):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0, equal_nan=True)
        assert original._version == version


@pytest.mark.parametrize('pool', POOLS)
@pytest.mark.parametrize('input_index', [0, 1])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 1000.0001, -1000.0001])
def test_invalid_selected_values_are_rejected_while_the_same_padding_is_ignored(pool, input_index, bad):
    arguments = list(inputs())
    arguments[input_index][0, 1] = bad
    assert bool(torch.isfinite(pool(*arguments)).all())
    arguments[input_index][0, 0] = bad
    with pytest.raises(RuntimeError, match='finite.*1000'):
        pool(*arguments)


def malformed(case):
    values, logits, mask = inputs()
    if case == 'values_rank':
        values = values.flatten(0, 1)
    elif case == 'logits_rank':
        logits = logits.unsqueeze(-1)
    elif case == 'mask_rank':
        mask = mask.unsqueeze(-1)
    elif case == 'logits_shape':
        logits = logits[:3]
    elif case == 'mask_shape':
        mask = mask[:3]
    elif case == 'mask_dtype':
        mask = mask.to(torch.int64)
    elif case == 'mixed_dtype':
        logits = logits.float()
    elif case == 'half':
        values, logits = values.half(), logits.half()
    elif case == 'integer':
        values, logits = values.long(), logits.long()
    elif case == 'B':
        values, logits, mask = torch.zeros(129, 1, 1), torch.zeros(129, 1), torch.ones(129, 1, dtype=torch.bool)
    elif case == 'T':
        values, logits, mask = torch.zeros(0, 4097, 1), torch.zeros(0, 4097), torch.ones(0, 4097, dtype=torch.bool)
    elif case == 'slots':
        values, logits, mask = torch.zeros(2, 2049, 1), torch.zeros(2, 2049), torch.zeros(2, 2049, dtype=torch.bool)
    elif case == 'D_zero':
        values = values[:, :, :0]
    elif case == 'D':
        values = torch.zeros(4, 5, 129, dtype=values.dtype)
    elif case.endswith('_layout'):
        index = {'values_layout': 0, 'logits_layout': 1, 'mask_layout': 2}[case]
        arguments = [values, logits, mask]
        shape = list(arguments[index].shape)
        shape[-1] *= 2
        arguments[index] = torch.zeros(shape, dtype=arguments[index].dtype)[..., ::2]
        return arguments
    elif case.endswith('_sparse'):
        index = {'values_sparse': 0, 'logits_sparse': 1, 'mask_sparse': 2}[case]
        arguments = [values, logits, mask]
        arguments[index] = arguments[index].to_sparse()
        return arguments
    else:
        raise AssertionError(case)
    return values, logits, mask


@pytest.mark.parametrize('pool', POOLS)
@pytest.mark.parametrize('case', ['values_rank', 'logits_rank', 'mask_rank', 'logits_shape', 'mask_shape',
                                  'mask_dtype', 'mixed_dtype', 'half', 'integer', 'B', 'T', 'slots',
                                  'D_zero', 'D', 'values_layout', 'logits_layout', 'mask_layout',
                                  'values_sparse', 'logits_sparse', 'mask_sparse'])
def test_rejects_original_metadata_before_selection_can_hide_it(pool, case):
    with pytest.raises(RuntimeError):
        pool(*malformed(case))


@pytest.mark.parametrize('pool', POOLS)
@pytest.mark.parametrize('input_index', [0, 1, 2])
def test_other_devices_are_rejected_without_a_transfer(pool, input_index):
    arguments = list(inputs())
    arguments[input_index] = torch.empty_like(arguments[input_index], device='meta')
    with pytest.raises(RuntimeError, match='CPU'):
        pool(*arguments)
    assert arguments[input_index].device.type == 'meta'


@pytest.mark.parametrize('pool', POOLS)
@pytest.mark.parametrize('input_index', [0, 1])
def test_logical_negative_views_preserve_values_and_original_layout_admission(pool, input_index):
    arguments = list(inputs())
    expected = pool(*arguments)
    arguments[input_index] = torch._neg_view(-arguments[input_index])
    assert arguments[input_index].is_neg() and arguments[input_index].is_contiguous()
    torch.testing.assert_close(pool(*arguments), expected, **tolerance(arguments[0].dtype))
    bad = list(malformed(('values_layout', 'logits_layout')[input_index]))
    bad[input_index] = torch._neg_view(bad[input_index])
    assert bad[input_index].is_neg() and not bad[input_index].is_contiguous()
    with pytest.raises(RuntimeError, match='contiguous'):
        pool(*bad)


def test_native_adapter_compacts_both_inputs_in_row_order_and_calls_once(monkeypatch):
    values, logits, mask = inputs()
    original = torch.ops.ragged_pool.pool.default
    observed = []

    def record(packed_values, packed_logits, offsets):
        observed.append((packed_values.detach().clone(), packed_logits.detach().clone(), offsets.clone()))
        return original(packed_values, packed_logits, offsets)

    monkeypatch.setattr(torch.ops.ragged_pool.pool, 'default', record)
    ragged_pool_padded(values, logits, mask)
    assert len(observed) == 1
    torch.testing.assert_close(observed[0][0], torch.cat([values[row][mask[row]] for row in range(4)]))
    torch.testing.assert_close(observed[0][1], torch.cat([logits[row][mask[row]] for row in range(4)]))
    assert observed[0][2].tolist() == [0, 3, 3, 4, 7]


def test_reference_does_not_call_native_operation(monkeypatch):
    def must_not_execute(*args, **kwargs):
        raise AssertionError('The independent row reference reached the native operation')

    monkeypatch.setattr(torch.ops.ragged_pool.pool, 'default', must_not_execute)
    output = ragged_pool_padded_reference(*inputs())
    assert output.shape == (4, 3) and bool(torch.isfinite(output).all())


@pytest.mark.parametrize('layout', ['expanded', 'transposed', 'sliced', 'negative'])
def test_upstream_views_propagate_through_compaction_and_scatter(layout):
    values, logits, mask = inputs()
    left = [values.clone().requires_grad_(), logits.clone().requires_grad_()]
    right = [values.clone().requires_grad_(), logits.clone().requires_grad_()]
    actual = ragged_pool_padded(*left, mask)
    expected = ragged_pool_padded_reference(*right, mask)
    if layout == 'expanded':
        upstream = torch.tensor([[1.5]], dtype=values.dtype).expand(4, 3)
        assert upstream.stride() == (0, 0)
    elif layout == 'transposed':
        upstream = torch.arange(12, dtype=values.dtype).reshape(3, 4).t()
    elif layout == 'sliced':
        upstream = torch.arange(24, dtype=values.dtype).reshape(4, 6)[:, ::2]
    else:
        upstream = torch._neg_view(torch.arange(12, dtype=values.dtype).reshape(4, 3))
    actual_grads = torch.autograd.grad(actual, left, upstream)
    expected_grads = torch.autograd.grad(expected, right, upstream)
    for first, second in zip(actual_grads, expected_grads):
        torch.testing.assert_close(first, second, **tolerance(values.dtype))
        assert torch.count_nonzero(first[~mask]) == 0


@pytest.mark.parametrize('pool', POOLS)
def test_item_permutation_preserves_outputs_and_reorders_input_gradients(pool):
    values, logits, mask = inputs()
    order = torch.tensor([4, 1, 3, 0, 2])
    original = [values.clone().requires_grad_(), logits.clone().requires_grad_()]
    permuted = [values[:, order].clone().requires_grad_(), logits[:, order].clone().requires_grad_()]
    before = pool(*original, mask)
    after = pool(*permuted, mask[:, order].contiguous())
    upstream = torch.arange(12, dtype=values.dtype).reshape(4, 3)
    before_grads = torch.autograd.grad(before, original, upstream)
    after_grads = torch.autograd.grad(after, permuted, upstream)
    torch.testing.assert_close(before, after, **tolerance(values.dtype))
    for first, second in zip(before_grads, after_grads):
        torch.testing.assert_close(first[:, order], second, **tolerance(values.dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_maximum_storage_width_and_stable_endpoint_logits(dtype):
    values = torch.linspace(-1000, 1000, 4096 * 128, dtype=dtype).reshape(128, 32, 128)
    logits = torch.zeros(128, 32, dtype=dtype)
    logits[:, 0], logits[:, 1] = 1000., 999.
    mask = torch.ones(128, 32, dtype=torch.bool)
    actual = ragged_pool_padded(values, logits, mask)
    expected = ragged_pool_padded_reference(values, logits, mask)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    assert actual.shape == (128, 128) and bool(torch.isfinite(actual).all())
    one_long = ragged_pool_padded(values.reshape(1, 4096, 128), logits.reshape(1, 4096), mask.reshape(1, 4096))
    long_reference = ragged_pool_padded_reference(values.reshape(1, 4096, 128), logits.reshape(1, 4096), mask.reshape(1, 4096))
    torch.testing.assert_close(one_long, long_reference, **tolerance(dtype))
    assert one_long.shape == (1, 128) and bool(torch.isfinite(one_long).all())


def test_float64_finite_difference_gradcheck_with_holes_and_empty_row():
    values = torch.tensor([[[0.2, -0.7], [1., 2.], [0.4, 0.9]],
                           [[0.5, 0.1], [0.3, 0.2], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[0.3, -0.1, 0.7], [0.6, 0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    assert torch.autograd.gradcheck(lambda v, l: ragged_pool_padded(v, l, mask),
                                    (values, logits), eps=1e-6, atol=1e-5, rtol=1e-4,
                                    check_undefined_grad=True)


@pytest.mark.parametrize('pool', POOLS)
def test_output_has_fresh_storage_and_never_aliases_inputs(pool):
    arguments = inputs()
    first, second = pool(*arguments), pool(*arguments)
    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() not in {tensor.data_ptr() for tensor in arguments}
    first.add_(100)
    torch.testing.assert_close(second, pool(*arguments), **tolerance(arguments[0].dtype))
