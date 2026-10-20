"""Padded second derivatives against independent math and finite differences."""
import pytest
import torch

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


def inputs(dtype=torch.float64, case='mixed'):
    generator = torch.Generator().manual_seed(937)
    masks = {
        'mixed': [[True, False, True, False], [False]*4, [False, True, False, False]],
        'all_true': [[True]*4]*3,
        'singletons': [[True, False, False, False]]*3,
        'all_false': [[False]*4]*3,
    }
    mask = (torch.empty((0, 4), dtype=torch.bool) if case == 'zero_batch' else
            torch.empty((3, 0), dtype=torch.bool) if case == 'zero_steps' else
            torch.tensor(masks[case]))
    values = torch.randn((*mask.shape, 2), generator=generator, dtype=dtype)
    logits = torch.randn(mask.shape, generator=generator, dtype=dtype)
    g = torch.randn(mask.shape[0], 2, generator=generator, dtype=dtype)
    a = torch.randn(values.shape, generator=generator, dtype=dtype)
    b = torch.randn(logits.shape, generator=generator, dtype=dtype)
    return a, b, g, values, logits, mask


def second_vjp(pool, a, b, g, values, logits, mask, branch='both'):
    upstream, v, l = variables = tuple(t.detach().clone().requires_grad_() for t in (g, values, logits))
    first = torch.autograd.grad(pool(v, l, mask), (v, l), upstream, create_graph=True)
    # Select before arithmetic, including in the independent empty-path oracle.
    objective = (first[0][mask] * a[mask]).sum() if branch == 'values' else (first[1][mask] * b[mask]).sum()
    if branch == 'both':
        objective = (first[0][mask] * a[mask]).sum() + (first[1][mask] * b[mask]).sum()
    objective = objective + upstream.sum()*0 + v[mask].sum()*0 + l[mask].sum()*0
    result = torch.autograd.grad(objective, variables, allow_unused=True)
    return tuple(torch.zeros_like(t) if grad is None else grad for t, grad in zip(variables, result))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['mixed', 'all_true', 'singletons', 'all_false', 'zero_batch', 'zero_steps'])
@pytest.mark.parametrize('branch', ['both', 'values', 'logits'])
def test_all_second_vjp_outputs_match_independent_rows(dtype, case, branch):
    arguments = inputs(dtype, case)
    actual = second_vjp(ragged_pool_padded, *arguments, branch)
    expected = second_vjp(ragged_pool_padded_reference, *arguments, branch)
    tolerance = {'rtol': 4e-5, 'atol': 4e-6} if dtype == torch.float32 else {'rtol': 1e-9, 'atol': 1e-11}
    mask = arguments[-1]
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, **tolerance)
        assert bool(torch.isfinite(first).all())
    assert torch.count_nonzero(actual[0][~mask.any(dim=1)]) == 0
    assert torch.count_nonzero(actual[1][~mask]) == torch.count_nonzero(actual[2][~mask]) == 0
    if case == 'singletons':
        assert torch.count_nonzero(actual[1]) == torch.count_nonzero(actual[2]) == 0
    assert mask.grad is None


def test_float64_gradgradcheck_includes_output_cotangent():
    _, _, g, values, logits, mask = inputs()
    assert torch.autograd.gradgradcheck(
        lambda v, l: ragged_pool_padded(v, l, mask),
        (values.requires_grad_(), logits.requires_grad_()), grad_outputs=g.requires_grad_(),
        eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_two_selected_items_have_known_head_penalty_and_derivative():
    values = torch.tensor([[[1.], [99.], [3.], [99.]]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[0., 0., 1.0986122886681098, 0.]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True, False]])
    head = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    prediction = head * ragged_pool_padded(values, logits, mask).sum()
    gradient, = torch.autograd.grad(prediction, (values,), create_graph=True)
    penalty = 0.5 * gradient[mask].square().mean()
    derivative, = torch.autograd.grad(penalty, (head,))
    torch.testing.assert_close(penalty, penalty.new_tensor(0.625), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(derivative, derivative.new_tensor(0.625), rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(gradient[~mask]) == 0


def view(tensor, layout):
    if layout == 'expanded':
        return tensor.new_tensor(0.7).expand(tensor.shape)
    if layout == 'negative':
        return torch._neg_view(tensor)
    if layout in ('sliced', 'negative_sliced'):
        storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        storage[..., 0] = tensor
        result = storage[..., 0]
        return torch._neg_view(result) if layout == 'negative_sliced' else result
    return tensor.transpose(0, 1).contiguous().transpose(0, 1)


@pytest.mark.parametrize('layout', ['expanded', 'sliced', 'transposed', 'negative', 'negative_sliced'])
def test_private_second_backward_accepts_all_three_upstream_views(layout):
    a, b, g, values, logits, mask = inputs()
    a, b, g = (view(t, layout) for t in (a, b, g))
    arguments = (a, b, g, values, logits, mask)
    snapshots = [(t.clone(), t._version, t.is_neg(), t.stride()) for t in arguments]
    actual = torch.ops.ragged_pool._padded_double_backward.default(*arguments)
    expected = second_vjp(ragged_pool_padded_reference, *arguments)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-11)
        assert first.is_contiguous()
    for tensor, (snapshot, version, negative, stride) in zip(arguments, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        assert tensor._version == version and tensor.is_neg() == negative and tensor.stride() == stride


def test_ignored_nonfinite_inputs_and_second_upstreams_are_selected_out():
    arguments = list(inputs())
    a, b, g, values, logits, mask = arguments
    a[~mask], b[~mask], values[~mask], logits[~mask] = float('nan'), float('inf'), float('nan'), -float('inf')
    snapshots = [(t.clone(), t._version) for t in arguments]
    actual = torch.ops.ragged_pool._padded_double_backward.default(*arguments)
    expected = second_vjp(ragged_pool_padded_reference, *arguments)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-11)
        assert bool(torch.isfinite(first).all())
    assert torch.count_nonzero(actual[1][~mask]) == torch.count_nonzero(actual[2][~mask]) == 0
    for tensor, (snapshot, version) in zip(arguments, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)
        assert tensor._version == version


@pytest.mark.parametrize('changed', [0, 1, 2, 3])
def test_original_g_values_logits_mask_versions_survive_until_second_backward(changed):
    a, b, g, values, logits, mask = inputs()
    g, values, logits = (t.requires_grad_() for t in (g, values, logits))
    first = torch.autograd.grad(ragged_pool_padded(values, logits, mask), (values, logits), g, create_graph=True)
    originals = (g, values, logits, mask)
    with torch.no_grad():
        # For padded originals, mutate an ignored slot rather than a selected one.
        if changed == 0:
            g[0, 0] += 1
        else:
            originals[changed][0, 1] = True if changed == 3 else 12.
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        torch.autograd.grad((first[0]*a).sum() + (first[1]*b).sum(), (g, values, logits))


def test_third_derivative_is_explicitly_refused():
    a, b, g, values, logits, mask = inputs()
    g, values, logits = (t.requires_grad_() for t in (g, values, logits))
    first = torch.autograd.grad(ragged_pool_padded(values, logits, mask), (values, logits), g, create_graph=True)
    second = torch.autograd.grad((first[0]*a).sum() + (first[1]*b).sum(), (g, values, logits), create_graph=True)
    with pytest.raises(RuntimeError, match='third order is not supported'):
        torch.autograd.grad(sum(t.sum() for t in second), (g, values, logits))


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('defect', ['dtype', 'shape', 'sparse'])
def test_private_second_backward_refuses_bad_second_upstream_metadata(index, defect):
    arguments = list(inputs())
    tensor = arguments[index]
    arguments[index] = tensor.float() if defect == 'dtype' else tensor[:, :2] if defect == 'shape' else tensor.to_sparse()
    # Sparse inputs are refused by dispatcher selection before the CPU kernel.
    expected = 'SparseCPU' if defect == 'sparse' else 'second upstream'
    with pytest.raises(RuntimeError, match=expected):
        torch.ops.ragged_pool._padded_double_backward.default(*arguments)


@pytest.mark.parametrize('index', [3, 4])
def test_private_second_backward_retains_native_selected_finite_admission(index):
    arguments = list(inputs())
    if index in (0, 3):
        arguments[index][0, 0, 0] = float('nan')
    else:
        arguments[index][0, 0] = float('inf')
    with pytest.raises(RuntimeError, match='finite'):
        torch.ops.ragged_pool._padded_double_backward.default(*arguments)


def test_largest_padded_envelope_with_endpoint_inputs_has_finite_second_vjp():
    values = torch.linspace(-1000, 1000, 4096*128, dtype=torch.float64).reshape(128, 32, 128)
    logits = torch.full((128, 32), -1000., dtype=values.dtype)
    logits[:, ::4], logits[:, 1::4] = 1000., 999.
    mask = torch.ones((128, 32), dtype=torch.bool)
    mask[:, 3::4] = False
    values[~mask], logits[~mask] = float('nan'), float('inf')
    a = torch.full_like(values, 0.25)
    b = torch.linspace(-0.5, 0.5, 4096, dtype=values.dtype).reshape(128, 32)
    g = torch.linspace(-0.1, 0.1, 128*128, dtype=values.dtype).reshape(128, 128)
    actual = second_vjp(ragged_pool_padded, a, b, g, values, logits, mask)
    expected = second_vjp(ragged_pool_padded_reference, a, b, g, values, logits, mask)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-8)
        assert bool(torch.isfinite(first).all())
