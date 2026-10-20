"""Independent second derivatives, finite differences and separate registration."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from ragged_pool import ragged_pool, ragged_pool_reference


def inputs(dtype=torch.float64, case='mixed'):
    generator = torch.Generator().manual_seed(731)
    lengths = {'mixed': [2, 0, 1, 3], 'singletons': [1, 1, 1], 'empty': [0, 0], 'zero_batch': []}[case]
    offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    values = torch.randn(sum(lengths), 2, generator=generator, dtype=dtype)
    logits = torch.randn(sum(lengths), generator=generator, dtype=dtype)
    g = torch.randn(len(lengths), 2, generator=generator, dtype=dtype)
    a = torch.randn(values.shape, generator=generator, dtype=dtype)
    b = torch.randn(logits.shape, generator=generator, dtype=dtype)
    return a, b, g, values, logits, offsets


def second_vjp(pool, a, b, g, values, logits, offsets, branch='both'):
    variables = tuple(t.detach().clone().requires_grad_() for t in (g, values, logits))
    upstream, v, l = variables
    output = pool(v, l, offsets)
    first = torch.autograd.grad(output, (v, l), upstream, create_graph=True)
    objective = (first[0] * a).sum() if branch == 'values' else (first[1] * b).sum()
    if branch == 'both':
        objective = (first[0] * a).sum() + (first[1] * b).sum()
    # Empty pure-PyTorch expressions can have no further connection. Add a zero
    # path for comparison only; it does not supply any nonzero derivative.
    objective = objective + sum(t.sum() * 0 for t in variables)
    gradients = torch.autograd.grad(objective, variables, allow_unused=True)
    return tuple(torch.zeros_like(t) if grad is None else grad for t, grad in zip(variables, gradients))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['mixed', 'singletons', 'empty', 'zero_batch'])
@pytest.mark.parametrize('branch', ['both', 'values', 'logits'])
def test_all_three_second_vjp_outputs_match_independent_torch(dtype, case, branch):
    arguments = inputs(dtype, case)
    actual = second_vjp(ragged_pool, *arguments, branch)
    expected = second_vjp(ragged_pool_reference, *arguments, branch)
    tolerance = {'rtol': 4e-5, 'atol': 4e-6} if dtype == torch.float32 else {'rtol': 1e-9, 'atol': 1e-11}
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, **tolerance)
        assert bool(torch.isfinite(left).all())
    if case == 'mixed':
        assert torch.count_nonzero(actual[0][1]) == 0
    if case == 'singletons':
        assert torch.count_nonzero(actual[1]) == torch.count_nonzero(actual[2]) == 0
    assert arguments[-1].grad is None


def test_known_two_item_second_vjp_includes_upstream_gradient_derivative():
    values = torch.tensor([[1.], [3.]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([0., 1.0986122886681098], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[2.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[4.], [-2.]], dtype=torch.float64)
    b = torch.tensor([3., -1.], dtype=torch.float64)
    output = ragged_pool(values, logits, torch.tensor([0, 2]))
    first = torch.autograd.grad(output, (values, logits), g, create_graph=True)
    actual = torch.autograd.grad((first[0]*a).sum() + (first[1]*b).sum(), (g, values, logits))
    for left, right in zip(actual, (torch.tensor([[-2.]], dtype=torch.float64),
                                    torch.tensor([[1.5], [-1.5]], dtype=torch.float64),
                                    torch.tensor([0.75, -0.75], dtype=torch.float64))):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_input_gradient_penalty_has_the_hand_calculated_head_derivative():
    values = torch.tensor([[1.], [3.]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([0., 1.0986122886681098], dtype=torch.float64)
    head = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    prediction = head * ragged_pool(values, logits, torch.tensor([0, 2])).sum()
    input_gradient, = torch.autograd.grad(prediction, (values,), create_graph=True)
    penalty = 0.5 * input_gradient.square().sum()
    head_gradient, = torch.autograd.grad(penalty, (head,))
    torch.testing.assert_close(penalty, torch.tensor(1.25, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(head_gradient, torch.tensor(1.25, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_float64_gradgradcheck_perturbs_both_inputs_and_output_cotangent():
    _, _, g, values, logits, offsets = inputs()
    assert torch.autograd.gradgradcheck(
        lambda v, l: ragged_pool(v, l, offsets),
        (values.requires_grad_(), logits.requires_grad_()), grad_outputs=g.requires_grad_(),
        eps=1e-6, atol=1e-5, rtol=1e-4, gen_non_contig_grad_outputs=True,
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_hessian_vector_product_matches_reference_and_is_symmetric(dtype):
    a, b, g, values, logits, offsets = inputs(dtype)
    native = second_vjp(ragged_pool, a, b, g, values, logits, offsets)[1:]
    expected = second_vjp(ragged_pool_reference, a, b, g, values, logits, offsets)[1:]
    tolerance = {'rtol': 4e-5, 'atol': 4e-6} if dtype == torch.float32 else {'rtol': 1e-9, 'atol': 1e-11}
    for actual, reference in zip(native, expected):
        torch.testing.assert_close(actual, reference, **tolerance)
    c, d = a.flip(0).contiguous(), b.flip(0).contiguous()
    other = second_vjp(ragged_pool, c, d, g, values, logits, offsets)[1:]
    torch.testing.assert_close((c*native[0]).sum() + (d*native[1]).sum(),
                               (a*other[0]).sum() + (b*other[1]).sum(), **tolerance)


def test_largest_declared_native_shape_and_endpoint_inputs_have_finite_second_vjp():
    values = torch.linspace(-1000, 1000, 4096*128, dtype=torch.float64).reshape(4096, 128)
    logits = torch.full((4096,), -1000., dtype=torch.float64)
    logits[::32], logits[1::32] = 1000., 999.
    offsets = torch.arange(0, 4097, 32)
    a = torch.full_like(values, 0.25)
    b = torch.linspace(-0.5, 0.5, 4096, dtype=values.dtype)
    g = torch.linspace(-0.1, 0.1, 128*128, dtype=values.dtype).reshape(128, 128)
    actual = second_vjp(ragged_pool, a, b, g, values, logits, offsets)
    expected = second_vjp(ragged_pool_reference, a, b, g, values, logits, offsets)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-8)
        assert bool(torch.isfinite(left).all())


def view(tensor, layout):
    if layout == 'expanded':
        return tensor.new_tensor(0.7).expand(tensor.shape)
    if layout == 'negative':
        return torch._neg_view(tensor)
    if layout == 'sliced':
        storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        storage[..., 0] = tensor
        return storage[..., 0]
    if tensor.dim() == 2:
        return tensor.t().contiguous().t()
    return tensor


@pytest.mark.parametrize('layout', ['expanded', 'negative', 'sliced', 'transposed'])
def test_native_second_backward_normalizes_all_three_upstream_views(layout):
    a, b, g, values, logits, offsets = inputs()
    a, b, g = (view(t, layout) for t in (a, b, g))
    snapshots = [(t.clone(), t._version) for t in (a, b, g, values, logits, offsets)]
    actual = torch.ops.ragged_pool._double_backward.default(a, b, g, values, logits, offsets)
    expected = second_vjp(ragged_pool_reference, a, b, g, values, logits, offsets)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-11)
        assert left.is_contiguous()
    for tensor, (snapshot, version) in zip((a, b, g, values, logits, offsets), snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        assert tensor._version == version


@pytest.mark.parametrize('changed', [0, 1, 2, 3])
def test_second_backward_checks_saved_g_values_logits_and_offsets_versions(changed):
    a, b, g, values, logits, offsets = inputs()
    g, values, logits = (t.requires_grad_() for t in (g, values, logits))
    first = torch.autograd.grad(ragged_pool(values, logits, offsets), (values, logits), g, create_graph=True)
    arguments = (g, values, logits, offsets)
    with torch.no_grad():
        arguments[changed].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        torch.autograd.grad((first[0]*a).sum() + (first[1]*b).sum(), (g, values, logits))


@pytest.mark.parametrize('index', [0, 1, 2])
@pytest.mark.parametrize('bad', ['shape', 'dtype', 'sparse', 'meta'])
def test_private_second_backward_rejects_malformed_upstreams(index, bad):
    arguments = list(inputs())
    tensor = arguments[index]
    if bad == 'shape':
        arguments[index] = tensor.unsqueeze(0)
    elif bad == 'dtype':
        arguments[index] = tensor.float()
    elif bad == 'sparse':
        arguments[index] = tensor.to_sparse()
    else:
        arguments[index] = torch.empty_like(tensor, device='meta')
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.ragged_pool._double_backward.default(*arguments)


@pytest.mark.parametrize('index', [3, 4])
def test_negative_dispatch_does_not_hide_unsupported_original_forward_layout(index):
    arguments = list(inputs())
    tensor = arguments[index]
    storage = torch.zeros((*tensor.shape, 2), dtype=tensor.dtype)
    stepped = storage[..., 0]
    assert not stepped.is_contiguous()
    arguments[index] = torch._neg_view(stepped)
    with pytest.raises(RuntimeError, match='contiguous'):
        torch.ops.ragged_pool._double_backward.default(*arguments)


def test_differentiable_private_first_backward_registration():
    _, _, g, values, logits, offsets = inputs()
    results = torch.library.opcheck(torch.ops.ragged_pool._backward.default,
                                   (g.requires_grad_(), values.requires_grad_(), logits.requires_grad_(), offsets))
    assert results and all(value == 'SUCCESS' for value in results.values())


def test_last_native_operation_registration_and_fixed_fake_shapes():
    arguments = inputs()
    results = torch.library.opcheck(torch.ops.ragged_pool._double_backward.default, arguments,
                                   test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'))
    assert all(value == 'SUCCESS' for value in results.values())
    differentiable = tuple(t.requires_grad_() if t.is_floating_point() else t for t in arguments)
    results = torch.library.opcheck(torch.ops.ragged_pool._double_backward.default, differentiable,
                                   test_utils=('test_autograd_registration',))
    assert results == {'test_autograd_registration': 'SUCCESS'}
    with FakeTensorMode() as mode:
        fake = tuple(mode.from_tensor(t) for t in arguments)
        outputs = torch.ops.ragged_pool._double_backward.default(*fake)
        for output, shape in zip(outputs, (arguments[2].shape, arguments[3].shape, arguments[4].shape)):
            assert isinstance(output, FakeTensor) and output.shape == shape
            assert output.dtype == torch.float64 and output.device.type == 'cpu'
            assert output.is_contiguous()
