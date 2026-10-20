"""Padded registration and compiler checks, separate from numerical gradcheck."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


def inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(617)
    values = torch.randn(3, 4, 2, generator=generator, dtype=dtype)
    logits = torch.randn(3, 4, generator=generator, dtype=dtype)
    mask = torch.tensor([[True, False, True, False], [False]*4, [True]*4])
    return values.requires_grad_(), logits.requires_grad_(), mask


def test_padded_forward_registration():
    results = torch.library.opcheck(torch.ops.ragged_pool.padded.default, inputs())
    assert results and all(result == 'SUCCESS' for result in results.values())


def test_private_padded_backward_registration_has_two_separate_scopes():
    values, logits, mask = inputs()
    upstream = torch.ones(3, 2, dtype=values.dtype)
    # Detached inputs check schema, fake metadata and AOT execution, not a VJP.
    results = torch.library.opcheck(
        torch.ops.ragged_pool._padded_backward.default,
        (upstream, values.detach(), logits.detach(), mask),
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'),
    )
    assert all(result == 'SUCCESS' for result in results.values())
    # This requires-grad check is separate from numerical derivative checks.
    results = torch.library.opcheck(
        torch.ops.ragged_pool._padded_backward.default, (upstream, values, logits, mask),
        test_utils=('test_autograd_registration',),
    )
    assert results == {'test_autograd_registration': 'SUCCESS'}


@pytest.mark.parametrize('shape', [(3, 4, 2), (0, 4, 2), (3, 0, 2)])
def test_both_padded_operations_supply_only_fixed_shape_fake_metadata(shape):
    with FakeTensorMode():
        values = torch.empty(shape, dtype=torch.float64)
        logits = torch.empty(shape[:2], dtype=torch.float64)
        mask = torch.empty(shape[:2], dtype=torch.bool)
        output = torch.ops.ragged_pool.padded.default(values, logits, mask)
        grad_values, grad_logits = torch.ops.ragged_pool._padded_backward.default(
            torch.empty(shape[0], shape[2], dtype=values.dtype), values, logits, mask,
        )
        for tensor, expected in ((output, (shape[0], shape[2])), (grad_values, shape),
                                 (grad_logits, shape[:2])):
            assert isinstance(tensor, FakeTensor) and tensor.shape == expected
            assert tensor.dtype == values.dtype and tensor.device.type == 'cpu'
            assert tensor.is_contiguous()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_one_fullgraph_capture_reuses_changed_mask_contents_and_native_backward(dtype):
    from torch._dynamo.backends.registry import lookup_backend

    graphs = []
    aot_eager = lookup_backend('aot_eager')

    def record_graph(graph, example_inputs):
        graphs.append(graph)
        return aot_eager(graph, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(ragged_pool_padded, backend=record_graph, fullgraph=True, dynamic=False)
    values, logits, holes = inputs(dtype)
    masks = [holes, torch.zeros_like(holes), torch.ones_like(holes),
             torch.tensor([[False, True, False, False]]*3)]
    tolerance = {'rtol': 2e-5, 'atol': 2e-6} if dtype == torch.float32 else {'rtol': 1e-10, 'atol': 1e-12}
    for index, mask in enumerate(masks):
        v, l = values.detach().clone(), logits.detach().clone()
        v.add_(index / 10)
        l.sub_(index / 10)
        v[~mask], l[~mask] = float('nan'), float('inf')
        left = (v.clone().requires_grad_(), l.clone().requires_grad_())
        right = (v.clone().requires_grad_(), l.clone().requires_grad_())
        output = compiled(*left, mask)
        expected = ragged_pool_padded_reference(*right, mask)
        # A sum supplies an expanded zero-stride upstream gradient.
        actual_grads = torch.autograd.grad(output.sum(), left)
        expected_grads = torch.autograd.grad(expected.sum(), right)
        torch.testing.assert_close(output, expected, **tolerance)
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference, **tolerance)
            assert bool(torch.isfinite(actual).all())
            assert torch.count_nonzero(actual[~mask]) == 0
    assert len(graphs) == 1
    targets = [node.target for node in graphs[0].graph.nodes if node.op == 'call_function']
    assert targets.count(torch.ops.ragged_pool.padded.default) == 1
    assert not any('nonzero' in str(target) or 'index.Tensor' in str(target) for target in targets)
    # Same graph still validates selected contents inside its runtime kernel.
    invalid_values = values.detach().clone()
    invalid_values[0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='finite.*1000'):
        compiled(invalid_values.requires_grad_(), logits, holes)
    assert len(graphs) == 1


@pytest.mark.parametrize('shape', [(0, 4, 2), (3, 0, 2)])
def test_compiled_empty_dimensions_produce_zero_derivatives(shape):
    values = torch.empty(shape, requires_grad=True)
    logits = torch.empty(shape[:2], requires_grad=True)
    mask = torch.empty(shape[:2], dtype=torch.bool)
    compiled = torch.compile(ragged_pool_padded, backend='aot_eager', fullgraph=True, dynamic=False)
    output = compiled(values, logits, mask)
    assert output.shape == (shape[0], shape[2]) and torch.count_nonzero(output) == 0
    gradients = torch.autograd.grad(output.sum(), (values, logits))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('changed', [0, 1, 2])
def test_saved_originals_and_mask_cannot_change_before_backward(compiled, changed):
    arguments = inputs()
    pool = torch.compile(ragged_pool_padded, backend='aot_eager', fullgraph=True, dynamic=False) if compiled else ragged_pool_padded
    output = pool(*arguments)
    with torch.no_grad():
        # Even ignored padding is part of the saved original tensor version.
        arguments[changed][0, 1] = False if changed == 2 else 12.
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        output.sum().backward()


def test_private_padded_first_backward_full_differentiable_registration():
    values, logits, mask = inputs()
    upstream = torch.ones(3, 2, dtype=values.dtype, requires_grad=True)
    results = torch.library.opcheck(
        torch.ops.ragged_pool._padded_backward.default, (upstream, values, logits, mask))
    assert results and all(result == 'SUCCESS' for result in results.values())


def test_final_padded_backward_registration_has_separate_scopes():
    values, logits, mask = inputs()
    arguments = (torch.ones_like(values), torch.ones_like(logits),
                 torch.ones(3, 2, dtype=values.dtype), values.detach(), logits.detach(), mask)
    op = torch.ops.ragged_pool._padded_double_backward.default
    results = torch.library.opcheck(op, arguments,
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'))
    assert all(result == 'SUCCESS' for result in results.values())
    differentiable = tuple(t.requires_grad_() if t.is_floating_point() else t for t in arguments)
    results = torch.library.opcheck(op, differentiable, test_utils=('test_autograd_registration',))
    assert results == {'test_autograd_registration': 'SUCCESS'}


@pytest.mark.parametrize('shape', [(3, 4, 2), (0, 4, 2), (3, 0, 2)])
def test_final_padded_backward_fixed_shape_fake_metadata(shape):
    with FakeTensorMode():
        values = torch.empty(shape, dtype=torch.float64)
        logits = torch.empty(shape[:2], dtype=torch.float64)
        mask = torch.empty(shape[:2], dtype=torch.bool)
        output = torch.empty((shape[0], shape[2]), dtype=values.dtype)
        results = torch.ops.ragged_pool._padded_double_backward.default(
            values, logits, output, values, logits, mask)
        for tensor, expected in zip(results, (output.shape, values.shape, logits.shape)):
            assert isinstance(tensor, FakeTensor) and tensor.shape == expected
            assert tensor.dtype == values.dtype and tensor.device.type == 'cpu'
            assert tensor.is_contiguous()


@pytest.mark.parametrize('layout', ['transposed', 'sliced', 'negative'])
def test_compiled_backward_accepts_ordinary_upstream_views(layout):
    values, logits, mask = inputs()
    left = (values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_())
    right = (values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_())
    if layout == 'transposed':
        upstream = torch.arange(6, dtype=values.dtype).reshape(2, 3).t()
    elif layout == 'sliced':
        upstream = torch.arange(12, dtype=values.dtype).reshape(3, 4)[:, ::2]
    else:
        upstream = torch._neg_view(torch.arange(6, dtype=values.dtype).reshape(3, 2))
    compiled = torch.compile(ragged_pool_padded, backend='aot_eager', fullgraph=True, dynamic=False)
    actual = torch.autograd.grad(compiled(*left, mask), left, upstream)
    expected = torch.autograd.grad(ragged_pool_padded_reference(*right, mask), right, upstream)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('case', ['dtype', 'rank', 'shape'])
def test_private_backward_rejects_invalid_upstream_metadata(case):
    values, logits, mask = inputs()
    upstream = torch.ones(3, 2, dtype=values.dtype)
    if case == 'dtype':
        upstream = upstream.float()
    elif case == 'rank':
        upstream = upstream.unsqueeze(0)
    else:
        upstream = upstream[:2]
    with pytest.raises(RuntimeError, match='grad_output'):
        torch.ops.ragged_pool._padded_backward.default(upstream, values, logits, mask)
