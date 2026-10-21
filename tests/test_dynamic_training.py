"""First-order dynamic packed checks, separate from operator registration."""
import importlib.util
from pathlib import Path

import pytest
import torch

from ragged_pool import ragged_pool


ORDINARY = [(0, 2, 2, 5, 7, 9), (0, 3, 5, 5, 8, 10, 12),
            (0, 1, 5, 7, 11), (0, 3, 3, 8, 11), (0, 1, 4, 6, 6, 9)]
CORNERS = [(0,), (0, 0), (0, 0, 0, 0), (0, 1), (0, 0, 1, 1), (0, 3)]


@pytest.fixture(autouse=True)
def reset_compiler():
    # Pinned test instrumentation, not a production cache/configuration change.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def primitive(values, logits, boundaries):
    rows = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            rows.append(values[start:end].sum(0) + logits[start:end].sum() * 0)
        else:
            rows.append((logits[start:end].softmax(0)[:, None] * values[start:end]).sum(0))
    if not rows:
        return values[:0] + logits.sum() * 0
    return torch.stack(rows)


def inputs(boundaries, dtype):
    n = boundaries[-1]
    values = (torch.arange(n * 4, dtype=dtype).reshape(n, 4) / 17 - 1).requires_grad_()
    logits = torch.linspace(-0.7, 0.9, n, dtype=dtype).requires_grad_()
    return values, logits, torch.tensor(boundaries, dtype=torch.int64)


def compare(pool, boundaries, dtype, upstream=None):
    values, logits, offsets = inputs(boundaries, dtype)
    originals = [t.detach().clone() for t in (values, logits, offsets)]
    versions = [t._version for t in (values, logits, offsets)]
    rv = values.detach().clone().requires_grad_()
    rl = logits.detach().clone().requires_grad_()
    actual = pool(values, logits, offsets)
    expected = primitive(rv, rl, boundaries)
    tolerance = dict(rtol=2e-4, atol=2e-6) if dtype == torch.float32 else dict(rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(actual, expected, **tolerance)
    if upstream is None:
        actual_grads = torch.autograd.grad(actual.sum(), (values, logits))
        expected_grads = torch.autograd.grad(expected.sum(), (rv, rl))
    else:
        actual_grads = torch.autograd.grad(actual, (values, logits), upstream)
        expected_grads = torch.autograd.grad(expected, (rv, rl), upstream)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, **tolerance)
        assert bool(torch.isfinite(a).all())
    for t, original, version in zip((values, logits, offsets), originals, versions):
        torch.testing.assert_close(t, original, rtol=0, atol=0)
        assert t._version == version
    assert not offsets.requires_grad
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            assert torch.count_nonzero(actual[i]) == 0
        elif end - start == 1:
            torch.testing.assert_close(actual[i], values[start], rtol=0, atol=0)
            assert actual_grads[1][start] == 0
    if not boundaries[-1]:
        assert torch.count_nonzero(actual) == 0
        assert all(torch.count_nonzero(g) == 0 for g in actual_grads)
    return actual.detach()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_string_dynamic_pool_all_shape_families(dtype):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    for boundaries in ORDINARY + CORNERS + ORDINARY[:1]:
        compare(compiled, boundaries, dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_pinned_capture_diagnostics_distinguish_specializations(dtype):
    # These private v2.9.1 callbacks count graph submissions, not optimized kernels.
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.backends.debugging import boxed_nop
    dynamo, forward, backward = [], [], []
    aot = lookup_backend('aot_eager')
    def capture(target):
        def compiler(graph, example_inputs):
            target.append([str(node.target) for node in graph.graph.nodes])
            return boxed_nop(graph, example_inputs)
        return compiler
    def backend(graph, example_inputs):
        dynamo.append([str(node.target) for node in graph.graph.nodes])
        return aot(graph, example_inputs, fw_compiler=capture(forward), bw_compiler=capture(backward))
    compiled = torch.compile(ragged_pool, backend=backend, fullgraph=True, dynamic=True)
    for boundaries in ORDINARY:
        compare(compiled, boundaries, dtype)
    assert len(dynamo) == 1
    assert len(forward) == len(backward) == 1
    assert any('ragged_pool.pool.default' in graph for graph in forward)
    assert any('ragged_pool._backward.default' in graph for graph in backward)
    for boundaries in CORNERS:
        compare(compiled, boundaries, dtype)
    assert len(dynamo) == 7
    compare(compiled, ORDINARY[0], dtype)
    assert len(dynamo) == 7
    print({'dtype': str(dtype), 'dynamo_submissions': len(dynamo),
           'aot_forward_callbacks': len(forward), 'aot_backward_callbacks': len(backward)})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('layout', ['transposed', 'stepped', 'expanded', 'negative'])
def test_dynamic_backward_accepts_upstream_views(dtype, layout):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    compare(compiled, ORDINARY[0], dtype)
    boundaries = ORDINARY[1]
    b = len(boundaries) - 1
    if layout == 'transposed':
        upstream = torch.arange(b * 4, dtype=dtype).reshape(4, b).t()
    elif layout == 'stepped':
        upstream = torch.arange(b * 8, dtype=dtype).reshape(b, 8)[:, ::2]
    elif layout == 'expanded':
        upstream = torch.tensor(0.3, dtype=dtype).expand(b, 4)
        assert upstream.stride() == (0, 0)
    else:
        upstream = torch._neg_view(torch.arange(b * 4, dtype=dtype).reshape(b, 4))
    snapshot, version = upstream.clone(), upstream._version
    compare(compiled, boundaries, dtype, upstream)
    torch.testing.assert_close(upstream, snapshot, rtol=0, atol=0)
    assert upstream._version == version


@pytest.mark.parametrize('changed', [0, 1, 2])
def test_dynamic_saved_original_mutation_refused(changed):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    compare(compiled, ORDINARY[0], torch.float64)
    args = inputs(ORDINARY[1], torch.float64)
    output = compiled(*args)
    with torch.no_grad():
        args[changed].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        output.sum().backward()


@pytest.mark.parametrize('case', ['start', 'end', 'decreasing', 'negative', 'past_end',
                                  'values_nan', 'logits_inf', 'values_range', 'logits_range'])
def test_dynamic_runtime_admission_and_recovery(case):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    compare(compiled, ORDINARY[0], torch.float64)
    args = list(inputs(ORDINARY[0], torch.float64))
    with torch.no_grad():
        if case == 'start': args[2][0] = 1
        elif case == 'end': args[2][-1] = 8
        elif case == 'decreasing': args[2][2] = 1
        elif case == 'negative': args[2][1] = -1
        elif case == 'past_end': args[2][1] = 10
        elif case == 'values_nan': args[0][0, 0] = float('nan')
        elif case == 'logits_inf': args[1][0] = float('inf')
        elif case == 'values_range': args[0][0, 0] = 1001
        else: args[1][0] = -1001
    with pytest.raises(RuntimeError):
        compiled(*args)
    compare(compiled, ORDINARY[0], torch.float64)
    compare(compiled, ORDINARY[1], torch.float64)


@pytest.mark.parametrize('changed', [0, 1, 2])
def test_dynamic_noncontiguous_primals_still_refused(changed):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    compare(compiled, ORDINARY[0], torch.float64)
    args = list(inputs(ORDINARY[0], torch.float64))
    source = args[changed].detach()
    storage = torch.empty((*source.shape, 2), dtype=source.dtype)
    storage[..., 0] = source
    args[changed] = storage[..., 0]
    if changed < 2: args[changed].requires_grad_()
    assert not args[changed].is_contiguous()
    with pytest.raises((RuntimeError, torch._dynamo.exc.TorchRuntimeError)):
        compiled(*args)


@pytest.mark.parametrize('case', ['N_limit', 'B_limit', 'mismatched_N'])
def test_dynamic_symbolic_size_admission_remains_bounded(case):
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    compare(compiled, ORDINARY[0], torch.float64)
    args = list(inputs(ORDINARY[0], torch.float64))
    if case == 'N_limit':
        args = [torch.zeros(4097, 4, dtype=torch.float64, requires_grad=True),
                torch.zeros(4097, dtype=torch.float64, requires_grad=True), torch.tensor([0, 4097])]
    elif case == 'B_limit':
        args[2] = torch.tensor([0] * 129 + [9])
    else:
        args[1] = args[1][:-1].detach().clone().requires_grad_()
    with pytest.raises((RuntimeError, torch._dynamo.exc.TorchRuntimeError)):
        compiled(*args)
    compare(compiled, ORDINARY[1], torch.float64)


def test_dynamic_same_metadata_partition_and_analytic_gradient_witness():
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    values = torch.tensor([[1., 2.], [5., 6.], [9., 10.]], dtype=torch.float64, requires_grad=True)
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 3])
    output = compiled(values, logits, offsets)
    expected = torch.tensor([[3., 4.], [0., 0.], [9., 10.]], dtype=torch.float64)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(output.sum(), (values, logits))
    torch.testing.assert_close(gradients[0], torch.tensor([[.5, .5], [.5, .5], [1., 1.]], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], torch.tensor([-2., 2., 0.], dtype=torch.float64), rtol=0, atol=0)
    changed = compiled(values, logits, torch.tensor([0, 1, 3, 3]))
    new_expected = torch.tensor([[1., 2.], [7., 8.], [0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(changed, new_expected, rtol=0, atol=0)
    # A stale partition or detached pool must fail a deciding assertion.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(output, new_expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='does not require grad'):
        torch.autograd.grad(output.detach().sum(), (values, logits))
    another = compiled(values, logits, offsets)
    assert output.data_ptr() != another.data_ptr()
    assert output.data_ptr() not in {values.data_ptr(), logits.data_ptr(), offsets.data_ptr()}
    output.detach().add_(100)
    torch.testing.assert_close(another, expected, rtol=0, atol=0)


def test_dynamic_nonuniform_analytic_weights_and_empty_singleton_derivatives():
    compiled = torch.compile(ragged_pool, backend='aot_eager', fullgraph=True, dynamic=True)
    values = torch.tensor([[1.], [3.], [5.]], dtype=torch.float64, requires_grad=True)
    # The two-item softmax has weights 1/4 and 3/4.
    logits = torch.tensor([0., float(torch.tensor(3., dtype=torch.float64).log()), 7.],
                          dtype=torch.float64, requires_grad=True)
    output = compiled(values, logits, torch.tensor([0, 2, 2, 3]))
    torch.testing.assert_close(output, torch.tensor([[2.5], [0.], [5.]], dtype=torch.float64))
    gradients = torch.autograd.grad(output, (values, logits), torch.tensor([[2.], [9.], [4.]], dtype=torch.float64))
    torch.testing.assert_close(gradients[0], torch.tensor([[.5], [1.5], [4.]], dtype=torch.float64))
    torch.testing.assert_close(gradients[1], torch.tensor([-.75, .75, 0.], dtype=torch.float64))


def load_consumer():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'train_dynamic.py'
    spec = importlib.util.spec_from_file_location('dynamic_training_consumer', path)
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    return consumer


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_dynamic_consumer_compares_full_training_trajectory(dtype_name, monkeypatch):
    consumer = load_consumer()
    real_compile, constructions = torch.compile, []
    def spy(model, **options):
        constructions.append(options)
        return real_compile(model, **options)
    monkeypatch.setattr(torch, 'compile', spy)
    model, batches, tolerance = consumer.fixture(dtype_name)
    result = consumer.train(model, batches, tolerance)
    assert len(result) == 5
    assert constructions == [dict(backend='aot_eager', fullgraph=True, dynamic=True)]
    assert [tuple(row['offsets']) for row in result] == ORDINARY
    blocks = {'encoder.weight', 'encoder.bias', 'scorer.weight', 'head.weight', 'head.bias'}
    for row in result:
        assert row['same_parameter_objects'] and row['same_optimizer_objects']
        assert row['all_routes_all_parameter_blocks_changed'] and row['inputs_unchanged']
        for route in ('eager_native', 'reference'):
            assert set(row['parameter_gradient_errors'][route]) == blocks
            assert set(row['updated_parameter_errors'][route]) == blocks
        for route in ('compiled', 'eager_native', 'reference'):
            assert all(value > 0 for value in row['parameter_update_max_abs'][route].values())


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_pinned_model_diagnostic_reuses_one_submission_for_five_steps(dtype_name, monkeypatch):
    from torch._dynamo.backends.registry import lookup_backend
    consumer = load_consumer()
    real_compile, graphs = torch.compile, []
    aot = lookup_backend('aot_eager')
    def backend(graph, example_inputs):
        graphs.append([str(node.target) for node in graph.graph.nodes])
        return aot(graph, example_inputs)
    def instrument(model, **options):
        assert options == dict(backend='aot_eager', fullgraph=True, dynamic=True)
        return real_compile(model, backend=backend, fullgraph=True, dynamic=True)
    monkeypatch.setattr(torch, 'compile', instrument)
    model, batches, tolerance = consumer.fixture(dtype_name)
    consumer.train(model, batches, tolerance)
    assert len(graphs) == 1
    assert 'ragged_pool.pool.default' in graphs[0]
    print({'dtype': dtype_name, 'model_dynamo_submissions': len(graphs), 'optimizer_steps': 5})


def test_training_acceptance_rejects_detached_pool(monkeypatch):
    consumer = load_consumer()
    original = consumer.SetRegressor.forward
    def detached(self, raw, offsets, pool=ragged_pool):
        return original(self, raw, offsets, lambda v, l, o: pool(v, l, o).detach())
    monkeypatch.setattr(consumer.SetRegressor, 'forward', detached)
    model, batches, tolerance = consumer.fixture('float64')
    with pytest.raises(AssertionError, match='Each input and parameter block must receive'):
        consumer.train(model, batches, tolerance)
