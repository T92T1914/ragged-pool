"""Fresh forward microbatches, clipping semantics and sample-independence limits."""
import importlib.util
from pathlib import Path
import weakref

import pytest
import torch

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


_spec = importlib.util.spec_from_file_location(
    'ragged_pool_microbatched_example', Path(__file__).resolve().parents[1] / 'examples/train_clipped.py',
)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)


def _fixture(dtype_name):
    dtype = getattr(torch, dtype_name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = _example.PaddedRegressor().to(dtype)
    layout = _example.make_layouts(dtype_name)['fixed_holes']
    return model, layout['raw'], layout['mask'], layout['raw'].new_tensor([-.3, 8., -.5, .25, -4.])


def _tolerance(dtype_name):
    return dict(rtol=2e-5, atol=2e-6) if dtype_name == 'float32' else dict(rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 5, 9])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_real_microbatch_forwards_local_vjps_lifetimes_and_ownership(monkeypatch, dtype_name, chunk_size):
    model, raw, mask, targets = _fixture(dtype_name)
    named = dict(model.named_parameters())
    first = next(iter(named.values()))
    first.grad = torch.full_like(first, .125)
    grad_objects = {name: p.grad for name, p in named.items()}
    grad_before = {name: None if p.grad is None else (p.grad.clone(), p.grad._version)
                   for name, p in named.items()}
    parameter_before = {name: (p.detach().view(torch.uint8).clone(), p._version)
                        for name, p in named.items()}
    snapshots = [(t.view(torch.uint8).clone(), t._version) for t in (raw, mask, targets)]
    full = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded)
    serial = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded, 'serial')
    reference = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded_reference, 'serial')
    expected, expected_norms, expected_scales = _example.clip_per_example(full, 1.)
    serial_means, _, _ = _example.clip_per_example(serial, 1.)
    reference_means, _, _ = _example.clip_per_example(reference, 1.)
    retained, retained_norms, retained_scales, _ = _example.chunked_clipped_gradients(
        model, raw, mask, targets, ragged_pool_padded, 1., chunk_size)
    real_forward, real_grad = model.forward, torch.autograd.grad
    effective = min(chunk_size, len(targets))
    events, calls, previous_refs = [], [], []

    def observed_forward(local_raw, local_mask, pool):
        assert all(ref() is None for ref in previous_refs)
        previous_refs.clear()
        start = sum(call['rows'] for call in calls)
        rows = min(effective, len(targets) - start)
        assert len(events) == 2 * len(calls)
        assert local_raw.shape[0] == local_mask.shape[0] == rows
        # Byte comparisons preserve the ignored NaN payload and row order.
        assert torch.equal(local_raw.view(torch.uint8), raw[start:start + rows].view(torch.uint8))
        assert torch.equal(local_mask, mask[start:start + rows])
        assert pool is ragged_pool_padded
        result = real_forward(local_raw, local_mask, pool)
        events.append(('forward', start, rows))
        previous_refs.extend(weakref.ref(t) for t in result)
        return result

    def observed_grad(outputs, inputs, grad_outputs=None, **kwargs):
        start = sum(call['rows'] for call in calls)
        rows = min(effective, len(targets) - start)
        assert events[-1] == ('forward', start, rows)
        assert outputs.shape == (rows,)
        assert grad_outputs.shape == (rows, rows)
        assert torch.equal(grad_outputs, torch.eye(rows, dtype=outputs.dtype))
        assert len(inputs) == len(named) and all(actual is expected for actual, expected in zip(inputs, named.values()))
        assert kwargs == {'retain_graph': False, 'create_graph': False,
                          'allow_unused': False, 'is_grads_batched': True}
        result = real_grad(outputs, inputs, grad_outputs, **kwargs)
        assert len(result) == len(named)
        for actual, parameter in zip(result, named.values()):
            assert actual.shape == (rows, *parameter.shape)
        calls.append({'rows': rows, 'seed_shape': list(grad_outputs.shape),
                      'gradient_shapes': {name: list(g.shape) for name, g in zip(named, result)}})
        events.append(('backward', start, rows))
        previous_refs.extend(weakref.ref(g) for g in result)
        previous_refs.extend((weakref.ref(outputs), weakref.ref(grad_outputs)))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(model, 'forward', observed_forward)
        patch.setattr(torch.autograd, 'grad', observed_grad)
        means, norms, scales, metadata = _example.microbatched_clipped_gradients(
            model, raw, mask, targets, ragged_pool_padded, 1., chunk_size)
    assert all(ref() is None for ref in previous_refs)
    assert len(events) == 2 * len(calls)
    assert metadata == {'requested_size': chunk_size, 'effective_size': effective,
        'chunk_rows': [call['rows'] for call in calls], 'forward_rows': [call['rows'] for call in calls],
        'retain_graph': [False] * len(calls), 'seed_shapes': [call['seed_shape'] for call in calls],
        'gradient_shapes': [call['gradient_shapes'] for call in calls],
        'max_rows': effective, 'parameter_elements': 25, 'global_denominator': len(targets)}
    assert sum(metadata['forward_rows']) == len(targets)
    tolerance = _tolerance(dtype_name)
    for other_norms, other_scales in ((expected_norms, expected_scales), (retained_norms, retained_scales)):
        torch.testing.assert_close(norms, other_norms, **tolerance)
        torch.testing.assert_close(scales, other_scales, **tolerance)
    for name, parameter in named.items():
        for other in (expected, serial_means, reference_means, retained):
            torch.testing.assert_close(means[name], other[name], **tolerance)
        before_bytes, before_version = parameter_before[name]
        assert parameter._version == before_version
        assert torch.equal(parameter.detach().view(torch.uint8), before_bytes)
        assert parameter.grad is grad_objects[name]
        if grad_before[name] is None:
            assert parameter.grad is None
        else:
            before_grad, before_grad_version = grad_before[name]
            assert parameter.grad._version == before_grad_version
            assert torch.equal(parameter.grad, before_grad)
    for tensor, (before, version) in zip((raw, mask, targets), snapshots):
        assert tensor._version == version and torch.equal(tensor.view(torch.uint8), before)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_saved_backward_holders_are_released_before_each_next_forward(monkeypatch, dtype_name):
    model, raw, mask, targets = _fixture(dtype_name)
    full = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded)
    expected, _, _ = _example.clip_per_example(full, 1.)
    real_forward = model.forward
    holders, packed_counts, unpacked_counts = [], [], []
    current = [-1]

    class SavedValue:
        def __init__(self, tensor):
            # The packed object must not retain the original graph tensor.
            self.value = tensor.detach()
            self.microbatch = current[0]

    def pack(tensor):
        holder = SavedValue(tensor)
        holders.append(weakref.ref(holder))
        packed_counts[current[0]] += 1
        return holder

    def unpack(holder):
        unpacked_counts[holder.microbatch] += 1
        return holder.value

    def observed_forward(local_raw, local_mask, pool):
        # Qualify release of the actual graph's packed saved-state holders.
        # This does not measure allocator memory or process RSS.
        assert all(ref() is None for ref in holders)
        holders.clear()
        current[0] += 1
        packed_counts.append(0)
        unpacked_counts.append(0)
        return real_forward(local_raw, local_mask, pool)

    with monkeypatch.context() as patch:
        patch.setattr(model, 'forward', observed_forward)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            actual, _, _, metadata = _example.microbatched_clipped_gradients(
                model, raw, mask, targets, ragged_pool_padded, 1., 2)
    assert metadata['forward_rows'] == [2, 2, 1]
    assert all(count > 0 for count in packed_counts)
    assert all(count > 0 for count in unpacked_counts)
    assert len(packed_counts) == len(unpacked_counts) == 3
    assert all(ref() is None for ref in holders)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], **_tolerance(dtype_name))


@pytest.mark.parametrize('chunk_size', [0, -1, True, False, 1.5, 2.0, '2', None])
def test_microbatching_rejects_invalid_sizes(chunk_size):
    model, raw, mask, targets = _fixture('float64')
    with pytest.raises(ValueError, match='positive Python integer'):
        _example.microbatched_clipped_gradients(model, raw, mask, targets, ragged_pool_padded, 1., chunk_size)


@pytest.mark.parametrize('threshold', [0., -1., float('inf'), float('nan')])
def test_microbatching_rejects_nonpositive_or_nonfinite_thresholds(threshold):
    model, raw, mask, targets = _fixture('float64')
    with pytest.raises(ValueError, match='finite and positive'):
        _example.microbatched_clipped_gradients(model, raw, mask, targets, ragged_pool_padded, threshold, 2)


def test_microbatch_caller_rejects_zero_examples_without_narrowing_native_pool():
    model = _example.PaddedRegressor().double()
    raw = torch.empty((0, 3, 3), dtype=torch.float64)
    mask = torch.empty((0, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        _example.microbatched_clipped_gradients(
            model, raw, mask, torch.empty(0, dtype=raw.dtype), ragged_pool_padded, 1., 2)
    assert ragged_pool_padded(raw, torch.empty((0, 3), dtype=raw.dtype), mask).shape == (0, 3)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_microbatch_tail_uses_global_count_and_zero_norm_scale(dtype_name):
    class ScalarPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=getattr(torch, dtype_name)))

        def forward(self, raw, mask, pool):
            return self.weight.expand(raw.shape[0]), raw.new_zeros((raw.shape[0], 1))

    model = ScalarPrediction()
    raw = torch.zeros((5, 1, 1), dtype=getattr(torch, dtype_name))
    targets = raw.new_tensor([-.5, -.5, -.5, -.5, 0.])
    means, norms, scales, metadata = _example.microbatched_clipped_gradients(
        model, raw, torch.ones((5, 1), dtype=torch.bool), targets, None, 1., 2)
    torch.testing.assert_close(norms, torch.tensor([1., 1., 1., 1., 0.], dtype=torch.float64), rtol=0, atol=0)
    assert torch.equal(scales, torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(means['weight'], raw.new_tensor([4/5]), rtol=0, atol=0)
    assert not torch.allclose(means['weight'], raw.new_tensor([2/3]))
    assert metadata['forward_rows'] == [2, 2, 1] and metadata['global_denominator'] == 5
    assert metadata['retain_graph'] == [False, False, False]
    assert model.weight.grad is None and torch.count_nonzero(model.weight.detach()) == 0


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_batch_coupled_prediction_is_an_expected_counterexample(dtype_name):
    class CoupledPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=getattr(torch, dtype_name)))

        def forward(self, raw, mask, pool):
            shared = self.weight * raw[:, 0, 0].mean()
            return shared.expand(raw.shape[0]), raw.new_zeros((raw.shape[0], 1))

    model = CoupledPrediction()
    raw = torch.tensor([[[1.]], [[3.]]], dtype=getattr(torch, dtype_name))
    mask, targets = torch.ones((2, 1), dtype=torch.bool), raw.new_zeros(2)
    full = _example.per_example_gradients(model, raw, mask, targets, None)
    full_mean, full_norms, full_scales = _example.clip_per_example(full, 100.)
    micro_mean, micro_norms, micro_scales, metadata = _example.microbatched_clipped_gradients(
        model, raw, mask, targets, None, 100., 1)
    # Full forward uses mean input two: d((2*w)^2)/dw at w=1 is eight
    # for either row. Singleton forwards yield derivatives two and eighteen.
    torch.testing.assert_close(full_norms, torch.tensor([8., 8.], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(micro_norms, torch.tensor([2., 18.], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(full_mean['weight'], raw.new_tensor([8.]), rtol=0, atol=0)
    torch.testing.assert_close(micro_mean['weight'], raw.new_tensor([10.]), rtol=0, atol=0)
    assert torch.equal(full_scales, torch.ones(2, dtype=torch.float64))
    assert torch.equal(micro_scales, torch.ones(2, dtype=torch.float64))
    assert not torch.equal(full_mean['weight'], micro_mean['weight'])
    assert metadata['forward_rows'] == [1, 1]
    assert model.weight.grad is None and torch.equal(model.weight.detach(), raw.new_tensor([1.]))


@pytest.mark.parametrize('batch', [1, 5])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_microbatch_single_example_and_all_empty_nonempty_batches(dtype_name, batch):
    dtype = getattr(torch, dtype_name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = _example.PaddedRegressor().to(dtype)
    raw = torch.full((batch, 3, 3), float('nan'), dtype=dtype)
    mask = torch.zeros((batch, 3), dtype=torch.bool)
    targets = torch.linspace(-.5, 1., batch, dtype=dtype)
    full = _example.per_example_gradients(model, raw, mask, targets, ragged_pool_padded)
    expected, expected_norms, expected_scales = _example.clip_per_example(full, 1.)
    actual, norms, scales, metadata = _example.microbatched_clipped_gradients(
        model, raw, mask, targets, ragged_pool_padded, 1., 2)
    tolerance = _tolerance(dtype_name)
    torch.testing.assert_close(norms, expected_norms, **tolerance)
    torch.testing.assert_close(scales, expected_scales, **tolerance)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], **tolerance)
        if name != 'head.bias':
            assert torch.count_nonzero(actual[name]) == 0
    assert torch.count_nonzero(full['head.bias']) > 0
    assert metadata['effective_size'] == min(2, batch)
    assert metadata['global_denominator'] == batch
    assert not any(metadata['retain_graph'])


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 5, 9])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_microbatched_consumer_takes_two_updates_with_all_comparisons(dtype_name, chunk_size):
    receipt = _example.run(dtype_name, chunk_size, 'forward-microbatches')
    assert receipt['status'] == 'passed' and receipt['dtype'] == dtype_name
    assert receipt['method'] == 'autograd.grad.is_grads_batched'
    assert receipt['aggregation_method'] == 'microbatched_per_example_joint_clip'
    assert receipt['aggregation_route'] == 'forward-microbatches'
    assert receipt['chunk_size_requested'] == chunk_size
    assert receipt['optimizer_steps'] == receipt['reference_optimizer_steps'] == 2
    assert receipt['controls']['status'] == receipt['invariance']['status'] == receipt['empty']['status'] == 'passed'
    effective = min(chunk_size, 5)
    rows = [min(effective, 5 - start) for start in range(0, 5, effective)]
    for step in receipt['steps']:
        assert step['chunking']['forward_rows'] == step['chunking']['chunk_rows'] == rows
        assert step['chunking']['effective_size'] == step['chunking']['max_rows'] == effective
        assert step['chunking']['seed_shapes'] == [[r, r] for r in rows]
        assert step['chunking']['retain_graph'] == [False] * len(rows)
        assert step['chunking']['global_denominator'] == 5
        assert step['retained_graph_clipped_mean_errors'].keys() == step['full_clipped_mean_errors'].keys()
        assert all(value >= 0 for value in step['retained_graph_clipped_mean_errors'].values())


def test_consumer_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match='aggregation'):
        _example.run('float64', 2, 'not-a-route')
