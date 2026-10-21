"""Mathematical and admission checks for bounded packed function batching."""
import itertools

import pytest
import torch

from ragged_pool import ragged_pool, ragged_pool_forward_ad
from ragged_pool.forward_ad import _PackedBackward


DTYPES = (torch.float32, torch.float64)
PARTITIONS = ((0, 2, 2, 3, 6), (0, 1, 3, 3, 6), (0, 0, 2, 5, 6))


def tolerance(dtype):
    return {"rtol": 4e-5, "atol": 4e-6} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-11}


def data(dtype=torch.float64, count=3):
    generator = torch.Generator().manual_seed(1519)
    values = torch.randn(count, 6, 2, generator=generator, dtype=dtype) / 2
    logits = torch.randn(count, 6, generator=generator, dtype=dtype) / 3
    offsets = torch.tensor([PARTITIONS[index % len(PARTITIONS)] for index in range(count)])
    return values, logits, offsets


def reference(boundaries):
    def pool(values, logits):
        rows = []
        for start, stop in zip(boundaries, boundaries[1:]):
            v, l = values[start:stop], logits[start:stop]
            rows.append(v.sum(0) * 0 + l.sum() * 0 if start == stop
                        else (torch.softmax(l, 0)[:, None] * v).sum(0))
        return torch.stack(rows) if rows else (values.sum(0) * 0 + logits.sum() * 0).unsqueeze(0)[:0]
    return pool


def selected(arguments, dimensions, index):
    return tuple(argument if dimension is None else argument.select(dimension, index)
                 for argument, dimension in zip(arguments, dimensions))


def assert_pairs(actual, expected, dtype):
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle, **tolerance(dtype))


def upstream(shape, dtype, kind):
    if kind == "transpose":
        return torch.linspace(-.7, .8, shape[0] * shape[1], dtype=dtype).reshape(shape[::-1]).t()
    if kind == "stepped":
        return torch.linspace(-.7, .8, shape[0] * shape[1] * 2, dtype=dtype).reshape(shape[0], -1)[:, ::2]
    if kind == "expanded":
        return torch.tensor(.3, dtype=dtype).expand(shape)
    if kind == "negative":
        return torch._neg_view(torch.full(shape, .4, dtype=dtype))
    return torch.linspace(-.7, .8, shape[0] * shape[1], dtype=dtype).reshape(shape)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["ordinary", "transpose", "stepped", "expanded", "negative"])
def test_func_grad_and_vjp_match_independent_math_and_preserve_cotangent(dtype, kind):
    values, logits, offsets = (tensor[0] for tensor in data(dtype))
    pool = lambda v, l: ragged_pool_forward_ad(v, l, offsets)
    oracle = reference(PARTITIONS[0])
    actual_grad = torch.func.grad(lambda v, l: pool(v, l).square().sum(), argnums=(0, 1))(values, logits)
    expected_grad = torch.func.grad(lambda v, l: oracle(v, l).square().sum(), argnums=(0, 1))(values, logits)
    assert_pairs(actual_grad, expected_grad, dtype)
    result, pullback = torch.func.vjp(pool, values, logits)
    expected, expected_pullback = torch.func.vjp(oracle, values, logits)
    g = upstream(result.shape, dtype, kind)
    before, version = g.clone(), g._version
    assert_pairs(pullback(g), expected_pullback(g), dtype)
    torch.testing.assert_close(result, expected, **tolerance(dtype))
    torch.testing.assert_close(g, before, rtol=0, atol=0)
    assert g._version == version


def test_float64_vjp_matches_independent_directional_finite_difference():
    values, logits, offsets = (tensor[0] for tensor in data())
    a, b = values * .37 + .1, logits * -.2 + .3
    g = upstream((4, 2), values.dtype, "ordinary")
    _, pullback = torch.func.vjp(lambda v, l: ragged_pool_forward_ad(v, l, offsets), values, logits)
    dv, dl = pullback(g)
    epsilon = 1e-5
    pure = reference(PARTITIONS[0])
    finite = ((pure(values + epsilon * a, logits + epsilon * b) -
               pure(values - epsilon * a, logits - epsilon * b)) * g).sum() / (2 * epsilon)
    torch.testing.assert_close((dv * a).sum() + (dl * b).sum(), finite, rtol=1e-8, atol=1e-10)


MAPS = [dimensions for dimensions in itertools.product((None, 0), repeat=3)
        if dimensions != (None, None, None)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dimensions", MAPS)
def test_shared_and_mapped_primals_and_offsets_match_serial_and_independent_gradients(dtype, dimensions):
    full = data(dtype)
    arguments = tuple(tensor[0] if dim is None else tensor for tensor, dim in zip(full, dimensions))
    before = [(tensor.clone(), tensor._version) for tensor in arguments]
    actual = torch.vmap(ragged_pool_forward_ad, in_dims=dimensions)(*arguments)
    grad = torch.func.grad(lambda v, l, o: ragged_pool_forward_ad(v, l, o).square().sum(), argnums=(0, 1))
    actual_grad = torch.vmap(grad, in_dims=dimensions)(*arguments)
    native_rows, reference_rows, reference_gradients = [], [], []
    for index in range(3):
        v, l, o = selected(arguments, dimensions, index)
        oracle = reference(tuple(o.tolist()))
        native_rows.append(ragged_pool(v, l, o))
        reference_rows.append(oracle(v, l))
        reference_gradients.append(torch.func.grad(lambda x, y: oracle(x, y).square().sum(),
                                                   argnums=(0, 1))(v, l))
    torch.testing.assert_close(actual, torch.stack(native_rows), **tolerance(dtype))
    torch.testing.assert_close(actual, torch.stack(reference_rows), **tolerance(dtype))
    for branch in (0, 1):
        torch.testing.assert_close(actual_grad[branch],
                                   torch.stack([row[branch] for row in reference_gradients]), **tolerance(dtype))
    for tensor, (copy, version) in zip(arguments, before):
        torch.testing.assert_close(tensor, copy, rtol=0, atol=0)
        assert tensor._version == version
    assert actual.data_ptr() not in {tensor.data_ptr() for tensor in arguments}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dimensions,out_dim", [((0, 0, 0), 0), ((1, 1, 1), 1),
                                               ((2, 1, 1), 2), ((-1, -1, -1), -1)])
@pytest.mark.parametrize("negative", [False, True])
def test_selected_lane_metadata_supports_nonleading_map_and_output_dimensions(dtype, dimensions, out_dim, negative):
    full = data(dtype)
    arguments = tuple(tensor.movedim(0, dim) for tensor, dim in zip(full, dimensions))
    if negative:
        arguments = (torch._neg_view(arguments[0]), torch._neg_view(arguments[1]), arguments[2])
    expected = torch.stack([ragged_pool(*selected(arguments, dimensions, index)) for index in range(3)])
    actual = torch.vmap(ragged_pool_forward_ad, in_dims=dimensions, out_dims=out_dim)(*arguments)
    torch.testing.assert_close(actual, expected.movedim(0, out_dim), **tolerance(dtype))
    gradient = torch.func.grad(lambda v, l, o: ragged_pool_forward_ad(v, l, o).square().sum(),
                               argnums=(0, 1))
    mapped_gradients = torch.vmap(gradient, in_dims=dimensions)(*arguments)
    expected_gradients = []
    for index in range(3):
        v, l, o = selected(arguments, dimensions, index)
        oracle = reference(tuple(o.tolist()))
        expected_gradients.append(torch.func.grad(lambda x, y: oracle(x, y).square().sum(),
                                                  argnums=(0, 1))(v, l))
    assert_pairs(mapped_gradients, tuple(torch.stack([row[j] for row in expected_gradients])
                                        for j in (0, 1)), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_shared_floating_gradients_are_unequal_lane_sum_and_not_lane_mean(dtype):
    values, logits, partitions = data(dtype)
    v, l = values[0], logits[0]
    cotangent = torch.linspace(-.9, .7, 24, dtype=dtype).reshape(3, 4, 2)
    def loss(x, y):
        return (torch.vmap(ragged_pool_forward_ad, in_dims=(None, None, 0))(x, y, partitions)
                * cotangent).sum()
    actual = torch.func.grad(loss, argnums=(0, 1))(v, l)
    rows = []
    native_rows = []
    for index, boundaries in enumerate(PARTITIONS):
        pure = reference(boundaries)
        rows.append(torch.func.grad(lambda x, y: (pure(x, y) * cotangent[index]).sum(),
                                    argnums=(0, 1))(v, l))
        x, y = v.clone().requires_grad_(), l.clone().requires_grad_()
        native_rows.append(torch.autograd.grad((ragged_pool(x, y, partitions[index]) *
                                               cotangent[index]).sum(), (x, y)))
    for branch in (0, 1):
        expected = torch.stack([row[branch] for row in rows]).sum(0)
        native = torch.stack([row[branch] for row in native_rows]).sum(0)
        torch.testing.assert_close(actual[branch], expected, **tolerance(dtype))
        torch.testing.assert_close(actual[branch], native, **tolerance(dtype))
        assert not torch.allclose(actual[branch], expected / 3, **tolerance(dtype))
    assert not torch.allclose(rows[0][0], rows[1][0], **tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("case", ["mixed", "all_empty", "zero_batch", "singleton"])
def test_lane_endpoints_and_empty_groups_keep_mathematical_zero_gradients(dtype, count, case):
    generator = torch.Generator().manual_seed(1523)
    boundaries = {"mixed": (0, 2, 2, 3, 6), "all_empty": (0, 0, 0),
                  "zero_batch": (0,), "singleton": (0, 1, 2, 3)}[case]
    n = boundaries[-1]
    values = torch.randn(count, n, 2, generator=generator, dtype=dtype)
    logits = torch.randn(count, n, generator=generator, dtype=dtype)
    offsets = torch.tensor(boundaries)
    result = torch.vmap(lambda v, l: ragged_pool_forward_ad(v, l, offsets))(values, logits)
    gradients = torch.vmap(torch.func.grad(
        lambda v, l: ragged_pool_forward_ad(v, l, offsets).square().sum(), argnums=(0, 1)))(values, logits)
    pure = reference(boundaries)
    expected = torch.stack([pure(v, l) for v, l in zip(values, logits)])
    torch.testing.assert_close(result, expected, **tolerance(dtype))
    expected_gradients = [torch.func.grad(lambda v, l: pure(v, l).square().sum(), argnums=(0, 1))(v, l)
                          for v, l in zip(values, logits)]
    assert_pairs(gradients, tuple(torch.stack([row[j] for row in expected_gradients])
                                 for j in (0, 1)), dtype)
    if case in ("all_empty", "zero_batch"):
        assert torch.count_nonzero(result) == 0
        assert all(torch.count_nonzero(g) == 0 for g in gradients)


@pytest.mark.parametrize("shared", [False, True])
def test_aggregate_endpoints_are_accepted_and_shared_inputs_are_charged_per_lane(shared):
    count, n, b = 8, 512, 16
    values = torch.zeros(count, n, 1, dtype=torch.float64)
    logits = torch.zeros(count, n, dtype=torch.float64)
    offsets = torch.arange(b + 1) * (n // b)
    if shared:
        result = torch.vmap(ragged_pool_forward_ad, in_dims=(None, 0, None))(values[0], logits, offsets)
    else:
        result = torch.vmap(ragged_pool_forward_ad, in_dims=(0, 0, None))(values, logits, offsets)
    assert result.shape == (count, b, 1)
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("case", ["zero_lanes", "nine_lanes", "aggregate_n", "aggregate_b", "shared_n"])
def test_map_resource_violations_are_rejected_before_native_forward(monkeypatch, case):
    count, n, b = {"zero_lanes": (0, 6, 4), "nine_lanes": (9, 6, 4),
                   "aggregate_n": (8, 513, 16), "aggregate_b": (8, 6, 17),
                   "shared_n": (8, 513, 1)}[case]
    values = torch.zeros(count, n, 1, dtype=torch.float64)
    logits = torch.zeros(count, n, dtype=torch.float64)
    offsets = torch.tensor([0] * b + [n])
    def forbidden(*args):
        raise AssertionError("invalid map envelope reached native execution")
    monkeypatch.setattr(torch.ops.ragged_pool.pool, "default", forbidden)
    arguments, dimensions = (values, logits, offsets), (0, 0, None)
    if case == "shared_n":
        arguments, dimensions = (values[0], logits, offsets), (None, 0, None)
    with pytest.raises(RuntimeError, match="mapped (lane count|aggregate)"):
        torch.vmap(ragged_pool_forward_ad, in_dims=dimensions)(*arguments)


@pytest.mark.parametrize("case", ["stepped_values", "negative_stepped_values", "stepped_logits",
                                  "stepped_offsets", "mixed_dtype", "wrong_offset_dtype",
                                  "decreasing", "wrong_end", "nonfinite", "range"])
def test_each_mapped_primal_lane_keeps_native_admission(case):
    values, logits, offsets = data()
    if case in ("stepped_values", "negative_stepped_values"):
        values = values.repeat_interleave(2, dim=-1)[..., ::2]
        if case == "negative_stepped_values":
            values = torch._neg_view(values)
    elif case == "stepped_logits":
        logits = logits.repeat_interleave(2, dim=-1)[..., ::2]
    elif case == "stepped_offsets":
        offsets = offsets.repeat_interleave(2, dim=-1)[..., ::2]
    elif case == "mixed_dtype":
        logits = logits.float()
    elif case == "wrong_offset_dtype":
        offsets = offsets.int()
    elif case == "decreasing":
        offsets[1] = torch.tensor([0, 3, 2, 4, 6])
    elif case == "wrong_end":
        offsets[1, -1] = 5
    elif case == "nonfinite":
        logits[1, 0] = float("nan")
    else:
        values[1, 0, 0] = 1001
    with pytest.raises(RuntimeError):
        torch.vmap(ragged_pool_forward_ad)(values, logits, offsets)


def test_native_pool_and_backward_are_serially_delegated(monkeypatch):
    values, logits, offsets = data()
    calls = {"pool": 0, "backward": 0}
    for operation, key in ((torch.ops.ragged_pool.pool, "pool"),
                           (torch.ops.ragged_pool._backward, "backward")):
        native = operation.default
        def delegated(*args, _native=native, _key=key):
            calls[_key] += 1
            return _native(*args)
        monkeypatch.setattr(operation, "default", delegated)
    result = torch.vmap(torch.func.grad(
        lambda v, l, o: ragged_pool_forward_ad(v, l, o).square().sum(),
        argnums=(0, 1)))(values, logits, offsets)
    assert len(result) == 2
    assert calls == {"pool": 3, "backward": 3}


@pytest.mark.parametrize("case", ["zero_lanes", "nine_lanes", "aggregate_n", "aggregate_b"])
def test_backward_map_bounds_are_enforced_before_native_derivative(monkeypatch, case):
    count, n, b = {"zero_lanes": (0, 6, 4), "nine_lanes": (9, 6, 4),
                   "aggregate_n": (8, 513, 16), "aggregate_b": (8, 6, 17)}[case]
    g = torch.zeros(count, b, 1, dtype=torch.float64)
    values = torch.zeros(n, 1, dtype=torch.float64)
    logits = torch.zeros(n, dtype=torch.float64)
    offsets = torch.tensor([0] * b + [n])
    def forbidden(*args):
        raise AssertionError("invalid map envelope reached native backward")
    monkeypatch.setattr(torch.ops.ragged_pool._backward, "default", forbidden)
    with pytest.raises(RuntimeError, match="mapped (lane count|aggregate)"):
        torch.vmap(_PackedBackward.apply, in_dims=(0, None, None, None))(g, values, logits, offsets)


@pytest.mark.parametrize("which", ["grad_output", "values", "logits", "offsets"])
def test_backward_bridge_saves_original_versions_for_second_reverse(which):
    values, logits, offsets = (tensor[0] for tensor in data())
    values.requires_grad_()
    logits.requires_grad_()
    grad_output = torch.randn(4, 2, dtype=values.dtype, requires_grad=True)
    arguments = dict(grad_output=grad_output, values=values, logits=logits, offsets=offsets)
    first = _PackedBackward.apply(grad_output, values, logits, offsets)
    with torch.no_grad():
        arguments[which].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(first[0].sum() + first[1].sum(),
                            (grad_output, values, logits))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("branch", ["values", "logits", "both"])
def test_backward_bridge_second_reverse_includes_absent_gradient_branches(dtype, branch):
    values, logits, offsets = (tensor[0] for tensor in data(dtype))
    g = upstream((4, 2), dtype, "transpose").requires_grad_()
    v, l = values.requires_grad_(), logits.requires_grad_()
    first = _PackedBackward.apply(g, v, l, offsets)
    selected_first = first if branch == "both" else (first[0] if branch == "values" else first[1],)
    objective = sum(part.square().sum() for part in selected_first)
    actual = torch.autograd.grad(objective, (g, v, l))
    rg, rv, rl = g.detach().clone().requires_grad_(), v.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
    pure = reference(PARTITIONS[0])
    expected_first = torch.autograd.grad(pure(rv, rl), (rv, rl), rg, create_graph=True)
    selected_expected = expected_first if branch == "both" else (expected_first[0] if branch == "values" else expected_first[1],)
    expected_objective = sum(part.square().sum() for part in selected_expected)
    expected = torch.autograd.grad(expected_objective, (rg, rv, rl), allow_unused=True)
    expected = tuple(torch.zeros_like(original) if gradient is None else gradient
                     for gradient, original in zip(expected, (rg, rv, rl)))
    assert_pairs(actual, expected, dtype)


@pytest.mark.parametrize("outer_mapped", ["values", "logits", "offsets"])
def test_nested_map_is_rejected_before_any_native_call(monkeypatch, outer_mapped):
    values, logits, offsets = data()
    outer = {"values": values.unsqueeze(0).expand(2, -1, -1, -1),
             "logits": logits.unsqueeze(0).expand(2, -1, -1),
             "offsets": offsets.unsqueeze(0).expand(2, -1, -1)}
    arguments = {"values": values, "logits": logits, "offsets": offsets}
    arguments[outer_mapped] = outer[outer_mapped]
    dimensions = tuple(0 if name == outer_mapped else None for name in ("values", "logits", "offsets"))
    def forbidden(*args):
        raise AssertionError("nested map reached native execution")
    monkeypatch.setattr(torch.ops.ragged_pool.pool, "default", forbidden)
    inner = torch.vmap(ragged_pool_forward_ad)
    with pytest.raises(RuntimeError, match="ragged-pool nested vmap is not supported"):
        torch.vmap(inner, in_dims=dimensions)(*arguments.values())


def test_forward_derivative_of_reverse_transform_is_explicitly_rejected():
    values, logits, offsets = (tensor[0] for tensor in data())
    gradient = torch.func.grad(lambda v, l: ragged_pool_forward_ad(v, l, offsets).square().sum(),
                               argnums=(0, 1))
    with pytest.raises(RuntimeError, match="forward AD of reverse derivatives is not supported"):
        torch.func.jvp(gradient, (values, logits), (torch.ones_like(values), torch.ones_like(logits)))


def test_mapped_jvp_is_rejected_instead_of_discarding_a_direction():
    values, logits, offsets = data()
    def mapped(v, l):
        return torch.vmap(ragged_pool_forward_ad, in_dims=(0, 0, None))(v, l, offsets[0])
    with pytest.raises((RuntimeError, NotImplementedError), match="(mapped forward AD|vmap|jvp)"):
        torch.func.jvp(mapped, (values, logits), (torch.ones_like(values), torch.ones_like(logits)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_chunked_axis_can_exceed_eight_with_nine_actual_native_calls(monkeypatch, dtype, chunk_size):
    values, logits, offsets = data(dtype, count=9)
    expected = torch.stack([reference(tuple(o.tolist()))(v, l)
                            for v, l, o in zip(values, logits, offsets)])
    original = torch.ops.ragged_pool.pool.default
    calls = []
    def delegated(*arguments):
        calls.append(arguments[0].shape)
        return original(*arguments)
    monkeypatch.setattr(torch.ops.ragged_pool.pool, "default", delegated)
    result = torch.vmap(ragged_pool_forward_ad, chunk_size=chunk_size)(values, logits, offsets)
    torch.testing.assert_close(result, expected, **tolerance(dtype))
    assert result.shape[0] == 9
    assert calls == [torch.Size((6, 2))] * 9


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["aggregate_n", "aggregate_b"])
def test_aggregate_limits_apply_to_each_chunk_not_the_complete_axis(monkeypatch, dtype, case):
    # One lane is admitted, but a chunk of four exceeds its declared envelope.
    n, b = (1025, 1) if case == "aggregate_n" else (6, 33)
    values = torch.zeros(9, n, 1, dtype=dtype)
    logits = torch.zeros(9, n, dtype=dtype)
    offsets = torch.tensor([0] * b + [n])
    original = torch.ops.ragged_pool.pool.default
    calls = []
    def delegated(*arguments):
        calls.append(arguments[0].shape)
        return original(*arguments)
    monkeypatch.setattr(torch.ops.ragged_pool.pool, "default", delegated)
    with pytest.raises(RuntimeError, match="mapped aggregate"):
        torch.vmap(ragged_pool_forward_ad, in_dims=(0, 0, None), chunk_size=4)(
            values, logits, offsets)
    assert calls == []
    result = torch.vmap(ragged_pool_forward_ad, in_dims=(0, 0, None), chunk_size=1)(
        values, logits, offsets)
    assert result.shape == (9, b, 1)
    assert torch.count_nonzero(result) == 0
    assert calls == [torch.Size((n, 1))] * 9


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["transpose", "stepped", "expanded", "negative"])
def test_native_backward_map_accepts_cotangent_views_without_mutation(dtype, kind):
    values, logits, offsets = data(dtype)
    if kind == "transpose":
        g = torch.linspace(-.7, .8, 24, dtype=dtype).reshape(3, 2, 4).transpose(1, 2)
    elif kind == "stepped":
        g = torch.linspace(-.7, .8, 48, dtype=dtype).reshape(3, 4, 4)[:, :, ::2]
    elif kind == "expanded":
        g = torch.tensor(.3, dtype=dtype).expand(3, 4, 2)
    else:
        g = torch._neg_view(torch.full((3, 4, 2), .4, dtype=dtype))
    tensors = (g, values, logits, offsets)
    snapshots = [(tensor.clone(), tensor._version) for tensor in tensors]
    actual = torch.vmap(_PackedBackward.apply)(*tensors)
    native_rows, expected_rows = [], []
    for index, boundaries in enumerate(PARTITIONS):
        native_rows.append(torch.ops.ragged_pool._backward.default(
            g[index], values[index], logits[index], offsets[index]))
        v, l = values[index].clone().requires_grad_(), logits[index].clone().requires_grad_()
        expected_rows.append(torch.autograd.grad(reference(boundaries)(v, l), (v, l), g[index]))
    for branch in (0, 1):
        torch.testing.assert_close(actual[branch], torch.stack([row[branch] for row in native_rows]),
                                   **tolerance(dtype))
        torch.testing.assert_close(actual[branch], torch.stack([row[branch] for row in expected_rows]),
                                   **tolerance(dtype))
    for tensor, (before, version) in zip(tensors, snapshots):
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
        assert tensor._version == version


@pytest.mark.parametrize("dtype", DTYPES)
def test_nested_reverse_transform_retains_pinned_setup_context_rejection(dtype):
    values, logits, offsets = (tensor[0] for tensor in data(dtype))
    first = torch.func.grad(lambda v: ragged_pool_forward_ad(v, logits, offsets).square().sum())
    with pytest.raises(RuntimeError, match="setup_context"):
        torch.func.grad(lambda v: first(v).square().sum())(values)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vmap_of_jvp_rejects_data_dependent_tangent_admission_instead_of_losing_it(dtype):
    values, logits, offsets = data(dtype)
    def tangent(v, l, a, b):
        return torch.func.jvp(
            lambda x, y: ragged_pool_forward_ad(x, y, offsets[0]),
            (v, l), (a, b),
        )[1]
    with pytest.raises(RuntimeError, match="(data-dependent control flow|[.]item|vmap)"):
        torch.vmap(tangent)(values, logits, torch.ones_like(values), torch.ones_like(logits))
