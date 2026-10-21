"""First-order forward AD math, native delegation and the explicit boundaries."""
import math

import pytest
import torch
from torch.autograd import forward_ad

from ragged_pool import ragged_pool, ragged_pool_forward_ad


DTYPES = (torch.float32, torch.float64)


def tolerance(dtype):
    return {"rtol": 4e-5, "atol": 4e-6} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-11}


def sample(dtype=torch.float64, case="mixed"):
    lengths = {"mixed": [2, 0, 1, 3], "singletons": [1, 1, 1],
               "empty": [0, 0], "zero_batch": []}[case]
    generator = torch.Generator().manual_seed(1403)
    values = torch.randn(sum(lengths), 2, generator=generator, dtype=dtype)
    logits = torch.randn(sum(lengths), generator=generator, dtype=dtype)
    direction_values = torch.randn(values.shape, generator=generator, dtype=dtype) / 3
    direction_logits = torch.randn(logits.shape, generator=generator, dtype=dtype) / 3
    offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    return values, logits, offsets, direction_values, direction_logits


def reference(boundaries):
    # Freeze ordinary Python boundaries before entering any function transform.
    def pool(values, logits):
        zero = values.sum(dim=0) * 0 + logits.sum() * 0
        rows = []
        for start, stop in zip(boundaries, boundaries[1:]):
            if start == stop:
                rows.append(zero)
            else:
                weights = torch.softmax(logits[start:stop], dim=0)
                rows.append((weights[:, None] * values[start:stop]).sum(dim=0))
        return torch.stack(rows) if rows else zero.unsqueeze(0)[:0].clone()
    return pool


def jvp(values, logits, offsets, direction_values, direction_logits, route="func"):
    if route == "func":
        directions = tuple(torch.zeros_like(primal) if direction is None else direction
                           for primal, direction in ((values, direction_values), (logits, direction_logits)))
        return torch.func.jvp(lambda v, l: ragged_pool_forward_ad(v, l, offsets),
                              (values, logits), directions)
    with forward_ad.dual_level():
        dual_values = values if direction_values is None else forward_ad.make_dual(values, direction_values)
        dual_logits = logits if direction_logits is None else forward_ad.make_dual(logits, direction_logits)
        return forward_ad.unpack_dual(ragged_pool_forward_ad(dual_values, dual_logits, offsets))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("route", ["func", "dual"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_hand_calculated_primal_tangent_and_reverse_boundary(dtype, route, requires_grad):
    values = torch.tensor([[2., 0.], [0., 4.]], dtype=dtype, requires_grad=requires_grad)
    logits = torch.tensor([0., math.log(3)], dtype=dtype, requires_grad=requires_grad)
    offsets = torch.tensor([0, 2, 2])
    direction_values = torch.tensor([[1., -1.], [2., 1.]], dtype=dtype)
    direction_logits = torch.tensor([.1, -.2], dtype=dtype)
    primal, tangent = jvp(values, logits, offsets, direction_values, direction_logits, route)
    torch.testing.assert_close(primal, values.new_tensor([[.5, 3.], [0., 0.]]), **tolerance(dtype))
    torch.testing.assert_close(tangent, values.new_tensor([[1.8625, .275], [0., 0.]]), **tolerance(dtype))
    assert primal.requires_grad == tangent.requires_grad == requires_grad
    if requires_grad:
        ordinary = torch.autograd.grad(ragged_pool(values, logits, offsets).sum(),
                                       (values, logits), retain_graph=True)
        actual = torch.autograd.grad(primal.sum(), (values, logits), retain_graph=True)
        for actual_gradient, expected_gradient in zip(actual, ordinary):
            torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance(dtype))
        with pytest.raises(RuntimeError, match="first-order tangents only"):
            torch.autograd.grad(tangent.sum(), (values, logits))
    assert offsets.grad is None


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("route", ["func", "dual"])
@pytest.mark.parametrize("case", ["mixed", "singletons", "empty", "zero_batch"])
@pytest.mark.parametrize("branch", ["both", "values", "logits", "zero"])
def test_jvp_matches_independent_primitive_torch(dtype, route, case, branch):
    values, logits, offsets, a, b = sample(dtype, case)
    if branch in ("logits", "zero"):
        a = None if branch == "logits" else torch.zeros_like(a)
    if branch in ("values", "zero"):
        b = None if branch == "values" else torch.zeros_like(b)
    expected = torch.func.jvp(reference(offsets.tolist()), (values, logits),
                              (torch.zeros_like(values) if a is None else a,
                               torch.zeros_like(logits) if b is None else b))
    actual = jvp(values, logits, offsets, a, b, route)
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle, **tolerance(dtype))
        assert bool(torch.isfinite(result).all())
        assert result.shape == (offsets.numel() - 1, values.shape[1])
        assert result.dtype == dtype and result.device.type == "cpu"
    if case == "mixed":
        assert torch.count_nonzero(actual[1][1]) == 0
        torch.testing.assert_close(actual[1][2], torch.zeros_like(values[2]) if a is None else a[2],
                                   rtol=0, atol=0)
    if case == "singletons":
        torch.testing.assert_close(actual[1], torch.zeros_like(values) if a is None else a,
                                   rtol=0, atol=0)


def test_float64_jvp_matches_central_directional_difference_of_independent_math():
    values, logits, offsets, a, b = sample()
    pool = reference(offsets.tolist())
    epsilon = 1e-5
    difference = (pool(values + epsilon * a, logits + epsilon * b) -
                  pool(values - epsilon * a, logits - epsilon * b)) / (2 * epsilon)
    _, tangent = jvp(values, logits, offsets, a, b)
    torch.testing.assert_close(tangent, difference, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_jvp_and_independent_vjp_obey_adjoint_identity(dtype):
    values, logits, offsets, a, b = sample(dtype)
    _, tangent = jvp(values, logits, offsets, a, b)
    cotangent = torch.linspace(-.6, .8, tangent.numel(), dtype=dtype).reshape(tangent.shape)
    v, l = values.clone().requires_grad_(), logits.clone().requires_grad_()
    gradient_values, gradient_logits = torch.autograd.grad(reference(offsets.tolist())(v, l),
                                                          (v, l), cotangent)
    torch.testing.assert_close((tangent * cotangent).sum(),
                               (gradient_values * a).sum() + (gradient_logits * b).sum(),
                               **tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_tangent_is_linear_and_group_constant_logit_directions_cancel(dtype):
    values, logits, offsets, a, b = sample(dtype)
    c, d = a.flip(0).contiguous(), b.flip(0).contiguous()
    _, left = jvp(values, logits, offsets, a, b)
    _, right = jvp(values, logits, offsets, c, d)
    _, combined = jvp(values, logits, offsets, .7 * a - .3 * c, .7 * b - .3 * d)
    torch.testing.assert_close(combined, .7 * left - .3 * right, **tolerance(dtype))
    constant = logits.new_tensor([.7, .7, -.2, .4, .4, .4])
    _, invariant = jvp(values, logits, offsets, None, constant, "dual")
    torch.testing.assert_close(invariant, torch.zeros_like(invariant), rtol=0, atol=1e-12)
    _, shifted = jvp(values, logits + constant, offsets, a, b)
    torch.testing.assert_close(shifted, left, **tolerance(dtype))


def direction_view(tensor, layout):
    if layout == "expanded":
        return tensor.new_tensor(.25).expand(tensor.shape)
    if layout == "negative":
        return torch._neg_view(tensor)
    if layout in ("stepped", "negative_stepped"):
        storage = tensor.new_empty((*tensor.shape, 2))
        storage[..., 0] = tensor
        result = storage[..., 0]
        return torch._neg_view(result) if layout == "negative_stepped" else result
    if tensor.dim() == 2:
        return tensor.t().contiguous().t()
    return tensor


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["expanded", "negative", "stepped", "transposed", "negative_stepped"])
def test_strided_direction_views_are_normalized_without_input_mutation(dtype, layout):
    values, logits, offsets, a, b = sample(dtype)
    a, b = direction_view(a, layout), direction_view(b, layout)
    arguments = (values, logits, offsets, a, b)
    snapshots = [(tensor.clone(), tensor._version) for tensor in arguments]
    expected = torch.func.jvp(reference(offsets.tolist()), (values, logits), (a, b))
    first = jvp(values, logits, offsets, a, b, "dual")
    second = jvp(values, logits, offsets, a, b, "dual")
    for result, oracle, repeated in zip(first, expected, second):
        torch.testing.assert_close(result, oracle, **tolerance(dtype))
        assert result.is_contiguous()
        assert result.data_ptr() != repeated.data_ptr()
        assert result.data_ptr() not in {tensor.data_ptr() for tensor in arguments}
    assert first[0].data_ptr() != first[1].data_ptr()
    first[0].add_(17)
    first[1].add_(19)
    for tensor, (snapshot, version) in zip(arguments, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        assert tensor._version == version


@pytest.mark.parametrize("dtype", DTYPES)
def test_direction_bounds_include_both_finite_endpoints(dtype):
    values = torch.tensor([[1000., -1000.], [-1000., 1000.]], dtype=dtype)
    logits = torch.tensor([-1000., 1000.], dtype=dtype)
    offsets = torch.tensor([0, 2, 2])
    a, b = torch.full_like(values, -1000), torch.full_like(logits, 1000)
    primal, tangent = jvp(values, logits, offsets, a, b, "dual")
    torch.testing.assert_close(primal, values.new_tensor([[-1000., 1000.], [0., 0.]]), rtol=0, atol=0)
    torch.testing.assert_close(tangent, values.new_tensor([[-1000., -1000.], [0., 0.]]), rtol=0, atol=0)


@pytest.mark.parametrize("route", ["func", "dual"])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1001., -1001.])
def test_nonfinite_and_out_of_range_directions_are_rejected_before_native_jvp(monkeypatch, route, index, bad):
    values, logits, offsets, a, b = sample()
    directions = [a, b]
    directions[index].view(-1)[0] = bad

    def forbidden(*args):
        raise AssertionError("invalid directions reached the native derivative")

    monkeypatch.setattr(torch.ops.ragged_pool._double_backward, "default", forbidden)
    with pytest.raises(RuntimeError, match="direction must be finite"):
        jvp(values, logits, offsets, *directions, route)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("case", ["shape", "dtype", "sparse", "meta"])
def test_malformed_direction_metadata_is_rejected_without_device_transfer(index, case):
    values, logits, offsets, a, b = sample()
    directions = [a, b]
    if case == "shape":
        directions[index] = directions[index].unsqueeze(0)
    elif case == "dtype":
        directions[index] = directions[index].float()
    elif case == "sparse":
        directions[index] = directions[index].to_sparse()
    else:
        directions[index] = torch.empty_like(directions[index], device="meta")
    original_device = directions[index].device
    # make_dual may reject a malformed direction before the bridge is reached.
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
        jvp(values, logits, offsets, *directions, "dual")
    assert directions[index].device == original_device


@pytest.mark.parametrize("case", ["values_layout", "negative_values_layout", "mixed_dtype",
                                  "meta_values", "sparse_values", "bad_offsets_dtype",
                                  "decreasing_offsets", "wrong_offsets_end", "nonfinite", "range"])
def test_bridge_preserves_original_primal_admission(case):
    values, logits, offsets, _, _ = sample()
    if case in ("values_layout", "negative_values_layout"):
        values = direction_view(values, "stepped")
        if case == "negative_values_layout":
            values = torch._neg_view(values)
    elif case == "mixed_dtype":
        logits = logits.float()
    elif case == "meta_values":
        values = torch.empty_like(values, device="meta")
    elif case == "sparse_values":
        values = values.to_sparse()
    elif case == "bad_offsets_dtype":
        offsets = offsets.int()
    elif case == "decreasing_offsets":
        offsets = torch.tensor([0, 3, 2, 6])
    elif case == "wrong_offsets_end":
        offsets = torch.tensor([0, 2, 5])
    elif case == "nonfinite":
        logits[0] = float("nan")
    else:
        values[0, 0] = 1001
    for pool in (ragged_pool, ragged_pool_forward_ad):
        with pytest.raises((RuntimeError, NotImplementedError)):
            pool(values, logits, offsets)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_saved_original_versions_are_checked_before_primal_backward(index):
    values, logits, offsets, _, _ = sample()
    values.requires_grad_()
    logits.requires_grad_()
    arguments = (values, logits, offsets)
    output = ragged_pool_forward_ad(*arguments)
    with torch.no_grad():
        arguments[index].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output.sum(), (values, logits))


def reverse_pair(pool, values, logits, a, b):
    v, l = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    output = pool(v, l)
    first = torch.autograd.grad(output.square().sum(), (v, l), create_graph=True)
    objective = (first[0] * a).sum() + (first[1] * b).sum() + v.sum() * 0 + l.sum() * 0
    second = torch.autograd.grad(objective, (v, l), allow_unused=True)
    return first, tuple(torch.zeros_like(primal) if gradient is None else gradient
                        for primal, gradient in zip((v, l), second))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["mixed", "empty", "zero_batch"])
def test_ordinary_primal_reverse_and_second_reverse_match_independent_torch(dtype, case):
    values, logits, offsets, a, b = sample(dtype, case)
    native = reverse_pair(lambda v, l: ragged_pool_forward_ad(v, l, offsets), values, logits, a, b)
    expected = reverse_pair(reference(offsets.tolist()), values, logits, a, b)
    for order_native, order_expected in zip(native, expected):
        for actual, oracle in zip(order_native, order_expected):
            torch.testing.assert_close(actual, oracle, **tolerance(dtype))


@pytest.mark.parametrize("route", ["func", "dual"])
def test_direction_gradients_are_explicitly_refused_even_with_constant_primals(route):
    values, logits, offsets, a, b = sample()
    a.requires_grad_()
    _, tangent = jvp(values, logits, offsets, a, b, route)
    assert tangent.requires_grad
    with pytest.raises(RuntimeError, match="first-order tangents only"):
        torch.autograd.grad(tangent.sum(), a)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("outer_variables", ["primals", "directions"])
def test_nested_function_forward_mode_is_explicitly_rejected(dtype, outer_variables):
    values, logits, offsets, a, b = sample(dtype)

    def tangent(v, l):
        pool = lambda x, y: ragged_pool_forward_ad(x, y, offsets)
        if outer_variables == "directions":
            return torch.func.jvp(pool, (values, logits), (v, l))[1]
        return torch.func.jvp(pool, (v, l), (a, b))[1]

    primals = (a, b) if outer_variables == "directions" else (values, logits)
    with pytest.raises(RuntimeError, match="ragged-pool nested forward AD is not supported"):
        torch.func.jvp(tangent, primals, (a, b))


def test_first_order_jvp_delegates_to_the_existing_native_second_backward(monkeypatch):
    values, logits, offsets, a, b = sample()
    native = torch.ops.ragged_pool._double_backward.default
    calls = []

    def delegated(direction_values, direction_logits, zero, v, l, o):
        torch.testing.assert_close(direction_values, a, rtol=0, atol=0)
        torch.testing.assert_close(direction_logits, b, rtol=0, atol=0)
        torch.testing.assert_close(v, values, rtol=0, atol=0)
        torch.testing.assert_close(l, logits, rtol=0, atol=0)
        torch.testing.assert_close(o, offsets, rtol=0, atol=0)
        assert zero.shape == (offsets.numel() - 1, values.shape[1])
        assert zero.dtype == values.dtype and zero.device.type == "cpu"
        assert torch.count_nonzero(zero) == 0
        result = native(direction_values, direction_logits, zero, v, l, o)
        calls.append(result)
        return result

    monkeypatch.setattr(torch.ops.ragged_pool._double_backward, "default", delegated)
    _, tangent = jvp(values, logits, offsets, a, b, "dual")
    assert len(calls) == 1
    torch.testing.assert_close(tangent, calls[0][0], rtol=0, atol=0)
    assert tangent.data_ptr() != calls[0][0].data_ptr()
