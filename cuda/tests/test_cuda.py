"""Functional CUDA candidate checks, separate from native diagnostic acceptance."""
import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from ragged_pool_cuda import ragged_pool, ragged_pool_reference


DTYPES = (torch.float32, torch.float64)
REJECTION = (RuntimeError, ValueError, NotImplementedError)


@pytest.fixture(scope="module", autouse=True)
def declared_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128", "Use the declared isolated CUDA environment"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.is_available(), "An unavailable CUDA prerequisite is not a skipped pass"
    torch.cuda.set_device(0)


def tolerance(dtype, endpoints=False):
    if dtype == torch.float32:
        return {"rtol": 2e-5, "atol": 2e-6}
    return {"rtol": 2e-10, "atol": 2e-10 if endpoints else 2e-12}


def inputs(dtype=torch.float64, boundaries=(0, 2, 2, 5, 6), width=3, requires_grad=False):
    count = boundaries[-1]
    values = (torch.arange(count * width, dtype=dtype).reshape(count, width) / 7 - 1).to("cuda")
    logits = torch.linspace(-1, 1, count, dtype=dtype).to("cuda")
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda")
    return values.requires_grad_(requires_grad), logits.requires_grad_(requires_grad), offsets


def primitive(values, logits, boundaries):
    """Use known Python boundaries and independent differentiable double math."""
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        selected_values = values[start:stop].to(torch.float64)
        selected_logits = logits[start:stop].to(torch.float64)
        if start == stop:
            row = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
        else:
            weights = torch.softmax(selected_logits, dim=0)
            row = (weights[:, None] * selected_values).sum(dim=0)
        rows.append(row.to(values.dtype))
    if rows:
        return torch.stack(rows)
    return values.sum(dim=0).unsqueeze(0)[:0] * 0 + logits.sum() * 0


def native(values, logits, offsets):
    return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets)


def private_backward(upstream, values, logits, offsets):
    return torch.ops.ragged_pool_cuda._backward.default(upstream, values, logits, offsets)


def valid_after_refusal():
    values, logits, offsets = inputs()
    actual = ragged_pool(values, logits, offsets)
    expected = primitive(values, logits, (0, 2, 2, 5, 6))
    torch.testing.assert_close(actual, expected, **tolerance(torch.float64))
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_and_public_reference_forward_and_both_gradients(dtype):
    boundaries = (0, 2, 2, 5, 6)
    values, logits, offsets = inputs(dtype, boundaries, requires_grad=True)
    rv, rl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    pv, pl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    upstream = torch.linspace(-0.7, 0.9, 12, dtype=dtype, device="cuda").reshape(4, 3)
    actual = ragged_pool(values, logits, offsets)
    public_reference = ragged_pool_reference(pv, pl, offsets)
    expected = primitive(rv, rl, boundaries)
    gradients = torch.autograd.grad(actual, (values, logits), upstream)
    reference_gradients = torch.autograd.grad(expected, (rv, rl), upstream)
    public_gradients = torch.autograd.grad(public_reference, (pv, pl), upstream)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    torch.testing.assert_close(public_reference, expected, **tolerance(dtype))
    for actual_gradient, public_gradient, expected_gradient in zip(gradients, public_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance(dtype))
        torch.testing.assert_close(public_gradient, expected_gradient, **tolerance(dtype))
        assert bool(torch.isfinite(actual_gradient).all())
    assert float(gradients[1][-1]) == 0
    assert offsets.grad is None


def test_hand_calculated_pool_and_first_derivatives():
    values = torch.tensor([[1., 2.], [3., -2.], [7., 5.]], dtype=torch.float64,
                          device="cuda", requires_grad=True)
    logits = torch.tensor([0., math.log(3), 0.], dtype=torch.float64, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 3], dtype=torch.int64, device="cuda")
    upstream = torch.tensor([[2., -3.], [4., 8.], [-1., 2.]], dtype=torch.float64, device="cuda")
    actual = ragged_pool(values, logits, offsets)
    dv, dl = torch.autograd.grad(actual, (values, logits), upstream)
    expected_output = torch.tensor([[2.5, -1.], [0., 0.], [7., 5.]], dtype=torch.float64, device="cuda")
    expected_dv = torch.tensor([[0.5, -0.75], [1.5, -2.25], [-1., 2.]], dtype=torch.float64, device="cuda")
    expected_dl = torch.tensor([-3., 3., 0.], dtype=torch.float64, device="cuda")
    for actual_value, expected in ((actual, expected_output), (dv, expected_dv), (dl, expected_dl)):
        torch.testing.assert_close(actual_value, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ("constant", "near_ties", "wide_spread"))
def test_range_endpoints_and_cancellation_match_independent_math(dtype, case):
    boundaries = (0, 2, 2, 4, 5)
    values = torch.tensor([[-1000., 1000.], [1000., -1000.], [-1000., 1000.],
                           [1000., -1000.], [-1000., 1000.]], dtype=dtype, device="cuda", requires_grad=True)
    scores = {"constant": [1000] * 5, "near_ties": [1000, 999, -999, -1000, 1000],
              "wide_spread": [1000, -1000, -1000, 1000, -1000]}[case]
    logits = torch.tensor(scores, dtype=dtype, device="cuda", requires_grad=True)
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda")
    rv, rl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    upstream = torch.tensor([[1., -2.], [3., -4.], [0.5, -0.25], [1., 1.]], dtype=dtype, device="cuda")
    actual, expected = ragged_pool(values, logits, offsets), primitive(rv, rl, boundaries)
    actual_gradients = torch.autograd.grad(actual, (values, logits), upstream)
    expected_gradients = torch.autograd.grad(expected, (rv, rl), upstream)
    torch.testing.assert_close(actual, expected, **tolerance(dtype, endpoints=True))
    for observed, wanted in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted, **tolerance(dtype, endpoints=True))


def test_native_float64_finite_difference_gradcheck():
    values = torch.tensor([[0.2, -0.1], [0.4, 0.7], [-0.3, 0.5]], dtype=torch.float64,
                          device="cuda", requires_grad=True)
    logits = torch.tensor([-0.2, 0.3, 0.1], dtype=torch.float64, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 3], dtype=torch.int64, device="cuda")
    assert torch.autograd.gradcheck(lambda v, l: ragged_pool(v, l, offsets), (values, logits),
                                    eps=1e-6, atol=1e-5, rtol=1e-4, check_undefined_grad=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boundaries", ((0,), (0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3)))
def test_empty_and_singleton_output_and_gradient_paths(dtype, boundaries):
    values, logits, offsets = inputs(dtype, boundaries, requires_grad=True)
    rv, rl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    actual, expected = ragged_pool(values, logits, offsets), primitive(rv, rl, boundaries)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    actual_gradients = torch.autograd.grad(actual.sum(), (values, logits))
    expected_gradients = torch.autograd.grad(expected.sum(), (rv, rl))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance(dtype))
    torch.testing.assert_close(actual_gradients[1], torch.zeros_like(logits), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_output_row_cotangents_contribute_no_input_gradient(dtype):
    values, logits, offsets = inputs(dtype, (0, 0, 2, 2), requires_grad=True)
    actual = ragged_pool(values, logits, offsets)
    actual[[0, 2]].sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values), rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ("sum", "expanded", "transposed", "stepped", "negative"))
def test_upstream_gradient_views_match_independent_backward(layout):
    boundaries = (0, 2, 2, 5, 6)
    values, logits, offsets = inputs(requires_grad=True)
    rv, rl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    actual, expected = ragged_pool(values, logits, offsets), primitive(rv, rl, boundaries)
    if layout == "sum":
        observed = torch.autograd.grad(actual.sum(), (values, logits))
        wanted = torch.autograd.grad(expected.sum(), (rv, rl))
    else:
        if layout == "expanded":
            upstream = torch.tensor([[1.5]], dtype=torch.float64, device="cuda").expand(4, 3)
            assert upstream.stride() == (0, 0)
        elif layout == "transposed":
            upstream = torch.arange(12, dtype=torch.float64, device="cuda").reshape(3, 4).t() / 10
        elif layout == "stepped":
            upstream = torch.arange(24, dtype=torch.float64, device="cuda").reshape(4, 6)[:, ::2]
        else:
            upstream = torch._neg_view(torch.ones(4, 3, dtype=torch.float64, device="cuda"))
            assert upstream.is_neg()
        if layout in ("expanded", "transposed", "stepped"):
            assert not upstream.is_contiguous()
        observed = torch.autograd.grad(actual, (values, logits), upstream)
        wanted = torch.autograd.grad(expected, (rv, rl), upstream)
        direct = private_backward(upstream, values.detach(), logits.detach(), offsets)
        for direct_gradient, expected_gradient in zip(direct, wanted):
            torch.testing.assert_close(direct_gradient, expected_gradient, **tolerance(torch.float64))
    for actual_gradient, expected_gradient in zip(observed, wanted):
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance(torch.float64))


@pytest.mark.parametrize("dtype", DTYPES)
def test_declared_maximum_shape_has_complete_output_and_gradient_writes(dtype):
    values = torch.linspace(-1000, 1000, 4096 * 128, dtype=dtype, device="cuda").reshape(4096, 128)
    values.requires_grad_()
    logits = torch.zeros(4096, dtype=dtype, device="cuda", requires_grad=True)
    offsets = torch.arange(0, 4097, 32, dtype=torch.int64, device="cuda")
    actual = ragged_pool(values, logits, offsets)
    grouped = values.detach().double().reshape(128, 32, 128)
    means = grouped.mean(dim=1)
    expected = means.to(dtype)
    torch.testing.assert_close(actual, expected, **tolerance(dtype, endpoints=True))
    dv, dl = torch.autograd.grad(actual.sum(), (values, logits))
    torch.testing.assert_close(dv, torch.full_like(dv, 1 / 32), rtol=0, atol=0)
    expected_dl = ((grouped - means[:, None, :]).sum(dim=2) / 32).reshape(-1).to(dtype)
    torch.testing.assert_close(dl, expected_dl, **tolerance(dtype, endpoints=True))
    assert bool(torch.isfinite(dl).all())
    assert dl.shape == logits.shape


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_group_reaches_the_maximum_item_count(dtype):
    values = torch.linspace(-1, 1, 4096, dtype=dtype, device="cuda").reshape(4096, 1)
    values.requires_grad_()
    logits = torch.zeros(4096, dtype=dtype, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 4096], dtype=torch.int64, device="cuda")
    mean = values.detach().double().mean(dim=0, keepdim=True)
    actual = ragged_pool(values, logits, offsets)
    torch.testing.assert_close(actual, mean.to(dtype), **tolerance(dtype))
    dv, dl = torch.autograd.grad(actual.sum(), (values, logits))
    torch.testing.assert_close(dv, torch.full_like(dv, 1 / 4096), rtol=0, atol=0)
    expected_dl = ((values.detach().double() - mean)[:, 0] / 4096).to(dtype)
    torch.testing.assert_close(dl, expected_dl, **tolerance(dtype))


@pytest.mark.parametrize("pool_name", ("public", "native", "reference"))
@pytest.mark.parametrize("bad_offsets", ((1, 2, 6), (0, 2, 5), (0, 3, 2, 6), (0, -1, 6),
                                       (0, 7, 6), (0, -(2**63), 6), (0, 2**63 - 1, 6)))
def test_invalid_device_offset_contents_refuse_and_valid_call_recovers(pool_name, bad_offsets):
    values, logits, _ = inputs()
    offsets = torch.tensor(bad_offsets, dtype=torch.int64, device="cuda")
    pool = {"public": ragged_pool, "native": native, "reference": ragged_pool_reference}[pool_name]
    with pytest.raises(REJECTION):
        pool(values, logits, offsets)
    with pytest.raises(REJECTION):
        private_backward(torch.ones(len(bad_offsets) - 1, 3, dtype=values.dtype, device="cuda"),
                         values, logits, offsets)
    valid_after_refusal()


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -float("inf"), 1000.01, -1000.01))
def test_invalid_value_contents_refuse_before_forward_or_backward_indexing(index, bad_value):
    arguments = list(inputs())
    arguments[index].view(-1)[0] = bad_value
    with pytest.raises(REJECTION):
        ragged_pool(*arguments)
    with pytest.raises(REJECTION):
        private_backward(torch.ones(4, 3, dtype=torch.float64, device="cuda"), *arguments)
    valid_after_refusal()


@pytest.mark.parametrize("case", ("rank", "logit_rank", "offset_rank", "n_mismatch", "mixed_dtype",
                                 "half", "integer", "offset_int32", "missing_offsets", "d_zero",
                                 "d_too_large", "n_too_large", "b_too_large", "values_view",
                                 "logits_view", "offsets_view", "sparse", "complex_conjugate"))
@pytest.mark.parametrize("pool_name", ("public", "native", "reference"))
def test_malformed_metadata_is_rejected_without_breaking_later_calls(case, pool_name):
    pool = {"public": ragged_pool, "native": native, "reference": ragged_pool_reference}[pool_name]
    values, logits, offsets = inputs()
    if case == "rank": values = values.flatten()
    elif case == "logit_rank": logits = logits[:, None]
    elif case == "offset_rank": offsets = offsets[:, None]
    elif case == "n_mismatch": logits = logits[:-1]
    elif case == "mixed_dtype": logits = logits.float()
    elif case == "half": values, logits = values.half(), logits.half()
    elif case == "integer": values, logits = values.long(), logits.long()
    elif case == "offset_int32": offsets = offsets.int()
    elif case == "missing_offsets": offsets = offsets[:0]
    elif case == "d_zero": values = values[:, :0]
    elif case == "d_too_large": values = torch.zeros(6, 129, dtype=values.dtype, device="cuda")
    elif case == "n_too_large":
        values = torch.zeros(4097, 1, dtype=values.dtype, device="cuda")
        logits = torch.zeros(4097, dtype=values.dtype, device="cuda")
        offsets = torch.tensor([0, 4097], dtype=torch.int64, device="cuda")
    elif case == "b_too_large":
        offsets = torch.cat((torch.zeros(129, dtype=torch.int64, device="cuda"), offsets[-1:]))
    elif case == "values_view": values = torch.zeros(3, 6, dtype=values.dtype, device="cuda").t()
    elif case == "logits_view": logits = torch.zeros(12, dtype=values.dtype, device="cuda")[::2]
    elif case == "offsets_view":
        offsets = torch.tensor([0, -1, 2, -1, 2, -1, 5, -1, 6, -1], dtype=torch.int64, device="cuda")[::2]
    elif case == "sparse": values = values.to_sparse()
    elif case == "complex_conjugate": values = values.to(torch.complex128).conj()
    with pytest.raises(REJECTION):
        pool(values, logits, offsets)
    with pytest.raises(REJECTION):
        private_backward(torch.ones(4, 3, dtype=torch.float64, device="cuda"), values, logits, offsets)
    valid_after_refusal()


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("pool_name", ("public", "native", "reference"))
def test_cpu_mixed_device_inputs_and_logical_negative_primals_refuse(index, pool_name):
    pool = {"public": ragged_pool, "native": native, "reference": ragged_pool_reference}[pool_name]
    arguments = list(inputs())
    arguments[index] = arguments[index].cpu()
    with pytest.raises(REJECTION):
        pool(*arguments)
    assert arguments[index].device.type == "cpu"
    negative = list(inputs())
    negative[index] = torch._neg_view(-negative[index])
    assert negative[index].is_neg()
    with pytest.raises(REJECTION):
        pool(*negative)
    with pytest.raises(REJECTION):
        private_backward(torch.ones(4, 3, dtype=torch.float64, device="cuda"), *negative)
    valid_after_refusal()


@pytest.mark.parametrize("case", ("shape", "dtype", "cpu", "sparse", "nan", "infinite", "range"))
def test_invalid_upstream_is_rejected_and_valid_backward_recovers(case):
    values, logits, offsets = inputs()
    upstream = torch.ones(4, 3, dtype=values.dtype, device="cuda")
    if case == "shape": upstream = upstream[:, :1]
    elif case == "dtype": upstream = upstream.float()
    elif case == "cpu": upstream = upstream.cpu()
    elif case == "sparse": upstream = upstream.to_sparse()
    elif case == "nan": upstream[0, 0] = float("nan")
    elif case == "infinite": upstream[0, 0] = float("inf")
    elif case == "range": upstream[0, 0] = 1000.01
    with pytest.raises(REJECTION):
        private_backward(upstream, values, logits, offsets)
    valid = torch.ones(4, 3, dtype=values.dtype, device="cuda")
    gradients = private_backward(valid, values, logits, offsets)
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    torch.cuda.synchronize()


def test_outputs_gradients_and_saved_input_storage_are_owned_correctly():
    values, logits, offsets = inputs(requires_grad=True)
    arguments = (values, logits, offsets)
    copies = [value.detach().clone() for value in arguments]
    versions = [value._version for value in arguments]
    first, second = ragged_pool(*arguments), ragged_pool(*arguments)
    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() not in {value.data_ptr() for value in arguments}
    gradients = torch.autograd.grad(first.sum(), (values, logits))
    assert len({value.data_ptr() for value in (*gradients, *arguments)}) == 5
    with torch.no_grad():
        first.add_(7)
    torch.testing.assert_close(second, ragged_pool(*arguments), **tolerance(values.dtype))
    for value, copy, version in zip(arguments, copies, versions):
        torch.testing.assert_close(value, copy, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize("index", (0, 1, 2))
def test_mutating_each_original_saved_tensor_is_reported(index):
    arguments = list(inputs(requires_grad=True))
    actual = ragged_pool(*arguments)
    with torch.no_grad():
        arguments[index].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        actual.sum().backward()
    valid_after_refusal()


def test_second_reverse_derivative_is_explicitly_refused():
    values, logits, offsets = inputs(requires_grad=True)
    with pytest.raises(RuntimeError, match="(?i)second|first.order|higher"):
        first = torch.autograd.grad(ragged_pool(values, logits, offsets).square().sum(),
                                    (values, logits), create_graph=True)
        torch.autograd.grad(sum(value.square().sum() for value in first), (values, logits))
    valid_after_refusal()


def test_nondefault_stream_with_explicit_caller_dependencies():
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        boundaries = (0, 2, 2, 5, 6)
        values, logits, offsets = inputs(requires_grad=True)
        rv, rl = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
        actual = ragged_pool(values, logits, offsets)
        expected = primitive(rv, rl, boundaries)
        actual_gradients = torch.autograd.grad(actual.square().sum(), (values, logits))
        expected_gradients = torch.autograd.grad(expected.square().sum(), (rv, rl))
        # Consume on this stream before any default-stream comparison can wait
        # for an accidentally misplaced default-stream kernel.
        actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach().cpu()
        actual_gradients_cpu = [value.detach().cpu() for value in actual_gradients]
        expected_gradients_cpu = [value.detach().cpu() for value in expected_gradients]
    current.wait_stream(stream)
    stream.synchronize()
    torch.testing.assert_close(actual_cpu, expected_cpu, **tolerance(values.dtype))
    for observed, wanted in zip(actual_gradients_cpu, expected_gradients_cpu):
        torch.testing.assert_close(observed, wanted, **tolerance(values.dtype))


def test_registration_forward_and_backward_schema_autograd_and_fake_metadata():
    values, logits, offsets = inputs(requires_grad=True)
    checks = ("test_schema", "test_autograd_registration", "test_faketensor")
    forward = torch.library.opcheck(torch.ops.ragged_pool_cuda.pool.default,
                                    (values, logits, offsets), test_utils=checks)
    backward = torch.library.opcheck(torch.ops.ragged_pool_cuda._backward.default,
                                     (torch.ones(4, 3, dtype=values.dtype, device="cuda"),
                                      values.detach(), logits.detach(), offsets), test_utils=checks)
    assert forward and all(value == "SUCCESS" for value in forward.values())
    assert backward and all(value == "SUCCESS" for value in backward.values())


def test_registration_fake_tensor_shapes_for_both_operations():
    values, logits, offsets = inputs()
    with FakeTensorMode() as mode:
        fv, fl, fo = (mode.from_tensor(value) for value in (values, logits, offsets))
        output = torch.ops.ragged_pool_cuda.pool.default(fv, fl, fo)
        dv, dl = torch.ops.ragged_pool_cuda._backward.default(torch.ones_like(output), fv, fl, fo)
        for value, shape in ((output, (4, 3)), (dv, (6, 3)), (dl, (6,))):
            assert isinstance(value, FakeTensor)
            assert tuple(value.shape) == shape and value.dtype == torch.float64
            assert value.device.type == "cuda" and value.is_contiguous()
