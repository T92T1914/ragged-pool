import math

import pytest
import torch

from ragged_pool import ragged_pool, ragged_pool_reference


def tolerances(dtype):
    return {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["unequal", "constant_logits", "large_logits", "singleton"])
def test_forward_against_reference(dtype, case):
    values = torch.tensor(
        [[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12], [13, 14, -15], [16, -17, 18]],
        dtype=dtype,
    )
    offsets = torch.tensor([0, 2, 2, 5, 6])
    logits = torch.tensor([-1, 2, 0.25, -3, 1.5, 0], dtype=dtype)
    if case == "constant_logits":
        logits.fill_(1000)
    elif case == "large_logits":
        logits = torch.tensor([1000, 999, -1000, 0, 1000, -1000], dtype=dtype)
    elif case == "singleton":
        offsets = torch.arange(7)
    actual = ragged_pool(values, logits, offsets)
    expected = ragged_pool_reference(values, logits, offsets)
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    assert actual.shape == (len(offsets) - 1, 3)
    assert actual.dtype == dtype
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_both_gradients_against_independent_reference_autograd(dtype):
    generator = torch.Generator().manual_seed(413)
    values = torch.randn(11, 5, generator=generator, dtype=dtype, requires_grad=True)
    logits = torch.randn(11, generator=generator, dtype=dtype, requires_grad=True)
    offsets = torch.tensor([0, 3, 3, 4, 11, 11])
    upstream = torch.randn(5, 5, generator=generator, dtype=dtype)
    reference_values = values.detach().clone().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    actual = ragged_pool(values, logits, offsets)
    expected = ragged_pool_reference(reference_values, reference_logits, offsets)
    actual_grads = torch.autograd.grad(actual, (values, logits), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_values, reference_logits), upstream)
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, **tolerances(dtype))
    assert offsets.grad is None
    assert torch.count_nonzero(actual_grads[1][3:4]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("case", ["constant", "large_near_ties", "large_spread"])
def test_endpoint_value_gradients_with_constant_and_large_logits(dtype, width, case):
    values = torch.tensor(
        [[-1000.0 if (i + d) % 2 == 0 else 1000.0 for d in range(width)] for i in range(5)],
        dtype=dtype, requires_grad=True,
    )
    logits_by_case = {
        "constant": [1000, 1000, 1000, 1000, 1000],
        "large_near_ties": [1000, 999, -999, -1000, 1000],
        "large_spread": [1000, -1000, -1000, 1000, -1000],
    }
    logits = torch.tensor(logits_by_case[case], dtype=dtype, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 4, 5])
    upstream = torch.arange(1, 4 * width + 1, dtype=dtype).reshape(4, width) / 4
    reference_values = values.detach().clone().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    actual = ragged_pool(values, logits, offsets)
    expected = ragged_pool_reference(reference_values, reference_logits, offsets)
    actual_grads = torch.autograd.grad(actual, (values, logits), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_values, reference_logits), upstream)
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, **tolerances(dtype))


def test_small_case_has_hand_calculated_output_and_gradients():
    values = torch.tensor([[1.0, 2.0], [3.0, -2.0], [7.0, 5.0]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([0.0, math.log(3), 0.0], dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 3])
    upstream = torch.tensor([[2.0, -3.0], [4.0, 8.0], [-1.0, 2.0]], dtype=torch.float64)
    pooled = ragged_pool(values, logits, offsets)
    grad_values, grad_logits = torch.autograd.grad(pooled, (values, logits), upstream)
    torch.testing.assert_close(pooled, torch.tensor([[2.5, -1.0], [0.0, 0.0], [7.0, 5.0]], dtype=torch.float64))
    torch.testing.assert_close(grad_values, torch.tensor([[0.5, -0.75], [1.5, -2.25], [-1.0, 2.0]], dtype=torch.float64))
    torch.testing.assert_close(grad_logits, torch.tensor([-3.0, 3.0, 0.0], dtype=torch.float64))


def test_float64_finite_difference_gradcheck():
    generator = torch.Generator().manual_seed(62)
    values = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(4, generator=generator, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 3, 4])
    assert torch.autograd.gradcheck(
        lambda v, l: ragged_pool(v, l, offsets), (values, logits),
        eps=1e-6, atol=1e-5, rtol=1e-4, check_undefined_grad=True,
    )


@pytest.mark.parametrize("offset_list", [[0], [0, 0, 0, 0]])
@pytest.mark.parametrize("pool", [ragged_pool, ragged_pool_reference])
def test_empty_inputs_keep_zero_gradient_paths(offset_list, pool):
    values = torch.empty(0, 3, dtype=torch.float64, requires_grad=True)
    logits = torch.empty(0, dtype=torch.float64, requires_grad=True)
    pooled = pool(values, logits, torch.tensor(offset_list))
    assert pooled.shape == (len(offset_list) - 1, 3)
    torch.testing.assert_close(pooled, torch.zeros_like(pooled))
    pooled.sum().backward()
    assert values.grad is not None and values.grad.shape == (0, 3)
    assert logits.grad is not None and logits.grad.shape == (0,)


def test_empty_output_rows_have_no_input_gradient_contribution():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    pooled = ragged_pool(values, logits, torch.tensor([0, 0, 2, 2]))
    pooled[[0, 2]].sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("layout", ["sum", "expanded", "transposed", "sliced", "negative_view"])
def test_backward_accepts_training_gradient_layouts(layout):
    generator = torch.Generator().manual_seed(511)
    values = torch.randn(6, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(6, generator=generator, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 5, 6])
    reference_values = values.detach().clone().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    actual = ragged_pool(values, logits, offsets)
    expected = ragged_pool_reference(reference_values, reference_logits, offsets)
    if layout == "sum":
        actual_grads = torch.autograd.grad(actual.sum(), (values, logits))
        expected_grads = torch.autograd.grad(expected.sum(), (reference_values, reference_logits))
    else:
        if layout == "expanded":
            upstream = torch.tensor([[1.5]], dtype=torch.float64).expand(4, 3)
            assert upstream.stride() == (0, 0)
        elif layout == "transposed":
            upstream = torch.randn(3, 4, generator=generator, dtype=torch.float64).t()
        elif layout == "negative_view":
            upstream = torch._neg_view(torch.randn(4, 3, generator=generator, dtype=torch.float64))
            assert upstream.is_neg()
        else:
            upstream = torch.randn(4, 6, generator=generator, dtype=torch.float64)[:, ::2]
        if layout != "negative_view":
            assert not upstream.is_contiguous()
        actual_grads = torch.autograd.grad(actual, (values, logits), upstream)
        expected_grads = torch.autograd.grad(expected, (reference_values, reference_logits), upstream)
        private_grads = torch.ops.ragged_pool._backward.default(
            upstream, values.detach(), logits.detach(), offsets,
        )
        for private_grad, expected_grad in zip(private_grads, expected_grads):
            torch.testing.assert_close(private_grad, expected_grad, rtol=1e-10, atol=1e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


def test_largest_declared_dimensions_and_range_endpoints():
    values = torch.linspace(-1000, 1000, 4096 * 128, dtype=torch.float64).reshape(4096, 128)
    logits = torch.zeros(4096, dtype=torch.float64)
    offsets = torch.arange(0, 4097, 32)
    actual = ragged_pool(values, logits, offsets)
    expected = values.reshape(128, 32, 128).mean(dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert actual.shape == (128, 128)
