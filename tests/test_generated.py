import importlib.util
from pathlib import Path

import pytest
import torch

from ragged_pool import ragged_pool_reference


_spec = importlib.util.spec_from_file_location("ragged_pool_generated_cases", Path(__file__).with_name("generated_cases.py"))
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [
    pytest.param(0, id="empty-batch"),
    pytest.param(2, id="empty-groups"),
    pytest.param(4, id="singletons"),
    pytest.param(6, id="long-group"),
    pytest.param(8, id="unequal-groups"),
    pytest.param(10, id="range-endpoints"),
    pytest.param(12, id="extreme-logits"),
    pytest.param(14, id="maximum-items"),
    pytest.param(30, id="maximum-width"),
    pytest.param(46, id="maximum-groups"),
    pytest.param(54, id="negative-upstream"),
    pytest.param(184, id="score-cancellation-oracle"),
])
def test_generated_boundary_and_gradient_properties(seed, dtype):
    report = _cases.run_case(seed, dtype)
    assert report["counters"]["differential_private_backward"] > 0
    assert report["counters"]["negative_nondense_direct"] == 3


@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_contiguous_negative_direct_gradients_keep_saved_input_versions(input_index):
    inputs = [torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64),
              torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64), torch.tensor([0, 2, 2, 3])]
    inputs[input_index] = torch._neg_view(-inputs[input_index])
    assert inputs[input_index].is_neg() and inputs[input_index].is_contiguous()
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()
    reference_inputs = [tensor.resolve_neg().detach().clone() for tensor in inputs]
    reference_inputs[0].requires_grad_()
    reference_inputs[1].requires_grad_()
    actual = torch.ops.ragged_pool.pool.default(*inputs)
    expected = ragged_pool_reference(*reference_inputs)
    upstream = torch.tensor([[1.0, -0.5], [3.0, 2.0], [-1.5, 0.25]], dtype=torch.float64)
    actual_gradients = torch.autograd.grad(actual, inputs[:2], upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, reference_inputs[:2], upstream)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    with torch.no_grad():
        inputs[input_index].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        actual.sum().backward()
