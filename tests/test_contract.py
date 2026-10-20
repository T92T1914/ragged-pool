import pytest
import torch

from ragged_pool import ragged_pool, ragged_pool_reference


def valid_inputs():
    return torch.arange(6, dtype=torch.float64).reshape(3, 2), torch.zeros(3, dtype=torch.float64), torch.tensor([0, 2, 3])


def malformed_inputs(case):
    values, logits, offsets = valid_inputs()
    if case == "values_rank":
        values = values.reshape(-1)
    elif case == "logits_rank":
        logits = logits[:, None]
    elif case == "offsets_rank":
        offsets = offsets[:, None]
    elif case == "mismatched_n":
        logits = logits[:2]
    elif case == "mixed_float_dtype":
        logits = logits.float()
    elif case == "half":
        values, logits = values.half(), logits.half()
    elif case == "bfloat16":
        values, logits = values.bfloat16(), logits.bfloat16()
    elif case == "integer_values":
        values, logits = values.long(), logits.long()
    elif case == "int32_offsets":
        offsets = offsets.int()
    elif case == "no_offsets":
        offsets = torch.empty(0, dtype=torch.int64)
    elif case == "nonzero_start":
        offsets = torch.tensor([1, 2, 3])
    elif case == "wrong_end":
        offsets = torch.tensor([0, 2, 2])
    elif case == "decreasing":
        offsets = torch.tensor([0, 2, 1, 3])
    elif case == "negative_offset":
        offsets = torch.tensor([0, -1, 3])
    elif case == "oversized_intermediate_offset":
        offsets = torch.tensor([0, 4, 3])
    elif case == "n_above_limit":
        values, logits, offsets = torch.zeros(4097, 1), torch.zeros(4097), torch.tensor([0, 4097])
    elif case == "d_zero":
        values = torch.empty(3, 0, dtype=torch.float64)
    elif case == "d_above_limit":
        values = torch.zeros(3, 129, dtype=torch.float64)
    elif case == "b_above_limit":
        offsets = torch.cat((torch.zeros(129, dtype=torch.int64), torch.tensor([3])))
    elif case == "values_noncontiguous":
        values = torch.zeros(2, 3, dtype=torch.float64).t()
    elif case == "logits_noncontiguous":
        logits = torch.zeros(6, dtype=torch.float64)[::2]
    elif case == "offsets_noncontiguous":
        offsets = torch.tensor([0, -1, 2, -1, 3, -1])[::2]
    elif case == "sparse_values":
        values = values.to_sparse()
    elif case == "sparse_logits":
        logits = logits.to_sparse()
    elif case == "sparse_offsets":
        offsets = offsets.to_sparse()
    else:
        raise AssertionError(case)
    return values, logits, offsets


@pytest.mark.parametrize("pool", [ragged_pool, ragged_pool_reference])
@pytest.mark.parametrize("case", [
    "values_rank", "logits_rank", "offsets_rank", "mismatched_n", "mixed_float_dtype",
    "half", "bfloat16", "integer_values", "int32_offsets", "no_offsets", "nonzero_start",
    "wrong_end", "decreasing", "negative_offset", "oversized_intermediate_offset",
    "n_above_limit", "d_zero", "d_above_limit", "b_above_limit", "values_noncontiguous",
    "logits_noncontiguous", "offsets_noncontiguous", "sparse_values", "sparse_logits", "sparse_offsets",
])
def test_rejects_inputs_outside_forward_contract(pool, case):
    with pytest.raises((RuntimeError, NotImplementedError)):
        pool(*malformed_inputs(case))


@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_contiguous_logical_negative_views_match_resolved_inputs(input_index):
    inputs = list(valid_inputs())
    inputs[input_index] = torch._neg_view(-inputs[input_index])
    assert inputs[input_index].is_neg() and inputs[input_index].is_contiguous()
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()
    resolved = [t.resolve_neg().detach().clone() for t in inputs]
    resolved[0].requires_grad_()
    resolved[1].requires_grad_()
    actual = ragged_pool(*inputs)
    expected = ragged_pool_reference(*resolved)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), inputs[:2])
    expected_grads = torch.autograd.grad(expected.sum(), resolved[:2])
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("pool", [ragged_pool, ragged_pool_reference])
@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1000.0001, -1000.0001])
def test_rejects_nonfinite_and_out_of_range_data(pool, input_index, bad_value):
    inputs = list(valid_inputs())
    inputs[input_index].view(-1)[0] = bad_value
    with pytest.raises(RuntimeError, match="finite.*1000"):
        pool(*inputs)


@pytest.mark.parametrize("pool", [ragged_pool, ragged_pool_reference])
@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_rejects_other_device_without_hidden_transfer(pool, input_index):
    inputs = list(valid_inputs())
    inputs[input_index] = torch.empty_like(inputs[input_index], device="meta")
    with pytest.raises(RuntimeError, match="CPU"):
        pool(*inputs)
    assert inputs[input_index].device.type == "meta"


@pytest.mark.parametrize("pool", [ragged_pool, ragged_pool_reference])
def test_output_storage_is_fresh_and_inputs_are_not_mutated(pool):
    values, logits, offsets = valid_inputs()
    inputs = (values, logits, offsets)
    snapshots = tuple(t.clone() for t in inputs)
    versions = tuple(t._version for t in inputs)
    first = pool(*inputs)
    second = pool(*inputs)
    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() not in {t.data_ptr() for t in inputs}
    first.add_(100)
    torch.testing.assert_close(second, pool(*inputs))
    for tensor, snapshot, version in zip(inputs, snapshots, versions):
        torch.testing.assert_close(tensor, snapshot)
        assert tensor._version == version


@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_saved_input_mutation_is_reported_by_autograd(input_index):
    values, logits, offsets = valid_inputs()
    values.requires_grad_()
    logits.requires_grad_()
    inputs = (values, logits, offsets)
    pooled = ragged_pool(*inputs)
    with torch.no_grad():
        inputs[input_index].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        pooled.sum().backward()


def test_double_backward_is_explicitly_refused():
    values, logits, offsets = valid_inputs()
    values.requires_grad_()
    logits.requires_grad_()
    grad_values, grad_logits = torch.autograd.grad(
        ragged_pool(values, logits, offsets).sum(), (values, logits), create_graph=True,
    )
    with pytest.raises(RuntimeError, match="first order autograd only"):
        torch.autograd.grad(grad_values.sum() + grad_logits.sum(), (values, logits))


@pytest.mark.parametrize("case", ["wrong_shape", "wrong_dtype", "sparse", "meta"])
def test_private_backward_rejects_malformed_upstream(case):
    values, logits, offsets = valid_inputs()
    upstream = torch.ones(2, 2, dtype=torch.float64)
    if case == "wrong_shape":
        upstream = upstream[:, :1]
    elif case == "wrong_dtype":
        upstream = upstream.float()
    elif case == "sparse":
        upstream = upstream.to_sparse()
    elif case == "meta":
        upstream = torch.empty(2, 2, dtype=torch.float64, device="meta")
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.ragged_pool._backward.default(upstream, values, logits, offsets)
