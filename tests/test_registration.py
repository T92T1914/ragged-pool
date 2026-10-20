import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from ragged_pool import ragged_pool


def registration_inputs(requires_grad=False):
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], requires_grad=requires_grad)
    logits = torch.tensor([0.0, 1.0, -0.5], requires_grad=requires_grad)
    offsets = torch.tensor([0, 2, 2, 3])
    return values, logits, offsets


def test_forward_operator_registration():
    results = torch.library.opcheck(torch.ops.ragged_pool.pool.default, registration_inputs(requires_grad=True))
    assert results and all(result == "SUCCESS" for result in results.values())


def test_private_backward_operator_registration():
    values, logits, offsets = registration_inputs()
    upstream = torch.ones(3, 2)
    results = torch.library.opcheck(
        torch.ops.ragged_pool._backward.default, (upstream, values, logits, offsets),
    )
    assert results and all(result == "SUCCESS" for result in results.values())


def test_both_operations_supply_fake_tensor_metadata():
    values, logits, offsets = registration_inputs()
    with FakeTensorMode() as mode:
        fake_values, fake_logits, fake_offsets = (mode.from_tensor(t) for t in (values, logits, offsets))
        pooled = torch.ops.ragged_pool.pool.default(fake_values, fake_logits, fake_offsets)
        grad_values, grad_logits = torch.ops.ragged_pool._backward.default(
            torch.ones_like(pooled), fake_values, fake_logits, fake_offsets,
        )
        for actual, shape in ((pooled, (3, 2)), (grad_values, (3, 2)), (grad_logits, (3,))):
            assert isinstance(actual, FakeTensor)
            assert actual.shape == shape
            assert actual.dtype == torch.float32
            assert actual.device.type == "cpu"
            assert actual.is_contiguous()


def test_aot_eager_fullgraph_forward_and_backward_match_eager():
    values, logits, offsets = registration_inputs(requires_grad=True)
    compiled_values = values.detach().clone().requires_grad_()
    compiled_logits = logits.detach().clone().requires_grad_()
    upstream = torch.tensor([[0.5, -1.0], [2.0, 3.0], [1.5, 0.25]])
    eager_output = ragged_pool(values, logits, offsets)
    eager_grads = torch.autograd.grad(eager_output, (values, logits), upstream)
    with torch._dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(ragged_pool, backend="aot_eager", fullgraph=True, dynamic=False)
        compiled_output = compiled(compiled_values, compiled_logits, offsets)
        compiled_grads = torch.autograd.grad(compiled_output, (compiled_values, compiled_logits), upstream)
        with pytest.raises(RuntimeError, match="nondecreasing"):
            compiled(compiled_values, compiled_logits, torch.tensor([0, 2, 1, 3]))
    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(actual, expected)
