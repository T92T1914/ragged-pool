"""Metadata-only fake rules and first-order native CUDA reverse differentiation."""
import torch

from ._validation import check_grad_metadata, check_metadata


@torch.library.register_fake("ragged_pool_cuda::pool")
def _fake_pool(values, logits, offsets):
    check_metadata(values, logits, offsets)
    return values.new_empty((offsets.shape[0] - 1, values.shape[1]))


@torch.library.register_fake("ragged_pool_cuda::_backward")
def _fake_backward(grad_output, values, logits, offsets):
    check_metadata(values, logits, offsets)
    check_grad_metadata(grad_output, values, offsets)
    return torch.empty_like(values), torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    values, logits, offsets = inputs
    ctx.save_for_backward(values, logits, offsets)


def _backward(ctx, grad_output):
    values, logits, offsets = ctx.saved_tensors
    grad_values, grad_logits = torch.ops.ragged_pool_cuda._backward.default(grad_output, values, logits, offsets)
    return grad_values, grad_logits, None


def _reject_second_order(ctx, *grads):
    raise RuntimeError("ragged-pool-cuda supports first-order reverse differentiation only; second derivatives are not supported")


torch.library.register_autograd("ragged_pool_cuda::pool", _backward, setup_context=_setup_context)
torch.library.register_autograd("ragged_pool_cuda::_backward", _reject_second_order)
