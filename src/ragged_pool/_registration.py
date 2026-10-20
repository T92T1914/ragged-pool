"""Metadata and first order autograd for both native dispatcher operations."""
import torch

from ._validation import check_metadata


@torch.library.register_fake("ragged_pool::pool")
def _fake_pool(values, logits, offsets):
    check_metadata(values, logits, offsets)
    return values.new_empty((offsets.shape[0] - 1, values.shape[1]))


@torch.library.register_fake("ragged_pool::_backward")
def _fake_backward(grad_output, values, logits, offsets):
    check_metadata(values, logits, offsets)
    torch._check(grad_output.device == values.device, lambda: "grad_output must be on CPU")
    torch._check(grad_output.layout == torch.strided, lambda: "grad_output must be strided")
    torch._check(grad_output.dtype == values.dtype, lambda: "grad_output dtype must match values")
    torch._check(grad_output.dim() == 2, lambda: "grad_output must have shape [B,D]")
    torch._check(grad_output.shape[0] == offsets.shape[0] - 1, lambda: "grad_output B must match")
    torch._check(grad_output.shape[1] == values.shape[1], lambda: "grad_output D must match")
    return torch.empty_like(values), torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    values, logits, offsets = inputs
    ctx.save_for_backward(values, logits, offsets)


def _backward(ctx, grad_output):
    values, logits, offsets = ctx.saved_tensors
    grad_values, grad_logits = torch.ops.ragged_pool._backward.default(grad_output, values, logits, offsets)
    return grad_values, grad_logits, None


def _reject_double_backward(ctx, *grads):
    raise RuntimeError("ragged-pool supports first order autograd only")


torch.library.register_autograd("ragged_pool::pool", _backward, setup_context=_setup_context)
torch.library.register_autograd("ragged_pool::_backward", _reject_double_backward)
