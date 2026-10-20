"""Metadata and packed autograd through second order for native CPU operations."""
import torch

from ._validation import check_metadata


@torch.library.register_fake("ragged_pool::pool")
def _fake_pool(values, logits, offsets):
    check_metadata(values, logits, offsets)
    return values.new_empty((offsets.shape[0] - 1, values.shape[1]))


@torch.library.register_fake("ragged_pool::_backward")
def _fake_backward(grad_output, values, logits, offsets):
    check_metadata(values, logits, offsets)
    _check_grad_metadata(grad_output, values, offsets)
    return torch.empty_like(values), torch.empty_like(logits)


def _check_grad_metadata(grad_output, values, offsets):
    torch._check(grad_output.device == values.device, lambda: "grad_output must be on CPU")
    torch._check(grad_output.layout == torch.strided, lambda: "grad_output must be strided")
    torch._check(grad_output.dtype == values.dtype, lambda: "grad_output dtype must match values")
    torch._check(grad_output.dim() == 2, lambda: "grad_output must have shape [B,D]")
    torch._check(grad_output.shape[0] == offsets.shape[0] - 1, lambda: "grad_output B must match")
    torch._check(grad_output.shape[1] == values.shape[1], lambda: "grad_output D must match")


@torch.library.register_fake("ragged_pool::_double_backward")
def _fake_double_backward(grad_grad_values, grad_grad_logits, grad_output, values, logits, offsets):
    check_metadata(values, logits, offsets)
    _check_grad_metadata(grad_output, values, offsets)
    for tensor, original in ((grad_grad_values, values), (grad_grad_logits, logits)):
        torch._check(tensor.device == values.device, lambda: "second upstream must be on CPU")
        torch._check(tensor.layout == torch.strided, lambda: "second upstream must be strided")
        torch._check(tensor.dtype == values.dtype, lambda: "second upstream dtype must match values")
        torch._check(tensor.shape == original.shape, lambda: "second upstream shape must match its first gradient")
    return (values.new_empty((offsets.shape[0] - 1, values.shape[1])),
            torch.empty_like(values), torch.empty_like(logits))


def _setup_context(ctx, inputs, output):
    values, logits, offsets = inputs
    ctx.save_for_backward(values, logits, offsets)


def _backward(ctx, grad_output):
    values, logits, offsets = ctx.saved_tensors
    grad_values, grad_logits = torch.ops.ragged_pool._backward.default(grad_output, values, logits, offsets)
    return grad_values, grad_logits, None


def _setup_backward_context(ctx, inputs, output):
    # The output cotangent can depend on a model head, so retain its connection.
    ctx.save_for_backward(*inputs)


def _double_backward(ctx, grad_grad_values, grad_grad_logits):
    grad_output, values, logits, offsets = ctx.saved_tensors
    if grad_grad_values is None:
        grad_grad_values = torch.zeros_like(values)
    if grad_grad_logits is None:
        grad_grad_logits = torch.zeros_like(logits)
    grad_grad_output, grad_values, grad_logits = torch.ops.ragged_pool._double_backward.default(
        grad_grad_values, grad_grad_logits, grad_output, values, logits, offsets,
    )
    return grad_grad_output, grad_values, grad_logits, None


def _reject_third_order(ctx, *grads):
    raise RuntimeError("ragged-pool packed autograd supports up to second order; third order is not supported")


torch.library.register_autograd("ragged_pool::pool", _backward, setup_context=_setup_context)
torch.library.register_autograd("ragged_pool::_backward", _double_backward, setup_context=_setup_backward_context)
torch.library.register_autograd("ragged_pool::_double_backward", _reject_third_order)
