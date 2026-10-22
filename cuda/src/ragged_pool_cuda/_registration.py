"""Metadata-only fake rules and explicit native CUDA derivative boundaries."""
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


@torch.library.register_fake("ragged_pool_cuda::_jvp")
def _fake_jvp(values, logits, offsets, dvalues, dlogits):
    check_metadata(values, logits, offsets)
    for tangent, primal in ((dvalues, values), (dlogits, logits)):
        torch._check(tangent.device == primal.device, lambda: "tangents must be on the primal CUDA device")
        torch._check(tangent.layout == torch.strided, lambda: "tangents must have strided layout")
        torch._check(tangent.dtype == primal.dtype, lambda: "tangent dtype must match its primal")
        torch._check(tangent.dim() == primal.dim(), lambda: "tangent shape must match its primal")
        torch._check(not tangent.is_conj(), lambda: "conjugate tangent views are not supported")
        for actual, expected in zip(tangent.shape, primal.shape):
            torch._check(actual == expected, lambda: "tangent shape must match its primal")
    return values.new_empty((offsets.shape[0] - 1, values.shape[1]))


def _setup_context(ctx, inputs, output):
    values, logits, offsets = inputs
    ctx.save_for_backward(values, logits, offsets)


def _backward(ctx, grad_output):
    values, logits, offsets = ctx.saved_tensors
    grad_values, grad_logits = torch.ops.ragged_pool_cuda._backward.default(grad_output, values, logits, offsets)
    return grad_values, grad_logits, None


def _reject_second_order(ctx, *grads):
    raise RuntimeError("ragged-pool-cuda supports first-order reverse differentiation only; second derivatives are not supported")


def _reject_jvp_reverse(ctx, *grads):
    raise RuntimeError("ragged-pool-cuda supports first-order JVP only; differentiating tangents is not supported")


torch.library.register_autograd("ragged_pool_cuda::pool", _backward, setup_context=_setup_context)
torch.library.register_autograd("ragged_pool_cuda::_backward", _reject_second_order)
torch.library.register_autograd("ragged_pool_cuda::_jvp", _reject_jvp_reverse)
