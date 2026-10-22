"""Opaque padded CUDA boundaries reusing the qualified packed native math."""
from typing import Tuple

import torch


def _check_metadata(values, logits, mask):
    for tensor in (values, logits, mask):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Padded inputs must be tensors")
        if tensor.device.type != "cuda" or tensor.device != values.device:
            raise RuntimeError("Padded inputs must share one CUDA device")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise RuntimeError("Padded inputs must have contiguous strided layout")
        if tensor.is_neg() or tensor.is_conj():
            raise RuntimeError("Logical negative or conjugate padded primal views are not supported")
    if values.dim() != 3 or logits.dim() != 2 or mask.dim() != 2:
        raise RuntimeError("Expected padded values[B,T,D], logits[B,T] and mask[B,T]")
    if values.dtype not in (torch.float32, torch.float64) or logits.dtype != values.dtype:
        raise RuntimeError("Padded values and logits must share float32 or float64 dtype")
    if mask.dtype != torch.bool:
        raise RuntimeError("Padded mask must have bool dtype")
    if logits.shape != values.shape[:2] or mask.shape != values.shape[:2]:
        raise RuntimeError("Padded values, logits and mask must share B and T")
    batches, steps, width = values.shape
    if not 0 <= batches <= 128 or not 0 <= steps <= 4096:
        raise RuntimeError("Padded B must be in[0,128] and T in[0,4096]")
    if batches * steps > 4096 or not 1 <= width <= 128:
        raise RuntimeError("Padded storage requires B*T<=4096 and D in[1,128]")


def _pack(values, logits, mask):
    # Select before arithmetic. Ignored NaN/Inf never enters native pooling.
    # Boolean compaction allocates and can synchronize for its selected count.
    packed_values, packed_logits = values[mask].contiguous(), logits[mask].contiguous()
    counts = mask.sum(dim=1, dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros((1,)), counts.cumsum(dim=0)))
    return packed_values, packed_logits, offsets


@torch.library.custom_op("ragged_pool_cuda::padded", mutates_args=(), device_types="cuda")
def _padded(values: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _check_metadata(values, logits, mask)
    return torch.ops.ragged_pool_cuda.pool.default(*_pack(values, logits, mask))


@_padded.register_fake
def _fake_padded(values, logits, mask):
    _check_metadata(values, logits, mask)
    return values.new_empty((values.shape[0], values.shape[2]))


def _check_backward_metadata(grad_output, values, logits, mask):
    _check_metadata(values, logits, mask)
    if (not isinstance(grad_output, torch.Tensor) or grad_output.device != values.device
            or grad_output.layout != torch.strided or grad_output.dtype != values.dtype):
        raise RuntimeError("Padded grad_output must be strided on the primal CUDA device and dtype")
    if grad_output.dim() != 2 or grad_output.shape != (values.shape[0], values.shape[2]):
        raise RuntimeError("Padded grad_output must have shape[B,D]")


@torch.library.custom_op("ragged_pool_cuda::_padded_backward", mutates_args=(), device_types="cuda")
def _padded_backward(grad_output: torch.Tensor, values: torch.Tensor,
                     logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_backward_metadata(grad_output, values, logits, mask)
    selected_values, selected_logits, offsets = _pack(values, logits, mask)
    grad_selected_values, grad_selected_logits = torch.ops.ragged_pool_cuda._backward.default(
        grad_output, selected_values, selected_logits, offsets)
    grad_values, grad_logits = torch.zeros_like(values), torch.zeros_like(logits)
    grad_values[mask], grad_logits[mask] = grad_selected_values, grad_selected_logits
    return grad_values, grad_logits


@_padded_backward.register_fake
def _fake_padded_backward(grad_output, values, logits, mask):
    _check_backward_metadata(grad_output, values, logits, mask)
    return torch.empty_like(values), torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    # Save originals so even ignored-slot or mask mutation invalidates backward.
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_output):
    values, logits, mask = ctx.saved_tensors
    grad_values, grad_logits = _padded_backward(grad_output, values, logits, mask)
    return grad_values, grad_logits, None


def _reject_second_order(ctx, *grads):
    raise RuntimeError("ragged-pool-cuda padded supports first-order reverse differentiation only; second derivatives are not supported")


_padded.register_autograd(_backward, setup_context=_setup_context)
_padded_backward.register_autograd(_reject_second_order)


def ragged_pool_padded(values, logits, mask):
    """Pool contiguous same-device CUDA padded inputs, with first gradients.

    Shapes are values[B,T,D], logits[B,T] and bool mask[B,T]. Bounds are
    B<=128, T<=4096, B*T<=4096 and1<=D<=128, including B=0 or T=0.
    Selected float32/float64 entries must be finite in[-1000,1000]. Ignored
    entries may be nonfinite. Empty rows and ignored gradients are zero.
    Originals, including ignored entries, must not change before backward.
    Packing allocates and may synchronize. Inputs stay on their CUDA device.
    """
    # Check originals before dispatcher view fallbacks can normalize them.
    # These static metadata conditions are also traceable by the pinned route.
    _check_metadata(values, logits, mask)
    return torch.ops.ragged_pool_cuda.padded.default(values, logits, mask)


def ragged_pool_padded_reference(values, logits, mask):
    """Independent per-row double-softmax GPU math, without native packing."""
    _check_metadata(values, logits, mask)
    if values.shape[0] == 0:
        zero = values.sum(dim=(0, 1)) * 0 + logits.sum() * 0
        return zero.unsqueeze(0)[:0].clone()
    rows = []
    for row in range(values.shape[0]):
        selected_values, selected_logits = values[row][mask[row]], logits[row][mask[row]]
        for selected in (selected_values, selected_logits):
            if not bool((torch.isfinite(selected) & (selected.abs() <= 1000)).all()):
                raise RuntimeError("Selected padded entries must be finite in[-1000,1000]")
        if selected_logits.numel() == 0:
            result = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
        else:
            weights = torch.softmax(selected_logits.double(), dim=0)
            result = (weights[:, None] * selected_values.double()).sum(dim=0).to(values.dtype)
        rows.append(result)
    return torch.stack(rows)
