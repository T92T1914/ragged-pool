"""Opaque padded CPU boundaries around the existing packed native math."""
from typing import Tuple

import torch

from .padded import _check_metadata


def _pack(values, logits, mask):
    # The selected length stays inside each opaque kernel, outside the graph.
    packed_values = values[mask].contiguous()
    packed_logits = logits[mask].contiguous()
    counts = mask.sum(dim=1, dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros((1,)), counts.cumsum(dim=0)))
    return packed_values, packed_logits, offsets


@torch.library.custom_op("ragged_pool::padded", mutates_args=(), device_types="cpu")
def _padded(values: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _check_metadata(values, logits, mask)
    return torch.ops.ragged_pool.pool.default(*_pack(values, logits, mask))


@_padded.register_fake
def _fake_padded(values, logits, mask):
    _check_metadata(values, logits, mask)
    return values.new_empty((values.shape[0], values.shape[2]))


def _check_backward_metadata(grad_output, values, logits, mask):
    _check_metadata(values, logits, mask)
    if grad_output.device != values.device or grad_output.layout != torch.strided:
        raise RuntimeError('Padded grad_output must be a strided CPU tensor')
    if grad_output.dtype != values.dtype:
        raise RuntimeError('Padded grad_output dtype must match values')
    if grad_output.dim() != 2 or grad_output.shape != (values.shape[0], values.shape[2]):
        raise RuntimeError('Padded grad_output must have shape [B,D]')


@torch.library.custom_op("ragged_pool::_padded_backward", mutates_args=(), device_types="cpu")
def _padded_backward(grad_output: torch.Tensor, values: torch.Tensor,
                     logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_backward_metadata(grad_output, values, logits, mask)
    packed_values, packed_logits, offsets = _pack(values, logits, mask)
    packed_grad_values, packed_grad_logits = torch.ops.ragged_pool._backward.default(
        grad_output, packed_values, packed_logits, offsets,
    )
    grad_values, grad_logits = torch.zeros_like(values), torch.zeros_like(logits)
    grad_values[mask] = packed_grad_values
    grad_logits[mask] = packed_grad_logits
    return grad_values, grad_logits


@_padded_backward.register_fake
def _fake_padded_backward(grad_output, values, logits, mask):
    _check_backward_metadata(grad_output, values, logits, mask)
    return torch.empty_like(values), torch.empty_like(logits)


def _check_double_backward_metadata(grad_grad_values, grad_grad_logits,
                                    grad_output, values, logits, mask):
    _check_backward_metadata(grad_output, values, logits, mask)
    for tensor, original in ((grad_grad_values, values), (grad_grad_logits, logits)):
        if tensor.device != values.device or tensor.layout != torch.strided:
            raise RuntimeError('Padded second upstream must be a strided CPU tensor')
        if tensor.dtype != values.dtype or tensor.shape != original.shape:
            raise RuntimeError('Padded second upstream dtype and shape must match its first gradient')


@torch.library.custom_op("ragged_pool::_padded_double_backward", mutates_args=(), device_types="cpu")
def _padded_double_backward(grad_grad_values: torch.Tensor, grad_grad_logits: torch.Tensor,
                            grad_output: torch.Tensor, values: torch.Tensor,
                            logits: torch.Tensor, mask: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_double_backward_metadata(grad_grad_values, grad_grad_logits,
                                    grad_output, values, logits, mask)
    packed_values, packed_logits, offsets = _pack(values, logits, mask)
    # Select before arithmetic so ignored second-upstream NaNs cannot enter math.
    grad_grad_output, packed_grad_values, packed_grad_logits = torch.ops.ragged_pool._double_backward.default(
        grad_grad_values[mask].contiguous(), grad_grad_logits[mask].contiguous(),
        grad_output, packed_values, packed_logits, offsets,
    )
    grad_values, grad_logits = torch.zeros_like(values), torch.zeros_like(logits)
    grad_values[mask], grad_logits[mask] = packed_grad_values, packed_grad_logits
    return grad_grad_output, grad_values, grad_logits


@_padded_double_backward.register_fake
def _fake_padded_double_backward(grad_grad_values, grad_grad_logits,
                                 grad_output, values, logits, mask):
    _check_double_backward_metadata(grad_grad_values, grad_grad_logits,
                                    grad_output, values, logits, mask)
    return (values.new_empty((values.shape[0], values.shape[2])),
            torch.empty_like(values), torch.empty_like(logits))


def _setup_context(ctx, inputs, output):
    # Repacking in backward requires unchanged originals, including the mask.
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_output):
    values, logits, mask = ctx.saved_tensors
    grad_values, grad_logits = _padded_backward(grad_output, values, logits, mask)
    return grad_values, grad_logits, None


def _setup_backward_context(ctx, inputs, output):
    # Keep the original g and padded tensors, not packed copies, for both their
    # derivative connections and version checks through the second backward.
    ctx.save_for_backward(*inputs)


def _double_backward(ctx, grad_grad_values, grad_grad_logits):
    grad_output, values, logits, mask = ctx.saved_tensors
    if grad_grad_values is None:
        grad_grad_values = torch.zeros_like(values)
    if grad_grad_logits is None:
        grad_grad_logits = torch.zeros_like(logits)
    grad_grad_output, grad_values, grad_logits = _padded_double_backward(
        grad_grad_values, grad_grad_logits, grad_output, values, logits, mask,
    )
    return grad_grad_output, grad_values, grad_logits, None


def _reject_third_order(ctx, *grads):
    raise RuntimeError('ragged-pool padded autograd supports up to second order; third order is not supported')


_padded.register_autograd(_backward, setup_context=_setup_context)
_padded_backward.register_autograd(_double_backward, setup_context=_setup_backward_context)
_padded_double_backward.register_autograd(_reject_third_order)
