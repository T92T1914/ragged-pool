"""Eager padded-batch adapters for the bounded packed pooling operation."""
import torch


def _check_metadata(values, logits, mask):
    for tensor in (values, logits, mask):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Padded inputs must be tensors')
        if tensor.device.type != 'cpu':
            raise RuntimeError('Padded inputs must be on CPU')
        if tensor.layout != torch.strided:
            raise RuntimeError('Padded inputs must have strided layout')
        if not tensor.is_contiguous():
            raise RuntimeError('Padded inputs must be contiguous')
    if values.dim() != 3 or logits.dim() != 2 or mask.dim() != 2:
        raise RuntimeError('Expected values [B,T,D], logits [B,T] and mask [B,T]')
    if values.dtype not in (torch.float32, torch.float64) or logits.dtype != values.dtype:
        raise RuntimeError('Padded values and logits must have the same float32 or float64 dtype')
    if mask.dtype != torch.bool:
        raise RuntimeError('Padded mask must have bool dtype')
    if logits.shape != values.shape[:2] or mask.shape != values.shape[:2]:
        raise RuntimeError('Padded values, logits and mask must have matching B and T')
    batches, steps, width = values.shape
    if not 0 <= batches <= 128 or not 0 <= steps <= 4096:
        raise RuntimeError('Padded B must be in [0,128] and T in [0,4096]')
    if not 1 <= width <= 128:
        raise RuntimeError('Padded D must be in [1,128]')
    if batches * steps > 4096:
        raise RuntimeError('Padded storage must have B*T <=4096 slots')


def ragged_pool_padded(values, logits, mask):
    """Pool [B,T,D] values using [B,T] logits and a bool mask of valid items.

    Inputs must be contiguous CPU tensors with matching float32/float64 values
    and logits. Bounds are B <=128, T <=4096, 1 <=D <=128 and B*T <=4096.
    Mask holes are allowed. Only selected entries must be finite and within
    [-1000,1000]. Ignored padding is selected out before numerical operations.
    Empty rows return zeros and masked inputs receive zero gradients.

    This eager adapter copies selected entries into packed storage and invokes
    the existing native operation once. It does not change or transfer inputs.
    It inherits the native operation's first order autograd limit.
    """
    _check_metadata(values, logits, mask)
    packed_values = values[mask].contiguous()
    packed_logits = logits[mask].contiguous()
    counts = mask.sum(dim=1, dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros((1,)), counts.cumsum(dim=0)))
    return torch.ops.ragged_pool.pool.default(packed_values, packed_logits, offsets)


def ragged_pool_padded_reference(values, logits, mask):
    """Compute the same eager contract using independent per-row PyTorch math.

    Each row selects its own valid entries and uses ordinary softmax/reduction,
    without packing rows, constructing offsets or calling a custom operation.
    The native implementation accumulates in float64, so rounding may differ.
    """
    _check_metadata(values, logits, mask)
    if values.shape[0] == 0:
        # These inputs have no elements, so sums retain both derivative paths
        # without touching any ignored non-finite padding.
        zero = values.sum(dim=(0, 1)) * 0 + logits.sum() * 0
        return zero.unsqueeze(0)[:0].clone()
    rows = []
    for index in range(values.shape[0]):
        selected_values = values[index][mask[index]]
        selected_logits = logits[index][mask[index]]
        for selected in (selected_values, selected_logits):
            if not bool((torch.isfinite(selected) & (selected.abs() <= 1000)).all()):
                raise RuntimeError('Selected values and logits must be finite and within [-1000,1000]')
        if selected_logits.numel() == 0:
            rows.append(selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0)
        else:
            weights = torch.softmax(selected_logits, dim=0)
            rows.append((weights[:, None] * selected_values).sum(dim=0))
    return torch.stack(rows)
