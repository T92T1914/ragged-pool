"""Straightforward differentiable PyTorch baseline, independent of native kernels."""
import torch

from ._validation import check_contents, check_metadata


def ragged_pool_reference(values, logits, offsets):
    check_metadata(values, logits, offsets)
    boundaries = check_contents(values, logits, offsets)
    # Even an empty result retains a zero derivative path to both inputs.
    zero = values.sum(dim=0) * 0 + logits.sum() * 0
    groups = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            groups.append(zero)
        else:
            weights = torch.softmax(logits[start:end], dim=0)
            groups.append((weights[:, None] * values[start:end]).sum(dim=0))
    if not groups:
        return zero.unsqueeze(0)[:0].clone()
    return torch.stack(groups)
