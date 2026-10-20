"""Equivalent PyTorch composition used by the bounded CPU study.

The checked entry point includes the package's Python admission checks. The
known-valid entry point is for already admitted inputs, never malformed offsets.
Float32 operations accumulate in ordinary PyTorch float32. The native kernel
instead uses double accumulation, so these are equivalent formulas with distinct
rounding behavior.
"""
import torch

from ragged_pool._validation import check_contents, check_metadata


def ragged_pool_builtin_known_valid(values, logits, offsets):
    """Pool admitted, resolved CPU inputs using segmented max and sum operations."""
    groups = offsets.shape[0] - 1
    if values.shape[0] == 0:
        # Keep a zero derivative to both inputs, including an empty batch.
        zero = values.sum(dim=0) * 0 + logits.sum() * 0
        return zero.unsqueeze(0).expand(groups, values.shape[1]).clone()
    lengths = offsets[1:] - offsets[:-1]
    indices = torch.repeat_interleave(
        torch.arange(groups, device=offsets.device), lengths,
        output_size=values.shape[0],
    )
    # The softmax is invariant to this shift. Detaching the maximum avoids a
    # needless max-gradient path, including ties and empty segment maxima.
    maxima = torch.segment_reduce(logits.detach(), "max", offsets=offsets)
    unnormalized = (logits - maxima[indices]).exp()
    totals = torch.segment_reduce(unnormalized, "sum", offsets=offsets)
    weights = unnormalized / totals[indices]
    return torch.segment_reduce(values * weights[:, None], "sum", offsets=offsets)


def ragged_pool_builtin(values, logits, offsets):
    """Check the original bounded input contract, then evaluate the composition."""
    # Admission precedes resolve_neg so a stepped negative view cannot be
    # densified into the supported layout before it is checked.
    check_metadata(values, logits, offsets)
    check_contents(values, logits, offsets)
    return ragged_pool_builtin_known_valid(
        values.resolve_neg(), logits.resolve_neg(), offsets.resolve_neg(),
    )
