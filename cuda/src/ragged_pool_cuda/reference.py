"""Independent primitive CUDA reference with explicit offset transfer to CPU."""
import torch

from ._validation import check_metadata


def ragged_pool_reference(values, logits, offsets):
    """Evaluate admitted groups with GPU float64 softmax and weighted sums.

    This readable reference explicitly transfers offsets to CPU for Python
    group boundaries and synchronizes scalar content checks. Its numerical
    operations and differentiable inputs stay on the original CUDA device.
    It is an oracle for comparisons, without a performance claim.
    """
    check_metadata(values, logits, offsets)
    boundaries = offsets.detach().cpu().tolist()
    if (boundaries[0] != 0 or boundaries[-1] != values.shape[0]
            or any(value < 0 or value > values.shape[0] for value in boundaries)
            or any(left > right for left, right in zip(boundaries, boundaries[1:]))):
        raise RuntimeError("offsets must start at zero, end at N, remain in range and be nondecreasing")
    for tensor in (values, logits):
        if not bool((torch.isfinite(tensor) & (tensor.abs() <= 1000)).all()):
            raise RuntimeError("values and logits must be finite and within [-1000,1000]")
    precise_values, precise_logits = values.to(torch.float64), logits.to(torch.float64)
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        selected_values, selected_logits = precise_values[start:stop], precise_logits[start:stop]
        if start == stop:
            row = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
        else:
            weights = torch.softmax(selected_logits, dim=0)
            row = (weights[:, None] * selected_values).sum(dim=0)
        rows.append(row)
    if not rows:
        dependency = (precise_values.sum() + precise_logits.sum()) * 0
        return values.new_zeros((0, values.shape[1])) + dependency.to(values.dtype)
    return torch.stack(rows).to(values.dtype)
