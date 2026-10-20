"""Shared metadata checks. Tensor contents are checked at execution time."""
import torch


def check_metadata(values, logits, offsets):
    for tensor in (values, logits, offsets):
        torch._check(tensor.device.type == "cpu", lambda: "inputs must be on CPU")
        torch._check(tensor.layout == torch.strided, lambda: "inputs must have strided layout")
        torch._check(tensor.is_contiguous(), lambda: "forward inputs must be contiguous")
    torch._check(values.dim() == 2, lambda: "values must have shape [N,D]")
    torch._check(logits.dim() == 1, lambda: "logits must have shape [N]")
    torch._check(offsets.dim() == 1, lambda: "offsets must have shape [B+1]")
    torch._check(values.dtype in (torch.float32, torch.float64), lambda: "values must be float32 or float64")
    torch._check(logits.dtype == values.dtype, lambda: "logits and values must have the same dtype")
    torch._check(offsets.dtype == torch.int64, lambda: "offsets must be int64")
    torch._check(values.shape[0] == logits.shape[0], lambda: "values and logits N must match")
    torch._check(values.shape[0] <= 4096, lambda: "N must be <=4096")
    torch._check(values.shape[1] >= 1, lambda: "D must be >=1")
    torch._check(values.shape[1] <= 128, lambda: "D must be <=128")
    torch._check(offsets.shape[0] >= 1, lambda: "offsets must contain at least zero")
    torch._check(offsets.shape[0] <= 129, lambda: "B must be <=128")


def check_contents(values, logits, offsets):
    boundaries = offsets.tolist()
    if boundaries[0] != 0 or boundaries[-1] != values.shape[0]:
        raise RuntimeError("offsets must start at zero and end at N")
    if any(a > b for a, b in zip(boundaries, boundaries[1:])):
        raise RuntimeError("offsets must be nondecreasing")
    for tensor in (values, logits):
        if not bool((torch.isfinite(tensor) & (tensor.abs() <= 1000)).all()):
            raise RuntimeError("values and logits must be finite and within [-1000,1000]")
    return boundaries
