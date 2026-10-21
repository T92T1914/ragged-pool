"""Shared CUDA metadata checks, without reading tensor contents."""
import torch


def check_metadata(values, logits, offsets):
    for tensor in (values, logits, offsets):
        torch._check(tensor.device.type == "cuda", lambda: "inputs must be CUDA tensors")
        torch._check(tensor.device == values.device, lambda: "inputs must be on the same CUDA device")
        torch._check(tensor.layout == torch.strided, lambda: "inputs must have strided layout")
        torch._check(tensor.is_contiguous(), lambda: "forward inputs must be contiguous")
        torch._check(not tensor.is_neg(), lambda: "logical negative primal views are not supported")
        torch._check(not tensor.is_conj(), lambda: "conjugate primal views are not supported")
    torch._check(values.dim() == 2, lambda: "values must have shape [N,D]")
    torch._check(logits.dim() == 1, lambda: "logits must have shape [N]")
    torch._check(offsets.dim() == 1, lambda: "offsets must have shape [B+1]")
    torch._check(values.dtype in (torch.float32, torch.float64), lambda: "values must be float32 or float64")
    torch._check(logits.dtype == values.dtype, lambda: "logits dtype must match values")
    torch._check(offsets.dtype == torch.int64, lambda: "offsets must be int64")
    torch._check(values.shape[0] == logits.shape[0], lambda: "values and logits N must match")
    torch._check(values.shape[0] <= 4096, lambda: "N must be <=4096")
    torch._check(values.shape[1] >= 1, lambda: "D must be >=1")
    torch._check(values.shape[1] <= 128, lambda: "D must be <=128")
    torch._check(offsets.shape[0] >= 1, lambda: "offsets must contain at least zero")
    torch._check(offsets.shape[0] <= 129, lambda: "B must be <=128")


def check_grad_metadata(grad_output, values, offsets):
    torch._check(grad_output.device == values.device, lambda: "grad_output must be on the primal CUDA device")
    torch._check(grad_output.layout == torch.strided, lambda: "grad_output must have strided layout")
    torch._check(grad_output.dtype == values.dtype, lambda: "grad_output dtype must match values")
    torch._check(grad_output.dim() == 2, lambda: "grad_output must have shape [B,D]")
    torch._check(grad_output.shape[0] == offsets.shape[0] - 1, lambda: "grad_output B must match")
    torch._check(grad_output.shape[1] == values.shape[1], lambda: "grad_output D must match")
