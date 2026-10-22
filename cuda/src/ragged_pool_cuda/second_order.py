"""Eager first and second reverse derivatives for bounded packed CUDA pooling."""
import torch


_NO_JVP = "ragged_pool_second_order does not support forward-mode AD or jvp"


def _require_versioned(*tensors):
    if torch.is_inference_mode_enabled() or any(torch.is_inference(tensor) for tensor in tensors):
        raise RuntimeError("ragged_pool_second_order requires ordinary versioned tensors, not inference tensors")


class _SecondOrderBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, values, logits, offsets):
        _require_versioned(grad_output, values, logits, offsets)
        # Saving the original cotangent preserves nonlinear model-head curvature.
        # Ordinary autograd checks saved tensor versions before this backward.
        ctx.save_for_backward(grad_output, values, logits, offsets)
        ctx.set_materialize_grads(False)
        return torch.ops.ragged_pool_cuda._backward.default(grad_output, values, logits, offsets)

    @staticmethod
    def backward(ctx, grad_grad_values, grad_grad_logits):
        grad_output, values, logits, offsets = ctx.saved_tensors
        if grad_grad_values is None and grad_grad_logits is None:
            return None, None, None, None
        if grad_grad_values is None:
            grad_grad_values = torch.zeros_like(values)
        if grad_grad_logits is None:
            grad_grad_logits = torch.zeros_like(logits)
        _require_versioned(grad_grad_values, grad_grad_logits, grad_output, values, logits, offsets)
        grad_grad_output, grad_values, grad_logits = torch.ops.ragged_pool_cuda._double_backward.default(
            grad_grad_values, grad_grad_logits, grad_output, values, logits, offsets,
        )
        return grad_grad_output, grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)


class _SecondOrderPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, logits, offsets):
        _require_versioned(values, logits, offsets)
        ctx.save_for_backward(values, logits, offsets)
        ctx.set_materialize_grads(False)
        return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        values, logits, offsets = ctx.saved_tensors
        grad_values, grad_logits = _SecondOrderBackward.apply(grad_output, values, logits, offsets)
        return grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)


def ragged_pool_second_order(values, logits, offsets):
    """Pool packed CUDA inputs through up to two ordinary reverse derivatives.

    The original contiguous same-device float32/float64 primal contract applies.
    Native backward accepts strided derivative inputs, resolves logical negative
    views, and checks all supplied values within the finite [-1000,1000] envelope.
    Offsets receive no gradient. Empty groups have zero outputs and derivatives.
    Singleton groups follow their feature input with no logit derivatives.

    This separate eager API retains saved original tensor version checks and the
    cotangent dependency needed by nonlinear heads. Third derivatives, forward AD,
    torch.func transforms, vmap, compile, export and inference tensors are outside
    its contract. The classic autograd.Function boundary deliberately provides no
    setup_context transform bridge. No existing API gains higher-order support.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("ragged_pool_second_order is eager-only and does not support compile or export")
    _require_versioned(values, logits, offsets)
    return _SecondOrderPool.apply(values, logits, offsets)
