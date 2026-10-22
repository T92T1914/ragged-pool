"""First-order eager torch.func reverse differentiation for packed CUDA pooling.

The bridge reuses the installed native forward and backward. It does not add
forward-mode AD, batching, higher derivatives, compilation or export support.
"""
import torch


_FIRST_ORDER = "ragged_pool_functional supports first-order reverse differentiation only"
_NO_JVP = "ragged_pool_functional does not support forward-mode AD or jvp"
_NO_VMAP = "ragged_pool_functional does not support vmap or batched transforms"


class _NativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad_output, values, logits, offsets):
        return torch.ops.ragged_pool_cuda._backward.default(grad_output, values, logits, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Keep all original arguments, including a differentiable cotangent.
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *gradients):
        raise RuntimeError(_FIRST_ORDER)

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)

    @staticmethod
    def vmap(info, in_dims, *inputs):
        raise RuntimeError(_NO_VMAP)


class _NativePool(torch.autograd.Function):
    @staticmethod
    def forward(values, logits, offsets):
        return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Saving the caller's primals preserves ordinary version-counter checks.
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        values, logits, offsets = ctx.saved_tensors
        grad_values, grad_logits = _NativeBackward.apply(grad_output, values, logits, offsets)
        return grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)

    @staticmethod
    def vmap(info, in_dims, *inputs):
        raise RuntimeError(_NO_VMAP)


def ragged_pool_functional(values, logits, offsets):
    """Pool packed CUDA inputs through eager first-order grad and VJP transforms.

    Input admission and upstream-gradient normalization belong to the existing
    native operators. Inputs are neither copied nor coerced by this bridge.
    Offsets receive no gradient. Use the original API for its qualified compiled
    path. Nested differentiation, jvp and vmap are outside this API's contract.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("ragged_pool_functional is eager-only and does not support compile or export")
    return _NativePool.apply(values, logits, offsets)
