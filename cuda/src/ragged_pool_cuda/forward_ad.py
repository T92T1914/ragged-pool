"""One eager first-order JVP for packed CUDA pooling, with ordinary reverse AD."""
import torch

from .functional import _NativeBackward


_NO_TANGENT_GRAD = (
    "ragged_pool_forward_ad supports first-order JVP only; "
    "differentiating tangents is not supported"
)
_NO_NESTED_JVP = "ragged_pool_forward_ad does not support nested JVP or forward AD"
_NO_VMAP = "ragged_pool_forward_ad does not support vmap or batched transforms"


class _FirstOrderTangent(torch.autograd.Function):
    @staticmethod
    def forward(tensor, expected_version):
        if torch.is_inference(tensor):
            raise RuntimeError("ragged_pool_forward_ad requires ordinary versioned inputs, not inference tensors")
        # torch.func wrappers can carry counters of their own. This plain
        # Function boundary sees the original counter before making a copy.
        if expected_version is not None and tensor._version != expected_version:
            raise RuntimeError("ragged_pool_forward_ad saved input version changed by an inplace operation")
        return tensor.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(_NO_TANGENT_GRAD)

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_NESTED_JVP)

    @staticmethod
    def vmap(info, in_dims, *inputs):
        raise RuntimeError(_NO_VMAP)


class _PackedForwardAD(torch.autograd.Function):
    @staticmethod
    def forward(values, logits, offsets):
        if torch.is_inference_mode_enabled() or any(
            torch.is_inference(tensor) for tensor in (values, logits, offsets)
        ):
            raise RuntimeError("ragged_pool_forward_ad requires ordinary versioned primals, not inference tensors")
        versions = tuple(tensor._version for tensor in (values, logits, offsets))
        # Admit the original primals before any guard copy can change strides.
        return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets), versions

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.versions = output[1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output, ignored_versions):
        if grad_output is None:
            return None, None, None
        values, logits, offsets = ctx.saved_tensors
        grad_values, grad_logits = _NativeBackward.apply(
            grad_output, values, logits, offsets, ctx.versions,
        )
        return grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, dvalues, dlogits, doffsets):
        values, logits, offsets = ctx.saved_tensors
        if doffsets is not None:
            raise RuntimeError("ragged_pool_forward_ad offsets must stay fixed and have no tangent")
        if dvalues is None:
            dvalues = torch.zeros_like(values)
        if dlogits is None:
            dlogits = torch.zeros_like(logits)
        # Guard every floating dependency before the opaque native call. An
        # outer transform may vary only a primal or only a supplied direction.
        values, logits, offsets = tuple(
            _FirstOrderTangent.apply(tensor, version)
            for tensor, version in zip((values, logits, offsets), ctx.versions)
        )
        dvalues, dlogits = tuple(
            _FirstOrderTangent.apply(tensor, None) for tensor in (dvalues, dlogits)
        )
        tangent = torch.ops.ragged_pool_cuda._jvp.default(
            values, logits, offsets, dvalues, dlogits,
        )
        return _FirstOrderTangent.apply(tangent, None), None

    @staticmethod
    def vmap(info, in_dims, *inputs):
        raise RuntimeError(_NO_VMAP)


def ragged_pool_forward_ad(values, logits, offsets):
    """Pool packed CUDA groups through one eager forward-mode derivative.

    Explicit dual tensors and torch.func.jvp reuse native pool and JVP kernels.
    The original contiguous CUDA primal contract still applies. Floating
    directions must match their primal shape, dtype and device, have strided
    layout, and be finite within [-1000,1000]. Native code normalizes logical
    negative and noncontiguous directions. Empty groups have zero tangents.

    Ordinary first-order reverse AD uses the established native backward.
    Keep offsets fixed. Reverse differentiation of a tangent, nested JVP,
    higher reverse derivatives, vmap, compilation and export are excluded.
    Guards copy saved inputs only after original native admission and retain
    dependency boundaries so unsupported transforms fail explicitly. Primals
    and directions must be ordinary versioned tensors. Mutation checks use
    pinned PyTorch Tensor._version at plain Function boundaries and do not
    cover external-memory writes or promise compatibility with other versions.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("ragged_pool_forward_ad is eager-only and does not support compile or export")
    return _PackedForwardAD.apply(values, logits, offsets)[0]
