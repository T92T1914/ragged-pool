"""Packed eager forward-mode AD using the existing native derivative math."""
import torch


def _check_direction(direction, primal, name):
    if direction.device.type != "cpu" or direction.layout != torch.strided:
        raise RuntimeError(f"{name} direction must be a strided CPU tensor")
    if direction.dtype != primal.dtype:
        raise RuntimeError(f"{name} direction dtype must match its primal")
    if direction.shape != primal.shape:
        raise RuntimeError(f"{name} direction shape must match its primal")
    if not bool((torch.isfinite(direction) & (direction.abs() <= 1000)).all()):
        raise RuntimeError(f"{name} direction must be finite and within [-1000,1000]")


class _FirstOrderTangent(torch.autograd.Function):
    @staticmethod
    def forward(tangent):
        return tangent.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "ragged-pool forward AD supports first-order tangents only; "
            "differentiating tangents is not supported"
        )

    @staticmethod
    def jvp(ctx, grad_input):
        raise RuntimeError("ragged-pool nested forward AD is not supported")


class _PackedForwardAD(torch.autograd.Function):
    @staticmethod
    def forward(values, logits, offsets):
        # Native admission still checks the original primal metadata and data.
        return torch.ops.ragged_pool.pool.default(values, logits, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output):
        values, logits, offsets = ctx.saved_tensors
        if grad_output is None:
            return None, None, None
        grad_values, grad_logits = torch.ops.ragged_pool._backward.default(
            grad_output, values, logits, offsets,
        )
        return grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, direction_values, direction_logits, direction_offsets):
        values, logits, offsets = ctx.saved_tensors
        if direction_offsets is not None:
            raise RuntimeError("ragged-pool offsets must remain fixed and have no tangent")
        if direction_values is None:
            direction_values = torch.zeros_like(values)
        if direction_logits is None:
            direction_logits = torch.zeros_like(logits)
        _check_direction(direction_values, values, "values")
        _check_direction(direction_logits, logits, "logits")
        # Reject outer forward tangents before an opaque native call could lose
        # them. Directions also need guards when only they vary under nesting.
        values, logits, direction_values, direction_logits = tuple(
            _FirstOrderTangent.apply(tensor)
            for tensor in (values, logits, direction_values, direction_logits)
        )
        zero_upstream = values.new_zeros((offsets.numel() - 1, values.shape[1]))
        # The existing derivative with respect to the output cotangent is J*d.
        # Its other two outputs are unused. This adds no dispatcher operation.
        tangent, _, _ = torch.ops.ragged_pool._double_backward.default(
            direction_values, direction_logits, zero_upstream, values, logits, offsets,
        )
        return _FirstOrderTangent.apply(tangent)


def ragged_pool_forward_ad(values, logits, offsets):
    """Pool packed CPU groups with an eager first-order forward-mode AD bridge.

    Primal admission matches ragged_pool. Floating directions must match their
    primals' shape and dtype, be strided CPU tensors, and be finite within
    [-1000,1000]. Public Function guards copy inputs before native JVP reuse.
    Keep offsets fixed. Tangent differentiation and nested transforms are
    excluded. Ordinary primal reverse differentiation retains second order.
    """
    return _PackedForwardAD.apply(values, logits, offsets)
