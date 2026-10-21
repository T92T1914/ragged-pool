"""Bounded packed eager transforms using the existing native derivative math."""
import torch


class _SingleMap(torch.autograd.Function):
    @staticmethod
    def forward(value):
        # Preserve strides and the original version counter. Cloning here
        # would admit a noncontiguous primal before native validation.
        return value

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def vmap(info, in_dims, value):
        raise RuntimeError("ragged-pool nested vmap is not supported")

    @staticmethod
    def jvp(ctx, direction):
        raise RuntimeError("ragged-pool mapped forward AD is not supported")


def _lane_shape(tensor, dim):
    shape = tuple(tensor.shape)
    if dim is None:
        return shape
    dim %= tensor.dim()
    return shape[:dim] + shape[dim + 1:]


def _serial_map(function, info, in_dims, arguments, value_index, offset_index):
    # PyTorch supplies the current chunk size here. These limits apply to
    # each rule invocation, not to the complete user axis under chunk_size.
    count = info.batch_size
    if not 1 <= count <= 8:
        raise RuntimeError("ragged-pool mapped lane count must be in [1,8]")
    value_shape = _lane_shape(arguments[value_index], in_dims[value_index])
    offset_shape = _lane_shape(arguments[offset_index], in_dims[offset_index])
    if len(value_shape) != 2 or len(offset_shape) != 1 or offset_shape[0] < 1:
        raise RuntimeError("ragged-pool mapped lanes need values [N,D] and offsets [B+1]")
    # Shared arguments are charged once for every actual native lane call.
    if count * value_shape[0] > 4096 or count * (offset_shape[0] - 1) > 128:
        raise RuntimeError("ragged-pool mapped aggregate requires K*N <=4096 and K*B <=128")
    # A remaining outer map reaches this public Function and raises before
    # native execution. The identity also keeps original primal lane strides.
    arguments = tuple(_SingleMap.apply(tensor) for tensor in arguments)
    results = []
    for index in range(count):
        lane = tuple(tensor if dim is None else tensor.select(dim, index)
                     for tensor, dim in zip(arguments, in_dims))
        results.append(function.apply(*lane))
    if isinstance(results[0], tuple):
        return (tuple(torch.stack([result[j] for result in results])
                      for j in range(len(results[0]))),
                (0,) * len(results[0]))
    return torch.stack(results), 0


class _SecondOrderTangent(torch.autograd.Function):
    @staticmethod
    def forward(tangent):
        return tangent.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("ragged-pool differentiating Hessian-vector products is not supported")

    @staticmethod
    def jvp(ctx, direction):
        raise RuntimeError("ragged-pool differentiating Hessian-vector products is not supported")

    @staticmethod
    def vmap(info, in_dims, tangent):
        raise RuntimeError("ragged-pool mapped Hessian-vector products are not supported")


class _PackedBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad_output, values, logits, offsets):
        return torch.ops.ragged_pool._backward.default(grad_output, values, logits, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_grad_values, grad_grad_logits):
        grad_output, values, logits, offsets = ctx.saved_tensors
        if grad_grad_values is None:
            grad_grad_values = torch.zeros_like(values)
        if grad_grad_logits is None:
            grad_grad_logits = torch.zeros_like(logits)
        grad_grad_output, grad_values, grad_logits = torch.ops.ragged_pool._double_backward.default(
            grad_grad_values, grad_grad_logits, grad_output, values, logits, offsets,
        )
        return grad_grad_output, grad_values, grad_logits, None

    @staticmethod
    def vmap(info, in_dims, *arguments):
        return _serial_map(_PackedBackward, info, in_dims, arguments, 1, 3)

    @staticmethod
    def jvp(ctx, direction_output, direction_values, direction_logits, direction_offsets):
        grad_output, values, logits, offsets = ctx.saved_tensors
        if direction_offsets is not None:
            raise RuntimeError("ragged-pool offsets must remain fixed and have no tangent")
        if direction_values is None:
            direction_values = torch.zeros_like(values)
        if direction_logits is None:
            direction_logits = torch.zeros_like(logits)
        if direction_output is None:
            direction_output = torch.zeros_like(grad_output)
        _check_direction(direction_values, values, "values")
        _check_direction(direction_logits, logits, "logits")
        # Protect every input before opaque native calls. In particular, an
        # outer direction only on the cotangent must not disappear silently.
        grad_output, values, logits, direction_output, direction_values, direction_logits = tuple(
            _SecondOrderTangent.apply(tensor)
            for tensor in (grad_output, values, logits, direction_output,
                           direction_values, direction_logits)
        )
        # B(g,x)=J(x)^T*g, so dB=B(dg,x)+D_x B(g,x)[dx]. A nonlinear
        # surrounding loss generally changes g and needs both contributions.
        cotangent_values, cotangent_logits = torch.ops.ragged_pool._backward.default(
            direction_output, values, logits, offsets,
        )
        _, primal_values, primal_logits = torch.ops.ragged_pool._double_backward.default(
            direction_values, direction_logits, grad_output, values, logits, offsets,
        )
        return tuple(_SecondOrderTangent.apply(tensor)
                     for tensor in (cotangent_values + primal_values,
                                    cotangent_logits + primal_logits))


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
        grad_values, grad_logits = _PackedBackward.apply(
            grad_output, values, logits, offsets,
        )
        return grad_values, grad_logits, None

    @staticmethod
    def vmap(info, in_dims, *arguments):
        return _serial_map(_PackedForwardAD, info, in_dims, arguments, 0, 2)

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
    """Pool packed CPU groups through selected eager function transforms.

    Primal admission matches ragged_pool. Floating directions must match their
    primals' shape and dtype, be strided CPU tensors, and be finite within
    [-1000,1000]. Public Function guards copy inputs before native JVP reuse.
    Keep offsets fixed under differentiation. Scalar-loss torch.func.grad,
    torch.func.vjp and one vmap layer, including vmap(grad), reuse native math.
    Every selected primal lane retains the original contiguous CPU contract.
    Per rule invocation or chunk, bounds are 1 <=K <=8, K*N <=4096 and
    K*B <=128, charging shared inputs K times. PyTorch chunk_size can process
    a larger complete axis through multiple bounded invocations. These bounds
    do not limit total work or total process memory. Manual batching calls
    native code serially without a speed claim.
    A single jvp(grad(scalar_loss)) composition returns a Hessian-vector
    product using both primal and output-cotangent directions. Further
    differentiation of that product, nested maps and mapped forward AD are
    excluded. Ordinary primal reverse differentiation retains second order.
    """
    return _PackedForwardAD.apply(values, logits, offsets)
