"""First-order eager torch.func reverse differentiation for packed CUDA pooling.

The bridge reuses the installed native forward and backward. One manual map
evaluates bounded native lanes serially. Forward AD, higher derivatives,
nested maps, compilation and export remain outside this route.
"""
import torch


_FIRST_ORDER = "ragged_pool_functional supports first-order reverse differentiation only"
_NO_JVP = "ragged_pool_functional does not support forward-mode AD or jvp"
_NO_NESTED_MAP = "ragged_pool_functional does not support nested vmap"


class _SingleMap(torch.autograd.Function):
    @staticmethod
    def forward(value):
        # An alias preserves lane strides and the original version counter.
        return value

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def vmap(info, in_dims, value):
        raise RuntimeError(_NO_NESTED_MAP)

    @staticmethod
    def jvp(ctx, direction):
        raise RuntimeError(_NO_JVP)


def _lane_shape(tensor, dim):
    shape = tuple(tensor.shape)
    if dim is None:
        return shape
    dim %= tensor.dim()
    return shape[:dim] + shape[dim + 1:]


def _mapped_lanes(info, in_dims, arguments, value_index, offset_index):
    # chunk_size changes the current invocation's K. These are per-invocation
    # bounds, not a limit on the enclosing map's total work or memory.
    count = info.batch_size
    if not 1 <= count <= 8:
        raise RuntimeError("ragged_pool_functional mapped lane count must be in [1,8]")
    value_shape = _lane_shape(arguments[value_index], in_dims[value_index])
    offset_shape = _lane_shape(arguments[offset_index], in_dims[offset_index])
    if len(value_shape) != 2 or len(offset_shape) != 1 or offset_shape[0] < 1:
        raise RuntimeError("ragged_pool_functional mapped lanes need values [N,D] and offsets [B+1]")
    if count * value_shape[0] > 4096 or count * (offset_shape[0] - 1) > 128:
        raise RuntimeError("ragged_pool_functional mapped aggregate requires K*N <=4096 and K*B <=128")
    # A remaining outer map reaches the guard before native execution.
    # Never clone or normalize primals before native lane admission.
    guarded = tuple(_SingleMap.apply(value) if isinstance(value, torch.Tensor) else value
                    for value in arguments)
    return [tuple(value.select(dim, lane) if isinstance(value, torch.Tensor) and dim is not None else value
                  for value, dim in zip(guarded, in_dims)) for lane in range(count)]


class _NativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad_output, values, logits, offsets, versions):
        # This plain Function boundary sees underlying tensors, unlike the
        # transform wrappers saved in outer backward. _version is pin-specific.
        if tuple(tensor._version for tensor in (values, logits, offsets)) != versions:
            raise RuntimeError("ragged_pool_functional saved input version changed by an inplace operation")
        return torch.ops.ragged_pool_cuda._backward.default(grad_output, values, logits, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Keep all original arguments, including a differentiable cotangent.
        ctx.save_for_backward(*inputs[:4])
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *gradients):
        raise RuntimeError(_FIRST_ORDER)

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)

    @staticmethod
    def vmap(info, in_dims, *inputs):
        lanes = _mapped_lanes(info, in_dims, inputs, 1, 3)
        results = [_NativeBackward.apply(*lane) for lane in lanes]
        return tuple(torch.stack([result[index] for result in results]) for index in (0, 1)), (0, 0)


class _NativePool(torch.autograd.Function):
    @staticmethod
    def forward(values, logits, offsets):
        if any(torch.is_inference(tensor) for tensor in (values, logits, offsets)):
            raise RuntimeError("ragged_pool_functional requires ordinary versioned primals, not inference tensors")
        # Capture the real counters here. A torch.func wrapper's TensorImpl can
        # have its own counter, so setup_context cannot supply this metadata.
        versions = tuple(tensor._version for tensor in (values, logits, offsets))
        return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets), versions

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.versions = output[1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output, ignored_versions):
        if grad_output is None:
            return None, None, None
        values, logits, offsets = ctx.saved_tensors
        grad_values, grad_logits = _NativeBackward.apply(grad_output, values, logits, offsets, ctx.versions)
        return grad_values, grad_logits, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_NO_JVP)

    @staticmethod
    def vmap(info, in_dims, *inputs):
        lanes = _mapped_lanes(info, in_dims, inputs, 0, 2)
        results = [_NativePool.apply(*lane) for lane in lanes]
        # Every selected lane aliases its original tensor, so each argument's
        # real counter is common across lanes. Preserve that non-Tensor tuple
        # for grad-over-map as well as each plain lane's saved context.
        versions = results[0][1]
        if any(result[1] != versions for result in results[1:]):
            raise RuntimeError("ragged_pool_functional mapped input version counters differ")
        return (torch.stack([result[0] for result in results]), versions), (0, None)


def ragged_pool_functional(values, logits, offsets):
    """Pool packed CUDA inputs through bounded eager reverse transforms.

    Input admission and upstream-gradient normalization belong to the existing
    native operators. Inputs are neither copied nor coerced by this bridge.
    Offsets receive no gradient. Use the original API for its qualified compiled
    path. One vmap layer, including vmap(grad), evaluates native lanes serially.
    Each selected lane retains the native contiguous primal contract. Per rule
    invocation or chunk, 1 <=K <=8, K*N <=4096 and K*B <=128. Shared inputs
    count K times. chunk_size may process a larger total axis through several
    bounded invocations. These bounds do not limit total work or memory.
    Nested differentiation, nested maps, jvp, compile and export are excluded.
    Primals must have ordinary version counters. The mutation guard reads the
    pinned PyTorch _version property at plain Function boundaries; this internal
    property is not a cross-version or external-memory-write guarantee.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("ragged_pool_functional is eager-only and does not support compile or export")
    return _NativePool.apply(values, logits, offsets)[0]
