"""Independent numerical, ownership and refusal checks for bounded CUDA maps."""
import pytest
import torch

from ragged_pool_cuda import ragged_pool
from ragged_pool_cuda.functional import ragged_pool_functional as pool


DTYPES = (torch.float32, torch.float64)
PARTITIONS = ((0, 2, 2, 5, 6), (0, 0, 1, 3, 6), (0, 1, 2, 6, 6))
REJECT = (RuntimeError, ValueError, NotImplementedError)


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128" and torch.version.cuda == "12.8"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)


def tolerance(dtype):
    return {"rtol": 2e-4, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-10, "atol": 2e-12}


def inputs(dtype=torch.float64, count=3, partitions=PARTITIONS, width=3):
    rows = partitions[0][-1]
    assert all(part[-1] == rows and len(part) == len(partitions[0]) for part in partitions)
    values = (torch.arange(count * rows * width, dtype=dtype, device="cuda:0")
              .reshape(count, rows, width) / 73 - 0.4)
    logits = torch.linspace(-0.8, 0.6, count * rows, dtype=dtype, device="cuda:0").reshape(count, rows)
    offsets = torch.tensor([partitions[k % len(partitions)] for k in range(count)],
                           dtype=torch.int64, device="cuda:0").reshape(count, len(partitions[0]))
    return values, logits, offsets


def primitive(values, logits, boundaries):
    result = []
    for begin, end in zip(boundaries, boundaries[1:]):
        v, l = values[begin:end].double(), logits[begin:end].double()
        row = ((v * torch.softmax(l, 0)[:, None]).sum(0) if end > begin
               else v.sum(0) * 0 + l.sum() * 0)
        result.append(row.to(values.dtype))
    return torch.stack(result) if result else values[:0] * 0 + logits.sum() * 0


def objective(output):
    return (0.17 * output.square() + 0.3 * output.sin()).sum()


def snapshot(tensors):
    return [(t.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8).clone(),
             t._version, t.untyped_storage()._cdata, tuple(t.stride()), t.storage_offset(), t.is_neg())
            for t in tensors]


def unchanged(tensors, before):
    for tensor, (data, version, storage, strides, offset, negative) in zip(tensors, before):
        assert torch.equal(tensor.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8), data)
        assert (tensor._version, tensor.untyped_storage()._cdata, tuple(tensor.stride()),
                tensor.storage_offset(), tensor.is_neg()) == (version, storage, strides, offset, negative)


def forbid_native_calls(monkeypatch):
    calls = []

    def forward(*args):
        calls.append("forward")
        raise AssertionError("mapped refusal reached the native forward operation")

    def backward(*args):
        calls.append("backward")
        raise AssertionError("mapped refusal reached the native backward operation")

    monkeypatch.setattr(torch.ops.ragged_pool_cuda.pool, "default", forward)
    monkeypatch.setattr(torch.ops.ragged_pool_cuda._backward, "default", backward)
    return calls


def serial(values, logits, offsets, dimensions, partitions, native, upstreams=None):
    count = next(t.shape[d] for t, d in zip((values, logits, offsets), dimensions) if d is not None)
    outputs, value_grads, logit_grads = [], [], []
    for k in range(count):
        v, l, o = (t if d is None else t.select(d, k)
                   for t, d in zip((values, logits, offsets), dimensions))
        v, l = v.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
        bounds = partitions[0] if dimensions[2] is None else partitions[k % len(partitions)]
        output = ragged_pool(v, l, o) if native else primitive(v, l, bounds)
        gradients = (torch.autograd.grad(objective(output), (v, l)) if upstreams is None
                     else torch.autograd.grad(output, (v, l), upstreams[k]))
        outputs.append(output)
        value_grads.append(gradients[0])
        logit_grads.append(gradients[1])
    return torch.stack(outputs), (torch.stack(value_grads), torch.stack(logit_grads))


def valid_after_refusal():
    v, l, o = inputs()
    got = torch.vmap(torch.func.grad(lambda x, s, b: objective(pool(x, s, b)), argnums=(0, 1)))(v, l, o)
    _, expected = serial(v, l, o, (0, 0, 0), PARTITIONS, False)
    for actual, wanted in zip(got, expected):
        torch.testing.assert_close(actual, wanted, **tolerance(v.dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dimensions", ((0, 0, 0), (0, None, None), (None, 0, None),
                                         (None, None, 0), (0, None, 0), (0, 0, None), (None, 0, 0)))
def test_mapped_forward_and_per_lane_gradients_match_two_oracles(dtype, dimensions):
    batched = inputs(dtype)
    args = tuple(t if d is not None else t[0] for t, d in zip(batched, dimensions))
    for tensor in args:
        tensor.copy_(tensor.clone())
        assert tensor._version > 0
    before = snapshot(args)
    output = torch.vmap(pool, in_dims=dimensions)(*args)
    gradients = torch.vmap(torch.func.grad(lambda v, l, o: objective(pool(v, l, o)), argnums=(0, 1)),
                           in_dims=dimensions)(*args)
    for native in (False, True):
        expected, wanted = serial(*args, dimensions, PARTITIONS, native)
        torch.testing.assert_close(output, expected, **tolerance(dtype))
        for actual, oracle in zip(gradients, wanted):
            torch.testing.assert_close(actual, oracle, **tolerance(dtype))
    assert all(t.grad is None for t in args)
    sources = {t.untyped_storage()._cdata for t in args if t.numel()}
    assert not sources & {t.untyped_storage()._cdata for t in (output, *gradients) if t.numel()}
    unchanged(args, before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boundaries", ((0,), (0, 0, 0), (0, 1, 2, 3)))
def test_mapped_zero_empty_and_singleton_lanes(dtype, boundaries):
    v, l, o = inputs(dtype, partitions=(boundaries,))
    before = snapshot((v, l, o))
    output = torch.vmap(pool)(v, l, o)
    gradients = torch.vmap(torch.func.grad(lambda x, s, b: objective(pool(x, s, b)), argnums=(0, 1)))(v, l, o)
    expected, wanted = serial(v, l, o, (0, 0, 0), (boundaries,), False)
    torch.testing.assert_close(output, expected, **tolerance(dtype))
    for actual, oracle in zip(gradients, wanted):
        torch.testing.assert_close(actual, oracle, **tolerance(dtype))
    assert torch.equal(gradients[1], torch.zeros_like(l))
    if boundaries[-1] == 0:
        assert torch.equal(output, torch.zeros_like(output))
    unchanged((v, l, o), before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunk_size", (1, 2, 5))
def test_uneven_chunking_preserves_lane_order_and_gradients(dtype, chunk_size):
    v, l, o = inputs(dtype, count=5)
    before = snapshot((v, l, o))
    output = torch.vmap(pool, chunk_size=chunk_size)(v, l, o)
    gradients = torch.vmap(torch.func.grad(lambda x, s, b: objective(pool(x, s, b)), argnums=(0, 1)),
                           chunk_size=chunk_size)(v, l, o)
    for native in (False, True):
        expected, wanted = serial(v, l, o, (0, 0, 0), PARTITIONS, native)
        torch.testing.assert_close(output, expected, **tolerance(dtype))
        for actual, oracle in zip(gradients, wanted):
            torch.testing.assert_close(actual, oracle, **tolerance(dtype))
    unchanged((v, l, o), before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonleading_map_axes_preserve_original_contiguous_lane_views(dtype):
    leading = inputs(dtype)
    args = tuple(t.movedim(0, 1) for t in leading)
    assert all(t.select(1, 0).is_contiguous() for t in args)
    before = snapshot(args)
    output = torch.vmap(pool, in_dims=(1, 1, 1), out_dims=1)(*args)
    got = torch.vmap(torch.func.grad(lambda v, l, o: objective(pool(v, l, o)), argnums=(0, 1)),
                     in_dims=(1, 1, 1))(*args)
    expected, wanted = serial(*args, (1, 1, 1), PARTITIONS, False)
    torch.testing.assert_close(output.movedim(1, 0), expected, **tolerance(dtype))
    for actual, oracle in zip(got, wanted):
        torch.testing.assert_close(actual, oracle, **tolerance(dtype))
    unchanged(args, before)


@pytest.mark.parametrize("fault", ("values_stride", "logits_stride", "offsets_stride", "negative_values",
                                    "negative_logits", "negative_offsets", "negative_strided_values"))
def test_map_does_not_coerce_unsupported_primal_lanes(fault):
    v, l, o = inputs()
    args = [v, l, o]
    field = 2 if "offsets" in fault else 1 if "logits" in fault else 0
    if "stride" in fault:
        original = args[field]
        shape = (*original.shape[:-1], original.shape[-1] * 2)
        args[field] = torch.empty(shape, dtype=original.dtype, device=original.device)[..., ::2].copy_(original)
        assert not args[field][0].is_contiguous()
    if "negative" in fault:
        args[field] = torch._neg_view(args[field])
    before = snapshot(args)
    with pytest.raises(REJECT, match="contiguous|negative"):
        torch.vmap(pool)(*args)
    unchanged(args, before)
    valid_after_refusal()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ("contiguous", "transpose", "stepped", "expanded", "negative"))
def test_mapped_pullbacks_normalize_upstream_views_and_remain_reusable(dtype, layout):
    v, l, o = (t[0] for t in inputs(dtype))
    output, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    base = torch.linspace(-0.5, 0.8, 36, dtype=dtype, device=v.device)
    if layout == "transpose": upstream = base.reshape(3, 3, 4).transpose(1, 2)
    elif layout == "stepped": upstream = torch.linspace(-0.5, 0.8, 72, dtype=dtype, device=v.device).reshape(3, 4, 6)[:, :, ::2]
    elif layout == "expanded": upstream = base[:12].reshape(3, 4, 1).expand(3, 4, 3)
    elif layout == "negative": upstream = torch._neg_view(base.reshape(3, 4, 3))
    else: upstream = base.reshape(3, 4, 3)
    second = (-0.23 * upstream.resolve_neg() + 0.19).contiguous()
    before = snapshot((v, l, o, upstream, second, output))
    first_grads = torch.vmap(pullback)(upstream)
    retained = snapshot(first_grads)
    second_grads = torch.vmap(pullback)(second)
    for upstreams, gradients in ((upstream, first_grads), (second, second_grads)):
        for k in range(3):
            rv, rl = v.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
            wanted = torch.autograd.grad(primitive(rv, rl, PARTITIONS[0]), (rv, rl), upstreams[k])
            for actual, oracle in zip(gradients, wanted):
                torch.testing.assert_close(actual[k], oracle, **tolerance(dtype))
    assert any(not torch.equal(a, b) for a, b in zip(first_grads, second_grads))
    result_storage = {t.untyped_storage()._cdata for t in (*first_grads, *second_grads)}
    assert len(result_storage) == 4
    assert not result_storage & {t.untyped_storage()._cdata for t in (v, l, o, upstream, second, output)}
    unchanged(first_grads, retained)
    unchanged((v, l, o, upstream, second, output), before)


@pytest.mark.parametrize("field", (0, 1, 2), ids=("values", "logits", "offsets"))
@pytest.mark.parametrize("shared", (False, True))
def test_mapped_saved_input_guards_use_real_nonzero_versions(field, shared):
    full = inputs()
    dimensions = tuple(None if shared and k == field else 0 for k in range(3))
    args = [t[0] if d is None else t for t, d in zip(full, dimensions)]
    for tensor in args:
        tensor.copy_(tensor.clone())
        assert tensor._version > 0
    v, l, o = args
    _, pullback = torch.func.vjp(lambda x, s: torch.vmap(pool, in_dims=dimensions)(x, s, o), v, l)
    before = snapshot(args)
    expected, _ = serial(v, l, o, dimensions, PARTITIONS, False)
    valid = pullback(torch.ones_like(expected))
    assert all(torch.isfinite(t).all() for t in valid)
    unchanged(args, before)
    with torch.no_grad():
        if field < 2: args[field].add_(0.01)
        elif shared: o[1] = 1
        else: o[0, 1] = 1
    with pytest.raises(RuntimeError, match="version|modified by an inplace operation"):
        pullback(torch.ones_like(expected))
    valid_after_refusal()


@pytest.mark.parametrize("fault", ("zero_lanes", "lanes", "items", "groups"))
def test_per_invocation_bounds_refuse_before_native_execution_and_recover(fault, monkeypatch):
    if fault == "zero_lanes": args = inputs(count=0)
    elif fault == "lanes": args = inputs(count=9)
    elif fault == "items": args = inputs(count=2, partitions=((0, 2049),), width=1)
    else: args = inputs(count=2, partitions=((0,) * 65 + (1,),), width=1)
    before = snapshot(args)
    with monkeypatch.context() as patch:
        calls = forbid_native_calls(patch)
        with pytest.raises(REJECT, match="lane|aggregate|4096|128"):
            torch.vmap(pool)(*args)
        assert calls == []
    primals = tuple(t[0] for t in (inputs() if fault == "zero_lanes" else args))
    # Build a valid pullback before guarding the mapped backward invocation.
    _, pullback = torch.func.vjp(lambda v, l: pool(v, l, primals[2]), *primals[:2])
    upstreams = torch.ones((args[0].shape[0], primals[2].numel() - 1, primals[0].shape[1]),
                           dtype=primals[0].dtype, device=primals[0].device)
    with monkeypatch.context() as patch:
        calls = forbid_native_calls(patch)
        with pytest.raises(REJECT, match="lane|aggregate|4096|128"):
            torch.vmap(pullback)(upstreams)
        assert calls == []
    unchanged(args, before)
    valid_after_refusal()


def test_shared_inputs_are_charged_per_call_and_complete_axis_can_be_chunked(monkeypatch):
    v, l, o = inputs(count=9)
    with monkeypatch.context() as patch:
        calls = forbid_native_calls(patch)
        with pytest.raises(REJECT, match="lane|8"):
            torch.vmap(pool)(v, l, o)
        assert calls == []
    actual = torch.vmap(pool, chunk_size=2)(v, l, o)
    expected, _ = serial(v, l, o, (0, 0, 0), PARTITIONS, False)
    torch.testing.assert_close(actual, expected, **tolerance(v.dtype))
    shared_v, shared_l, mapped_o = inputs(count=2, partitions=((0, 2049),), width=1)
    with monkeypatch.context() as patch:
        calls = forbid_native_calls(patch)
        with pytest.raises(REJECT, match="aggregate|4096"):
            torch.vmap(pool, in_dims=(None, None, 0))(shared_v[0], shared_l[0], mapped_o)
        assert calls == []
    valid_after_refusal()


def test_exact_mapped_item_and_group_envelopes_are_admitted():
    boundaries = tuple(range(0, 513, 32))
    v, l, o = inputs(count=8, partitions=(boundaries,), width=1)
    assert v.shape[0] * v.shape[1] == 4096 and 8 * (o.shape[1] - 1) == 128
    before = snapshot((v, l, o))
    output = torch.vmap(pool)(v, l, o)
    got = torch.vmap(torch.func.grad(lambda x, s, b: objective(pool(x, s, b)), argnums=(0, 1)))(v, l, o)
    expected, wanted = serial(v, l, o, (0, 0, 0), (boundaries,), False)
    torch.testing.assert_close(output, expected, **tolerance(v.dtype))
    for actual, oracle in zip(got, wanted):
        torch.testing.assert_close(actual, oracle, **tolerance(v.dtype))
    unchanged((v, l, o), before)


@pytest.mark.parametrize("outer_dimensions", ((0, None, None), (None, 0, None), (None, None, 0)),
                         ids=("outer_values_only", "outer_logits_only", "outer_offsets_only"))
def test_outer_map_on_any_single_primal_refuses_before_native_execution(outer_dimensions, monkeypatch):
    inner = inputs()
    arguments = tuple(torch.stack((tensor, tensor.clone())) if dimension is not None else tensor
                      for tensor, dimension in zip(inner, outer_dimensions))
    before = snapshot(arguments)
    with monkeypatch.context() as patch:
        calls = forbid_native_calls(patch)
        with pytest.raises(REJECT, match="nested.*vmap|vmap.*nested"):
            torch.vmap(torch.vmap(pool), in_dims=outer_dimensions)(*arguments)
        assert calls == []
    unchanged(arguments, before)
    valid_after_refusal()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction", ("sum", "mean"))
def test_shared_model_parameter_gradients_reduce_with_unequal_lane_weights(dtype, reduction):
    v, l, o = inputs(dtype)
    targets = torch.tensor([[0.2, -0.3, 0.1, 0.4], [-0.5, 0.1, 0.7, -0.1],
                            [0.1, 0.3, -0.2, 0.2]], dtype=dtype, device=v.device)
    parameters = {
        "encoder": torch.nn.Parameter(torch.arange(6, dtype=dtype, device=v.device).reshape(3, 2) / 17 - 0.1),
        "scorer": torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=dtype, device=v.device)),
        "head": torch.nn.Parameter(torch.tensor([0.4, -0.5], dtype=dtype, device=v.device)),
        "bias": torch.nn.Parameter(torch.tensor(0.17, dtype=dtype, device=v.device)),
    }
    for k, parameter in enumerate(parameters.values()):
        parameter.grad = torch.full_like(parameter, k + 2)
    existing_grads = tuple(p.grad for p in parameters.values())
    original = (v, l, o, targets, *parameters.values(), *existing_grads)
    before = snapshot(original)

    def loss(p, x, scores, bounds, target, operation):
        features = torch.tanh(x @ p["encoder"]).contiguous()
        learned_scores = (scores + features @ p["scorer"]).contiguous()
        prediction = operation(features, learned_scores, bounds) @ p["head"] + p["bias"]
        return (prediction - target).square().mean()

    mapped_loss = lambda p, x, scores, bounds, target: loss(p, x, scores, bounds, target, pool)
    actual = torch.vmap(torch.func.grad(mapped_loss), in_dims=(None, 0, 0, 0, 0))(parameters, v, l, o, targets)
    oracle_rows = {name: [] for name in parameters}
    for native in (False, True):
        rows = {name: [] for name in parameters}
        for lane in range(3):
            copied = {name: p.detach().clone().requires_grad_() for name, p in parameters.items()}
            operation = ragged_pool if native else lambda x, scores, bounds: primitive(x, scores, PARTITIONS[lane])
            gradients = torch.autograd.grad(loss(copied, v[lane], l[lane], o[lane], targets[lane], operation), tuple(copied.values()))
            for name, gradient in zip(copied, gradients): rows[name].append(gradient)
        for name in parameters:
            torch.testing.assert_close(actual[name], torch.stack(rows[name]), **tolerance(dtype))
        if not native: oracle_rows = rows
    weights = torch.tensor([0.7, -0.2, 1.1], dtype=dtype, device=v.device)

    def combined(p):
        losses = torch.vmap(mapped_loss, in_dims=(None, 0, 0, 0, 0))(p, v, l, o, targets)
        weighted = losses * weights
        return weighted.sum() if reduction == "sum" else weighted.mean()

    combined_grads = torch.func.grad(combined)(parameters)
    for name in parameters:
        expected = sum(weights[k] * oracle_rows[name][k] for k in range(3))
        if reduction == "mean": expected = expected / 3
        torch.testing.assert_close(combined_grads[name], expected, **tolerance(dtype))
    assert all(p.grad is old for p, old in zip(parameters.values(), existing_grads))
    assert all(t.grad is None for t in (v, l, o, targets))
    unchanged(original, before)


@pytest.mark.parametrize("route", ("nested_vmap", "mapped_jvp", "mapped_second", "mapped_nested_grad", "compile", "export"))
def test_excluded_map_compositions_refuse_and_valid_map_recovers(route):
    v, l, o = inputs()
    mapped_pool = lambda x, s: torch.vmap(pool)(x, s, o)
    mapped_grad = torch.vmap(torch.func.grad(lambda x, s, b: objective(pool(x, s, b)), argnums=(0, 1)))
    message = ("nested.*vmap|vmap.*nested" if route == "nested_vmap"
               else "jvp|forward-mode|mapped forward" if route == "mapped_jvp"
               else "eager-only|Observed exception" if route in ("compile", "export")
               else "first-order")
    with pytest.raises(REJECT, match=message):
        if route == "nested_vmap":
            torch.vmap(torch.vmap(pool))(torch.stack((v, v + 0.1)), torch.stack((l, l + 0.1)), torch.stack((o, o)))
        elif route == "mapped_jvp":
            torch.func.jvp(mapped_pool, (v, l), (torch.ones_like(v), torch.ones_like(l)))
        elif route == "mapped_second":
            v.requires_grad_()
            l.requires_grad_()
            first = torch.autograd.grad(objective(mapped_pool(v, l)), (v, l), create_graph=True)
            torch.autograd.grad(sum(t.sum() for t in first), (v, l))
        elif route == "mapped_nested_grad":
            torch.func.grad(lambda x: mapped_grad(x, l, o)[0].sum())(v)
        elif route == "compile":
            torch.compile(mapped_pool, backend="aot_eager", fullgraph=True)(v, l)
        else:
            class Layer(torch.nn.Module):
                def forward(self, values, logits, offsets):
                    return torch.vmap(pool)(values, logits, offsets)
            torch.export.export(Layer(), (v, l, o), strict=True)
    valid_after_refusal()
