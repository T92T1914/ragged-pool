"""Mathematical and ownership checks for one eager CUDA reverse bridge."""
import pytest
import torch

from ragged_pool_cuda import ragged_pool
from ragged_pool_cuda.functional import ragged_pool_functional as pool


DTYPES = (torch.float32, torch.float64)
BOUNDARIES = (0, 2, 2, 5, 6)
REJECT = (RuntimeError, ValueError, NotImplementedError)


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128" and torch.version.cuda == "12.8"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)


def inputs(dtype=torch.float64, boundaries=BOUNDARIES, requires_grad=False):
    count = boundaries[-1]
    values = (torch.arange(count * 3, dtype=dtype).reshape(count, 3) / 13 - 0.6).to("cuda:0")
    logits = torch.linspace(-0.7, 0.8, count, dtype=dtype, device="cuda:0")
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda:0")
    return values.requires_grad_(requires_grad), logits.requires_grad_(requires_grad), offsets


def primitive(values, logits, boundaries):
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        selected, scores = values[start:stop].double(), logits[start:stop].double()
        row = ((torch.softmax(scores, dim=0)[:, None] * selected).sum(0) if stop > start
               else selected.sum(0) * 0 + scores.sum() * 0)
        rows.append(row.to(values.dtype))
    return torch.stack(rows) if rows else values[:0] * 0 + logits.sum() * 0


def tolerance(dtype):
    return {"rtol": 2e-4, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-10, "atol": 2e-12}


def nonlinear(result):
    return (0.17 * result.square() + 0.3 * result.sin()).sum()


def snapshot(values):
    return [(v.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).clone(),
             v._version, v.untyped_storage()._cdata, tuple(v.stride()), v.storage_offset(), v.is_neg())
            for v in values]


def unchanged(values, before):
    for value, (data, version, storage, stride, offset, negative) in zip(values, before):
        actual = value.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual, data)
        assert (value._version, value.untyped_storage()._cdata, tuple(value.stride()),
                value.storage_offset(), value.is_neg()) == (version, storage, stride, offset, negative)


def reference_grad(values, logits, offsets, upstream=None, boundaries=BOUNDARIES, native=False):
    v, l = values.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
    output = ragged_pool(v, l, offsets) if native else primitive(v, l, boundaries)
    gradients = (torch.autograd.grad(nonlinear(output), (v, l)) if upstream is None
                 else torch.autograd.grad(output, (v, l), upstream))
    return output, gradients


def valid_after_refusal():
    v, l, o = inputs()
    got = torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l)
    _, wanted = reference_grad(v, l, o)
    for actual, expected in zip(got, wanted):
        torch.testing.assert_close(actual, expected, **tolerance(v.dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boundaries", (BOUNDARIES, (0,), (0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3)))
def test_grad_and_ordinary_reverse_match_native_and_primitive(dtype, boundaries):
    v, l, o = inputs(dtype, boundaries)
    before = snapshot((v, l, o))
    got = torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l)
    output = pool(v, l, o)
    for native in (False, True):
        expected, gradients = reference_grad(v, l, o, boundaries=boundaries, native=native)
        torch.testing.assert_close(output, expected, **tolerance(dtype))
        for actual, wanted in zip(got, gradients):
            torch.testing.assert_close(actual, wanted, **tolerance(dtype))
    rv, rl = v.clone().requires_grad_(), l.clone().requires_grad_()
    ordinary = torch.autograd.grad(nonlinear(pool(rv, rl, o)), (rv, rl))
    for actual, wanted in zip(ordinary, got):
        torch.testing.assert_close(actual, wanted, **tolerance(dtype))
    if all(b - a <= 1 for a, b in zip(boundaries, boundaries[1:])):
        assert torch.equal(got[1], torch.zeros_like(l))
    assert all(value.grad is None for value in (v, l, o))
    unchanged((v, l, o), before)


def cotangent(dtype, layout):
    flat = torch.linspace(-0.7, 0.9, 12, dtype=dtype, device="cuda:0")
    if layout == "transpose": return flat.reshape(3, 4).t()
    if layout == "stepped": return torch.linspace(-0.7, 0.9, 24, dtype=dtype, device="cuda:0").reshape(4, 6)[:, ::2]
    if layout == "expanded": return flat[:4, None].expand(4, 3)
    if layout == "negative": return torch._neg_view(flat.reshape(4, 3))
    return flat.reshape(4, 3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ("contiguous", "transpose", "stepped", "expanded", "negative"))
def test_repeated_distinct_vjp_cotangents_keep_sources_and_fresh_storage(dtype, layout):
    v, l, o = inputs(dtype)
    output, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    upstream = cotangent(dtype, layout)
    before = snapshot((v, l, o, upstream))
    first = pullback(upstream)
    retained = snapshot(first)
    second_upstream = (-0.31 * upstream.resolve_neg() + 0.23).contiguous()
    second = pullback(second_upstream)
    for cot, actual in ((upstream, first), (second_upstream, second)):
        for native in (False, True):
            expected, gradients = reference_grad(v, l, o, cot, native=native)
            torch.testing.assert_close(output, expected, **tolerance(dtype))
            for got, wanted in zip(actual, gradients):
                torch.testing.assert_close(got, wanted, **tolerance(dtype))
    assert any(not torch.equal(a, b) for a, b in zip(first, second))
    sources = {value.untyped_storage()._cdata for value in (v, l, o, upstream, second_upstream)}
    results = [output, *first, *second]
    assert len({value.untyped_storage()._cdata for value in results}) == len(results)
    assert not sources & {value.untyped_storage()._cdata for value in results}
    unchanged(first, retained)
    unchanged((v, l, o, upstream), before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_functional_calls_do_not_accumulate_or_replace_existing_leaf_grads(dtype):
    v, l, o = inputs(dtype, requires_grad=True)
    v.grad, l.grad = torch.full_like(v, -2), torch.full_like(l, 3)
    original = (v.grad, l.grad)
    before = snapshot((v, l, o, *original))
    torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l)
    _, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    pullback(cotangent(dtype, "expanded"))
    assert v.grad is original[0] and l.grad is original[1] and o.grad is None
    unchanged((v, l, o, *original), before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vjp_cotangent_dtype_follows_the_ordinary_autograd_engine(dtype):
    v, l, o = inputs(dtype)
    other = torch.float64 if dtype == torch.float32 else torch.float32
    upstream = cotangent(other, "contiguous")
    before = snapshot((upstream,))
    _, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    actual = pullback(upstream)
    _, expected = reference_grad(v, l, o, upstream, native=True)
    for got, wanted in zip(actual, expected):
        assert got.dtype == dtype
        torch.testing.assert_close(got, wanted, **tolerance(dtype))
    unchanged((upstream,), before)


@pytest.mark.parametrize("field", ("values", "logits", "offsets"))
def test_vjp_retains_original_saved_input_version_checks(field):
    v, l, o = inputs()
    _, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    with torch.no_grad():
        if field == "values": v.add_(0.01)
        elif field == "logits": l.add_(0.05)
        else: o[1] = 1  # Still a valid partition, so a bounds error is not the witness.
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        pullback(torch.ones(4, 3, dtype=v.dtype, device=v.device))
    valid_after_refusal()


@pytest.mark.parametrize("fault", ("values_cpu", "offsets_cpu", "half", "dtype_mismatch", "values_stride",
    "logits_stride", "offsets_stride", "negative_primal", "nan_values", "inf_logits", "large_values",
    "start", "end", "descending", "outside", "zero_width", "large_n", "large_b"))
def test_transform_preserves_primal_admission_and_recovers(fault):
    v, l, o = inputs()
    if fault == "values_cpu": v = v.cpu()
    elif fault == "offsets_cpu": o = o.cpu()
    elif fault == "half": v, l = v.half(), l.half()
    elif fault == "dtype_mismatch": l = l.float()
    elif fault == "values_stride": v = v.t().contiguous().t()
    elif fault == "logits_stride": l = torch.zeros(l.numel() * 2, dtype=l.dtype, device=l.device)[::2].copy_(l)
    elif fault == "offsets_stride": o = torch.zeros(o.numel() * 2, dtype=o.dtype, device=o.device)[::2].copy_(o)
    elif fault == "negative_primal": v = torch._neg_view(v)
    elif fault == "nan_values": v[0, 0] = float("nan")
    elif fault == "inf_logits": l[0] = float("inf")
    elif fault == "large_values": v[0, 0] = 1001
    elif fault == "start": o[0] = 1
    elif fault == "end": o[-1] = 5
    elif fault == "descending": o[2] = 1
    elif fault == "outside": o[1] = 7
    elif fault == "zero_width": v = v[:, :0]
    elif fault == "large_n":
        v, l, o = torch.zeros(4097, 1, dtype=v.dtype, device=v.device), torch.zeros(4097, dtype=l.dtype, device=l.device), torch.tensor([0, 4097], device=o.device)
    elif fault == "large_b": o = torch.tensor([0] * 129 + [6], dtype=torch.int64, device=o.device)
    with pytest.raises(REJECT):
        torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l)
    valid_after_refusal()


@pytest.mark.parametrize("fault", ("shape", "nan", "inf", "large"))
def test_invalid_pullback_cotangent_refuses_then_same_pullback_recovers(fault):
    v, l, o = inputs()
    _, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), v, l)
    upstream = torch.ones(4, 3, dtype=v.dtype, device=v.device)
    if fault == "shape": upstream = upstream[:3]
    elif fault == "nan": upstream[0, 0] = float("nan")
    elif fault == "inf": upstream[0, 0] = float("inf")
    else: upstream[0, 0] = 1001
    with pytest.raises(REJECT): pullback(upstream)
    valid = torch.ones(4, 3, dtype=v.dtype, device=v.device)
    actual = pullback(valid)
    _, expected = reference_grad(v, l, o, valid)
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, **tolerance(v.dtype))


@pytest.mark.parametrize("route", ("ordinary_second", "nested_grad", "cotangent_grad", "jvp", "vmap", "compile", "export"))
def test_unsupported_derivative_or_capture_fails_without_poisoning_first_order(route):
    v, l, o = inputs(requires_grad=True)
    function = lambda x, s: pool(x, s, o)
    loss = lambda x, s: nonlinear(function(x, s))
    message = {"ordinary_second": "first-order", "nested_grad": "first-order",
               "cotangent_grad": "first-order", "jvp": "jvp|forward-mode", "vmap": "vmap",
               "compile": "eager-only|Observed exception", "export": "eager-only|Observed exception"}[route]
    with pytest.raises(REJECT, match=message):
        if route == "ordinary_second":
            first = torch.autograd.grad(loss(v, l), (v, l), create_graph=True)
            torch.autograd.grad(sum(g.sum() for g in first), (v, l))
        elif route == "nested_grad":
            torch.func.grad(lambda s: torch.func.grad(lambda x: loss(v, x))(s).sum())(l)
        elif route == "cotangent_grad":
            fixed_v, fixed_l = v.detach(), l.detach()
            _, pullback = torch.func.vjp(lambda x, s: pool(x, s, o), fixed_v, fixed_l)
            torch.func.grad(lambda g: pullback(g)[0].sum())(torch.ones(4, 3, dtype=v.dtype, device=v.device))
        elif route == "jvp": torch.func.jvp(function, (v, l), (torch.ones_like(v), torch.ones_like(l)))
        elif route == "vmap": torch.vmap(function)(v[None], l[None])
        elif route == "compile": torch.compile(function, backend="aot_eager", fullgraph=True)(v, l)
        else:
            class Layer(torch.nn.Module):
                def forward(self, values, logits, offsets): return pool(values, logits, offsets)
            torch.export.export(Layer(), (v, l, o), strict=True)
    valid_after_refusal()


def test_float64_directional_finite_difference_and_logit_shift():
    v, l, o = inputs()
    dv = torch.linspace(-0.2, 0.3, v.numel(), dtype=v.dtype, device=v.device).reshape_as(v)
    dl = torch.linspace(0.1, -0.2, l.numel(), dtype=l.dtype, device=l.device)
    gradients = torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l)
    directional = (gradients[0] * dv).sum() + (gradients[1] * dl).sum()
    epsilon = 1e-5
    numeric = (nonlinear(pool(v + epsilon * dv, l + epsilon * dl, o))
               - nonlinear(pool(v - epsilon * dv, l - epsilon * dl, o))) / (2 * epsilon)
    torch.testing.assert_close(directional, numeric, rtol=1e-7, atol=1e-9)
    shifted = torch.func.grad(lambda x, s: nonlinear(pool(x, s, o)), argnums=(0, 1))(v, l + 17)
    torch.testing.assert_close(pool(v, l, o), pool(v, l + 17, o), **tolerance(v.dtype))
    for actual, expected in zip(shifted, gradients):
        torch.testing.assert_close(actual, expected, **tolerance(v.dtype))
