"""Check JVP mathematics independently from registration and native admission."""
import pytest
import torch

from ragged_pool_cuda import ragged_pool
from ragged_pool_cuda.forward_ad import ragged_pool_forward_ad as pool


DTYPES = (torch.float32, torch.float64)
BOUNDARIES = (0, 2, 2, 5, 6)
REJECT = (RuntimeError, ValueError, NotImplementedError)


@pytest.fixture(scope='module', autouse=True)
def environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == '2.9.1+cu128' and torch.version.cuda == '12.8'
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)


def fixture(dtype=torch.float64, boundaries=BOUNDARIES):
    n = boundaries[-1]
    v = (torch.arange(n * 3, dtype=dtype, device='cuda:0').reshape(n, 3) / 13 - 0.6)
    l = torch.linspace(-0.7, 0.8, n, dtype=dtype, device='cuda:0')
    o = torch.tensor(boundaries, dtype=torch.int64, device='cuda:0')
    dv = torch.linspace(0.3, -0.2, n * 3, dtype=dtype, device='cuda:0').reshape(n, 3)
    dl = torch.linspace(-0.4, 0.7, n, dtype=dtype, device='cuda:0')
    return v, l, o, dv, dl


def primitive(v, l, boundaries):
    rows = []
    for a, b in zip(boundaries, boundaries[1:]):
        x, s = v[a:b].double(), l[a:b].double()
        rows.append(((s.softmax(0)[:, None] * x).sum(0) if b > a
                     else x.sum(0) * 0 + s.sum() * 0).to(v.dtype))
    return torch.stack(rows) if rows else v[:0] * 0 + l.sum() * 0


def tol(dtype):
    return dict(rtol=2e-4, atol=2e-6) if dtype == torch.float32 else dict(rtol=2e-10, atol=2e-12)


def stamp(items):
    return [(id(x), x._version, x.untyped_storage()._cdata, x.stride(), x.storage_offset(),
             x.is_neg(), x.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8).clone()) for x in items]


def unchanged(items, before):
    for got, expected in zip(stamp(items), before):
        assert got[:-1] == expected[:-1]
        assert torch.equal(got[-1], expected[-1])


def native(v, l, o, dv, dl):
    return torch.ops.ragged_pool_cuda._jvp.default(v, l, o, dv, dl)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('boundaries', (BOUNDARIES, (0,), (0, 0, 0), (0, 1, 2, 3), (0, 6)))
def test_public_and_native_jvp_independent_primitive_duality_and_ownership(dtype, boundaries):
    v, l, o, dv, dl = fixture(dtype, boundaries)
    before = stamp((v, l, o, dv, dl))
    y, tangent = torch.func.jvp(lambda x, s: pool(x, s, o), (v, l), (dv, dl))
    expected, expected_t = torch.func.jvp(lambda x, s: primitive(x, s, boundaries), (v, l), (dv, dl))
    torch.testing.assert_close(y, expected, **tol(dtype))
    torch.testing.assert_close(tangent, expected_t, **tol(dtype))
    torch.testing.assert_close(native(v, l, o, dv, dl), expected_t, **tol(dtype))
    assert y.untyped_storage()._cdata != tangent.untyped_storage()._cdata
    assert not {y.untyped_storage()._cdata, tangent.untyped_storage()._cdata} & {x.untyped_storage()._cdata for x in (v, l, o, dv, dl)}
    rv, rl = v.clone().requires_grad_(), l.clone().requires_grad_()
    g = torch.linspace(-0.3, 0.8, y.numel(), dtype=dtype, device=v.device).reshape(y.shape)
    gv, gl = torch.autograd.grad(ragged_pool(rv, rl, o), (rv, rl), g)
    torch.testing.assert_close((tangent.double() * g).sum(), (gv.double() * dv).sum() + (gl.double() * dl).sum(), **tol(dtype))
    unchanged((v, l, o, dv, dl), before)


@pytest.mark.parametrize('layout', ('transpose', 'step', 'expand', 'negative', 'negative_step'))
@pytest.mark.parametrize('dtype', DTYPES)
def test_direct_native_direction_layouts_are_normalized_without_changing_sources(dtype, layout):
    v, l, o, dv, dl = fixture(dtype)
    if layout == 'transpose': dv = dv.t().contiguous().t()
    if layout == 'step': dv, dl = dv.repeat_interleave(2, 1)[:, ::2], dl.repeat_interleave(2)[::2]
    if layout == 'expand': dv, dl = dv[:, :1].expand_as(dv), dl[:1].expand_as(dl)
    if layout == 'negative': dv, dl = torch._neg_view(dv), torch._neg_view(dl)
    if layout == 'negative_step': dv, dl = torch._neg_view(dv.repeat_interleave(2, 1)[:, ::2]), torch._neg_view(dl.repeat_interleave(2)[::2])
    before = stamp((v, l, o, dv, dl))
    expected = torch.func.jvp(lambda x, s: primitive(x, s, BOUNDARIES), (v, l), (dv, dl))[1]
    torch.testing.assert_close(native(v, l, o, dv, dl), expected, **tol(dtype))
    unchanged((v, l, o, dv, dl), before)


@pytest.mark.parametrize('which', ('both', 'values', 'logits'))
def test_explicit_duals_allow_missing_directions(which):
    v, l, o, dv, dl = fixture()
    from torch.autograd import forward_ad
    with forward_ad.dual_level():
        x = forward_ad.make_dual(v, dv) if which != 'logits' else v
        s = forward_ad.make_dual(l, dl) if which != 'values' else l
        y, t = forward_ad.unpack_dual(pool(x, s, o))
        expected = native(v, l, o, dv if which != 'logits' else torch.zeros_like(dv), dl if which != 'values' else torch.zeros_like(dl))
        torch.testing.assert_close(t, expected, **tol(v.dtype))
        torch.testing.assert_close(y, ragged_pool(v, l, o), **tol(v.dtype))


def test_analytic_weights_score_centering_shift_singleton_and_linearity():
    v = torch.tensor([[1., 2.], [3., -2.], [7., 4.]], dtype=torch.float64, device='cuda:0')
    l = torch.tensor([0., 1.0986122886681098, -0.2], dtype=v.dtype, device=v.device)
    o = torch.tensor([0, 2, 2, 3], dtype=torch.int64, device=v.device)
    dv = torch.tensor([[2., -1.], [-2., 3.], [0.5, -0.8]], dtype=v.dtype, device=v.device)
    dl = torch.tensor([0.4, -0.2, 0.7], dtype=v.dtype, device=v.device)
    expected = torch.tensor([[-1.225, 2.45], [0., 0.], [0.5, -0.8]], dtype=v.dtype, device=v.device)
    actual = native(v, l, o, dv, dl)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    shifted = native(v, l, o, torch.zeros_like(dv), torch.ones_like(dl))
    torch.testing.assert_close(shifted, torch.zeros_like(shifted), rtol=0, atol=1e-14)
    wrong = torch.tensor([0.25, 0.75], dtype=v.dtype, device=v.device)[:, None] * (dv[:2] + v[:2] * dl[:2, None])
    assert float((wrong.sum(0) - expected[0]).abs().max()) > 0.1
    torch.testing.assert_close(native(v, l, o, 0.3 * dv, 0.3 * dl), 0.3 * actual, rtol=1e-13, atol=1e-13)


def test_centered_difference_of_native_primal():
    v, l, o, dv, dl = fixture()
    h = 1e-5
    centered = (ragged_pool(v + h * dv, l + h * dl, o) - ragged_pool(v - h * dv, l - h * dl, o)) / (2 * h)
    torch.testing.assert_close(native(v, l, o, dv, dl), centered, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize('field', ('values', 'logits', 'offsets'))
def test_saved_primal_mutation_is_rejected_on_ordinary_reverse(field):
    v, l, o, _, _ = fixture()
    v.requires_grad_(); l.requires_grad_()
    output = pool(v, l, o)
    with torch.no_grad():
        if field == 'values': v.add_(0.01)
        elif field == 'logits': l.add_(0.01)
        else: o[1] = 1
    with pytest.raises(RuntimeError, match='version|inplace'):
        output.sum().backward()


@pytest.mark.parametrize('field', ('primal', 'direction'))
def test_reverse_of_tangent_and_nested_jvp_refuse_instead_of_detaching(field):
    v, l, o, dv, dl = fixture()
    varied = v if field == 'primal' else dv
    varied.requires_grad_()
    tangent = torch.func.jvp(lambda x, s: pool(x, s, o), (v, l), (dv, dl))[1]
    assert tangent.requires_grad, 'The first-order guard must retain dependency rather than silently detach'
    with pytest.raises(RuntimeError, match='first-order JVP|differentiating tangents'):
        torch.autograd.grad(tangent.sum(), varied)
    def inner(x):
        return torch.func.jvp(lambda a, b: pool(a, b, o), (x, l) if field == 'primal' else (v, l),
                              (dv, dl) if field == 'primal' else (x, dl))[1]
    with pytest.raises(RuntimeError, match='nested JVP|forward AD'):
        torch.func.jvp(inner, (varied.detach(),), (torch.ones_like(varied),))


@pytest.mark.parametrize('n,width,boundaries', ((4096, 128, (0, 4096)), (5, 128, (0, 1, 1, 5)), (0, 128, (0,) * 129)))
def test_maximum_width_item_group_envelopes_and_finite_endpoints(n, width, boundaries):
    v = ((torch.arange(n * width, dtype=torch.float64, device='cuda:0') % 7) * 300 - 900).reshape(n, width)
    l = (torch.arange(n, dtype=v.dtype, device=v.device) % 2) * 2000 - 1000
    dv = ((torch.arange(n * width, dtype=v.dtype, device=v.device) % 5) * 400 - 800).reshape(n, width)
    dl = (torch.arange(n, dtype=v.dtype, device=v.device) % 3) * 1000 - 1000
    o = torch.tensor(boundaries, dtype=torch.int64, device=v.device)
    expected = torch.func.jvp(lambda x, s: primitive(x, s, boundaries), (v, l), (dv, dl))[1]
    actual = native(v, l, o, dv, dl)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-9)
    assert bool(torch.isfinite(actual).all())


@pytest.mark.parametrize('case', ('nan', 'inf', 'bound', 'dtype', 'shape', 'cpu', 'primal_stride', 'bad_offsets'))
def test_private_native_admission_refusal_then_valid_recovery(case):
    v, l, o, dv, dl = fixture()
    if case == 'nan': dv[0, 0] = float('nan')
    elif case == 'inf': dl[0] = float('inf')
    elif case == 'bound': dl[0] = 1001
    elif case == 'dtype': dv = dv.float()
    elif case == 'shape': dl = dl[:-1]
    elif case == 'cpu': dl = dl.cpu()
    elif case == 'primal_stride': v = v.t().contiguous().t()
    elif case == 'bad_offsets': o[2] = 1000000
    before = stamp((v, l, o, dv, dl))
    with pytest.raises(REJECT): native(v, l, o, dv, dl)
    unchanged((v, l, o, dv, dl), before)
    v, l, o, dv, dl = fixture()
    assert bool(torch.isfinite(native(v, l, o, dv, dl)).all())


def test_registration_is_separate_from_math():
    v, l, o, dv, dl = fixture()
    torch.library.opcheck(torch.ops.ragged_pool_cuda._jvp.default, (v, l, o, dv, dl))


def test_current_nondefault_stream_and_first_order_primal_reverse():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        v, l, o, dv, dl = fixture()
        v.requires_grad_(); l.requires_grad_()
        v.grad, l.grad = torch.full_like(v, 2), torch.full_like(l, -3)
        before = stamp((v, l, o, dv, dl, v.grad, l.grad))
        output = pool(v, l, o)
        gradients = torch.autograd.grad(output.square().sum(), (v, l))
        expected = torch.autograd.grad(ragged_pool(v, l, o).square().sum(), (v, l))
        for a, b in zip(gradients, expected): torch.testing.assert_close(a, b, **tol(v.dtype))
        actual = native(v, l, o, dv, dl)
        expected_t = torch.func.jvp(lambda x, s: primitive(x, s, BOUNDARIES), (v, l), (dv, dl))[1]
        torch.testing.assert_close(actual, expected_t, **tol(v.dtype))
        unchanged((v, l, o, dv, dl, v.grad, l.grad), before)
    stream.synchronize()


def test_jvp_accepts_existing_nonzero_versions_and_keeps_gradient_sentinels():
    v, l, o, dv, dl = fixture()
    for value in (v, l, o, dv, dl): value.add_(0)
    v.requires_grad_(); l.requires_grad_()
    v.grad, l.grad = torch.full_like(v, 0.7), torch.full_like(l, -0.2)
    original = (v.grad, l.grad)
    before = stamp((v, l, o, dv, dl, *original))
    actual = torch.func.jvp(lambda x, s: pool(x, s, o), (v, l), (dv, dl))[1]
    expected = torch.func.jvp(lambda x, s: primitive(x, s, BOUNDARIES), (v, l), (dv, dl))[1]
    torch.testing.assert_close(actual, expected, **tol(v.dtype))
    assert v.grad is original[0] and l.grad is original[1]
    unchanged((v, l, o, dv, dl, *original), before)


@pytest.mark.parametrize('case', ('map', 'compile', 'inference', 'second_reverse'))
def test_unsupported_routes_refuse(case):
    v, l, o, dv, dl = fixture()
    with pytest.raises(REJECT):
        if case == 'map': torch.vmap(lambda x: pool(x, l, o))(v.unsqueeze(0).repeat(2, 1, 1))
        elif case == 'compile': torch.compile(lambda x: pool(x, l, o), backend='aot_eager', fullgraph=True)(v)
        elif case == 'inference':
            with torch.inference_mode(): pool(v.clone(), l.clone(), o.clone())
        else:
            v.requires_grad_(); l.requires_grad_()
            first = torch.autograd.grad(pool(v, l, o).square().sum(), v, create_graph=True)[0]
            first.sum().backward()
