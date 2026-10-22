"""Check native second reverse math separately from registration and admission."""
import pytest
import torch

from ragged_pool_cuda import ragged_pool
from ragged_pool_cuda.second_order import ragged_pool_second_order as pool


DTYPES = (torch.float32, torch.float64)
BOUNDARIES = (0, 2, 2, 5, 6)


@pytest.fixture(scope='module', autouse=True)
def environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    assert str(torch.__version__) == '2.9.1+cu128' and torch.version.cuda == '12.8'
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)


def fixture(dtype=torch.float64, boundaries=BOUNDARIES, width=3):
    n, b = boundaries[-1], len(boundaries) - 1
    v = (torch.arange(n * width, dtype=dtype, device='cuda:0').reshape(n, width) / max(n * width, 1) - 0.4)
    l = torch.linspace(-0.7, 0.8, n, dtype=dtype, device='cuda:0')
    o = torch.tensor(boundaries, dtype=torch.int64, device=v.device)
    p = torch.linspace(0.3, -0.2, v.numel(), dtype=dtype, device=v.device).reshape_as(v)
    q = torch.linspace(-0.4, 0.7, n, dtype=dtype, device=v.device)
    g = torch.linspace(-0.6, 0.8, b * width, dtype=dtype, device=v.device).reshape(b, width)
    return p, q, g, v, l, o


def primitive(v, l, boundaries):
    rows = []
    for a, b in zip(boundaries, boundaries[1:]):
        x, s = v[a:b].double(), l[a:b].double()
        rows.append(((s.softmax(0)[:, None] * x).sum(0) if b > a else x.sum(0) * 0 + s.sum() * 0).to(v.dtype))
    return torch.stack(rows) if rows else v[:0] * 0 + l.sum() * 0


def tol(dtype):
    return dict(rtol=3e-4, atol=3e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=3e-12)


def native(*args): return torch.ops.ragged_pool_cuda._double_backward.default(*args)


def oracle(p, q, g, v, l, o, boundaries=BOUNDARIES):
    x, s, upstream = (t.detach().clone().requires_grad_() for t in (v, l, g))
    y = primitive(x, s, boundaries)
    first = torch.autograd.grad(y, (x, s), upstream, create_graph=True)
    scalar = (first[0] * p).sum() + (first[1] * q).sum()
    return torch.autograd.grad(scalar, (upstream, x, s), materialize_grads=True)


def stamp(items):
    return [(id(x), x._version, x.untyped_storage()._cdata, x.stride(), x.storage_offset(), x.is_neg(),
             x.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8).clone()) for x in items]


def unchanged(items, before):
    for got, expected in zip(stamp(items), before):
        assert got[:-1] == expected[:-1] and torch.equal(got[-1], expected[-1])


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('boundaries', (BOUNDARIES, (0,), (0, 0, 0), (0, 1, 2, 3), (0, 6)))
def test_three_native_outputs_against_independent_autograd_and_fresh_storage(dtype, boundaries):
    args = fixture(dtype, boundaries)
    before = stamp(args)
    actual, expected = native(*args), oracle(*args, boundaries=boundaries)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, **tol(dtype))
    source = {x.untyped_storage()._cdata for x in args}
    assert len({x.untyped_storage()._cdata for x in actual}) == 3
    assert not source & {x.untyped_storage()._cdata for x in actual}
    unchanged(args, before)


@pytest.mark.parametrize('layout', ('transpose', 'step', 'expand', 'negative', 'negative_step'))
@pytest.mark.parametrize('dtype', DTYPES)
def test_first_and_second_upstream_layout_normalization(dtype, layout):
    p, q, g, v, l, o = fixture(dtype)
    if layout == 'transpose': p, g = p.t().contiguous().t(), g.t().contiguous().t()
    elif layout == 'step': p, q, g = p.repeat_interleave(2, 1)[:, ::2], q.repeat_interleave(2)[::2], g.repeat_interleave(2, 1)[:, ::2]
    elif layout == 'expand': p, q, g = p[:, :1].expand_as(p), q[:1].expand_as(q), g[:, :1].expand_as(g)
    elif layout == 'negative': p, q, g = (torch._neg_view(x) for x in (p, q, g))
    else: p, q, g = torch._neg_view(p.repeat_interleave(2, 1)[:, ::2]), torch._neg_view(q.repeat_interleave(2)[::2]), torch._neg_view(g.repeat_interleave(2, 1)[:, ::2])
    args = (p, q, g, v, l, o)
    before = stamp(args)
    for a, b in zip(native(*args), oracle(*args)): torch.testing.assert_close(a, b, **tol(dtype))
    unchanged(args, before)


def test_singleton_nonlinear_loss_needs_output_cotangent_derivative():
    p, q, g, v, l, o = fixture(boundaries=(0, 1), width=1)
    p.fill_(3); q.fill_(0.4); g.fill_(0.7)
    dg, dv, dl = native(p, q, g, v, l, o)
    torch.testing.assert_close(dg, p, rtol=0, atol=0)
    assert torch.equal(dv, torch.zeros_like(v)) and torch.equal(dl, torch.zeros_like(l))
    v.requires_grad_(); l.requires_grad_()
    y = pool(v, l, o)
    first = torch.autograd.grad(0.5 * y.square().sum(), v, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), v)[0]
    torch.testing.assert_close(second, torch.ones_like(v), rtol=0, atol=0)
    assert float((second - dv).abs().max()) == 1  # Omitting dg falsely leaves zero curvature.


@pytest.mark.parametrize('dtype', DTYPES)
def test_nonlinear_loss_second_reverse_and_existing_grad_sentinels(dtype):
    _, _, _, v, l, o = fixture(dtype)
    v.requires_grad_(); l.requires_grad_()
    v.grad, l.grad = torch.full_like(v, 2), torch.full_like(l, -3)
    gradients = (v.grad, l.grad)
    before = stamp((v, l, o, *gradients))
    results = []
    for reference in (False, True):
        x, s = v.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
        y = primitive(x, s, BOUNDARIES) if reference else pool(x, s, o)
        loss = (y.sin() + 0.23 * y.square()).sum()
        first = torch.autograd.grad(loss, (x, s), create_graph=True)
        second = torch.autograd.grad(sum(t.square().sum() for t in first), (x, s))
        results.append((y, *first, *second))
    for a, b in zip(*results): torch.testing.assert_close(a, b, **tol(dtype))
    first = torch.autograd.grad(pool(v, l, o).square().sum(), (v, l), create_graph=True)
    torch.autograd.grad(sum(t.square().sum() for t in first), (v, l))
    assert v.grad is gradients[0] and l.grad is gradients[1]
    unchanged((v, l, o, *gradients), before)


def test_finite_difference_of_first_vjp_and_gradgradcheck():
    p, q, g, v, l, o = fixture(boundaries=(0, 2, 2), width=2)
    def first(x, s, upstream):
        x, s = x.detach().requires_grad_(), s.detach().requires_grad_()
        return torch.autograd.grad(ragged_pool(x, s, o), (x, s), upstream)
    h = 1e-5
    actual = native(p, q, g, v, l, o)
    # Check each coordinate of dS/dg,dS/dv,dS/dl independently.
    for argument, result in zip((g, v, l), actual):
        expected = torch.zeros_like(argument)
        for index in range(argument.numel()):
            delta = torch.zeros_like(argument); delta.reshape(-1)[index] = h
            plus = first(v + delta if argument is v else v, l + delta if argument is l else l, g + delta if argument is g else g)
            minus = first(v - delta if argument is v else v, l - delta if argument is l else l, g - delta if argument is g else g)
            expected.reshape(-1)[index] = ((plus[0] * p).sum() + (plus[1] * q).sum() - (minus[0] * p).sum() - (minus[1] * q).sum()) / (2*h)
        torch.testing.assert_close(result, expected, rtol=2e-8, atol=2e-10)
    v.requires_grad_(); l.requires_grad_()
    assert torch.autograd.gradcheck(lambda x, s: pool(x, s, o), (v, l), eps=1e-5, atol=2e-7, rtol=2e-5)
    assert torch.autograd.gradgradcheck(lambda x, s: pool(x, s, o), (v, l), grad_outputs=(g.requires_grad_(),), eps=1e-5, atol=3e-7, rtol=3e-5)


@pytest.mark.parametrize('selected', ('values', 'logits'))
def test_undefined_second_upstream_is_zero(selected):
    p, q, g, v, l, o = fixture()
    v.requires_grad_(); l.requires_grad_(); g.requires_grad_()
    first = torch.autograd.grad(pool(v, l, o), (v, l), g, create_graph=True)
    target = (first[0] * p).sum() if selected == 'values' else (first[1] * q).sum()
    actual = torch.autograd.grad(target, (g, v, l))
    expected = native(p if selected == 'values' else torch.zeros_like(p), q if selected == 'logits' else torch.zeros_like(q), g, v, l, o)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, **tol(v.dtype))


@pytest.mark.parametrize('stage,field', [('first', key) for key in ('values','logits','offsets')] + [('second', key) for key in ('values','logits','offsets','upstream')])
def test_original_saved_mutation_is_refused_at_both_stages(stage, field):
    p, q, g, v, l, o = fixture()
    v.requires_grad_(); l.requires_grad_(); g.requires_grad_()
    y = pool(v, l, o)
    first = torch.autograd.grad(y, (v, l), g, create_graph=True) if stage == 'second' else None
    with torch.no_grad():
        if field == 'values': v.add_(0.01)
        elif field == 'logits': l.add_(0.01)
        elif field == 'upstream': g.add_(0.01)
        else: o[1] = 1  # Remains a valid partition.
    with pytest.raises(RuntimeError, match='version|inplace'):
        if stage == 'first': torch.autograd.grad(y.sum(), (v, l))
        else: torch.autograd.grad((first[0]*p).sum() + (first[1]*q).sum(), (g, v, l))


@pytest.mark.parametrize('case', ('p_nan', 'q_inf', 'p_bound', 'g_bound', 'p_dtype', 'q_shape', 'g_shape', 'q_cpu', 'offsets', 'primal_stride', 'negative_primal'))
def test_direct_native_refusal_preserves_inputs_and_valid_call_recovers(case):
    p, q, g, v, l, o = fixture()
    if case == 'p_nan': p[0, 0] = float('nan')
    elif case == 'q_inf': q[0] = float('inf')
    elif case == 'p_bound': p[0, 0] = 1001
    elif case == 'g_bound': g[0, 0] = -1001
    elif case == 'p_dtype': p = p.float()
    elif case == 'q_shape': q = q[:-1]
    elif case == 'g_shape': g = g[:-1]
    elif case == 'q_cpu': q = q.cpu()
    elif case == 'offsets': o[2] = 1000000
    elif case == 'primal_stride': v = v.t().contiguous().t()
    else: v = torch._neg_view(v)
    args = (p, q, g, v, l, o)
    before = stamp(args)
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)): native(*args)
    unchanged(args, before)
    assert all(bool(torch.isfinite(x).all()) for x in native(*fixture()))


@pytest.mark.parametrize('boundaries,width', (((0, 4096), 128), ((0,) * 129, 128), ((0, 1, 1, 5), 128)))
def test_maximum_envelopes_and_finite_upstream_endpoints(boundaries, width):
    p, q, g, v, l, o = fixture(boundaries=boundaries, width=width)
    p.fill_(1000); q = torch.linspace(-1000,1000,q.numel(),dtype=q.dtype,device=q.device); g.fill_(1000)
    for a, b in zip(native(p,q,g,v,l,o), oracle(p,q,g,v,l,o,boundaries)):
        torch.testing.assert_close(a, b, rtol=3e-10, atol=3e-7)


def test_nondefault_current_stream_result_is_consumed():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        args = fixture()
        for a, b in zip(native(*args), oracle(*args)): torch.testing.assert_close(a, b, **tol(args[3].dtype))
    stream.synchronize()


def test_registration_is_separate_from_mathematics():
    torch.library.opcheck(torch.ops.ragged_pool_cuda._double_backward.default, fixture())


def test_third_order_retains_dependency_and_explicitly_refuses():
    p, q, g, v, l, o = fixture()
    v.requires_grad_(); l.requires_grad_(); g.requires_grad_()
    first = torch.autograd.grad(pool(v, l, o).square().sum(), (v, l), create_graph=True)
    second = torch.autograd.grad(sum(x.square().sum() for x in first), (v, l), create_graph=True)
    assert all(x.requires_grad for x in second)
    with pytest.raises(RuntimeError, match='third|second.order'):
        torch.autograd.grad(sum(x.sum() for x in second), (v, l))


@pytest.mark.parametrize('route', ('grad', 'jvp', 'map', 'compile', 'inference'))
def test_out_of_scope_transforms_are_refused(route):
    p,q,g,v,l,o = fixture()
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
        if route == 'grad': torch.func.grad(lambda x: pool(x,l,o).sum())(v)
        elif route == 'jvp': torch.func.jvp(lambda x: pool(x,l,o), (v,), (p,))
        elif route == 'map': torch.vmap(lambda x: pool(x,l,o))(v[None].repeat(2,1,1))
        elif route == 'compile': torch.compile(lambda x: pool(x,l,o), backend='aot_eager', fullgraph=True)(v)
        else:
            with torch.inference_mode(): pool(v.clone(),l.clone(),o.clone())
