"""Independent directional second derivatives for the packed eager bridge."""
from types import SimpleNamespace
import pytest
import torch
from ragged_pool import ragged_pool, ragged_pool_forward_ad
from ragged_pool.forward_ad import _PackedBackward

DTYPES=(torch.float32,torch.float64)
PARTITIONS={"mixed":(0,2,2,3,6),"singleton":(0,1,2),"empty":(0,0,0),"zero_batch":(0,)}

def fixture(dtype,kind="mixed"):
    boundaries=PARTITIONS[kind]
    generator=torch.Generator().manual_seed(1603)
    values=torch.randn((boundaries[-1],3),generator=generator,dtype=dtype)*.2
    logits=torch.randn((boundaries[-1],),generator=generator,dtype=dtype)*.3
    dv=torch.randn(values.shape,generator=generator,dtype=dtype)*.1
    dl=torch.randn(logits.shape,generator=generator,dtype=dtype)*.1
    return values,logits,torch.tensor(boundaries),dv,dl

def primitive(boundaries):
    def pool(values,logits,offsets):
        rows=[]
        for start,stop in zip(boundaries,boundaries[1:]):
            if start==stop:
                rows.append(values.sum(0)*0+logits.sum()*0)
            else:
                rows.append((values[start:stop]*torch.softmax(logits[start:stop],0)[:,None]).sum(0))
        return torch.stack(rows) if rows else (values.sum(0)[None,:]+logits.sum()*0)[:0]
    return pool

def objective(pool,v,l,o,nonlinear=True):
    y=pool(v,l,o)
    return (y.square()+.1*y.sin()).sum() if nonlinear else (y*torch.arange(1,4,dtype=v.dtype)).sum()

def hvp(pool,v,l,o,dv,dl,nonlinear=True):
    gradient=torch.func.grad(lambda a,b:objective(pool,a,b,o,nonlinear),argnums=(0,1))
    return torch.func.jvp(gradient,(v,l),(dv,dl))

def close(actual,expected,dtype,fd=False):
    tolerance=({"rtol":3e-3,"atol":5e-5} if dtype==torch.float32 else {"rtol":2e-7,"atol":3e-9}) if fd else ({"rtol":4e-5,"atol":4e-6} if dtype==torch.float32 else {"rtol":3e-11,"atol":3e-12})
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,**tolerance)

def ror(v,l,o,dv,dl,nonlinear=True):
    a=v.detach().requires_grad_();b=l.detach().requires_grad_()
    g=torch.autograd.grad(objective(ragged_pool,a,b,o,nonlinear),(a,b),create_graph=True)
    dot=sum((x*d).sum() for x,d in zip(g,(dv,dl)))
    if not dot.requires_grad:
        return g,(torch.zeros_like(v),torch.zeros_like(l))
    h=torch.autograd.grad(dot,(a,b),allow_unused=True)
    return g,tuple(torch.zeros_like(x) if z is None else z for z,x in zip(h,(v,l)))

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("kind",PARTITIONS)
@pytest.mark.parametrize("direction",["joint","values","logits","zero"])
def test_hvp_matches_independent_math_and_native_reverse(dtype,kind,direction):
    v,l,o,dv,dl=fixture(dtype,kind)
    if direction in ("logits","zero"):dv=torch.zeros_like(v)
    if direction in ("values","zero"):dl=torch.zeros_like(l)
    tensors=(v,l,o,dv,dl)
    before=[(x.clone(),x._version) for x in tensors]
    g,h=hvp(ragged_pool_forward_ad,v,l,o,dv,dl)
    rg,rh=hvp(primitive(PARTITIONS[kind]),v,l,o,dv,dl)
    ag,ah=ror(v,l,o,dv,dl)
    close(g,rg,dtype);close(g,ag,dtype);close(h,rh,dtype);close(h,ah,dtype)
    for x,(old,version) in zip(tensors,before):
        torch.testing.assert_close(x,old,rtol=0,atol=0)
        assert x._version==version
    if kind in ("empty","zero_batch") or direction=="zero":
        assert all(torch.count_nonzero(x)==0 for x in h)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("nonlinear",[False,True])
def test_directional_finite_difference_uses_independent_gradients(dtype,nonlinear):
    v,l,o,dv,dl=fixture(dtype)
    _,h=hvp(ragged_pool_forward_ad,v,l,o,dv,dl,nonlinear)
    grad=torch.func.grad(lambda a,b:objective(primitive(PARTITIONS["mixed"]),a,b,o,nonlinear),argnums=(0,1))
    step=1e-2 if dtype==torch.float32 else 1e-5
    plus=grad(v+step*dv,l+step*dl);minus=grad(v-step*dv,l-step*dl)
    close(h,tuple((a-b)/(2*step) for a,b in zip(plus,minus)),dtype,fd=True)

@pytest.mark.parametrize("dtype",DTYPES)
def test_singleton_squared_loss_requires_cotangent_direction(dtype):
    v,l,o,dv,dl=fixture(dtype,"singleton")
    gradient=torch.func.grad(lambda a,b:ragged_pool_forward_ad(a,b,o).square().sum(),argnums=(0,1))
    g,h=torch.func.jvp(gradient,(v,l),(dv,dl))
    close(g,(2*v,torch.zeros_like(l)),dtype)
    close(h,(2*dv,torch.zeros_like(l)),dtype)
    # The pool is linear in singleton values. Its fixed-cotangent Hessian is
    # zero, so omitting B(dg,x) would return zero instead of 2*dv.
    _,bad_v,bad_l=torch.ops.ragged_pool._double_backward.default(dv,dl,2*v,v,l,o)
    assert torch.count_nonzero(bad_v)==torch.count_nonzero(bad_l)==0
    assert (h[0]-bad_v).abs().max()>1e-2

@pytest.mark.parametrize("dtype",DTYPES)
def test_two_item_squared_loss_has_hand_derived_cross_block_hvp(dtype):
    v=torch.tensor([[2.],[0.]],dtype=dtype)
    l=torch.zeros(2,dtype=dtype);o=torch.tensor([0,2])
    dv=torch.tensor([[.3],[-.1]],dtype=dtype)
    dl=torch.tensor([.2,-.4],dtype=dtype)
    gradient=torch.func.grad(lambda a,b:ragged_pool_forward_ad(a,b,o).square().sum(),argnums=(0,1))
    g,h=torch.func.jvp(gradient,(v,l),(dv,dl))
    total=dv.sum();difference=dl[0]-dl[1]
    expected_v=torch.stack((.5*total+difference,.5*total)).reshape(2,1)
    first=dv[0,0]+.5*difference
    expected_l=torch.stack((first,-first))
    close(g,(torch.ones_like(v),torch.tensor([1.,-1.],dtype=dtype)),dtype)
    close(h,(expected_v,expected_l),dtype)

@pytest.mark.parametrize("dtype",DTYPES)
def test_linear_loss_requires_pool_curvature_with_zero_cotangent_direction(dtype):
    v,l,o,dv,dl=fixture(dtype)
    _,h=hvp(ragged_pool_forward_ad,v,l,o,dv,dl,nonlinear=False)
    _,reference=hvp(primitive(PARTITIONS["mixed"]),v,l,o,dv,dl,nonlinear=False)
    close(h,reference,dtype)
    assert max(x.abs().max().item() for x in h)>1e-3
    # B(dg,x) alone is zero when this fixed linear objective has dg=0.
    wrong=torch.ops.ragged_pool._backward.default(torch.zeros((4,3),dtype=dtype),v,l,o)
    assert all(torch.count_nonzero(x)==0 for x in wrong)
    assert max((x-y).abs().max().item() for x,y in zip(h,wrong))>1e-3

@pytest.mark.parametrize("dtype",DTYPES)
def test_hessian_bilinear_symmetry_direction_linearity_and_logit_shift(dtype):
    v,l,o,dv,dl=fixture(dtype)
    ev=dv.flip(0)*.7+.03;el=dl.flip(0)*-.4+.02
    _,h=hvp(ragged_pool_forward_ad,v,l,o,dv,dl)
    _,k=hvp(ragged_pool_forward_ad,v,l,o,ev,el)
    _,combined=hvp(ragged_pool_forward_ad,v,l,o,2*dv-.5*ev,2*dl-.5*el)
    close(combined,tuple(2*a-.5*b for a,b in zip(h,k)),dtype)
    a=sum((u*z).sum() for u,z in zip((ev,el),h))
    b=sum((u*z).sum() for u,z in zip((dv,dl),k))
    torch.testing.assert_close(a,b,rtol=4e-5 if dtype==torch.float32 else 3e-11,atol=3e-7 if dtype==torch.float32 else 3e-12)
    _,shifted=hvp(ragged_pool_forward_ad,v,l+2,o,dv,dl)
    close(h,shifted,dtype)

def view_like(shape,dtype,kind):
    count=1
    for n in shape:count*=n
    if kind=="expanded":return torch.tensor(.15,dtype=dtype).expand(shape)
    if kind=="stepped":return torch.linspace(-.2,.3,2*count,dtype=dtype).reshape(*shape[:-1],2*shape[-1])[...,::2]
    if kind=="negative":return torch._neg_view(torch.full(shape,.2,dtype=dtype))
    if kind=="transpose" and len(shape)==2:
        return torch.linspace(-.2,.3,count,dtype=dtype).reshape(shape[1],shape[0]).t()
    return torch.linspace(-.2,.3,count,dtype=dtype).reshape(shape)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("kind",["contiguous","transpose","stepped","expanded","negative"])
def test_backward_jvp_accepts_upstream_and_direction_views(dtype,kind):
    v,l,o,_,_=fixture(dtype)
    g=view_like((4,3),dtype,kind)*5000
    dg=view_like((4,3),dtype,kind)
    dv=view_like(v.shape,dtype,kind);dl=view_like(l.shape,dtype,kind)
    # Keep both actual upstreams as the intended views, including a scale
    # exceeding the public primal-direction magnitude bound.
    if kind=="expanded":g=torch.tensor(1500.,dtype=dtype).expand(4,3)
    elif kind=="transpose":g=torch.full((3,4),1500.,dtype=dtype).t()
    elif kind=="stepped":g=torch.full((4,6),1500.,dtype=dtype)[:,::2]
    elif kind=="negative":g=torch._neg_view(torch.full((4,3),1500.,dtype=dtype))
    dg=dg*10000
    if kind=="expanded":dg=torch.tensor(1600.,dtype=dtype).expand(4,3)
    elif kind=="transpose":dg=torch.full((3,4),1600.,dtype=dtype).t()
    elif kind=="stepped":dg=torch.full((4,6),1600.,dtype=dtype)[:,::2]
    elif kind=="negative":dg=torch._neg_view(torch.full((4,3),1600.,dtype=dtype))
    tensors=(g,v,l,o,dg,dv,dl)
    snapshots=[(x.clone(),x._version) for x in tensors]
    primal,tangent=torch.func.jvp(lambda a,b,c:_PackedBackward.apply(a,b,c,o),(g,v,l),(dg,dv,dl))
    ref_pool=primitive(PARTITIONS["mixed"])
    def backward(a,b,c):
        return torch.func.grad(lambda x,y:(ref_pool(x,y,o)*a).sum(),argnums=(0,1))(b,c)
    rg,rh=torch.func.jvp(backward,(g,v,l),(dg,dv,dl))
    close(primal,rg,dtype);close(tangent,rh,dtype)
    for x,(before,version) in zip(tensors,snapshots):
        torch.testing.assert_close(x,before,rtol=0,atol=0)
        assert x._version==version

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("missing",["g","values","logits","all"])
def test_backward_jvp_missing_directions_are_zero(dtype,missing):
    v,l,o,dv,dl=fixture(dtype);g=torch.full((4,3),.4,dtype=dtype);dg=torch.full_like(g,.2)
    directions=[dg,dv,dl,None]
    for index,name in enumerate(("g","values","logits")):
        if missing in (name,"all"):directions[index]=None
    actual=_PackedBackward.jvp(SimpleNamespace(saved_tensors=(g,v,l,o)),*directions)
    normalized=tuple(torch.zeros_like(x) if d is None else d for x,d in zip((g,v,l),directions[:3]))
    _,expected=torch.func.jvp(lambda a,b,c:_PackedBackward.apply(a,b,c,o),(g,v,l),normalized)
    close(actual,expected,dtype)

@pytest.mark.parametrize("dtype",DTYPES)
def test_hvp_delegates_both_native_contributions(monkeypatch,dtype):
    v,l,o,dv,dl=fixture(dtype);g=torch.full((4,3),.4,dtype=dtype);dg=torch.full_like(g,.2)
    calls=[]
    for name in ("_backward","_double_backward"):
        packet=getattr(torch.ops.ragged_pool,name);original=packet.default
        def wrapped(*args,_name=name,_original=original):
            calls.append(_name);return _original(*args)
        monkeypatch.setattr(packet,"default",wrapped)
    torch.func.jvp(lambda a,b,c:_PackedBackward.apply(a,b,c,o),(g,v,l),(dg,dv,dl))
    assert calls==["_backward","_backward","_double_backward"]

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("failure",["nan","infinity","magnitude"])
def test_hvp_retains_primal_direction_admission(dtype,failure):
    v,l,o,dv,dl=fixture(dtype)
    dv[0,0]={"nan":float("nan"),"infinity":float("inf"),"magnitude":1001.}[failure]
    with pytest.raises(RuntimeError,match="direction must be finite"):
        hvp(ragged_pool_forward_ad,v,l,o,dv,dl)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("failure",["noncontiguous","nonfinite"])
def test_hvp_keeps_original_primal_admission(dtype,failure):
    v,l,o,dv,dl=fixture(dtype)
    if failure=="noncontiguous":v=torch.zeros((6,6),dtype=dtype)[:,::2]
    else:v[0,0]=float("nan")
    with pytest.raises(RuntimeError,match="(contiguous|finite)"):
        hvp(ragged_pool_forward_ad,v,l,o,dv,dl)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("target",["primal","direction"])
def test_hvp_cannot_be_further_reverse_differentiated(dtype,target):
    v,l,o,dv,dl=fixture(dtype)
    if target=="primal":v.requires_grad_()
    else:dv.requires_grad_()
    _,h=hvp(ragged_pool_forward_ad,v,l,o,dv,dl)
    with pytest.raises(RuntimeError,match="(differentiating Hessian-vector products|first-order tangents|setup_context)"):
        torch.autograd.grad(sum(x.sum() for x in h),v if target=="primal" else dv)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("target",["primal","direction"])
def test_hvp_cannot_be_further_forward_differentiated(dtype,target):
    v,l,o,dv,dl=fixture(dtype)
    def more(x):
        return hvp(ragged_pool_forward_ad,x if target=="primal" else v,l,o,x if target=="direction" else dv,dl)[1]
    with pytest.raises((RuntimeError,NotImplementedError),match="(nested forward AD|differentiating Hessian-vector products|not supported)"):
        torch.func.jvp(more,(v if target=="primal" else dv,),(dv,))

@pytest.mark.parametrize("dtype",DTYPES)
def test_scaled_nonlinear_loss_keeps_large_internal_cotangent_direction(dtype):
    v,l,o,dv,dl=fixture(dtype)
    def gradient(pool):
        return torch.func.grad(lambda a,b:2000*objective(pool,a,b,o),argnums=(0,1))
    actual=torch.func.jvp(gradient(ragged_pool_forward_ad),(v,l),(dv,dl))
    expected=torch.func.jvp(gradient(primitive(PARTITIONS["mixed"])),(v,l),(dv,dl))
    close(actual[0],expected[0],dtype);close(actual[1],expected[1],dtype)

@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("target",["cotangent","cotangent_direction"])
def test_outer_cotangent_only_forward_direction_rejects_before_native(monkeypatch,dtype,target):
    v,l,o,dv,dl=fixture(dtype);g=torch.full((4,3),.4,dtype=dtype);dg=torch.full_like(g,.2)
    calls=[]
    def forbidden(*args):
        calls.append(True)
        raise AssertionError("Excluded outer direction reached opaque native math")
    for name in ("_backward","_double_backward"):
        monkeypatch.setattr(getattr(torch.ops.ragged_pool,name),"default",forbidden)
    def outer(x):
        context=SimpleNamespace(saved_tensors=(x if target=="cotangent" else g,v,l,o))
        return _PackedBackward.jvp(context,x if target=="cotangent_direction" else dg,dv,dl,None)
    with pytest.raises(RuntimeError,match="differentiating Hessian-vector products"):
        torch.func.jvp(outer,(g if target=="cotangent" else dg,),(torch.ones_like(g),))
    assert calls==[]

@pytest.mark.parametrize("dtype",DTYPES)
def test_mapped_hvp_rejects_the_excluded_composition(dtype):
    v,l,o,dv,dl=fixture(dtype)
    tensors=tuple(torch.stack((x,x*.7)) for x in (v,l,dv,dl))
    def mapped(a,b,c,d):
        return hvp(ragged_pool_forward_ad,a,b,o,c,d)[1]
    with pytest.raises((RuntimeError,NotImplementedError),match="(vmap|mapped|data-dependent|[.]item|not supported)"):
        torch.vmap(mapped)(*tensors)
