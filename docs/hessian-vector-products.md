# Packed eager Hessian-vector products

The existing `ragged_pool_forward_ad` bridge supports one eager `torch.func.jvp(torch.func.grad(scalar_loss))` composition on the pinned CPU environment. A gradient describes how a scalar loss changes with each input or parameter. A Hessian-vector product (HVP) describes how that gradient changes along one supplied direction. This route evaluates the product without constructing a full Hessian matrix.

The packed result, original primal admission, first-order JVP, ordinary second reverse derivative and single-map routes retain their own contracts. The new composition uses the existing native first and second backward operations. It adds no public pooling API, C++ kernel, dispatcher operation, fake kernel or dependency.

## Differentiate both parts of backward

Let x combine the pool's values and logits, y=P(x), J(x) be its Jacobian and g the output cotangent supplied by the surrounding loss. Native backward returns B(g,x)=J(x)^T*g. Along primal direction dx and cotangent direction dg, its derivative is:

```text
dB = J(x)^T*dg + Hessian_x <g,P(x)>*dx
```

The cotangent term is native `_backward(dg, values, logits, offsets)`. The other term is the value/logit pair returned by `_double_backward(dx_values, dx_logits, g, values, logits, offsets)`. Its first result, the separate pool JVP, is unused in this call. Adding the two pairs gives the tangent of backward.

For a nonlinear scalar loss, g changes as y changes. Dropping dg computes only the curvature of a frozen-cotangent objective and can return an incorrect zero. The symmetry of each smooth scalar pooled component's Hessian lets the existing second-backward VJP supply the needed directional derivative. These are established differentiation rules, implemented with the package's existing floating-point accumulation and casts.

The private backward `autograd.Function` saves its original inputs through both `save_for_backward` and `save_for_forward`. Its `jvp` supplies the expression above. This follows [PyTorch 2.9's custom Function transform guidance](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) and [Function.jvp contract](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.Function.jvp.html). Version-specific decisions were checked against [exact v2.9.1 Function source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/autograd/function.py).

## A singleton that catches the missing term

Use one item and an empty second group. The singleton's pooled value is exactly its input, regardless of its logit. For loss `sum(pooled**2)`, the value gradient is twice the value and its HVP is twice the value direction. The logit gradient and HVP are zero.

```python
import torch
from ragged_pool import ragged_pool_forward_ad

values = torch.tensor([[2., -1.]], dtype=torch.float64)
logits = torch.tensor([0.7], dtype=torch.float64)
offsets = torch.tensor([0, 1, 1], dtype=torch.int64)
dv = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
dl = torch.tensor([0.2], dtype=torch.float64)

def loss(v, l):
    return ragged_pool_forward_ad(v, l, offsets).square().sum()

gradient, hvp = torch.func.jvp(
    torch.func.grad(loss, argnums=(0, 1)),
    (values, logits), (dv, dl),
)
# gradient: (tensor([[4., -2.]]), tensor([0.]))
# hvp:      (tensor([[0.6, -0.8]]), tensor([0.]))
```

Here g=2*y and dg=2*dv for the singleton. Native pool curvature at fixed g is zero because its value mapping is linear. The entire nonzero HVP comes from `J^T*dg`. An omitted-cotangent control must therefore disagree. The empty group contributes nothing.

## Admission and derivative limits

Primals retain the [packed CPU contract](contract.md): contiguous float32/float64 values and logits, contiguous int64 offsets, N <=4096, B <=128, 1 <=D <=128 and finite values/logits within `[-1000,1000]`. Offsets remain fixed and have no tangent or gradient. Empty groups, all-empty inputs and a zero-group batch retain their structural zero behavior.

Directions for values and logits match their primals' shape and dtype, use CPU strided layout, and are finite within `[-1000,1000]`. Transposed, stepped, expanded zero-stride and logical negative direction views are supported. Public guards and native derivative normalization may copy those views without changing the caller's tensors. In a model-parameter HVP, it is the directions reaching pool values and logits that must meet this policy. A bound on the parameter direction alone does not prove that admission.

Output cotangents and their directions retain the native backward metadata and supported-view rules. They receive no new magnitude cap. Very large or nonfinite upstreams can overflow or propagate nonfinite results, so this is not a finite-output guarantee for arbitrary losses. Missing floating directions become matching zeros. Original saved cotangents, primals and offsets must remain unchanged until the derivative using them completes.

Public Function guards protect inputs before opaque native calls, then guard the returned HVPs. Further differentiation of an HVP is excluded, including differentiation with respect to its supplied direction. Nested forward derivatives, general nested reverse transforms, mapped HVP/JVP, full Jacobian/Hessian helpers, padded transforms and compiled/exported transforms remain outside this route. The existing [single-map contract](functional-models.md) does not admit HVP/map compositions. Ordinary first-order compiled training and saved inference export keep their separate acceptance.

## Installed training and fixed-parameter curvature

`examples/train_curvature.py` uses a deterministic encoder 3->4, tanh, scorer 4->1 without bias, packed pool and head 4->1. It has 25 trainable scalar parameters in five blocks. Matched bridge, original-native and independent primitive PyTorch models take two actual SGD steps, comparing predictions, loss, parameter gradients and updated parameters. Every block must change at each step.

After training, inputs and parameters stay fixed. The consumer evaluates parameter-dictionary HVPs for two fixed nonproportional directions through `functional_call`. It compares the new route with original-native reverse-over-reverse differentiation, independent primitive PyTorch HVPs and centered differences of independent gradients. It records d dot H d as directional curvature, bilinear symmetry, zero-direction and linearity controls. Separate mathematical tests in `tests/test_hessian_vector.py` retain a deliberately incomplete pool-level derivative that omits dg.

These curvature checks do not take additional optimizer steps or choose a better optimizer. The example demonstrates training integration and differentiation at fixed parameters. It establishes no model quality, adversarial robustness, privacy, speed, process-memory saving or optimizer improvement.

After installing the ordinary wheel with pinned dependencies into a fresh environment outside the checkout, copy the example to a separate consumer directory and use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\train_curvature.py" --dtype float32 --output "$consumer\curvature-float32.json"
& $python -I "$consumer\train_curvature.py" --dtype float64 --output "$consumer\curvature-float64.json"
```

Receipts identify the installed package and loaded native library, script bytes, dtype, thread settings, actual update counts, parameter shapes and comparison errors. Input and fixed-parameter ownership checks accompany the numerical results. Workflow files declare this execution, while completed jobs and exact revisions establish the actual evidence.

## Independent checks and debugging

The HVP tests compare both gradient and product with independently differentiated primitive PyTorch and native reverse-over-reverse results. Float64 centered differences of gradients use interior points. Direction linearity and bilinear symmetry provide additional relations. Values-only, logits-only, joint and zero directions, logit shifts, scaled nonlinear losses, empty/singleton groups, upstream views and saved-input versions exercise the boundary. Operator registration checks remain separate from these mathematical checks.

The independent transformed reference captures fixed Python group boundaries before entering a transform. Reading offsets with `tolist()` inside `jvp(grad(...))` can fail on a transform wrapper without ordinary storage. That reference failure must be resolved at the reference's Python boundary, not by weakening production admission or replacing the native operation with reference math.

An output guard alone cannot protect opaque native differentiation. Higher tangents may disappear before reaching it. Guards cover backward-JVP inputs as well as results, including a direction that varies only the cotangent. Supported compositions must agree with independent math, and excluded compositions must retain an actual rejection rather than silently returning a plausible zero.
