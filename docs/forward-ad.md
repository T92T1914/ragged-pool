# Packed forward automatic differentiation

`ragged_pool_forward_ad(values, logits, offsets)` has the same packed result and primal admission as `ragged_pool`. This separate eager API also supplies a first-order Jacobian-vector product (JVP). Given directions for the values and logits, it returns the local rate of change of each pooled vector. It does not estimate whether a model is accurate or robust.

## A worked direction

For values `[[2,0],[0,4]]`, logits `[0,log(3)]` and offsets `[0,2,2]`, the weights are one quarter and three quarters. The primal result is `[[0.5,3],[0,0]]`. Use value directions `[[1,-1],[2,1]]` and logit directions `[0.1,-0.2]`:

```python
import math
import torch
from ragged_pool import ragged_pool_forward_ad

values = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
logits = torch.tensor([0., math.log(3.)], dtype=torch.float64)
offsets = torch.tensor([0, 2, 2], dtype=torch.int64)
dv = torch.tensor([[1., -1.], [2., 1.]], dtype=torch.float64)
dl = torch.tensor([0.1, -0.2], dtype=torch.float64)

primal, tangent = torch.func.jvp(
    lambda v, l: ragged_pool_forward_ad(v, l, offsets),
    (values, logits), (dv, dl),
)
# primal:  [[0.5, 3.0], [0.0, 0.0]]
# tangent: [[1.8625, 0.275], [0.0, 0.0]]
```

For weights `w_i`, pooled value `y`, value direction `a_i` and logit direction `b_i`, define `b_bar = sum_i w_i*b_i`. The JVP is:

```text
dy = sum_i w_i*a_i + sum_i w_i*(b_i-b_bar)*(value_i-y)
```

In this example, the weighted value direction is `[1.75,0.5]` and the logit contribution is `[0.1125,-0.225]`. Empty groups contribute zero. A singleton passes through its value direction, and a common shift in every logit of a group has zero contribution.

The existing native second-backward routine already returns this expression as its first result, independently of its output cotangent. The bridge calls that routine with a zero cotangent and discards its other two results. This avoids another derivative kernel, while still allocating the unused results. No performance advantage follows from this choice.

## Supported route and bounds

Both `torch.autograd.forward_ad.dual_level()` with `make_dual`/`unpack_dual` and `torch.func.jvp` are declared routes on CPython 3.11.8 and PyTorch 2.9.1+cpu. Ordinary reverse differentiation of the primal result retains the native first and second reverse derivatives.

Primals retain the [packed CPU contract](contract.md): contiguous float32/float64 values and logits, int64 offsets, finite entries in `[-1000,1000]` and the existing size bounds. Floating tangents match their primals' shape, dtype and CPU strided layout, with finite entries in the same inclusive range. Tangents may be transposed, stepped, expanded with zero strides or logical negative views. PyTorch can normalize a supplied view while constructing a dual tensor. The bridge's public Function guards copy its four floating JVP inputs before native evaluation, which can also normalize views. Neither changes the caller's tensors. A missing floating tangent becomes zero. Offsets remain fixed and receive no derivative. Outputs and tangents use fresh storage.

This is an eager first-order forward AD route. Differentiating its tangent in reverse is explicitly rejected. Nested forward differentiation, padded JVP, mapped JVP, `jacfwd`, `hessian`, compiled/exported forward AD and general transform composition are outside this route. The same bridge separately supports the bounded [functional reverse and map routes](functional-models.md). Existing compiled and exported APIs keep their separate acceptance.

The implementation follows [PyTorch 2.9's separate forward/setup_context guidance](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) and [Function.jvp contract](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.Function.jvp.html). The original library registration path in [exact v2.9.1 source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_library/autograd.py) generates a different Function shape without this JVP method. The bridge adds no private framework dispatch mechanism, custom operator, fake kernel or new C++ implementation.

## Installed training and sensitivity check

After building the ordinary wheel and installing it with the pinned dependencies into a fresh environment outside the checkout, copy `examples/train_forward_ad.py` to a separate consumer directory. Use new output paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\train_forward_ad.py" --dtype float32 --output "$consumer\forward-ad-float32.json"
& $python -I "$consumer\train_forward_ad.py" --dtype float64 --output "$consumer\forward-ad-float64.json"
```

The seeded encoder, scorer, pool and head take two actual SGD updates. Their outputs, parameter gradients and updated parameters are compared with the original native API and independent primitive PyTorch composition. With trained parameters held fixed, both forward AD entry points propagate an input direction through the complete model. Independent PyTorch JVP and centered finite differences check the sensitivity. Receipts record the installed module/native identity, script digest, dtype, threads, update counts and observed numerical differences.

The mathematical tests separately check the hand result, empty and singleton groups, linearity, common logit shifts, float64 finite differences and the JVP/VJP inner-product identity. Contract tests exercise direction views, invalid data, ownership and saved-input versions. Existing operator registration checks remain separate from this mathematical evidence.

## Debugging boundaries

The checked pure PyTorch reference reads offsets into Python. Reading them with `tolist()` inside `torch.func.jvp` can fail because the transform supplies a wrapper without ordinary storage access. The independent transformed reference therefore captures fixed Python boundaries before entering the transform and performs only primitive tensor math inside it. The production bridge continues to validate the original tensor offsets in native execution.

Requires-grad primals can carry an ordinary reverse graph while forward AD computes their tangent. Reversing that tangent would ask for a derivative composition this bridge does not support. A small public-API Function guard rejects that request clearly rather than detaching the result and suggesting a valid zero derivative.

The first fresh installed candidate passed its first-order comparisons but failed the nested-rejection test: nested `torch.func.jvp` returned a result without invoking the output guard. The opaque native call had already lost the outer tangent. Guards now copy all four floating JVP inputs before that call, rejecting nesting over either primals or directions. The result guard separately rejects reverse differentiation of tangents. This adds bounded copies and is not a speed or memory claim.
