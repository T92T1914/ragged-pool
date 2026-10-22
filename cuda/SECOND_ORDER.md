# Train through a CUDA input gradient

`ragged_pool_cuda.second_order.ragged_pool_second_order` supports up to two ordinary reverse derivatives for packed CUDA pooling. This lets a small model train with a loss that depends on its input gradient. The original CUDA APIs retain their own derivative limits.

```python
import torch
from ragged_pool_cuda.second_order import ragged_pool_second_order

values = torch.tensor([[0.4]], device="cuda:0", dtype=torch.float64, requires_grad=True)
logits = torch.tensor([0.2], device="cuda:0", dtype=torch.float64, requires_grad=True)
offsets = torch.tensor([0, 1], device="cuda:0", dtype=torch.int64)
y = ragged_pool_second_order(values, logits, offsets)
first = torch.autograd.grad(0.5 * y.square().sum(), values, create_graph=True)[0]
second = torch.autograd.grad(first.sum(), values)[0]
# y = [[0.4]], first = [[0.4]], second = [[1.0]]
```

The singleton pool is simply `y=values`. With a fixed incoming gradient `g`, pooling's first value gradient is `g` and has zero derivative with respect to values. For the squared loss above, however, `g=y` changes with values. The full second derivative is one. A backward implementation that forgets the derivative with respect to `g` incorrectly returns zero.

## Native derivative and framework boundary

The new private `_double_backward(p,q,g,values,logits,offsets)` differentiates the first backward with respect to its incoming gradient, values and scores. Here `p[N,D]` and `q[N]` are upstreams for the first value and score gradients, and `g[B,D]` is the first backward's original incoming gradient.

For one nonempty group, use softmax-weighted means `E`, pooled `y=E[v]` and the following centered calculation:

```text
qbar  = E[q]
H     = E[p] + E[(q-qbar)*(v-y)]
dg    = H
dv[i] = w[i]*(q[i]-qbar)*g
dl[i] = w[i]*dot(g, p[i]-H+(q[i]-qbar)*(v[i]-y))
```

Stable weights, means, covariance and dot products use double intermediates, with one final cast. A singleton copies `p` exactly into `dg` and leaves the other outputs zero. Empty groups leave all corresponding zero-filled outputs unchanged. The existing CPU implementation supplied a design reference; independent GPU primitives and numerical checks qualify this CUDA implementation separately.

Two classic `torch.autograd.Function` boundaries call the existing native forward/backward and the new second-backward kernel. They save the original primals and the original differentiable `g` using `save_for_backward`. Returning `dg` keeps the surrounding nonlinear head connected. Undefined `p` or `q` becomes a zero tensor. The new private operation has its own metadata-only fake output and explicit third-order rejection. Existing forward, backward and JVP registrations are unchanged.

The [pinned autograd documentation](https://docs.pytorch.org/docs/2.9/notes/extending.html) explains saved tensors and declared higher-derivative support. [PyTorch 2.9 function-transform documentation](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) distinguishes classic Functions from the separate setup-context form. This route deliberately uses the classic form and supports ordinary reverse AD only.

## Admission and limits

Original primal admission remains matching contiguous CUDA float32/float64 values `[N,D]`, scores `[N]` and int64 offsets `[B+1]`. Bounds are `N<=4096`, `1<=D<=128`, `B<=128` and finite values/scores in `[-1000,1000]`. Offsets remain ordered partition boundaries and receive no gradient. Zero groups and entirely empty inputs are included.

The first and second upstreams match their corresponding shape, dtype and CUDA device. They use strided layout with finite entries in `[-1000,1000]`. Native code explicitly normalizes transposed, stepped, expanded zero-stride and logical-negative upstreams. Original primals are admitted without copying or coercion. Inputs are not mutated or transferred to another device.

The second-backward operation performs two explicit four-byte status transfers to the host, each synchronizing the current stream. Primal/offset and derivative contents are admitted before offset-indexed numerical work. This retains the existing device guard and current-stream/launch-check policy. These costs belong in a complete call and exclude CUDA graph capture. There is no speed or memory claim.

Saved original tensors retain ordinary version checks before either backward. Inference tensors/mode are refused. Third derivatives, torch.func transforms, vmap/batched backward, forward AD on this API, compile/export and padded second order are outside this route. Existing functional and JVP APIs do not acquire higher derivatives. No separate HVP helper or transform composition is qualified here. This is one pinned environment, without platform, ABI or saved-artifact compatibility promises.

## Installed model consumer

Build a fresh native wheel through the [explicit Windows route](README.md#one-explicit-installation-route), then install it normally in a fresh isolated environment outside the checkout. Copy `examples/train_second_order.py` and `examples/train_cuda.py` together outside the checkout:

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_second_order.py --dtype float32 --output C:\work\cuda-consumer\second-float32.json
& $acceptPython -I C:\work\cuda-consumer\train_second_order.py --dtype float64 --output C:\work\cuda-consumer\second-float64.json
```

The encoder/scorer/pool model uses a nonlinear `tanh(linear(pool))` prediction head. Its fixed objective is `MSE + 0.5*mean(input_gradient**2)`, where the input gradient is for the sum of predictions with respect to packed raw features. Two actual SGD updates compare predictions, pools, first input gradients, loss components, total input/parameter gradients and updated parameters against independently differentiated primitive GPU math.

A detached-penalty control has exactly the same scalar loss and penalty while its parameter gradients match MSE-only training. The proper penalty must contribute meaningful scorer/head gradients. A shared float64 singleton control checks the omitted-`dg` error, yielding one for full curvature and zero when that dependency is omitted. Both training dtypes run that float64 control; it is not a separate float32 analytic witness. Inputs, initial parameters and persistent model/optimizer ownership are checked.

The synthetic example demonstrates integration. It is not an experiment establishing model quality, adversarial robustness, privacy or operational security. Default comparisons use float32 `rtol=3e-4, atol=3e-6` and float64 `rtol=3e-10, atol=3e-12`. Coordinate differences and gradcheck/gradgradcheck declare their own small float64 fixture and tolerances.

## Executed evidence

The [validation record](evidence/second-order-windows-sm89-20261022.json) identifies the source, checked archive, freshly compiled wheel and ordinary external installation. The executed route is Windows x64, CPython 3.11.8, Torch 2.9.1+cu128/CUDA 12.8, NVCC 12.8.93, MSVC 19.44.35228/toolset 14.44.35207, C++17, sm89 and RTX 4090/CUDA0. Compiler-version probing retained a cl status 2 warning before successful actual compilation/linking, without a guard bypass.

All eleven serial installed commands passed at the same source revision. The complete CUDA suite reports 683 passed, four existing symlink-privilege skips and one existing exported-loader non-writable-buffer warning. New mathematical tests compare all three native outputs to primitives, nonlinear second reverse, coordinate finite differences and float64 gradcheck/gradgradcheck. Registration checks remain separate. Contract tests cover layouts, mutation at both stages, ownership/nonaccumulation, missing upstreams, direct admission/recovery, maximum envelopes and the explicit connected third-order refusal. Both new consumers and preceding JVP, functional, clipping and compiled consumers passed in both dtypes. Existing export/resume and other API tests passed in the full suite.

Independent native/source review requested a stronger exact third-order error assertion before the build. No runtime implementation defect was found in that review, and no mathematical/native execution failure was observed in the declared cohort. Earlier research and instrumented results retain their original identities.

No new Compute Sanitizer result qualifies this changed binary. The earlier Windows diagnostic prerequisite required an owner-administered privileged action and was restored afterward. This run changes no privileged settings. Hosted CPU CI is separate from CUDA acceptance. New source publication does not create a binary release/tag or establish other GPU backends, optimization, distributed training, performance or broad compatibility.
