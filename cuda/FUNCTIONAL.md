# Functional first gradients on CUDA

`ragged_pool_cuda.functional.ragged_pool_functional` exposes the packed native CUDA pool to selected eager reverse transforms. `torch.func.grad` returns gradients of a scalar loss as values. `torch.func.vjp` returns the output and a reusable pullback for chosen output cotangents. The separate submodule leaves the established ordinary, compiled and exported callers unchanged.

```python
import torch
from ragged_pool_cuda.functional import ragged_pool_functional

values = torch.tensor([[1., 2.], [3., -2.], [7., 5.]], device="cuda:0")
logits = torch.tensor([0., 1.0986123, 0.], device="cuda:0")
offsets = torch.tensor([0, 2, 2, 3], device="cuda:0")

def loss(v, l):
    pooled = ragged_pool_functional(v, l, offsets)
    return pooled.square().sum()

dv, dl = torch.func.grad(loss, argnums=(0, 1))(values, logits)
pooled, pullback = torch.func.vjp(
    lambda v, l: ragged_pool_functional(v, l, offsets), values, logits)
other_dv, other_dl = pullback(torch.tensor([[2., -3.], [4., 8.], [-1., 2.]], device="cuda:0"))
```

Offsets stay fixed closure data and receive no gradient. Ordinary primals need not have `requires_grad=True` before a functional call. Those calls return gradients without accumulating into or replacing leaf `.grad` fields. A consumer can then deliberately hand gradient copies to an optimizer.

## Contract and derivative boundary

Primal shape, contiguous layout, same-device dtype and finite bounds follow the [packed CUDA contract](README.md#api-and-bounds). This bridge additionally requires ordinary versioned tensors. Inference-mode-created primals are refused. Native forward receives the original primals, with no bridge clone, layout conversion or device transfer. Existing native validation still copies its explicit four-byte status to the host and synchronizes the current stream.

Pullbacks retain the native backward's strided, transposed, stepped, expanded zero-stride and logical-negative upstream handling. PyTorch's autograd engine can normalize an outer cotangent's floating dtype before the Function receives it. The bridge follows that ordinary behavior. Native backward validates the gradient it actually receives. Nonfinite or oversized upstreams remain errors, including when an excessively scaled surrounding loss creates them. The bridge does not clip or rescale them.

The [PyTorch2.9 Function guidance](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) explains the separate `forward` and `setup_context` route for transforms. The outer Function invokes native pool. Its backward calls another Function around native `_backward`. That inner boundary refuses further reverse differentiation, including a derivative with respect to the pullback cotangent. Both boundaries refuse JVP. The bounded manual mapping rules below extend this same bridge to one map layer. The public wrapper uses [is_compiling](https://docs.pytorch.org/docs/2.9/generated/torch.compiler.is_compiling.html) to refuse compile/export tracing. The existing qualified compiled API remains separate. No new dispatcher operation, derivative formula, CUDA kernel or fake rule is added.

## A mutation bug found during qualification

The first installed cohort passed its model consumers and 546 tests, but three tests showed that changing original values, logits or offsets after VJP did not raise. Saving primals through `save_for_backward` alone was insufficient for this transform path. Those failures remain in the [evidence record](evidence/functional-windows-sm89-20261022.json).

Pinned [transform machinery](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_functorch/autograd_function.py#L97-L145) unwraps operands for plain forward while setup/backward can see wrappers. Their [TensorImpl counters](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/functorch/TensorWrapper.cpp#L151-L170) can differ from the underlying tensors. A retained probe observed wrapper counter 0 while the actual tensor was already at 1, then actual counter 2 after mutation.

The correction captures actual integer counters in plain outer forward and returns them as hidden non-Tensor metadata. Context saves tensors normally and retains only those integers separately. Plain inner derivative forward checks the real counters before native backward. The public result remains the pooled tensor. No private functorch unwrapping, direct Tensor context attributes, primal copies or content comparison is used.

This guard reads the internal Tensor `_version` property at the exact pinned framework version. It is not a portable public version-query API or a detector for arbitrary external memory writes. Ordinary version-tracked in-place mutations are refused. Tests preserve all three original mutation cases and verify that nonzero versions established before capture do not cause false refusal.

## Installed model consumer

`examples/train_functional.py` loads its adjacent `train_cuda.py` helper explicitly. `functional_call(..., strict=True)` supplies the model's parameter dictionary. `grad(..., argnums=(0,1), has_aux=True)` returns all five parameter gradients, the raw-input gradient, loss, predictions and pools. Two actual SGD updates use learning rate 0.03, momentum 0, `foreach=False` and `fused=False`. Matched ordinary-native and independent primitive GPU models compare complete results and updates. A trained-model VJP reuses one pullback for two unequal prediction/pool cotangent tuples and compares every parameter and raw-input derivative.

Use the qualified Windows x64 Python 3.11.8, Torch 2.9.1+cu128, CUDA runtime 12.8, RTX 4090/CUDA0 environment. Install the new ordinary wheel, then copy `train_functional.py` and `train_cuda.py` from the checked source archive outside every checkout. Set OMP/MKL/OPENBLAS threads to 1 and build workers to at most 2.

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_functional.py --dtype float32 --output C:\work\functional-f32.json
& $acceptPython -I C:\work\cuda-consumer\train_functional.py --dtype float64 --output C:\work\functional-f64.json
```

The [installed cohort](evidence/functional-windows-sm89-20261022.json) passed 23 sequential commands and 552 CUDA tests, including 60 new functional cases. Four established unavailable symlink fixtures and one PyTorch PT2 warning remain recorded. Tests include nonlinear gradient comparisons, repeated unequal/layout-varied pullbacks, a float64 directional finite difference, logit-shift invariance, leaf-gradient preservation, admission/recovery and explicit unsupported-transform refusals. Registration checks remain distinct from mathematical checks.

Qualification used a newly packaged wheel from a checked source archive with the exact accepted native binary and five established Python modules, plus the new `functional.py`. Standard setuptools `build_py` and `bdist_wheel --skip-build` assembled that local wheel, followed by ordinary pip installation and RECORD/member verification. It is a new Python package artifact, without a new NVCC build or native instrumentation result. The normal source-build route still needs the declared CUDA compiler prerequisites. Wheel and native hashes remain separate.

Both dtypes passed the model consumer, eager resume, static/dynamic compiled training, fixed/dynamic export and consumption of the preceding milestone's actual fixed/dynamic artifacts under the new wheel. No broader artifact-compatibility matrix follows. This route excludes padded functional transforms, nested maps, forward/higher derivatives and transformed compilation/export, and makes no performance, model-quality, operational security, device/platform/version/ABI or general transform-composition claim.


## Map three independent models

The same functional API now supports one eager `vmap` layer, including `vmap(grad)` and differentiating a sum or mean of mapped results. A lane is a complete packed pooling problem. Values, logits and offsets may independently be mapped or shared. Logical N, D and B shapes match across lanes, while partition contents may differ. Empty, singleton and zero-group cases retain the native behavior. Every selected primal lane must already satisfy the contiguous CUDA contract. The bridge preserves original strides and refuses unsupported lane views without copying them into admissibility.

Each manual rule invocation admits `1 <= K <= 8`, `K*N <= 4096` and `K*B <= 128`. Shared inputs count once per actual native evaluation. A nonleading map dimension is supported when its selected lanes remain admissible. PyTorch `chunk_size` can divide a larger complete axis into multiple bounded invocations, so these limits do not bound the caller's total work or process memory. Native lane calls run serially. There is no fused batching kernel or performance claim.

The outer pool rule stacks tensor results and preserves one common tuple of real input version counters, asserting that all selected aliases agree. The backward rule selects its tensor arguments while carrying that metadata unchanged. An alias-only public Function guard rejects a remaining outer map or JVP and preserves lane strides. This retains the earlier mutation correction, including shared inputs and nonzero versions established before capture. The internal `_version` query remains tied to the pinned framework.

The existing consumer's `--models 3` mode trains three distinct seeded encoder/scorer/pool/head models through `vmap(grad)` and strict `functional_call`. Optimizers own new stacked parameter leaves, separate from the seed models and template. Two actual SGD updates compare complete predictions, pools, losses, raw-input gradients, all five parameter-gradient blocks per model and updated parameters against serial ordinary-native and independent primitive CUDA execution. The default one-model mode remains available.

After training, a fixed-parameter check differentiates the sum and mean of the three model losses with respect to shared raw inputs. Unequal model contributions make the reduction meaningful. If their individual input derivatives are g0, g1 and g2, the sum needs `g0 + g1 + g2`, while the mean needs that result divided by three. Dropping a shared contribution or confusing the two objectives changes this fixture's result.

Install the ordinary wheel into a fresh isolated pinned environment, then copy `train_functional.py` and its adjacent `train_cuda.py` helper outside every checkout. The same commands select either dtype:

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_functional.py --models 3 --dtype float32 --output C:\work\map-f32.json
& $acceptPython -I C:\work\cuda-consumer\train_functional.py --models 3 --dtype float64 --output C:\work\map-f64.json
```

The [executed map record](evidence/functional-map-windows-sm89-20261022.json) retains both complete consumer comparisons and artifact identities. The fresh installed cohort passed 15 serial commands and 812 CUDA tests, including 70 new map cases. Four existing Windows symlink privilege skips and one existing export-buffer warning remain. Tests compare map gradients with two numerical oracles, exercise all seven shared/mapped primal patterns, reusable mapped pullbacks and upstream layouts, shared learned-parameter sums/means, uneven chunks, exact envelopes and original mutation guards. Patched native sentinels require excessive envelopes and nested maps to be refused before native execution. Registration checks remain separate from mathematical checks.

Qualification assembled a new Python wheel using standard setuptools with only `functional.py` changed in the runtime payload. The accepted native binary and remaining runtime modules are literal matches. Complete wheel RECORD membership, installed bytes, source joins and `pip check` passed. No new NVCC build or instrumented native execution occurred. Observed model comparisons use the stated dtype tolerances, without a bitwise equivalence claim.

This is one bounded packed first-order integration path. It does not add padded maps, per-bag ragged batching, nested maps, mapped forward AD, higher derivatives or transformed compilation/export. It establishes no useful ensemble accuracy, differential privacy, security property, throughput or memory advantage. Execution remains limited to the declared pin and device.
