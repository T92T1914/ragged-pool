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

The [PyTorch2.9 Function guidance](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) explains the separate `forward` and `setup_context` route for transforms. The outer Function invokes native pool. Its backward calls another Function around native `_backward`. That inner boundary refuses further reverse differentiation, including a derivative with respect to the pullback cotangent. Both boundaries refuse JVP and vmap. The public wrapper uses [is_compiling](https://docs.pytorch.org/docs/2.9/generated/torch.compiler.is_compiling.html) to refuse compile/export tracing. The existing qualified compiled API remains separate. No new dispatcher operation, derivative formula, CUDA kernel or fake rule is added.

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

Both dtypes passed the model consumer, eager resume, static/dynamic compiled training, fixed/dynamic export and consumption of the preceding milestone's actual fixed/dynamic artifacts under the new wheel. No broader artifact-compatibility matrix follows. This route excludes padded functional transforms, maps, forward/higher derivatives and transformed compilation/export, and makes no performance, model-quality, operational security, device/platform/version/ABI or general transform-composition claim.
