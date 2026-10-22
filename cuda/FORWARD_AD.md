# Follow an input direction on CUDA

`ragged_pool_cuda.forward_ad.ragged_pool_forward_ad` returns the packed native pool and supports one eager first-order forward derivative. A Jacobian-vector product, or JVP, describes the output's local change when values and scores move along supplied directions. It does not construct the complete Jacobian.

```python
import torch
from ragged_pool_cuda.forward_ad import ragged_pool_forward_ad

values = torch.tensor([[1., 2.], [3., -2.]], device="cuda:0", dtype=torch.float64)
logits = torch.tensor([0., 1.0986122886681098], device="cuda:0", dtype=torch.float64)
offsets = torch.tensor([0, 2, 2], device="cuda:0", dtype=torch.int64)
dvalues = torch.tensor([[2., -1.], [-2., 3.]], device="cuda:0", dtype=torch.float64)
dlogits = torch.tensor([0.4, -0.2], device="cuda:0", dtype=torch.float64)
pooled, direction = torch.func.jvp(
    lambda v, l: ragged_pool_forward_ad(v, l, offsets),
    (values, logits), (dvalues, dlogits),
)
# pooled: [[2.5, -1.0], [0.0, 0.0]]
# direction: [[-1.225, 2.45], [0.0, 0.0]]
```

The first group's weights are 0.25 and 0.75. Changing its values alone contributes `[-1,2]`. Changing its scores also changes the weights, contributing `[-0.225,0.45]`. Both terms belong in the derivative. The empty group's output and tangent are zero. A singleton's tangent is its value direction, regardless of its score direction. Adding the same direction to every score in a group changes no weight.

For pooled vector `y`, weights `wi`, feature directions `dvi` and score directions `dli`, the native kernel computes:

```text
dy = sum_i wi * (dvi + (vi - y) * dli)
```

Stable weights, the recomputed pooled vector and the tangent use double intermediates, with one final cast to the primal dtype. The new private `_jvp` dispatcher operation has its own metadata-only fake rule. This is a new native kernel, rather than a claim that existing reverse autograd automatically supplies forward AD. Ordinary first-order reverse on this separate API still uses the established native backward. The original packed, padded, compiled, export and functional-reverse APIs retain their own contracts.

## Input and derivative boundary

Primal admission remains contiguous matching CUDA float32/float64 values `[N,D]`, logits `[N]` and int64 offsets `[B+1]`, with `N<=4096`, `1<=D<=128`, `B<=128` and finite values/scores in `[-1000,1000]`. Offsets retain their ordered partition contract and remain fixed under differentiation. Zero groups and empty groups are admitted. Inputs are not mutated or moved between devices.

Each floating direction matches its primal shape, dtype and CUDA device. It must have strided layout and finite entries in `[-1000,1000]`. Native code explicitly resolves logical-negative directions and normalizes noncontiguous or zero-stride directions. Original primal layout admission occurs first. The Python guard copies saved primals/directions after that admission to retain explicit transform boundaries. This allocates storage and adds work.

The native JVP validates offsets/primal contents and direction contents before offset-indexed arithmetic. It performs two separate four-byte device-to-host status transfers, each synchronizing the current stream. The public JVP also executes the primal call with its existing admission synchronization. These costs exclude CUDA graph capture and belong in any future complete-call measurement. No speed or memory advantage is claimed.

Supported routes are explicit dual tensors from `torch.autograd.forward_ad.make_dual` and one eager `torch.func.jvp`. Explicit duals can supply only a values direction or only a logits direction. Ordinary primal reverse is first order. Differentiating the tangent, nested JVP, vmap/batched transforms, higher reverse derivatives, compile/export of this API, inference mode and inference tensors are excluded and explicitly refused. This does not add parameter HVPs, CUDA second derivatives or general transform composition.

The guard retains dependencies so reverse differentiation of a tangent fails deliberately rather than returning a detached answer. Original saved tensors and real version counters are checked at plain Function boundaries. This uses the pinned internal `Tensor._version` property and does not detect external-memory writes or promise cross-version compatibility. [PyTorch's transform-extension notes](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) and [JVP contract](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.Function.jvp.html) explain the framework route, not the correctness of this implementation.

## Installed training and sensitivity example

Build through the [existing explicit Windows installation route](README.md#one-explicit-installation-route). This increment requires a newly compiled native wheel. Install it normally into a fresh isolated environment outside the checkout, copy `examples/train_forward_ad.py` and `examples/train_cuda.py` together into an external consumer directory, then run:

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_forward_ad.py --dtype float32 --output C:\work\cuda-consumer\forward-float32.json
& $acceptPython -I C:\work\cuda-consumer\train_forward_ad.py --dtype float64 --output C:\work\cuda-consumer\forward-float64.json
```

The consumer takes two actual SGD updates through an encoder, scorer, native pool and prediction head. It compares predictions, pooled vectors, loss, raw-input gradients, all five parameter gradients and updated parameters with ordinary-native and independent primitive GPU models. It then holds the trained parameters fixed and checks input sensitivity against primitive forward AD, explicit duals, centered differences of the native model and a native reverse-mode adjoint identity. The sensitivity phase preserves existing parameter/gradient values and gradient objects. Empty groups have zero prediction and pool input tangents in this model.

Float32 comparisons use `rtol=2e-4, atol=2e-6`; float64 uses `rtol=2e-10, atol=2e-12`. The consumer's centered differences declare separate dtype-specific step/tolerance values. The fixture is small, seeded and synthetic. Two optimizer steps and local sensitivity establish integration, without model-quality, adversarial robustness, privacy, operational security or production-scale claims.

## Executed result and limits

The [validation record](evidence/forward-ad-windows-sm89-20261022.json) identifies the fresh build, ordinary wheel, installed native payload and separately revised tests. The declared Windows x64, CPython 3.11.8, torch 2.9.1+cu128/CUDA 12.8, NVCC 12.8.93, MSVC 19.44.35228/toolset 14.44.35207, C++17, sm89 and RTX 4090 route passed. This is one pinned environment, not a compatibility matrix. The build retained a compiler-version probe warning before successful actual compilation and linking. Existing Python source has explicit Git/source-format joins where standard packaging preserved CRLF rather than the earlier wheel's LF bytes.

The complete installed CUDA suite passed 625 tests, with four existing symlink-privilege skips and one retained PyTorch non-writable-buffer warning. All nine cohort commands passed, including the new, functional, clipped and compiled consumers in both dtypes. A final test-only review follow-up passed 48 focused tests, strengthening explicit refusal checks and adding maximum N/D/B envelopes and finite endpoints. It changed no runtime or consumer bytes. Operator registration checks remain separate from independent mathematical checks.

Independent source review prompted stronger stream-result, nonzero-version and explicit guard-message assertions. The initial broad exception assertion could have accepted a silent detach; the final test requires a connected dependency and the intended error. No mathematical or native execution failure was observed in these cohorts. This test-review finding is retained as a debugging case rather than described as a fixed native defect.

No new Compute Sanitizer run qualifies this changed binary. The earlier Windows diagnostics required a manually elevated prerequisite and restoration. This run changed no privileged settings and supplies ordinary GPU execution plus independent numerical/source checks. Historical old-binary diagnostic results remain separate. Hosted CPU CI exercises the unchanged CPU package and is not CUDA execution. Previously saved CUDA artifacts bind their earlier binary/module identities and are not promised compatible with this new wheel. No new binary release/tag, GPU backend, mixed precision, optimized compiler, performance, portability or ABI guarantee follows from source publication.
