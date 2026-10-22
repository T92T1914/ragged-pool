# Pool a masked CUDA batch

`ragged_pool_padded(values, logits, mask)` accepts feature vectors already arranged as `[B,T,D]`, importance logits `[B,T]` and a boolean selection mask `[B,T]`. True selects a real item. The adapter gathers selected items in row order, builds CUDA offsets and reuses the packed native forward and backward. A caller does not need to pack a padded model batch by hand.

The inputs must be contiguous strided tensors on the same CUDA device. Values and logits share float32 or float64. The mask is bool. Bounds are `0<=B<=128`, `0<=T<=4096`, `B*T<=4096` and `1<=D<=128`. The slot bound limits padded storage, even if few entries are selected. Only selected values and logits must be finite and within `[-1000,1000]`. Ignored entries may contain NaN, infinity or larger finite sentinels. Eager public entry points refuse logical negative or conjugate primal views before dispatcher normalization. The compiled qualification uses ordinary primal tensors and does not claim original-view refusal.

```python
import torch
from ragged_pool_cuda import ragged_pool_padded

values = torch.tensor([[[1., 2.], [float('nan'), float('inf')], [3., -2.]],
                       [[float('nan'), 99.], [float('inf'), 99.], [99., 99.]]],
                      dtype=torch.float64, device='cuda:0', requires_grad=True)
logits = torch.tensor([[0., float('nan'), 1.0986122886681098],
                       [float('nan'), float('inf'), 99.]],
                      dtype=torch.float64, device='cuda:0', requires_grad=True)
mask = torch.tensor([[True, False, True], [False, False, False]], device='cuda:0')
pooled = ragged_pool_padded(values, logits, mask)
# Approximately [[2.5, -1.0], [0.0, 0.0]].
pooled.sum().backward()
```

The empty row returns an exact zero pool. Ignored values and logits receive exact zero gradients. A singleton's value gradient is its upstream gradient and its logit gradient is zero. The mask has no gradient. Inputs retain their device, dtype, contents and storage. Packing and gradient scattering allocate new tensors.

## Differentiation and capture

Two Python CUDA custom operations form opaque padded forward and private backward boundaries around the unchanged packed native implementation. Each has a metadata-only fake rule. Forward saves original values, logits and mask. Backward repacks those originals, runs native backward and scatters selected gradients into fresh padded zero tensors. PyTorch's saved-tensor version checks reject changes to any original, including ignored slots, before backward.

The route supports first-order reverse differentiation. A second derivative is explicitly refused. Matching finite upstream gradients within `[-1000,1000]` may be contiguous, transposed, stepped, expanded zero-stride or logical negative views. Private backward has its own metadata and native content admission. These are separate checks from the public function's original-primal representation checks.

The [PyTorch2.9 library API](https://docs.pytorch.org/docs/2.9/library.html) describes opaque custom operations, supported autograd registration, fake rules and registration checks. Data-dependent compaction stays inside both opaque implementations. The declared compiler route is `torch.compile(..., backend='aot_eager', fullgraph=True, dynamic=False)` at fixed padded outer shapes with changed mask contents. Pinned Dynamo cannot trace the bool-returning `Tensor.is_neg()` method, so the eager original-view check is excluded while compiling; all other metadata admission remains active. This tests graph and autograd integration, without optimizing code generation or a general graph-count promise.

Boolean CUDA compaction allocates and can synchronize to obtain the selected count. Exact upstream indexing uses nonzero construction, whose [CUDA synchronization behavior](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html) is documented. Native admission separately copies a four-byte status to the host and synchronizes before offset-indexed work. Inputs stay on CUDA, but this is not a zero-copy or asynchronous path. No memory-saving or performance result is assumed.

## Use the installed training consumer

Copy `examples/train_padded.py` and `examples/train_cuda.py` together outside the source checkout after installing the companion wheel normally into a fresh isolated environment. Use the same Python3.11.8/PyTorch2.9.1+cu128 Windows RTX4090 environment described in the [CUDA guide](README.md). Install NumPy2.2.6 and pytest8.4.2 when running the complete checks.

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_padded.py --dtype float32 --output C:\work\cuda-consumer\padded-float32.json
& $acceptPython -I C:\work\cuda-consumer\train_padded.py --dtype float64 --output C:\work\cuda-consumer\padded-float64.json
```

The consumer owns three persistent models and SGD optimizers: compiled native, eager native and independent per-row GPU primitives. Each performs two real updates. B=5, T=6, raw width3 and pooled width4 stay fixed, while raw data, targets, masks, empty/singleton rows and selected counts change. It compares predictions, pools, loss, raw-input gradients, all five parameter gradients and updated parameters. It also checks literal input bytes/versions, initial state, ownership and actual compiled backward execution. This is a synthetic integration fixture rather than a model-quality study.

Ignored raw features contain NaN and both infinities. The model replaces them with finite zeros through out-of-place `torch.where` before encoder arithmetic. Masking only at pooling can leave `0*NaN` in encoder parameter gradients. The [autograd masking note](https://docs.pytorch.org/docs/2.9/notes/autograd.html) explains the underlying issue. The pooling API cannot repair invalid arithmetic performed earlier by its caller.

## Evidence boundaries

The [executed record](evidence/padded-windows-sm89-20261021.json) identifies source `a545f16e0b8f037a236e5d8843fbd32b1a21b026`, the checked source archive, installed wrapper wheel and reused native payload. Local Windows RTX4090 checks passed 335 tests, with four symlink cases skipped for ordinary privilege and one existing PT2 loader warning retained. Both dtypes completed two real SGD updates through each of compiled native, eager native and independent per-row primitives, including changed masks and poisoned raw padding. Packed export production and fresh-process consumption also passed with the new package identity. These CUDA results are local execution evidence; hosted CPU CI is a separate regression gate.

The first attempt retained 333 passing tests, two failed compiler checks and two failed training consumers. Dynamo rejected the bool-returning `is_neg()` inspection. The corrected attempt restricts that eager inspection while retaining all other admission. This debugging result narrows compiled qualification to ordinary primals instead of claiming unsupported original-view refusal.

The package's existing native payload is reused unchanged. The new wheel packages the Python API and registrations through standard setuptools build and wheel commands. Its original NVCC12.8.93/MSVC19.44/C++17/sm89 provenance remains historical. This is not a fresh native compilation or new Compute Sanitizer cohort for the padded gather/scatter caller.

Mathematical tests use independent per-row double-softmax arithmetic and finite differences. Registration/schema/fake/AOT checks remain separate. Other checks cover selected admission and recovery, ignored sentinels, empty cases, bounds, original mutation and genuine upstream layouts. Full established CUDA regression and export checks retain their own source/artifact identities. Exact execution results belong in the accompanying revision-linked record rather than following merely from this source description.

This route does not qualify dynamic outer shapes, padded export, second derivatives, transforms, CUDA graph capture, Inductor/Triton, mixed precision, other architectures/platforms/ABIs, distributed training, speed, peak memory or operational security. The CPU padded operation's broader derivative support is separate. Source remains MIT licensed with the existing attribution notices.
