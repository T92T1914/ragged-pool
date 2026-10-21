# Packed CUDA pooling candidate

This optional companion adds native CUDA forward and backward to the CPU package's pooling problem, with a separate distribution, import and dependency pin. The root CPU installation, operators and broader derivative support retain their own contracts.

This is a private candidate. Required native diagnostic qualification is open: the local Compute Sanitizer prerequisite probe could not initialize the Windows WDDM debugging interface and requested an administrator action. Functional tests and installed training are separate evidence. Do not describe the candidate as accepted or publish a GPU acceptance claim until the required memory, race and synchronization checks cover the actual native implementation.

## API and bounds

```python
from ragged_pool_cuda import ragged_pool
pooled = ragged_pool(values, logits, offsets)
```

`values` has shape `[N,D]`, `logits` has shape `[N]`, and int64 `offsets` has shape `[B+1]`. All three must be ordinary contiguous strided tensors on the same CUDA device. Values and logits use matching float32 or float64, with finite entries in `[-1000,1000]`. Bounds are `N<=4096`, `1<=D<=128` and `0<=B<=128`. Offsets start at zero, end at N, stay in `[0,N]` and are nondecreasing. Duplicate offsets create empty groups. Logical negative or conjugate primal views are refused. Inputs are not mutated.

Each nonempty group subtracts its maximum logit, normalizes exponentials and returns the weighted sum. Normalization and accumulation use double internally, then cast to the input dtype. Empty groups return zero vectors. Native backward returns gradients for values and logits, with no offsets gradient. Its upstream must match output shape, dtype and device and be finite in `[-1000,1000]`. Ordinary strided, transposed, stepped, expanded zero-stride and logical negative upstreams are normalized inside backward.

For values `[[1,2],[3,-2]]` and logits `[0,log(3)]`, weights are `[0.25,0.75]` and the pooled vector is `[2.5,-1]`. With upstream `[2,-3]`, value gradients are `[[0.5,-0.75],[1.5,-2.25]]`. Logit gradients are `[-3,3]`, using `w_i * dot(upstream, value_i - pooled)`. A singleton logit gradient is zero. Backward recomputes the pooled intermediate in double rather than using a rounded output as exact.

## Admission and streams

Host metadata checks precede pointer access. A separate validation kernel reads offsets by their known linear array positions and checks content bounds. It never uses an unvalidated offset as an address. The caller explicitly copies one four-byte status value to the host and synchronizes the tensor device's current PyTorch CUDA stream. Invalid content is refused before offset-indexed numerical work. Backward validates its own callable boundary too.

That synchronization is a deliberate tradeoff for a small readable component. It excludes CUDA graph capture and belongs in a complete call cost. It is not a hidden transfer or a speed advantage. Numerical work uses the current stream, a scoped device guard and immediate launch-error checks. A launch check alone does not establish completion or memory safety. Temporary storage remains alive through its required use.

One thread handles each group, without shared-memory reductions or barriers. Valid offsets divide rows into disjoint groups. Zero-filled outputs and gradients cover empty cases, including zero items and zero groups. This simple implementation is a framework integration and learning component, with no algorithmic novelty or performance claim.

Autograd saves original tensors so their version checks remain effective. Both custom operations, including the private backward, have metadata-only fake rules. Native reverse differentiation is first order only. A second reverse derivative is explicitly refused. Fake registration does not establish compilation, export or transform support.

`ragged_pool_reference` uses independent primitive GPU arithmetic. It explicitly transfers offsets to CPU for Python boundaries and synchronizes scalar admission checks. Differentiable values, logits and numerical operations stay on their CUDA device. The mathematical tests and training consumer also use independently supplied Python boundaries, so native device offsets do not supply their only oracle.

## One explicit installation route

The selected local envelope is Windows x64, CPython 3.11.8, released `torch==2.9.1+cu128`, CUDA 12.8.1 components with NVCC 12.8.93, MSVC 19.44.35228/toolset 14.44.35207, C++17 and only `sm_89`. Setup requires `TORCH_CUDA_ARCH_LIST=8.9`. No compiler-guard bypass or fast-math flag is used. This names the selected combination, without promising broad wheel, ABI, platform or device compatibility.

The [PyTorch release instructions](https://pytorch.org/get-started/previous-versions/#v291) identify the official CUDA wheel index. [Extension documentation](https://docs.pytorch.org/docs/2.9/cpp_extension.html) explains the separate compiler dependency. The [NVIDIA manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) supplies project-local component paths and checksums. The driver remains a separate prerequisite. No system installer, global environment change or driver replacement is required by this source route.

Use a separate build environment and a qualified x64 Visual Studio command environment without launching a visible application. Set `CUDA_HOME` to the toolkit and add its `bin` to the child process's PATH. Set `TORCH_CUDA_ARCH_LIST=8.9`, `MAX_JOBS=2`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. From this `cuda` directory:

```powershell
& $buildPython -m pip --isolated install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $buildPython -m pip --isolated install setuptools==80.9.0 wheel==0.45.1
& $buildPython -m pip --isolated wheel . --no-build-isolation --no-deps --no-cache-dir --wheel-dir C:\work\cuda-wheels
```

Verification builds from a checked source archive and uses an uncached native build. Create a second fresh acceptance environment and consumer directory outside the checkout. Install CUDA PyTorch from its official index, install the companion wheel normally and run `pip check`. Copy `examples/train_cuda.py` and `tests` from the checked archive into the external consumer directory. Do not rely on source-tree imports.

```powershell
& $acceptPython -I -m pytest -q --import-mode=importlib C:\work\cuda-consumer\tests
& $acceptPython -I C:\work\cuda-consumer\train_cuda.py --dtype float32 --output C:\work\cuda-consumer\float32.json
& $acceptPython -I C:\work\cuda-consumer\train_cuda.py --dtype float64 --output C:\work\cuda-consumer\float64.json
```

Each consumer performs two actual eager SGD updates through an encoder, score layer, pool and prediction head. It compares complete outputs, losses, raw-input and parameter gradients, and updated parameters with independently differentiated GPU primitives. Default tolerances are float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. Endpoint tests state their separate absolute tolerance for cancellation near magnitude 1000. Registration checks remain separate from mathematical gradients. This synthetic example demonstrates integration, not model quality or real-world accuracy.

## Limits and open native diagnostics

The companion does not inherit CPU padded pooling, second derivatives, HVPs, JVPs, vmap, compile/export, graph capture, stochastic recovery or other CPU example capabilities. No mixed precision, distributed training, other GPU backend or architecture, cross-version/platform/ABI, coinstallation matrix, performance or operational security result is claimed.

Actual native memory/race/synchronization evidence remains required. [Compute Sanitizer's Windows requirements](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) identify the administrator-controlled debugging interface. Version output, compilation, ordinary GPU success, source review and failed instrumentation are different observations. No registry, driver or display timeout change is part of the current scope.

Source is MIT licensed. [NOTICE](NOTICE) and the included [PyTorch license](third_party/PYTORCH-LICENSE) preserve attribution and the truthful assistance statement. Documentation informs API choices. Research rationale is not measured implementation evidence. CPU studies and earlier adverse results retain their separate identities.
