# Packed CUDA pooling companion

This optional companion adds native CUDA forward and backward to the CPU package's pooling problem, with a separate distribution, import and dependency pin. The root CPU installation, operators and broader derivative support retain their own contracts.

A separate [cross-device training handoff](../docs/cross-device-resume.md) restores the fixed model and SGD momentum from CPU to CUDA0 or back, then completes two further updates in a fresh process. Both dtypes passed with the exact co-installed wheels. The example uses explicit caller transfers and preserves earlier saved-artifact contracts.

To use both ordinary packages in one environment, follow the explicit [CPU cu128 build route](../docs/coinstallation.md). Its CPU extension is compiled against this companion's exact `torch==2.9.1+cu128` runtime and its wheel carries build tag `1cu128`. The default CPU-only wheel still requires +cpu, so that original pair has conflicting dependencies. The installed four-route consumer checks two actual updates on CPU and CUDA against independent primitives in both dtypes, with no automatic device movement or broad portability claim.

The Windows `sm_89` route below passed a fresh external wheel installation, 138 functional tests, two eager training updates per dtype, and six bounded Compute Sanitizer runs against the installed native implementation. The [validation record](evidence/windows-sm89-20261021.json) identifies the tested source, wheel and execution cohort. These results qualify that specific route and workload, with the limits described below.

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

Autograd saves original tensors so their version checks remain effective. Both custom operations, including the private backward, have metadata-only fake rules. Native reverse differentiation is first order only. A second reverse derivative is explicitly refused. Fake registration alone does not establish compilation, export or transform support. The bounded compiled route below has its own executed checks.

`ragged_pool_reference` uses independent primitive GPU arithmetic. It explicitly transfers offsets to CPU for Python boundaries and synchronizes scalar admission checks. Differentiable values, logits and numerical operations stay on their CUDA device. The mathematical tests and training consumer also use independently supplied Python boundaries, so native device offsets do not supply their only oracle.

## One explicit installation route

The tested envelope is Windows x64, CPython 3.11.8, released `torch==2.9.1+cu128` with CUDA runtime 12.8, CUDA 12.8.1 toolkit components with NVCC 12.8.93, MSVC 19.44.35228/toolset 14.44.35207, C++17 and only `sm_89`. The device was an NVIDIA GeForce RTX 4090 with compute capability 8.9 and driver 617.42. Setup requires `TORCH_CUDA_ARCH_LIST=8.9`. No compiler-guard bypass or fast-math flag is used. This names the tested combination, without promising broad wheel, ABI, platform or device compatibility.

The [PyTorch release instructions](https://pytorch.org/get-started/previous-versions/#v291) identify the official CUDA wheel index. [Extension documentation](https://docs.pytorch.org/docs/2.9/cpp_extension.html) explains the separate compiler dependency. The [NVIDIA manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) supplies project-local component paths and checksums. The driver remains a separate prerequisite. No system installer, global environment change or driver replacement is required by this source route.

Use a separate build environment and a qualified x64 Visual Studio command environment without launching a visible application. Set `CUDA_HOME` to the toolkit and add its `bin` to the child process's PATH. Set `TORCH_CUDA_ARCH_LIST=8.9`, `MAX_JOBS=2`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. From this `cuda` directory:

```powershell
& $buildPython -m pip --isolated install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $buildPython -m pip --isolated install setuptools==80.9.0 wheel==0.45.1
& $buildPython -m pip --isolated wheel . --no-build-isolation --no-deps --no-cache-dir --wheel-dir C:\work\cuda-wheels
```

The original native qualification used an uncached build from a checked source archive. The compiled follow-up packages an updated Python dispatcher wrapper with that exact accepted native payload, without a new NVCC build. Create a separate acceptance environment and consumer directory outside the checkout. Install CUDA PyTorch from its official index, install the companion wheel normally and run `pip check`. Copy both `examples/train_cuda.py` and `examples/train_compiled.py`, plus `tests`, from the checked archive into the external consumer directory. Do not rely on source-tree imports.

```powershell
& $acceptPython -I -m pytest -q --import-mode=importlib C:\work\cuda-consumer\tests
& $acceptPython -I C:\work\cuda-consumer\train_cuda.py --dtype float32 --output C:\work\cuda-consumer\float32.json
& $acceptPython -I C:\work\cuda-consumer\train_cuda.py --dtype float64 --output C:\work\cuda-consumer\float64.json
```

Each consumer performs two actual eager SGD updates through an encoder, score layer, pool and prediction head. It compares complete outputs, losses, raw-input and parameter gradients, and updated parameters with independently differentiated GPU primitives. Default tolerances are float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. Endpoint tests state their separate absolute tolerance for cancellation near magnitude 1000. Registration checks remain separate from mathematical gradients. This synthetic example demonstrates integration, not model quality or real-world accuracy.

## Fixed-shape compiled training

`examples/train_compiled.py` compares three persistent models and SGD optimizers: compiled native, eager native and independent GPU primitives. It calls `torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)`. Two batches keep N=9, B=5, raw width3 and pooled width4 while changing raw inputs, targets and offset contents. Empty and singleton groups occur in the sequence. Both dtypes complete two real updates and compare predictions, pooled values, loss, raw-input gradients, every parameter gradient and updated parameters. Input bytes/versions, initial parameters and model/optimizer ownership remain checked.

Keep `train_cuda.py` beside `train_compiled.py`. The latter loads its sibling explicitly, reusing the model, primitive reference and installed-package checks without adding the source checkout to the import path.

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_compiled.py --dtype float32 --output C:\work\cuda-consumer\compiled-float32.json
& $acceptPython -I C:\work\cuda-consumer\train_compiled.py --dtype float64 --output C:\work\cuda-consumer\compiled-float64.json
```

The [PyTorch2.9 compiler FAQ](https://docs.pytorch.org/docs/2.9/torch.compiler_faq.html) explains the limit of `aot_eager`: Dynamo captures the forward and AOTAutograd traces backward, then eager PyTorch executes both graphs. This is graph integration, without backend optimization or generated fused kernels. Fullgraph refuses graph breaks. Static specialization does not promise one graph for every shape. The consumer records one compile API call without claiming a graph count. Its pinned backward-node witness and separate pinned callback tests establish actual AOT execution and native forward/private-backward graph targets for the exercised cases.

The first installed compile attempt failed at the public wrapper's duplicate metadata checks because Dynamo could not trace their error-message lambdas. The wrapper now dispatches directly, matching the CPU API's structure. Native admission still validates real metadata and content before pointer/indexed access. Fake rules validate tracing metadata. Tests separately check registration, mathematics, changed partition contents, refusal/recovery after capture, original saved-input mutation, empty groups, fresh output storage and strided upstreams. Unsupported backends may have different dispatcher error text, while remaining rejected.

The [compiled validation record](evidence/compiled-windows-sm89-20261021.json) separates this new installed functional cohort from the earlier native diagnostics. All175 functional tests passed on the declared Windows RTX4090 environment, including the original138 tests. The new Python wrapper was installed in an ordinary wheel with the exact prior native payload. No new compiled-consumer sanitizer execution is claimed, and no administrator or debugger-setting action is required by these ordinary checks.

## Changing packed shapes

The same installed `train_compiled.py` consumer also has an explicit `--dynamic` mode for changing packed N/B counts through five updates while retaining its model and SGD optimizer. The [changing-shape guide](DYNAMIC.md) gives the seeded trajectory, full three-route comparisons, symbolic graph checks and first-order limits. Its default remains the fixed-shape route above. This source/consumer extension reuses the accepted installed runtime wheel and native binary.

## Eager training state

For a bounded eager training continuation, [the CUDA resume example](RESUME.md) saves model parameters and SGD momentum after update two, exits the producer and completes the remaining three changing-shape updates in a fresh process. It checks the full trajectory against uninterrupted native and independent primitive routes, with a weights-only reset control that exposes missing optimizer history. This uses the same installed wheel and retains an explicit trusted-local-state boundary.

The [installed resume cohort](evidence/resume-windows-sm89-20261022.json) passed 17 commands. The CUDA suite reported 492 passed tests, 4 skipped symlink fixtures and 1 existing PyTorch loader warning. Both dtypes matched uninterrupted native records byte for byte. The first cohort's Windows launcher-ID assertion failure and its correction remain recorded separately from the passing training result. This consumer increment adds no new native build or instrumentation claim.

## Masked padded batches

`ragged_pool_padded(values[B,T,D], logits[B,T], mask[B,T])` accepts a contiguous masked CUDA batch and reuses the packed native math. Holes and empty rows are supported. Ignored nonfinite padding is selected out before arithmetic, with exact zero ignored gradients. Opaque padded forward/backward operations retain original-input version checks and metadata-only fake rules. The [padded guide](PADDED.md) gives finite storage bounds, a worked example, first-order eager/fixed-shape AOT training and the compaction/synchronization tradeoff. No new native kernel or performance result is implied.

## Trusted inference export

`examples/export_cuda.py` saves a trained inference graph with `torch.export.export(..., strict=True)`. Its default retains the fixed CUDA0 `[9,3]`/`[6]` artifact. `--dynamic` saves one graph for independent N2..32/B2..8 counts with fixed raw width3. `examples/predict_cuda.py` loads either supported owned schema in a separate process without importing the producer or model class. It retains the native CUDA pool and compares changed inputs/partitions against independent primitive math. The [model handoff guide](EXPORT.md) gives both float32/float64 routes, trusted-file boundary and limitations. They require the same installed native package and pinned environment. A file digest identifies bytes rather than authenticating a publisher or making arbitrary serialized models safe.

Both dtype round trips passed in an ordinary fresh isolated installation. The [export validation record](evidence/export-windows-sm89-20261021.json) separates256passed tests,4unavailable symlink fixtures and1PyTorch loader warning. It preserves the actual source/artifact identities, numerical differences and adverse attempts. This example reuses the accepted native binary without a new compiler or instrumentation result.

The later [dynamic export cohort](evidence/dynamic-export-windows-sm89-20261022.json) passed 17 commands and 444 CUDA tests, with 4 skips and 1 warning retained. One saved graph per dtype served 11 cases across independent item/group counts in a fresh process. Actual symbolic graph inspection, wrapper admission, saved guard behavior and native content refusal remain distinct checks. Both fixed-schema regressions and consumption of the preceding milestone's artifacts passed in the same pinned environment. No new native binary was built.

## Validation and limits

The separate [functional reverse bridge](FUNCTIONAL.md) adds eager first-order `torch.func.grad` and reusable VJP calls through `ragged_pool_cuda.functional.ragged_pool_functional`. Its installed model consumer performs two actual SGD updates with `functional_call`, full raw/parameter gradient comparisons and unequal model-output cotangents. The [552-test cohort](evidence/functional-windows-sm89-20261022.json) retains the original mutation-check failure and a pinned version-counter correction. A newly packaged Python wheel adds this module while preserving the existing native binary and five established runtime files. This route requires ordinary versioned primals and excludes inference tensors, JVP/higher derivatives and compile/export. The bounded map extension below has separate acceptance.

The [per-group clipping caller](CLIPPING.md) uses serial functional gradients to clip each group's joint five-block parameter vector before taking its mean. Both dtypes complete two SGD updates against full-batch native loss-row gradients and primitive GPU oracles. Its [580-test installed cohort](evidence/clipping-windows-sm89-20261022.json) includes zero/underflow/overflow, empty-group head-bias and clipping-order checks. It reuses the accepted functional wheel without adding runtime code or claiming a new compiler, sanitizer, privacy or speed result.

On 2026-10-21, an uncached build from source commit `15db410f153e6bcac498baa175fedbbd19c34f7f` produced the wheel tested in a fresh acceptance environment outside the checkout. All 138 installed functional tests passed. The float32 and float64 training consumers each completed two SGD updates and passed their independent GPU-reference comparisons. These are small synthetic fixtures, not evidence of training convergence or model quality.

Compute Sanitizer 2025.1.0.0, build 35583870, then ran `memcheck`, `racecheck` and `synccheck` separately for each dtype against the installed native consumer. All six invocations exited with code 0, without a timeout or forced cleanup. Memory and synchronization checks reported zero errors. Race checks reported zero hazards, errors and warnings. This is actual instrumentation of the native route used by those fixtures. The full 138-test suite was not run under every sanitizer tool, and zero reported findings are limited to the executed scenarios.

The first Windows WDDM prerequisite probe could not initialize the debugging interface. The owner manually launched a reviewed elevated helper to enable the required interface temporarily and run the checks. The setting was absent before the run, restoration completed, and an independent readback confirmed that it was absent again afterward. [NVIDIA's Windows requirements](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) explain this administrator-controlled diagnostic prerequisite. The failed probe and the later successful instrumentation are distinct observations in the validation record.

The original eager GPU evidence remains tied to the source commit above and the hashes in its validation record. That publication added documentation, the sanitized evidence summary and its source-archive inclusion. The later compiled cohort changes the public Python dispatcher wrapper and adds consumer/tests. Its native source, exact native binary, operator registrations and compiler settings remain unchanged. Each cohort retains its own source and artifact identities.

The separate [packed forward AD API](FORWARD_AD.md) supports one eager first-order JVP with a new native tangent kernel and explicit finite direction admission. The trained consumer checks input sensitivity after two matched SGD updates. Its [fresh-build validation record](evidence/forward-ad-windows-sm89-20261022.json) separates the new installed binary, 625-test cohort and 48-test review follow-up from earlier diagnostics. No new Compute Sanitizer result qualifies this changed binary.

The separate [eager second-reverse API](SECOND_ORDER.md) adds a native derivative of backward for one gradient-dependent training objective. Its installed nonlinear-head consumer compares two actual SGD updates with independent GPU math, including a detached-penalty control and a shared float64 omitted-cotangent witness. The [683-test installed cohort](evidence/second-order-windows-sm89-20261022.json) retains original first-order behavior on existing APIs. No new Compute Sanitizer result qualifies the changed binary.

The separate [stochastic resume consumer](STOCHASTIC.md) restores CUDA dropout RNG and a dedicated CPU shuffled-order generator, permutation, cursor and epoch alongside model parameters and SGD momentum. Its fresh process completes five updates after the checkpoint and crosses a real permutation rollover. Both dtypes passed exact native continuation, primitive comparisons and six causal omission controls. The [installed record](evidence/stochastic-resume-windows-sm89-20261022.json) qualifies that fixed trajectory using the unchanged accepted native wheel.

The original CUDA APIs do not inherit CPU second derivatives, HVP transforms, vmap, CUDA graph capture or other CPU example capabilities. The stochastic continuation above has its own fixed consumer contract. The separate second-reverse API above is eager packed ordinary autograd only. Bounded changing-shape packed training, padded APIs, fixed-shape first-order `aot_eager` training and strict inference-export examples retain their own contracts. Padded changing-shape training and padded export remain outside this route. No Inductor/Triton optimization, mixed precision, distributed training, other GPU backend or architecture, cross-version/platform/ABI, coinstallation matrix, performance or operational security result is claimed.

Source is MIT licensed. [NOTICE](NOTICE) and the included [PyTorch license](third_party/PYTORCH-LICENSE) preserve attribution and the truthful assistance statement. Documentation informs API choices. Research rationale is not measured implementation evidence. CPU studies and earlier adverse results retain their separate identities.


## Bounded functional model maps

The [functional bridge](FUNCTIONAL.md#map-three-independent-models) also supports one eager `vmap` and `vmap(grad)` layer over complete packed problems. Manual rules call native lanes serially, retaining original contiguous lane admission and saved-input mutation guards. Each rule invocation admits K1..8, K*N<=4096 and K*B<=128, including shared inputs. These are per-invocation bounds under chunking, with no total-work, memory or batching speed claim.

Use `examples/train_functional.py --models 3` for two actual SGD updates in three distinct stacked models, checked against serial ordinary-native and primitive CUDA routes in both dtypes. Shared raw-input sum/mean gradients have unequal model contributions. The [map cohort](evidence/functional-map-windows-sm89-20261022.json) records 812 passing CUDA tests, four existing skips, one warning and fifteen installed commands. This changes Python batching rules while reusing the accepted native binary. Nested maps, mapped JVP/higher derivatives, padded maps and transformed compilation/export remain excluded.
