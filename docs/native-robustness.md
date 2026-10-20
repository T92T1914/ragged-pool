# Native boundary checks

The bounded CPU operation accepts tensors from code already running in its process. It is not a sandbox for untrusted Python, models or native libraries. Malformed-argument rejection and memory diagnostics provide robustness evidence. They do not by themselves establish a remotely exploitable vulnerability or a secure service. This distinction agrees with [PyTorch's security policy](https://github.com/pytorch/pytorch/blob/9544ec5e23cef8a0a4ac3f44c27e0e94d651ad02/SECURITY.md).

## A combined view-state bug

The v0.1.0 installed Windows package admitted a stepped, noncontiguous forward input when that tensor also carried a logical negative bit. This was reproduced separately for values, logits and offsets. The ordinary noncontiguous case and contiguous negative case were already covered. Their combination was missing.

PyTorch's [v2.9.1 Negative fallback](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/MathBitsFallback.h) can clone a negative view before CPU redispatch. A stepped slice can become contiguous during that clone. The CPU kernel therefore saw an admitted layout after the original unsupported layout had disappeared.

The repair registers Negative-key handlers for both native operations. They validate original metadata before resolving negative views, then delegate to the existing CPU implementation. This keeps contiguous negative inputs and strided upstream gradients useful while protecting direct dispatcher entrypoints as well as the public function. It does not widen the forward contract or introduce another custom operation.

## Generated checks and mathematical properties

`tests/generated_cases.py` contains the deterministic case engine shared by pytest and the bounded campaign. It compares valid native output, autograd and direct private-backward results with independently differentiated reference composition. Malformed admission cases exercise both native operations, including extreme offsets and combined view states. Inputs are bounded before allocation. Category counters describe the cases executed, not branch or edge coverage.

A useful example is a consistent within-group permutation. Reordering both values and their logits preserves the weighted sum. The corresponding input gradients follow the same permutation. A logit shift is valid only when the shifted input remains within the finite numerical range. Uniform logits provide an independent mean-pooling check. These relations supplement reference comparisons and finite differences.

Consensus helped locate Segura et al.'s [survey on metamorphic testing](https://eprints.whiterose.ac.uk/id/eprint/110335/1/segura16-tse.pdf). Its discussion of relation preconditions and approximate numerical comparisons informs this test design. It supplies no measured fault-detection result for ragged-pool.

A wider uninstrumented seed sweep exposed a useful test-oracle problem at seed 184. The float32 PyTorch reference's score gradient differed from native by 3.14e-6 near cancellation, which exceeded the original absolute tolerance. Against the same reference composition run in float64 on the exact float32 inputs, native differed by at most 4.57e-8. Generated float32 gradient checks now use that higher precision derivative cast to float32; the float32 reference still checks forward output. This is a numerical test correction, not a tolerance increase or evidence of a native defect.

## Instrumented execution

The [native ASan workflow](../.github/workflows/native-asan.yml) selects Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GNU C++ 13 and C++17. The actual compiler and runner image belong to each run's log. It builds the same production extension with `RAGGED_POOL_ASAN=1`, using `-O1 -g -fno-omit-frame-pointer -fsanitize=address` and an ASan link flag. The matching shared runtime is resolved through that compiler and preloaded before Python starts. Ordinary wheels remain separate and uninstrumented.

The [first Linux PR run](https://github.com/T92T1914/ragged-pool/actions/runs/39384343118) used Clang 18. Its instrumented wheel built and its injected control reported the expected heap error, but the production extension could not load because a `TensorBase::const_data_ptr<float>` symbol was unresolved. No production sanitizer result follows from that run. The workflow now selects the runner's existing GNU C++ 13 and matching `libasan` to test another compiler route against the installed PyTorch binary. Only a subsequent executed run can establish whether that route works.

The [first GNU C++ 13 run](https://github.com/T92T1914/ragged-pool/actions/runs/39384861553) also built and passed the injected control. Its production step exited before pytest emitted a test result, and the campaign did not start. This failure is retained without assigning a cause. The next run separates an ASan-preloaded import probe from pytest and leaves pytest output uncaptured to expose the failure.

The job installs its instrumented wheel outside the checkout. It first qualifies the runtime/shared-library loading route with a disposable injected heap error, requiring the expected ASan diagnostic and a nonzero exit. This control is not a discovered project defect. The production campaign separately checks the loaded extension path, SHA256, dispatcher load identity, ASan symbols and runtime activation. It executes the existing tests, generated cases and both installed training consumers.

PR runs request a ten-minute single-worker campaign followed by a five-second replay. Main-branch and manually dispatched runs use a five-second campaign and replay. Seeds begin at 20261020. Tensor and case bounds are fixed in the case engine. The process uses one PyTorch thread, at most two build jobs, a four-GiB sampled RSS budget and a finite job timeout. RSS is checked between bounded cases, not enforced as a hard OS memory cap. An in-flight seed file and checkpointed JSON preserve the active case if native code terminates Python. A running or interrupted receipt is not a pass.

Instrumentation covers this extension's compiled native code. Python and the prebuilt PyTorch dependency are not rebuilt with ASan. Leak detection is disabled explicitly. This is an AddressSanitizer campaign, not general undefined-behavior checking, coverage-guided fuzzing, an edge-coverage measurement or proof of memory safety. No suppressions or relaxed runtime-order checks are used. [GCC 13 AddressSanitizer options](https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Instrumentation-Options.html).

The actual run and revision determine execution status. Workflow source is not a successful campaign. The installed Windows release-wheel checks remain separate from this Linux instrumentation configuration. No Linux distribution or cross-version ABI promise follows from it. The local MSVC 14.44 installation had x86 ASan components but no discovered x64 runtime, so local Windows sanitizer execution was not claimed.
