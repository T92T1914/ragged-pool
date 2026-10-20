# Local CPU acceptance

Goal A is accepted locally for the bounded CPU contract. The tested implementation revision is `87202e4239df4e9525788034805842ea774c63ce`. This note was added after execution without changing implementation, tests, build metadata or consumer code.

## Executed result

| Evidence | Result and limit |
| --- | --- |
| Native build | Built C++ CPU forward and backward into a Windows x64 wheel with MSVC 19.44.35228, C++17, `/O2`, `/fp:precise`, and `/CGTHREADS:2`. Compile is serial. |
| Environment | CPython 3.11.8, PyTorch 2.9.1+cpu, Windows x64. Dependencies are pinned in `requirements-lock.txt`. The CPU wheel has no CUDA runtime build. |
| Mathematical checks | 35 passed. Both dtypes, independent reference autograd for both gradients, analytic example, float64 finite differences, constant/extreme logits, range endpoints, empty groups/batches, D=1 and maximum declared dimensions. |
| Contract checks | 89 passed. Invalid offsets/shapes/ranges/dtypes/layouts, Meta rejection without transfer, fresh storage, nonmutation, saved-tensor version checks, logical views and explicit double-backward refusal. |
| Registration and compiler | Four passed. Separate opcheck for forward and private backward, fake metadata for both, `aot_eager`, `fullgraph=True`, `dynamic=False` forward/backward against eager. Invalid runtime offsets also fail through the compiled call. |
| Installed consumers | Float32 and float64 passed from outside the checkout using the final wheel in a fresh isolated virtual environment. Each performed one SGD step through learned logits, compared native/reference gradients and updates, and changed scorer/head weights. |
| Dependency and diff checks | `pip check` and `git diff --check` passed. |
| Independent review | A separate agent inspected native semantics, integration, test strength and public claims at the tested revision. No consequential defect remained in that inspected scope. This is source review, separate from executed tests. |

The final wheel is `ragged_pool-0.1.0-cp311-cp311-win_amd64.whl`, SHA256 `965455c8ebd22b4c4e9e39ba8d3fa83d31a6f10191e65184eb979306bec83045`. The native source SHA256 is `11a623c1cde80343611672c06d585d9bd4efcf650eecc473e1ee90c3ab7fe3ac`. Installed Python and native library bytes were compared with the wheel. Packaged Python bytes also matched this source revision, and the outside consumer copy matched `examples/train_sets.py`.

The seeded float32 consumer's maximum pooled difference from the reference was `1.1920928955078125e-7`, its maximum feature-gradient difference was `7.450580596923828e-9`, and updated model parameters matched after one step. The scorer's largest parameter change was `0.013225138187408447`. These are synthetic integration observations, not model accuracy, general numerical error bounds or performance measurements. Float64 had a maximum pooled difference of `2.220446049250313e-16` in its separately initialized example.

## Retained adverse results

The delivery includes the actual build logs, development failures and final runner reports. The first build succeeded after an early compiler-probe warning. An explicit compiler-environment attempt then stopped because `DISTUTILS_USE_SDK` was unset. Setting it within the child build process produced the successful native rebuild.

Initial tests exposed an unnecessary negative-view rejection rule. PyTorch resolved these views before CPU admission, so three rejection tests failed. Two attempted Python guards then broke the declared fullgraph path, first through a lambda check message and then through a boolean `is_neg()` query. The final implementation uses standard CPU dispatcher resolution and tests the logical values and gradients. The corrected full suite passed all 128 cases.

The first revision-linked acceptance report, `final-checks/result.json`, retains overall status `failed`: its identity helper incorrectly expected six package files rather than five. All mathematical, contract, registration/compiler, consumer, dependency and diff checks in that report passed. The helper had already compared the available files byte for byte before failing its incorrect count assertion. The corrected helper verifies the actual expected filenames and passed in the separate `final-identity/result.json` without package changes. The failed report has not been relabeled or overwritten.

An early preparation environment also installed an obsolete candidate through a raw `pip freeze` package URL. It was not used for final acceptance. The final `acceptance-env` was created separately, checked for package absence, and installed from the selected wheel using a dependency lock that excludes the package itself.

## Boundaries

The Code Review capability exposed for this task reads PR CI diagnostics. There was no semantic review action for this local repository, so independent agent source review was used and labeled accordingly. No remote CI ran.

No benchmark, CUDA execution, GPU profiling, other device backend, Inductor optimization, broad platform matrix or cross-version ABI trial was performed. Those are outside this CPU capability, not unfinished requirements for it. No foreground application check was needed. SHARED_PC remained active throughout file and command execution. No remote repository, publication or public profile edit is included.
