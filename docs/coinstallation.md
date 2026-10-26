# Install the CPU and CUDA packages together

The default CPU distribution requires `torch==2.9.1+cpu`. The CUDA companion requires `torch==2.9.1+cu128`. Those exact requirements cannot share an environment. Do not bypass the conflict with `--no-deps` when installing the two ordinary distributions.

The accepted historical CPU build variant uses `RAGGED_POOL_TORCH_VARIANT=cu128`, compiles the CPU extension against that exact framework, and produces `ragged_pool-0.1.0-1cu128-cp311-cp311-win_amd64.whl`. It still runs its pooling operations on CPU. The wheel build tag identifies its framework dependency separately from the project version. Both installed distributions then require the same cu128 framework.

This guide retains the qualified 0.1.0 pair. The current 0.2.0 checkout would build a 0.2.0 CPU wheel, which is outside that pair's acceptance and the strict cross-device resume admission. Use the immutable historical source below for this recipe. Do not rename a new wheel or weaken version checks to make it look like the accepted pair. The default 0.2.0 release remains CPU-only.

The build checks the selected Torch version and CUDA runtime before importing the extension builder. Its generated `_build_config.json` lives in the wheel, rather than the source tree. Import checks that configuration against the active framework before loading the CPU native library. Native builds force recompilation so switching flavors cannot reuse an earlier object file. These checks are a compatibility boundary, not authentication of untrusted packages or a cross-version ABI guarantee.

## Build and install

The [execution record](results/coinstallation-cu128.json) qualifies one Windows x64 / CPython 3.11.8 / Torch 2.9.1+cu128 / MSVC 19.44.35228, toolset 14.44.35207 route, with the unchanged accepted CUDA companion on an RTX 4090. CPU source compilation uses C++17 and the ATen/dispatcher API. The CPU C++ source is unchanged from the preceding revision, but this CPU binary was compiled afresh against cu128 from a checked source archive. The companion retains its accepted native and Python wheel bytes. No CUDA kernel change, new CUDA compiler result, performance result or broader platform compatibility is implied.

Use a separate build environment with the exact cu128 framework already installed. The [CPU execution source at `4fd3553f93ab59be0b5cef845aeb9f7bbe88adf4`](https://github.com/T92T1914/ragged-pool/tree/4fd3553f93ab59be0b5cef845aeb9f7bbe88adf4) declares version 0.1.0. Download that immutable Git snapshot into new owned paths without changing the current checkout:

```powershell
$cpuRevision = '4fd3553f93ab59be0b5cef845aeb9f7bbe88adf4'
$cpuArchive = 'C:\work\ragged-coinstall-source.zip'
$cpuExtract = 'C:\work\ragged-coinstall-source'
if ((Test-Path -LiteralPath $cpuArchive) -or (Test-Path -LiteralPath $cpuExtract)) { throw 'Use new source paths' }
Invoke-WebRequest -Uri "https://github.com/T92T1914/ragged-pool/archive/$cpuRevision.zip" -OutFile $cpuArchive
Expand-Archive -LiteralPath $cpuArchive -DestinationPath $cpuExtract
$cpuSource = Join-Path $cpuExtract "ragged-pool-$cpuRevision"
```

In a qualified x64 Visual Studio child command environment, build from `$cpuSource`, not `.` in the current checkout. The child environment must expose the existing MSVC 14.44 compiler and Windows SDK. The root [Windows build example](../README.md#build-and-run) shows the `vcvars64.bat` setup, while the immutable source above supplies this recipe's project version and code.

```powershell
& $buildPython -m pip --isolated install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $buildPython -m pip --isolated install setuptools==80.9.0 wheel==0.45.1
$env:RAGGED_POOL_TORCH_VARIANT = 'cu128'
$env:DISTUTILS_USE_SDK = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MAX_JOBS = '2'
& $buildPython -m pip --isolated wheel $cpuSource --no-build-isolation --no-deps --no-cache-dir --wheel-dir C:\work\coinstall-wheels
```

`--no-build-isolation` deliberately uses this exact prepared framework. The default isolated build requirements still select +cpu. `--no-deps` applies to wheel construction here. Runtime installation below uses normal dependency resolution. No CUDA toolkit is needed to compile this CPU extension. Building the companion from source has its own [CUDA toolchain requirements](../cuda/README.md#one-explicit-installation-route).

Create a new acceptance environment and consumer directory outside both source trees. Install cu128 from its official index, then install both exact wheel paths together. `$cpuWheel` must identify the 0.1.0 `1cu128` CPU wheel. `$cudaWheel` must identify the qualified 0.1.0 companion. Do not use the untagged +cpu wheel or substitute the 0.2.0 package in that pair. A rebuild produces new artifact bytes and needs its own acceptance. It does not inherit the historical wheel hashes or results merely because the version matches.

```powershell
py -3.11 -m venv C:\work\coinstall-accept
$acceptPython = 'C:\work\coinstall-accept\Scripts\python.exe'
& $acceptPython -m pip --isolated install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $acceptPython -m pip --isolated install $cpuWheel $cudaWheel --index-url https://download.pytorch.org/whl/cu128
& $acceptPython -m pip check
```

Copy [the immutable executed `train_backends.py`](https://github.com/T92T1914/ragged-pool/blob/4fd3553f93ab59be0b5cef845aeb9f7bbe88adf4/examples/train_backends.py) from `$cpuSource` into the separate consumer directory, then run it there with no checkout added to `PYTHONPATH`:

```powershell
New-Item -ItemType Directory -Path C:\work\coinstall-consumer | Out-Null
Copy-Item -LiteralPath (Join-Path $cpuSource 'examples\train_backends.py') -Destination C:\work\coinstall-consumer
& $acceptPython -I C:\work\coinstall-consumer\train_backends.py --dtype float32 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\coinstall-consumer\float32.json
& $acceptPython -I C:\work\coinstall-consumer\train_backends.py --dtype float64 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\coinstall-consumer\float64.json
```

`examples/train_backends.py` supplies two actual SGD updates through CPU native, CUDA native, CPU primitive and CUDA primitive routes. It checks complete predictions, pools, losses, raw-input and parameter gradients, and updated parameters for float32 and float64. It uses explicit device copies for its fixtures and model parameters. The pooling APIs themselves do not move inputs between devices. The installed example requires both exact wheel paths to join the loaded packages to their artifact bytes.

## Executed checks and a debugging case

Ordinary installation and `pip check` passed in a fresh external environment. Separate processes loaded both packages in either import order. Complete wheel RECORD, installed Python/native bytes and license files were checked against the exact artifacts and source. The CPU wheel contains 16 non-RECORD members and the reused CUDA wheel contains 15. Generated CPU configuration declares `cu128`, exact framework/runtime versions and `1cu128`; the loader refuses a different flavor before native loading.

Nine serial commands passed: installed identity, both import orders, 110 packaging checks, 30 CPU registration/compiler checks, 1,042 eligible CPU checks, 812 CUDA checks, and both dtype consumers. CUDA retained four unavailable Windows symlink fixtures and one existing PyTorch loader warning. The CPU subset retained one nested-tensor prototype warning. Registration checks test framework integration; mathematical gradient checks supply separate numerical evidence. Full default +cpu checks and hosted CPU instrumentation remain separate CI results, rather than evidence of hosted CUDA execution.

| Largest absolute difference across both updates and all route pairs | float32 | float64 |
| --- | ---: | ---: |
| Outputs, loss and raw-input gradients | 2.9802322387695312e-8 | 1.6653345369377348e-16 |
| Parameter gradients | 2.9802322387695312e-8 | 2.220446049250313e-16 |
| Updated parameters | 7.450580596923828e-9 | 1.3877787807814457e-17 |

These are tolerance comparisons for two seeded synthetic updates, not bitwise CPU/GPU equivalence, model quality or convergence. Every route changed all five parameter blocks while retaining its own model/optimizer objects and unchanged inputs. The complete per-route/per-step values and tolerances remain in the JSON record.

The old wheel pair first produced the expected ordinary `ResolutionImpossible`, without changing the environment. The first new installed identity attempt also stopped: its private verifier required literal equality for a reused CUDA Python source member whose accepted source archive had CRLF line endings. The corrected verifier first requires the exact accepted wheel-member hash, then complete literal or CRLF-to-LF source equality. Installed wheel bytes, native payloads and RECORD checks remain literal. The failed attempt and compiler's setuptools `Py_DEBUG` warning are retained. No numerical tolerance was changed to resolve either issue.

Existing CPU exported and training-resume artifacts retain their original `+cpu` environment admission. This opt-in package variant does not broaden those saved-artifact contracts. Its original two-update consumer adds no cross-device optimizer migration result. A separate [fresh-process handoff](cross-device-resume.md) now qualifies the fixed model and momentum in both directions using these same wheels. Neither route establishes AMP, distributed training, security hardening or model quality.

The wheel build tag follows [PyPA's wheel filename format](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention). Exact local framework pins follow the [version-specifier rules](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers). Setuptools documents [dynamic metadata](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#dynamic-metadata) and [custom build commands](https://setuptools.pypa.io/en/stable/userguide/extension.html). The native extension retains the pinned [PyTorch 2.9 API route](https://docs.pytorch.org/docs/2.9/cpp_extension.html).
