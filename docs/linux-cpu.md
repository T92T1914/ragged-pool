# Ordinary Linux CPU source distribution route

This route builds the existing CPU implementation with GCC 13, `-O2` and C++17 on Ubuntu 24.04 x86-64. It uses CPython 3.11.8, PyTorch 2.9.1+cpu and the existing pinned dependencies. It keeps the ATen `CppExtension` API and records the installed framework's ABI setting with the executed toolchain facts. No native algorithm or public tensor contract changes.

The [Linux CPU workflow](../.github/workflows/linux-cpu.yml) is the executable recipe. Its [revision-specific runs](https://github.com/T92T1914/ragged-pool/actions/workflows/linux-cpu.yml) provide the actual execution evidence. Select a completed successful run and inspect its source revision, logs and retained artifacts. Artifacts expire after 14 days. The original Windows release assets are separate and are not replaced by this workflow.

## Build from the archive

Use trusted project source and a separate build environment. The existing GCC 13 C++ toolchain is a prerequisite. These Bash commands show the source-to-wheel steps from the checkout. Use new output directories and CPython 3.11.8 for `python3.11`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2
export CC=gcc-13 CXX=g++-13
export PYTHONNOUSERSITE=1
unset PYTHONPATH RAGGED_POOL_ASAN LD_PRELOAD
python3.11 -m venv /tmp/ragged-build
build_python=/tmp/ragged-build/bin/python
"$build_python" -m pip --isolated install 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$build_python" -m pip --isolated install -r requirements-lock.txt
"$build_python" -I -c 'from setuptools.build_meta import build_sdist; build_sdist("/tmp/ragged-sdist")'
"$build_python" -I scripts/check_sdist.py --checkout "$PWD" --sdist /tmp/ragged-sdist/ragged_pool-0.1.0.tar.gz --extract /tmp/ragged-source --output /tmp/ragged-sdist.json
cd /tmp
"$build_python" -m pip --isolated wheel /tmp/ragged-sdist/ragged_pool-0.1.0.tar.gz --no-build-isolation --no-deps --wheel-dir /tmp/ragged-wheels
```

The backend call is the supported PEP 517 `build_sdist` hook in a fresh Python process, using the already installed pinned build requirements. Pip then builds from the archive. This is deliberately one route, without a custom backend or another build frontend. [PEP 517](https://peps.python.org/pep-0517/) and the [pinned setuptools backend](https://github.com/pypa/setuptools/blob/v80.9.0/setuptools/build_meta.py) describe those interfaces.

The archive includes the Python and C++ implementation, tests and their helpers, consumers, dependency pins and license/attribution files. The checker compares those bytes with the source, rejects ambiguous paths and previously built binaries, then extracts into a new owned directory. Baseline source packaging omitted test helpers even though it included the `test_*.py` files. Explicit manifest entries fix that practical reproduction gap.

## Installed acceptance

Create a distinct `/tmp/ragged-accept` environment and install the same dependencies and the single wheel. Copy `tests`, `examples` and the two checker scripts from `/tmp/ragged-source` to a new consumer directory. Run from that directory, outside both source trees, with `python -I`. The workflow supplies the exact commands and preserves their exit status.

Acceptance covers the installed package and license bytes, actual loaded native library, full contract/mathematical/registration suite, both float32 and float64 optimizer-step examples, the validation-only study smoke, and the full 200-update trained export producer followed by its separate prediction process. The consumer checks all eight declared cases. It retains the existing export limit `2 <= B <= 128`; the eager operation keeps its wider group-count contract. This route does not repeat the held-out model-quality study or benchmark results on another platform.

Compiler version, libc and libstdc++ facts, build flags, wheel tag and archive/wheel/native hashes belong to the run receipt. Ordinary-library inspection rejects AddressSanitizer symbols/dependencies, and execution uses no sanitizer preload. The existing [ASan workflow](../.github/workflows/native-asan.yml) remains a separate diagnostic route with its own limitations.

## Limits

A `cp311-cp311-linux_x86_64` wheel tested on this runner establishes that environment. It is not retagged manylinux and does not establish compatibility with arbitrary Linux distributions, compilers or PyTorch versions. [PyPA's platform tag specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/) explains the broader policies required for manylinux.

One successful source/archive build and installed consumer run does not establish byte-for-byte deterministic rebuilds, an external independent replication or protection against a compromised build host. Hashes identify bytes; they do not authenticate a publisher. Builds execute project/backend code and exported graphs remain trusted inputs. The loading boundary in [the exported-consumer note](exported-consumer.md) still applies.
