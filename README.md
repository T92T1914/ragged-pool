# ragged-pool

`ragged_pool(values, logits, offsets)` combines variable length groups of feature vectors using softmax weights. The CPU forward and backward run in C++. Python supplies a readable PyTorch reference, dispatcher metadata and first order autograd registration.

This is a small framework integration and learning project. Ordinary PyTorch composition already expresses the operation. No algorithmic novelty, production requirement or performance advantage is claimed.

```python
import torch
from ragged_pool import ragged_pool

values = torch.tensor([[2., 0.], [0., 4.]], requires_grad=True)
logits = torch.tensor([0., 1.0986123], requires_grad=True)
offsets = torch.tensor([0, 2, 2], dtype=torch.int64)
pooled = ragged_pool(values, logits, offsets)
# Approximately [[0.5, 3.0], [0.0, 0.0]]. The second group is empty.
pooled.sum().backward()
```

The bounded interface accepts contiguous CPU float32 or float64 values `[N,D]`, matching logits `[N]`, and contiguous int64 offsets `[B+1]`. Offsets begin at zero, end at N and are nondecreasing. Bounds are N <= 4096, B <= 128 and 1 <= D <= 128. Values and logits must be finite and within `[-1000,1000]`. Empty groups and an entirely empty batch are supported. Outputs use fresh storage. Inputs are neither changed nor moved between devices.

The native code uses maximum subtraction for stable softmax and float64 accumulation for either input dtype. It casts results back to that dtype. The reference uses ordinary PyTorch softmax and reductions, so floating point rounding can differ. See [the contract](docs/contract.md) for derivatives, numerical checks and excluded behavior.

## Build and run

The initial target is Windows x64, CPython 3.11.8, PyTorch 2.9.1+cpu and MSVC 19.44 with C++17. This is one pinned environment, with no cross-version ABI or other-platform compatibility promise. Building requires an existing Visual Studio C++ toolchain and Windows SDK. No CUDA toolchain is needed.

From PowerShell at the checkout, create an isolated environment. Replace the example environment paths as needed. Install PyTorch from its official CPU wheel index before building. The explicit `--no-build-isolation` uses that pinned environment, and `--no-deps` prevents the package build from choosing another framework wheel.

```powershell
py -3.11 -m venv C:\work\ragged-build
$python = 'C:\work\ragged-build\Scripts\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MAX_JOBS = '2'
& $python -m pip install 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements-lock.txt
$build = 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set DISTUTILS_USE_SDK=1 && "C:\work\ragged-build\Scripts\python.exe" -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist'
cmd.exe /d /c $build
& $python -m pip install --no-deps dist\ragged_pool-0.1.0-cp311-cp311-win_amd64.whl
& $python -m pytest -q
& $python examples\train_sets.py
```

Adjust the Visual Studio path if your existing installation lives elsewhere. `vcvars64.bat` and `DISTUTILS_USE_SDK=1` apply to that child build process. They do not change the machine's global compiler settings. Native compilation is serial and the linker uses at most two code generation threads.

`examples/train_sets.py` uses explicitly synthetic groups, learned importance logits and a seeded model. It compares native and reference outputs, gradients and a real optimizer step from the same starting parameters. It demonstrates training integration. Its loss is not a model quality or accuracy result.

For an installed consumer check, create a second fresh virtual environment outside the checkout, install the same pinned dependencies and the wheel there, copy `examples/train_sets.py` to a separate directory and run it from that directory. Do not add the checkout to `PYTHONPATH`. The example reports its installed module path.

## What the checks mean

`tests/test_math.py` compares the native result and both input gradients with independently differentiated PyTorch composition and checks float64 finite differences. Contract checks cover malformed inputs, empty and singleton groups, storage, version rules and upstream gradient views. Registration checks use `torch.library.opcheck` separately from the mathematical tests.

The declared compiler path is `torch.compile(..., backend="aot_eager", fullgraph=True)` for forward and backward. It exercises graph capture and autograd integration. It does not establish Inductor support or an optimized execution speed. Fake tensor implementations describe both forward and private native backward outputs without reading tensor contents. Native execution still validates offset contents and the numerical range.

Build and acceptance results are recorded in [the local acceptance note](docs/acceptance.md). No benchmark has been run. CUDA, other devices, float16/bfloat16, sparse inputs, arbitrary forward layouts, mixed precision, second derivatives and vmap are outside this version.

T set the project scope and acceptance requirements. The implementation, test authoring, source review and local command execution in this initial version used Codex assistance. The execution record establishes component behavior, not unaided expertise.
