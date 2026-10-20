# ragged-pool

`ragged_pool(values, logits, offsets)` combines variable length groups of feature vectors using softmax weights. The CPU forward and backward run in C++. Python supplies a readable PyTorch reference, dispatcher metadata and first order autograd registration.

This is a small framework integration and learning project. Ordinary PyTorch composition already expresses the operation. No algorithmic novelty, production requirement or general performance advantage is claimed.

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

The initial target is Windows x64, CPython 3.11.8, PyTorch 2.9.1+cpu and MSVC 19.44 with C++17. This is one pinned environment, with no cross-version ABI or other-platform compatibility promise. Building requires an existing Visual Studio C++ toolchain and Windows SDK. No CUDA toolchain is needed. Source and versioned package downloads belong to [this repository's releases](https://github.com/T92T1914/ragged-pool/releases).

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

## One set-learning and CPU cost study

`examples/learn_sets.py` trains a small synthetic bag-level classifier with native attention, reference attention, mean, max and sum pooling. It uses one fixed task, three paired initialization seeds, held-out in-range and longer-bag sets, and validation-only checkpoint selection. Attention adds 16 scorer parameters to the common 97-parameter encoder/head. In the recorded run, attention had the lowest mean BCE in both test sets. Longer-bag accuracy fell to 73.31%, below max pooling's 75.78%. These are descriptive synthetic results, with no real-world accuracy claim.

The accompanying `examples/cpu_cost.py` compares complete checked calls against the readable reference and an equivalent segmented PyTorch composition. Native had lower medians in the eight declared cases on the executed one-thread Windows environment, with substantial shared-machine variation. The native kernel was retained after the investigation. See the [fixed protocol, every model variant, timings, compiler outcome and reproduction commands](docs/cpu-study.md) and the [machine-readable results](docs/results/cpu-study.json).

For installed execution, copy `learn_sets.py`, `pool_baselines.py` and `cpu_cost.py` together outside the checkout after installing the ordinary wheel into a fresh isolated environment. Use the revision and wheel you actually built, and a new receipt path for each run.

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
$wheel = 'C:\work\ragged-pool\dist\ragged_pool-0.1.0-cp311-cp311-win_amd64.whl'
$revision = git -C C:\work\ragged-pool rev-parse HEAD
& $python -I "$consumer\learn_sets.py" --output "$consumer\study.json" --source-revision $revision --wheel $wheel
& $python -I "$consumer\cpu_cost.py" --timing --output "$consumer\timing.json" --source-revision $revision --wheel $wheel --budget-seconds 300
```

The study freezes data and training choices before held-out evaluation, records content digests and class/length histograms, checks native/reference gradients and an Adam update before training, and retains all selected seeds. Its optional `--smoke` mode performs 20 native/reference updates on training and validation only. It supplies a small CI check, distinct from both the full study and the original single-step consumer.

## What the checks mean

`tests/test_math.py` compares the native result and both input gradients with independently differentiated PyTorch composition and checks float64 finite differences. Contract checks cover malformed inputs, empty and singleton groups, storage, version rules and upstream gradient views. Registration checks use `torch.library.opcheck` separately from the mathematical tests.

The declared compiler acceptance path is `torch.compile(..., backend="aot_eager", fullgraph=True)` for forward and backward. It exercises graph capture and autograd integration. The separate CPU study also executed native Inductor forward/backward on one declared case in the pinned environment. Its equivalent built-in attempt failed during generated-code compilation, and no optimized speed advantage was established. Fake tensor implementations describe both forward and private native backward outputs without reading tensor contents. Native execution still validates offset contents and the numerical range.

Initial build and acceptance results are recorded in [the local acceptance note](docs/acceptance.md). The later [CPU study](docs/cpu-study.md) records its own source, installed wheel and bounded measurements. CUDA, other devices, float16/bfloat16, sparse inputs, arbitrary forward layouts, mixed precision, second derivatives and vmap are outside this version.

The [Windows CPU workflow](.github/workflows/cpu.yml) rebuilds the wheel, installs it into a fresh environment outside the checkout, checks package and license bytes, runs the test suite, and executes both one-step dtype consumers plus the validation-only study smoke. The actual run and revision determine CI status. A workflow file by itself is not an executed result.

[Native boundary checks](docs/native-robustness.md) describe generated mathematical/admission cases, a combined negative-view layout bug, and a separate Linux AddressSanitizer workflow. That workflow instruments the production extension and checks its loaded identity. It supplies bounded memory-diagnostic evidence when executed successfully, with prebuilt framework and leak-detection limits stated explicitly. Ordinary Windows wheel acceptance remains separate.

## License and attribution

Project source is licensed under [MIT](LICENSE). [NOTICE](NOTICE) records implementation assistance, upstream guidance and the separate PyTorch dependency license. The full PyTorch v2.9.1 main notice is retained with the wheel and source distribution. No PyTorch dependency binaries are bundled.

T set the project scope and acceptance requirements. The implementation, test authoring, source review and local command execution in this initial version used Codex assistance. The execution record establishes component behavior, not unaided expertise.
