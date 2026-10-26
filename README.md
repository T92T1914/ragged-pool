<a id="ragged-pool"></a>

# Ragged Pooling for PyTorch

Groups of feature vectors can contain different numbers of items. `ragged_pool(values, logits, offsets)` combines each group into one vector using softmax weights. C++ runs the CPU forward and backward operations. Python supplies a readable reference and connects the operation to PyTorch training.

This is a small framework integration and learning project. Ordinary PyTorch composition already expresses the operation. It does not claim a new algorithm, a production requirement or a general performance advantage.

[Read in another language](https://github.com/T92T1914/T92T1914/blob/main/TRANSLATION.md).

<picture>
  <source media="(prefers-color-scheme: dark) and (max-width: 680px)" srcset="docs/ragged-pool-obscur-narrow.png">
  <source media="(max-width: 680px)" srcset="docs/ragged-pool-clair-narrow.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/ragged-pool-obscur.png">
  <img src="docs/ragged-pool-clair.png" alt="Groups with two, three and no items produce one summary vector per group. Softmax weights determine the weighted sum. Empty groups produce zeros. Training sends gradients back to values and scores. Conceptual CPU example, not a benchmark." width="1600">
</picture>

Different group sizes produce one summary vector each. This is a conceptual illustration of the CPU layer, not a benchmark. [Diagram source](docs/render_pooling_example.py).

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

Start with the [v0.2.0 downloads and installed example](docs/install.md). The Windows CPU wheel installs without a compiler. It requires x64 CPython 3.11.8 and the exact PyTorch 2.9.1+cpu dependency. [Release notes](docs/releases/v0.2.0.md) explain what the assets contain and which results remain historical.

Building the Windows extension requires an existing Visual Studio C++ toolchain and Windows SDK, using MSVC 19.44 and C++17. The [ordinary Linux source distribution route](docs/linux-cpu.md) uses Ubuntu 24.04 x86-64 and GCC 13 with the same Python, PyTorch and C++ standard. Each route records its own CI result. Neither promises ABI compatibility across versions or compatibility with other platforms. No CUDA toolchain is needed for the default CPU package.

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
& $python -m pip install --no-deps dist\ragged_pool-0.2.0-cp311-cp311-win_amd64.whl
& $python -m pytest -q
& $python examples\train_sets.py
```

Adjust the Visual Studio path if your existing installation lives elsewhere. `vcvars64.bat` and `DISTUTILS_USE_SDK=1` apply to that child build process. They do not change the machine's global compiler settings. Native compilation is serial and the linker uses at most two code generation threads.

`examples/train_sets.py` uses explicitly synthetic groups, learned importance logits and a seeded model. It compares native and reference outputs, gradients and a real optimizer step from the same starting parameters. It demonstrates training integration. Its loss is not a model quality or accuracy result.

For an installed consumer check, create a second fresh virtual environment outside the checkout, install the same pinned dependencies and the wheel there, copy `examples/train_sets.py` to a separate directory and run it from that directory. Do not add the checkout to `PYTHONPATH`. The example reports its installed module path.

## Choose a demonstration

| What to inspect | Example and evidence |
| --- | --- |
| A native operation inside an ordinary model | `examples/train_sets.py` takes a real optimizer step and compares outputs, gradients and updated parameters with the readable reference. Begin with [the tensor contract](docs/contract.md) and [initial acceptance](docs/acceptance.md). |
| Derivatives and framework integration | [Native second derivatives](docs/second-order.md) support an input-gradient penalty. The separate [functional HVP example](docs/hessian-vector-products.md) checks how a trained loss gradient changes along a direction. Each route has its own admitted composition. |
| A distributed training update | Start with [two-rank Linux CPU training](docs/distributed-cpu.md), then follow the [shared numerical admission example](docs/distributed-admission.md). Unequal groups, wrong-objective controls and a deliberate post-backward fault expose different correctness questions. |

The distributed examples use cooperating CPU workers on one qualified Linux
host. Windows Gloo failed before training and remains adverse evidence.
Numerical agreement between live ranks does not establish crash recovery,
malicious-peer protection, secure transport or multi-node/GPU training.

## More examples

The [examples guide](docs/examples.md) keeps the detailed descriptions, commands
and limitations together. Use the route that matches the question:

<a id="pool-a-padded-batch"></a>

- [Pool padded batches](docs/examples.md#pool-a-padded-batch), including masking before arithmetic.

<a id="train-across-packed-shapes"></a>

- [Train across packed shapes](docs/examples.md#train-across-packed-shapes) through the declared dynamic compiler route.

<a id="resume-packed-training-in-a-fresh-process"></a>

- [Resume CPU training in a fresh process](docs/examples.md#resume-packed-training-in-a-fresh-process), including stochastic state.

<a id="recompute-activations-during-backward"></a>

- [Recompute activations](docs/examples.md#recompute-activations-during-backward) during backward.

<a id="train-with-an-input-gradient-penalty"></a>

- [Train with an input-gradient penalty](docs/examples.md#train-with-an-input-gradient-penalty).

<a id="train-with-per-example-clipped-gradients"></a>

- [Train with per-example clipped gradients](docs/examples.md#train-with-per-example-clipped-gradients).

<a id="follow-an-input-direction"></a>

- [Follow an input direction](docs/examples.md#follow-an-input-direction) with first-order forward AD.

<a id="functional-model-gradients"></a>

- [Compute functional model gradients](docs/examples.md#functional-model-gradients) through the bounded reverse/map bridge.

<a id="inspect-loss-curvature-in-one-direction"></a>

- [Inspect loss curvature](docs/examples.md#inspect-loss-curvature-in-one-direction) with a Hessian-vector product.

<a id="solve-one-fixed-curvature-system"></a>

- [Solve one fixed curvature system](docs/examples.md#solve-one-fixed-curvature-system) without applying the solution as an update.

<a id="one-set-learning-and-cpu-cost-study"></a>
<a id="one-synthetic-training-study-and-cpu-cost-comparison"></a>

- [Read the synthetic study and CPU cost comparison](docs/examples.md#one-synthetic-training-study-and-cpu-cost-comparison), including the weaker longer-bag accuracy and compiler result.

<a id="export-the-trained-consumer"></a>

- [Export the trained packed consumer](docs/examples.md#export-the-trained-consumer).

<a id="export-a-trained-padded-model"></a>

- [Export a trained padded model](docs/examples.md#export-a-trained-padded-model).

The separate [CUDA companion](docs/examples.md#cuda-companion) and
[CPU/CUDA co-installation route](docs/examples.md#cpu-and-cuda-in-one-environment)
have their own dependency pins, qualified Windows RTX 4090 execution and
sanitizer limits. The co-installation and cross-device resume guides retain the
exact 0.1.0 wheel pair and link immutable historical source recipes. The 0.2.0
CPU release does not qualify a replacement pair or expand the CPU wheel's contract.

## What the checks mean

Mathematical tests compare native outputs and input derivatives with independent
PyTorch composition and float64 finite differences. Operator registration checks
use `torch.library.opcheck` separately. Contract tests cover malformed inputs,
empty groups, storage and saved-input rules, and strided or expanded upstream
gradients.

The declared `aot_eager` paths exercise graph capture and autograd integration.
Fake tensor implementations describe forward and private backward outputs
without reading numerical contents. These checks do not imply optimized code
generation or a speed advantage. The [full verification notes](docs/examples.md#what-the-checks-mean)
retain the separate Inductor outcome, generated cases and native diagnostic limits.

The [Windows wheel](.github/workflows/cpu.yml),
[Linux source-to-wheel](.github/workflows/linux-cpu.yml) and
[Linux AddressSanitizer](.github/workflows/native-asan.yml) workflows each record
their actual tested revision, environment and result. Installed consumers run
outside the checkout. A workflow definition is not an executed result, and a
successful pinned build is not cross-version ABI or platform qualification.

Load only trusted model artifacts. Byte digests identify contents but do not
authenticate the publisher, and exported graphs are not a sandbox. The
[packed](docs/exported-consumer.md) and [padded](docs/padded-export.md) handoffs
declare their own input and loading boundaries. Small synthetic training runs
demonstrate integration, with no general accuracy, privacy, security or performance
claim.

## License and attribution

Project source is licensed under [MIT](LICENSE). [NOTICE](NOTICE) records implementation assistance, upstream guidance and the separate PyTorch dependency license. The full PyTorch v2.9.1 main notice is retained with the wheel and source distribution. No PyTorch dependency binaries are bundled.

T set the project scope and acceptance requirements. The implementation, test authoring, source review and local command execution in this initial version used Codex assistance. The execution record establishes component behavior, not unaided expertise.
