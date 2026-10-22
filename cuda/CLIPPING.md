# Clip each group's gradient before averaging

`examples/train_clipped.py` uses the eager [CUDA functional bridge](FUNCTIONAL.md) to compute one scalar-loss parameter gradient per ragged group. It limits each complete gradient vector, averages the limited vectors, and performs two actual SGD updates. The fixed model has an encoder, tanh, scorer, native pool and scalar prediction head. A group is one example, regardless of its item count.

The CPU example already implements this training rule. This consumer qualifies the CUDA functional interface as a caller of that rule. It adds no algorithm, package API, runtime module, native derivative or CUDA kernel.

## The update

For group i, concatenate the gradients of all five named parameter blocks conceptually into g[i]. The implementation keeps those blocks named and uses float64 arithmetic:

```text
norm[i] = sqrt(sum of every parameter-gradient element squared)
scale[i] = 1                         if norm[i] == 0
           min(1, C / norm[i])       otherwise
mean_gradient[p] = sum_i(scale[i] * gradient[i,p]) / B
```

The demonstration fixes C=1, learning rate 0.03 and momentum 0, with foreach/fused false. Norms, scaling and accumulation use float64. Each completed mean is divided by global group count B and cast once to the model dtype. There is no epsilon adjustment. The helper refuses invalid thresholds, nonfinite gradients and overflowing squared norms. A nonzero vector whose squared norm underflows to zero is also refused rather than mistaken for an exact zero vector.

One norm covers all 25 scalar parameters. Clipping each parameter block separately defines another update. Blocks containing 3 and 4 have joint norm 5, so threshold 1 produces 0.6 and 0.8. Independent scalar clipping would produce 1 and 1. Clipping before the mean also differs from clipping the mean: opposing vectors can have a small average while each individual contribution still needs scaling. The executed controls retain those differences.

Empty groups count in B. Their pooled vectors and encoder/scorer/head-weight gradients are zero, but the scalar head bias can receive a loss gradient. In update one, the empty group's unmodified head-bias derivative is approximately 4.8. Omitting it or dividing by item count would change the intended optimizer input. Singleton groups have zero scorer-weight gradient because one softmax weight is always one.

## Functional calls and independent checks

The main route holds parameters fixed while processing groups serially. Each local packed slice gets offsets `(0, stop-start)`. Actual input offsets are checked against the declared full boundaries before rebasing. `functional_call(..., strict=True)` supplies the explicit parameter dictionary and `torch.func.grad(..., has_aux=True)` returns a group's parameter gradients and outputs. Those pure calls must preserve input and parameter bytes, version counters, objects, storage and existing nonzero `.grad` sentinels. Only after complete aggregation does the caller assign detached gradient copies and call its persistent SGD optimizer once.

The main helper retains one group's parameter-gradient blocks at a time, float64 sums and small output/norm/scale diagnostics. Acceptance oracles separately retain full rows. This is a statement about the named algorithm data, not a measured peak-memory reduction or hard memory cap. Serial differentiation repeats model work and has no assumed speed advantage.

The ordinary-native oracle evaluates the original full batch and differentiates individual loss rows. It does not repeat the functional route's slicing, so a rebasing mistake cannot pass merely because both routes slice the same way. Independent primitive GPU pooling supplies a second mathematical comparison. All per-group predictions, pools, losses and parameter gradients, completed clipped means and both updated parameter states are checked. The mean of unclipped rows must match the gradient of full-batch mean loss. Both updates must change every parameter block and preserve model/parameter/optimizer ownership.

The seeded fixtures contain five then six groups, with empty and singleton groups and both clipped and unclipped contributions at each update. Controls distinguish the result from unclipped training, clipping after averaging and per-block clipping. Tests add all-empty groups, different packed partitions, zero and opposing gradients, invalid batch metadata, sentinel preservation and refusal followed by valid recovery. Full-batch vectorized norm/mean calculations in the tests independently check the streamed arithmetic.

## Installed execution

Use the declared Windows x64 Python 3.11.8, Torch 2.9.1+cu128, CUDA runtime 12.8, RTX 4090/CUDA0 pin. Build/install through the [companion route](README.md), or use the exact qualified wheel. Copy `train_clipped.py`, `train_functional.py` and `train_cuda.py` from the checked source outside every checkout, then use new receipt paths:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MAX_JOBS = '2'
& $acceptPython -I C:\work\cuda-consumer\train_clipped.py --dtype float32 --output C:\work\clipped-f32.json
& $acceptPython -I C:\work\cuda-consumer\train_clipped.py --dtype float64 --output C:\work\clipped-f64.json
```

The [executed cohort](evidence/clipping-windows-sm89-20261022.json) passed six installed commands and 580 CUDA tests, including 28 new clipping cases. Four existing unavailable Windows symlink fixtures and one PT2 warning remain recorded. Both new consumers and both preceding functional model consumers passed. Registration checks remain distinct from mathematical checks.

This cohort reused the preceding accepted ordinary wheel and isolated external environment without changing dependencies. There was no new wheel installation, native compilation or CUDA instrumentation. All seven runtime payloads and the pinned dependencies were verified again. A tracked CUDA source archive supplied external consumer/test copies. Windows Git converted its 44 text members to CRLF; each member joined completely to its Git blob through only CRLF-to-LF conversion. Literal archive/consumer hashes remain separate from public Git LF hashes. The first archive assertion and six dependent wrapper startup failures remain recorded, without being counted as executed project tests.

## Limits

The [PyTorch 2.9 grad](https://docs.pytorch.org/docs/2.9/generated/torch.func.grad.html) and [functional_call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html) APIs support the selected scalar-loss/parameter-dictionary route. This caller is serial, eager and first order. It does not use vmap, batched gradients, higher derivatives or compile/export. The bridge's pinned internal `_version` mutation guard and inference-tensor exclusion remain as documented. The fixed model has independent deterministic groups. Dropout, batch normalization, mutable buffers and cross-group objectives require different reasoning.

Clipping alone does not make training differentially private or provide an adversarial defense. [Abadi et al., Algorithm 1](https://arxiv.org/pdf/1607.00133) also includes noise and privacy accounting. Neither is present here. The example makes no model-quality, novelty, production, security-operations, performance, peak-memory or version/platform/device/ABI compatibility claim.
