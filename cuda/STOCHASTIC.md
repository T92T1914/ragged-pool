# Continue CUDA dropout and shuffled training

`examples/resume_stochastic.py` saves one eager CUDA training run after update two, exits the producer, then restores a fresh process for updates three through seven. It preserves actual CUDA dropout state and shuffled-batch position alongside model weights and SGD momentum. This adds one fixed stochastic continuation to the [deterministic CUDA resume example](RESUME.md), using the same installed native operation.

## State needed for the next update

The seeded encoder/scorer/pool/head has five parameter tensors containing 25 scalars. Ordinary training dropout with probability 0.25 acts on its CUDA encoded values before scoring and pooling. SGD uses learning rate 0.03, momentum 0.8, no weight decay or Nesterov update, and explicit `foreach=False` and `fused=False`. Five fixtures vary packed group shapes and include empty and singleton groups.

A dedicated CPU `torch.Generator` produces a permutation of those five batches. The checkpoint after update two stores that generator, its current permutation, next position and epoch. Updates three through five consume the saved permutation's tail. Updates six and seven generate and consume a new permutation. The generator state therefore matters beyond the saved list.

The CUDA0 default generator supplies dropout randomness. Restoring its original seed would restart the sequence. Restoring its saved state continues from the next draw. The checkpoint contains model tensors, momentum buffers, training mode, completed-update metadata, order state and the current CUDA RNG bytes. Tensor snapshots own detached contiguous CPU storage. That explicit staging belongs to the consumer and does not change the native pooling operation's transfer contract.

The producer exits before the resumer starts. Both children finish before the parent constructs GPU fixtures and uninterrupted native/primitive baselines. Windows virtual-environment launcher and runtime IDs are checked using the existing provenance approach. One child has a 120-second timeout. If a timeout occurs, the consumer writes `timeout.json` and partial logs before attempting direct-launcher cleanup. Descendant completion remains unknown, so further GPU work must stop until the owned runtime is inspected. The two mocked timeout tests qualify those written halt/evidence paths, not whole-process-tree cleanup or a hard process-tree deadline.

## Admission before restoration

Only the fixed checkpoint after two completed updates is admitted. The existing target must already be in training mode and its optimizer must own the expected parameter objects, named order and options. Evaluation targets are refused without changing their mode. The caller may explicitly return the target to training before a valid restore.

Every saved model, optimizer, RNG, permutation, cursor, epoch and training field is checked before applying target state. The CUDA RNG form is an owned CPU uint8 tensor of exactly 16 bytes for this pinned release. A scratch CUDA generator validates those opaque bytes. A scratch CPU generator validates the order state, and its seeded permutation association is checked separately. Cross-field storage aliases are refused.

The pinned [CUDA generator implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L326-L373) sets the seed before validating its Philox offset. Trying malformed state on the live generator could therefore change the seed before throwing. The scratch check prevents that partial live change. Tests demonstrate the behavior on a scratch generator and require rejected restoration to preserve populated momentum, existing gradients, model bytes, training mode, sampler ownership, default CPU/CUDA RNG and current stream. A valid restore follows each malformed-state case.

Public model/optimizer state loading retains the original parameter objects. The admitted order state is applied next. Public [`torch.cuda.set_rng_state`](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.set_rng_state.html) restores CUDA0's dropout stream last, after construction, loads and admission. The global CPU stream is not part of this continuation because training uses CUDA dropout and the separate order generator.

The loader reads one bounded ordinary local file, verifies an expected SHA256 and deserializes those exact bytes with `weights_only=True, map_location="cpu"`. The expected digest must come from the trusted local workflow. It identifies bytes without authenticating a publisher. This is not hostile-checkpoint safety, an atomic persistence service or general crash recovery. No unrestricted loading fallback is used.

## Comparisons and controls

Every native resumed tensor record must match uninterrupted native execution byte for byte. Records include packed inputs, predictions, pooled vectors, loss, raw-input gradients, all parameter gradients, updated parameters, momentum, encoded/dropped observations, dropout masks, CUDA RNG bytes and order state. Source fixtures, initial model tensors and retained checkpoint bytes/versions/storage remain unchanged. Model and optimizer ownership remains stable.

The independent primitive GPU trajectory uses the same actual dropout draws and compares the complete seven-update sequence with float32 `rtol=2e-4, atol=2e-6` or float64 `rtol=2e-10, atol=2e-12`. The momentum recurrence is checked separately from optimizer output. Both native and primitive routes actually update parameters.

Six deliberately incomplete continuations are applied only after a valid restore:

| Changed dependency | Required witness |
| --- | --- |
| Restart dropout from its original seed | Same next batch, changed next dropout mask and forward result, then final parameter drift |
| Omit saved dropout state | Same next batch, changed next dropout mask and forward result, then final parameter drift |
| Reset only the order generator | Three exact remaining current-permutation updates, followed by changed post-rollover batch selection |
| Reset momentum | Exact next forward and gradients, followed by a changed first parameter update |
| Repeat the cursor | The declared wrong immediate batch, followed by final parameter drift |
| Skip the cursor | The declared wrong immediate batch, followed by final parameter drift |

These controls demonstrate missing recovery dependencies. They do not establish model quality, adversarial robustness, privacy or operational security.

## Run an installed copy

Use the companion's [qualified ordinary Windows installation](README.md). Copy `resume_stochastic.py`, `resume_training.py`, `train_compiled.py` and `train_cuda.py` together from a checked source archive to an external consumer directory. The installed `ragged_pool_cuda` package supplies native execution.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MAX_JOBS = '2'
& $acceptPython -I C:\work\cuda-consumer\resume_stochastic.py --dtype float32 --directory C:\work\stochastic-f32 --output C:\work\stochastic-f32.json
& $acceptPython -I C:\work\cuda-consumer\resume_stochastic.py --dtype float64 --directory C:\work\stochastic-f64 --output C:\work\stochastic-f64.json
```

Artifact directories and output receipt files must be new. The receipt retains checkpoint size/digest, real child process relationships, installed/source/native identities, every update's batch/cursor/RNG/mask identity, numerical comparisons and causal controls. Optional complete `--source-revision` and `--source-tree` labels accompany actual byte hashes. A caller label by itself does not prove source identity.

## Executed evidence and limits

The [validation record](evidence/stochastic-resume-windows-sm89-20261022.json) identifies the executed source, checked archive and fresh external ordinary installation. All 13 serial commands passed at one source revision. The complete CUDA suite reports 742 passed, four existing Windows symlink-privilege skips and one existing exported-loader non-writable-buffer warning. The stochastic consumers and preceding deterministic resume, second-reverse, JVP, functional and compiled consumers passed in both dtypes.

Both stochastic continuations match native records exactly. Primitive maximum differences are zero for float32 and about 2.22e-16 for float64. Each dtype completed seven real updates, including five after restoration and one actual permutation regeneration. Both checkpoints remain unchanged, at 12,351 and 12,479 bytes respectively. All six incomplete-state controls diverged with their declared causal witnesses.

This consumer reuses the accepted wheel/native binary built at b500a3bfd1f79cdc2ad03f93176871d4229b0066. Runtime, native, build and license source joins were checked, including explicit complete CRLF-to-LF handling where checkout formatting differs. There was no new native compilation, wheel build or instrumentation run. The original build used NVCC 12.8.93, MSVC 19.44.35228/toolset 14.44.35207, C++17 and sm89. Existing native and derivative contracts remain unchanged.

The declared environment is Windows x64, CPython 3.11.8, Torch 2.9.1+cu128/CUDA runtime 12.8 and RTX 4090/CUDA0. [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html) limits replay across releases and platforms. This example adds no cross-device/release guarantee, other GPU backend, DataLoader workers, Python/NumPy RNG recovery, scheduler, compiled state, distributed recovery, crash consistency, performance or memory claim. Hosted CPU CI remains separate from local CUDA acceptance. No binary release/tag is created for this source increment.
