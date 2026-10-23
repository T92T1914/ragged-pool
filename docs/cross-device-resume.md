# Continue training on the other device

`examples/resume_backends.py` continues one small deterministic training run across CPU and CUDA in fresh processes. Its producer completes two SGD momentum updates, writes an owned CPU snapshot, and exits. A new target process constructs the model on the other device, creates its optimizer, restores weights and momentum, then takes two further updates. CPU-to-CUDA0 and CUDA0-to-CPU passed separately in float32 and float64 through the [qualified co-installed packages](coinstallation.md).

This is a consumer of the existing native operations. Device copies are explicit caller actions. The pooling APIs retain their input and device contracts. The example adds no runtime API, native kernel or general checkpoint service.

## What crosses the boundary

The example reuses the encoder, scorer, pooling layer, head and two changing packed batches from `train_backends.py`. The five parameter blocks contain 25 scalars. Four updates alternate between N=9/B=5 and N=12/B=6, including empty groups. Raw feature width is 3 and hidden width is 4. Seed 1811 fixes initialization and fixtures. Forward has no stochastic operations, so this example does not recover a live random-number stream.

The checkpoint holds five model tensors and five momentum tensors, all finite, contiguous, independently owned CPU storage, plus an exact schema and cursor. Metadata binds ordered parameter names/shapes/dtype, optimizer options, direction, fixtures, consumer/helper bytes, exact wheels and native libraries. Optional source revision/tree fields are caller labels joined separately to the executed source archive, rather than self-authenticating source claims.

For these optimizer options, momentum changes the update to:

```text
buffer = 0.8 * previous_buffer + gradient
parameter = parameter - 0.03 * buffer
```

The first buffer starts from the first gradient. Dampening, weight decay and Nesterov are disabled. Foreach, differentiable and fused SGD are false. A model with correct weights but missing momentum can produce the correct next prediction and gradient, then take the wrong update. Optimizer state is part of this training trajectory.

The target model is placed before optimizer construction. Admission validates all known schema fields, tensors and live parameter bindings before either load. The model uses `load_state_dict(..., strict=True, assign=False)`. The optimizer receives cloned state. Exact checks inspect restored values, target device/dtype, stable parameter objects and storage disjoint from the checkpoint. PyTorch 2.9.1 maps optimizer state by parameter order and casts floating state to each target parameter's device/dtype. Parameter names alone do not control loading. See the [pinned implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/optimizer.py) and [loading API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html).

## Run the installed consumer

Use the fresh external environment and exact CPU `1cu128`/CUDA wheel pair from the [co-installation instructions](coinstallation.md). The default +cpu wheel is a different dependency contract. Keep both `examples/train_backends.py` and `examples/resume_backends.py` beside each other in a separate consumer folder. Do not put the checkout on `PYTHONPATH`. Each invocation needs a new scenario directory and output path. It retains subprocess logs, checkpoint bytes and full numerical receipts.

```powershell
$acceptPython = 'C:\work\coinstall-accept\Scripts\python.exe'
# Set $cpuWheel and $cudaWheel to the exact qualified wheel paths.
& $acceptPython -I -B C:\work\coinstall-consumer\resume_backends.py --dtype float32 --direction cpu-to-cuda --directory C:\work\handoff-cpu-cuda-f32 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\handoff-cpu-cuda-f32.json
& $acceptPython -I -B C:\work\coinstall-consumer\resume_backends.py --dtype float64 --direction cpu-to-cuda --directory C:\work\handoff-cpu-cuda-f64 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\handoff-cpu-cuda-f64.json
& $acceptPython -I -B C:\work\coinstall-consumer\resume_backends.py --dtype float32 --direction cuda-to-cpu --directory C:\work\handoff-cuda-cpu-f32 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\handoff-cuda-cpu-f32.json
& $acceptPython -I -B C:\work\coinstall-consumer\resume_backends.py --dtype float64 --direction cuda-to-cpu --directory C:\work\handoff-cuda-cpu-f64 --cpu-wheel $cpuWheel --cuda-wheel $cudaWheel --output C:\work\handoff-cuda-cpu-f64.json
```

Run serially. The script uses one CPU/inter-op thread, completes the producer before launching the resumer, and runs parent numerical oracles after both children exit. It distinguishes Windows virtual-environment launcher IDs from actual Python process IDs. A timeout records a stop marker and terminates its direct launcher with a bounded cleanup wait. It does not claim arbitrary descendant process-tree cleanup. Inspect owned processes before retrying a timeout.

## Executed checks

The [result record](results/cross-device-resume.json) identifies execution revision `3c471c30090050bbcf84107a11bede56c34835f7` and tree `d186485a727109aea68487eb357597f81f2da505`. Documentation was completed afterward without changing executed source. The pin is Windows x64, CPython 3.11.8, PyTorch 2.9.1+cu128, CUDA runtime 12.8 and one RTX 4090 (`sm_89`). Both existing wheels were installed ordinarily in a new environment outside the checkout. `pip check`, complete wheel RECORD/installed byte checks, loaded native identities and licenses passed. No new native build or instrumentation run was performed for this consumer.

All four cases passed exact restoration and separate tolerant continuation checks. The native target oracle copies named checkpoint tensors and assigns momentum directly instead of calling the restore helper. An independent primitive model supplies a second comparison, including the source prefix and explicit target handoff. Every step checks complete outputs/loss/raw-input gradients, parameter gradients, updated parameters and momentum. An independent SGD recurrence checks each optimizer update. All five parameter blocks changed while inputs, initial templates and parameter/optimizer identities stayed intact.

| Direction | Dtype | Largest native continuation difference | Largest primitive difference |
| --- | --- | ---: | ---: |
| CPU to CUDA0 | float32 | 0 | 0 |
| CPU to CUDA0 | float64 | 0 | 2.220446049250313e-16 |
| CUDA0 to CPU | float32 | 0 | 0 |
| CUDA0 to CPU | float64 | 0 | 2.220446049250313e-16 |

Float32 comparisons use rtol=2e-4/atol=2e-6. Float64 uses rtol=2e-10/atol=2e-12. These observations do not establish general bitwise CPU/GPU equivalence. Full per-step error fields, checkpoint identities, controls and raw receipt digests are public. Large tensor trajectories, local paths and ephemeral process IDs are omitted explicitly. Complete receipts and checkpoint bytes remain retained locally.

Two deliberate live-state controls run after a valid restore: reset momentum, or swap the same-shaped scorer/head momentum buffers. Both preserve the next forward result and gradients exactly, but diverge on the first and final parameter updates. Across the four cases, final maximum parameter error was at least 0.04139043018221855 for reset and 0.016418881714344025 for swap. These are sensitive counterexamples for this fixture, not an attack-detection guarantee.

Eleven serial check groups passed. New CPU-only schema/admission fixtures contributed 52 passes and one unavailable Windows symlink fixture. Existing checks supplied 110 packaging passes, 30 registration/compiler passes, 1,042 eligible CPU passes and 812 CUDA passes, with four further symlink skips. Both old co-installation consumers and all four new handoffs passed. Existing nested-tensor prototype and export-loader buffer warnings remain recorded. Registration checks are separate from mathematical gradient checks. Synthetic metadata in CPU-only admission fixtures is not CUDA execution evidence.

## Trust and limits

The bounded loader reads a regular, non-symlink/non-reparse checkpoint once, up to 128 KiB, verifies the expected digest, then uses `torch.load(BytesIO(...), weights_only=True, map_location="cpu")` on those same bytes. Result files have a separate 384 KiB bound. A digest identifies bytes. It does not authenticate a publisher. Use trusted local artifacts. PyTorch's [load documentation](https://docs.pytorch.org/docs/2.9/generated/torch.load.html) and [serialization notes](https://docs.pytorch.org/docs/2.9/notes/serialization.html) remain relevant. Bounded files and schema checks are not a hostile-checkpoint sandbox.

Known validation refusals occur before state application and preserve the live model/optimizer. Arbitrary exceptions during an applied load have no rollback guarantee. This route excludes arbitrary models/optimizers, stochastic RNG recovery, AMP, distributed recovery, broad hardware/platform portability, security hardening, performance and model quality. Earlier CPU-only and CUDA-only saved artifact schemas retain their environment restrictions.
