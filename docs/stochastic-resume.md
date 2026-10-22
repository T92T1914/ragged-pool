# Continue a stochastic CPU training run

`examples/resume_stochastic.py` adds dropout and shuffled batch order to the existing eager packed training model. A producer completes two SGD updates and exits. A fresh resumer restores its state and completes five more updates. The seven-update trajectory crosses the end of a five-batch permutation, so restoring the next shuffle matters as well as restoring the current position.

This is one bounded training recovery example for CPython 3.11.8 and PyTorch 2.9.1+cpu. It retains the existing float32/float64 inputs, native pooling operation, 25 model parameters, SGD momentum policy and primitive mathematical reference. It adds no package API or native kernel and makes no performance, model-quality, security or cross-platform replay claim.

## State that determines the next update

Dropout with probability 0.25 acts on the encoded hidden values before scoring and pooling. It uses the default CPU Torch random generator. A separate CPU `torch.Generator` supplies permutations of the five fixed packed batches. Keeping these streams separate makes their ownership explicit.

The completed-update-two checkpoint contains:

- Model parameters and ordered SGD momentum state, with the existing learning rate 0.03 and momentum 0.8.
- The default CPU random-generator state used by dropout.
- The independent shuffle generator's state after creating the current permutation.
- The current permutation, epoch and next-position cursor.
- The completed update count, training mode and exact fixture, parameter, dtype and runtime metadata.

The resumer constructs its model and optimizer before restoring the saved states. It restores the default CPU random state last. Model construction itself consumes random draws, so restoring it before construction would place dropout at the wrong point in its stream. Restarting from the original seed also loses the draws already consumed by completed updates.

The original [fixed-trajectory example](training_resume.md) remains available with its separate deterministic contract. Its exclusions have not been silently expanded.

## Admission and ownership

Only known, locally produced artifacts are accepted. The loader bounds and hashes the file, then passes the same bytes to `torch.load` with explicit `weights_only=True` and `map_location="cpu"`. It has no unrestricted fallback or custom allowlist. A digest establishes byte identity rather than publisher authentication. Restricted loading is not a hostile-input sandbox.

Before applying live state, the example checks the complete model and optimizer schema, fixed metadata, permutation and cursor. Both random states must be ordinary owned contiguous CPU byte tensors. A scratch CPU generator also tries each state before any model, optimizer, live shuffle or default-generator mutation. Shape and dtype alone cannot establish a usable generator state. The fixed shuffle state is checked against the declared first permutation and its generator progress.

Refused state must leave the model, optimizer, sampler and default random state unchanged. A successful restore retains the model's parameter objects and uses independent state storage. The original checkpoint and fixture bytes remain unchanged.

## What decides whether continuation worked

The producer and resumer have distinct process IDs and execute sequentially. Their combined records must match uninterrupted native execution exactly on the executed pinned environment. The comparison includes dropout results, pooled values, predictions, losses, raw-input and parameter gradients, updated parameters, momentum, batch order and random-state progression.

An independently differentiated primitive PyTorch pool checks every update. Comparisons retain float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. The checks also retain the independent SGD momentum recurrence. This establishes agreement for the declared fixture and environment, rather than accuracy on operational data.

Deliberately incomplete continuation resets dropout state, resets the shuffle generator or discards momentum after a valid restore. Each must produce a measurable final-parameter difference. Repeated and skipped cursors are refused as checkpoint states, with deliberate wrong-cursor controls showing why the boundary matters. These controls do not weaken normal admission.

The receipt records each batch ID, epoch, cursor and random-state digest, including the next permutation. It also records installed package and native-library identity, distinct process IDs, comparison errors, control thresholds and checkpoint byte identity. The CI receipt checker verifies this declared structure and the installed-source identity in addition to the example's numerical assertions. An independent source review and executed revision-linked results remain separate evidence.

At update six, the preselection cursor is five: the previous order is exhausted. Selecting the next batch generates a new permutation, advances the epoch to one and leaves its cursor at one. The receipt preserves that actual transition instead of relabeling the observed preselection state as zero.

## Run an installed copy

Use the [ordinary pinned wheel route](../README.md#build-and-run) in an isolated environment outside the checkout. Copy `resume_stochastic.py`, `resume_training.py` and `train_dynamic.py` into the same separate consumer directory. Use a new artifact directory and receipt path for each run:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\resume_stochastic.py" --dtype float32 --directory "$consumer\stochastic-float32" --output "$consumer\stochastic-float32.json"
& $python -I "$consumer\resume_stochastic.py" --dtype float64 --directory "$consumer\stochastic-float64" --output "$consumer\stochastic-float64.json"
```

Both runs use one Torch intra-op and one inter-op thread. The parent does not perform numerical work while a child is active. Checkpoints, bounded child result records and stdout/stderr logs remain in the new directory. Existing directories and receipt files are refused.

## Sources and limits

[PyTorch's reproducibility note](https://docs.pytorch.org/docs/2.9/notes/randomness.html) distinguishes control of random sources from broader deterministic execution and does not guarantee reproducibility across versions or platforms. The version-matched [CPU state API](https://docs.pytorch.org/docs/2.9/generated/torch.get_rng_state.html), [restore API](https://docs.pytorch.org/docs/2.9/generated/torch.set_rng_state.html) and [Generator API](https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html) define the two state owners used here. The exact [v2.9.1 CPU generator implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/CPUGeneratorImpl.cpp) checks generator contents as well as representation. The [dropout implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Dropout.cpp) establishes its random draws and inverse-retention scaling.

Xu and colleagues' [Checkpointing and Deterministic Training for Deep Learning](https://tao.aisec.world/assets/pdf/CAIN22.pdf), CAIN 2022, explains why generator progress and data-order state matter alongside learned weights. Selected sections inform this example's state choice. The paper is not validation of this implementation or a claim that every source of nondeterminism has been handled.

The scope excludes Python and NumPy random streams, DataLoader workers, asynchronous prefetch, schedulers, pending gradient accumulation, compiled caches, distributed or GPU recovery, crash-consistent power-loss recovery, authenticated storage and general untrusted checkpoints. The accepted CUDA companion retains its separate contract. Implementation and verification use Codex assistance, consistent with [NOTICE](../NOTICE), without implying unaided expertise, certification, employer qualification or operational defense experience.
