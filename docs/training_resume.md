# Resume one packed training trajectory

`examples/resume_training.py` saves a small eager packed model after two complete SGD updates, exits that producer process, then restores the model and optimizer in a different fresh process for updates three through five. Its parent runs the child processes sequentially. It compares the complete five-update trajectory with uninterrupted native and independently differentiated primitive PyTorch models.

This is a bounded training recovery example for the pinned CPU environment. It uses ordinary model and optimizer `state_dict` APIs and the existing native operation. It adds no package API, kernel or derivative, and makes no model-quality, speed or memory claim.

## What has to survive the process boundary

The model has the same 25 scalar parameters in five blocks and the same five packed batches as the [dynamic training example](dynamic_training.md). This route is eager first order. Raw width stays 3 and hidden width stays 4. Item count N, group count B and offset partitions change across the fixed sequence, including empty groups.

SGD uses learning rate `0.03` and momentum `0.8`, with all other options fixed by the example. A momentum buffer carries prior gradients into the next update. For the declared options, the recurrence after the first step is `buffer = 0.8 * previous_buffer + gradient`, followed by `parameter -= 0.03 * buffer`. Restoring weights alone preserves the next forward prediction but loses that history. The model-only control therefore continues all three remaining updates with a new optimizer, then requires its final parameters to diverge from the correct continuation.

The checkpoint is written at completed cursor 2 after gradients are cleared. It contains the model tensors, optimizer state and exact format, dtype, pin, fixture, ordered-parameter and cursor metadata. A newly created model and optimizer receive this state in the resumer. The regenerated fixed fixture and cursor identify the next batch. There is no stochastic forward or live sampler to continue.

Every producer and resumed step is compared through pooled values, predictions, loss, raw-input gradients, each parameter gradient, post-update parameters and momentum buffers. The example checks an independent momentum recurrence and that each optimizer owns the intended live model parameters. Same-native continuation must agree exactly on the executed pinned environment. Independent primitive comparisons use the existing tolerances: float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. Exact continuation here is not a cross-version or cross-platform replay guarantee.

## Validate before restoring

Successful deserialization is only the first boundary. The example checks the checkpoint's exact schema, fixed metadata, completed cursor, model keys and tensor metadata, finite bounded state, optimizer options, canonical integer IDs and ordered parameter identity before loading either target. Refused state must leave the target unchanged. Reordered or duplicate optimizer IDs, wrong cursors and malformed tensors are rejected rather than repaired.

PyTorch's [optimizer loading reference](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html) explains that parameter names do not control ordinary loading. Its [v2.9.1 optimizer implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/optimizer.py) pairs saved IDs with current parameters by order. Matching shapes and group lengths do not establish the intended association. The example's fixed ordered schema supplies that additional check.

The resumer bounds and hashes the known local checkpoint bytes, then loads that same byte buffer with explicit `weights_only=True` and `map_location="cpu"`. It uses no custom allowlist or unrestricted-load fallback. The [serialization note](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch-load-with-weights-only-true) describes the restricted loader. This route accepts locally produced known checkpoints. Restricted loading and schema checks do not make arbitrary hostile artifacts safe, and a hash establishes byte identity rather than authentication.

The receipt records distinct parent, producer and resumer process IDs, unchanged checkpoint bytes, fresh loaded storage, installed package/native-library identity, comparison results, controls and thread settings. The producer/resumer logs and tensor result records remain alongside the checkpoint. A workflow definition describes the gate. Completed revision-linked execution establishes that the gate passed.

## Run from an ordinary installation

Use the [pinned wheel installation route](../README.md#build-and-run) with a fresh isolated environment outside the checkout. Copy both `examples/resume_training.py` and its adjacent `examples/train_dynamic.py` into the same separate consumer directory. The latter supplies the unchanged model and fixture helpers. Use a new artifact directory and receipt path for each run:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\resume_training.py" --dtype float32 --directory "$consumer\resume-float32" --output "$consumer\resume-float32.json"
& $python -I "$consumer\resume_training.py" --dtype float64 --directory "$consumer\resume-float64" --output "$consumer\resume-float64.json"
```

The command orchestrates the producer and resumer without running their numerical work concurrently. The new directory retains `checkpoint.pt`, `producer-results.pt`, `resume-results.pt` and the two child processes' stdout/stderr logs. Existing directories and receipt paths are refused. Both precision runs use one Torch intra-op and one inter-op thread. The normal installed native-library checks remain part of each process's acceptance.

The scope excludes pending gradient accumulation, general RNG restoration, DataLoader workers, dropout, schedulers, compiled-cache persistence, distributed recovery, GPU, crash-consistent power-loss recovery, authenticated storage and cross-version/platform/ABI compatibility. Earlier inference export and dynamic compiled training retain their separate contracts. This example continues one declared local trajectory, rather than claiming a general checkpoint service.

Implementation and verification use Codex assistance, consistent with [NOTICE](../NOTICE). Executed records establish component behavior and preserve the distinction from unaided expertise, security certification or employer qualification.
