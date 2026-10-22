# Recompute activations during packed training

`examples/train_checkpointed.py` compares ordinary eager training with activation recomputation around the existing CPU packed layer. It takes two SGD updates through an encoder, dropout, scorer, native pool and head. Each checkpointed backward recomputes the declared segment rather than using its original saved intermediate tensors.

This guide describes the method and its required comparisons. Completed revision-linked execution establishes whether they pass. A guide or workflow definition is not an executed result.

## The recomputed segment

The model retains five parameter tensors with 25 scalar parameters. The two fixed batches come from the [dynamic training fixture](dynamic_training.md): N=9/B=5 followed by N=12/B=6, including empty groups. SGD uses learning rate 0.03 and momentum 0.8. Both CPU float32 and float64 are checked.

The segment runs the encoder, tanh, dropout with probability 0.25, scorer and pooling. The head runs outside it. The default CPU Torch generator supplies dropout, starting from seed 2211. Batch selection and optimizer updates also stay outside the segment. Raw inputs and offsets are explicit tensor arguments.

The call uses `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`, `preserve_rng_state=True` and the default determinism check. Its forward runs inside `set_checkpoint_early_stop(False)`. Non-reentrant checkpointing normally stops recomputation once the required saved tensors have been recovered. Disabling that optimization here makes full segment replay explicit and gives the example a complete forward/recomputation witness.

The segment must perform the same arithmetic on replay. It must not advance a sampler, step an optimizer, change training mode or mutate model observations. A forward/recomputation context may record passive execution evidence, but it must not choose different numerical behavior.

## What the comparisons establish

Direct native and preserved checkpointed native routes start from the same model and CPU random state. They must agree exactly on the executed pinned environment for pooled values, predictions, loss, raw-input gradients, every parameter gradient, updated parameters and momentum buffers. The checks also evaluate the SGD momentum recurrence independently and retain parameter-object ownership.

For both correct routes, the CPU RNG after backward must equal its state after forward. Corresponding RNG progressions must agree across routes. Recomputing dropout therefore cannot silently consume additional draws from the surrounding training stream.

An independently differentiated primitive PyTorch pool supplies a second mathematical comparison. It retains float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. The primitive route uses the same declared upstream dropout operation. It is an independent pooling/gradient comparison, not an independent random-number algorithm.

Two controls distinguish the mechanism:

- `without_rng_preservation` changes only the replay policy. Its initial forward must match the correct route, while the declared backward RNG and gradient/update differences must expose the incomplete replay.
- `dropout_disabled` uses probability zero as a positive control. Direct and checkpointed numerical results and random-state behavior must still agree.

The default checkpoint determinism check compares saved/recomputed tensor shape, dtype and device. It does not compare their numerical values or prove semantic equivalence. The explicit output, derivative, update and RNG checks remain necessary.

## Observe saved tensors precisely

A separate observation records `original_forward_pack_hook_events` and their logical payload: `numel * element_size` for each original-forward pack event. A tensor or alias saved several times contributes several events. Recomputed backward saves are outside that original-forward total.

These are hook-event and logical-payload observations. They do not measure unique allocations, bytes freed, allocator peak, process RSS or elapsed-time savings. Hooks can also affect tensor-object packing. The small model qualifies this declared integration rather than demonstrating a useful memory or speed advantage.

## Run an installed copy

Use the [ordinary pinned wheel route](../README.md#build-and-run) in an isolated environment outside the checkout. Copy `train_checkpointed.py` and its adjacent `train_dynamic.py` into the same separate consumer directory. Use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\train_checkpointed.py" --dtype float32 --output "$consumer\checkpointed-float32.json"
& $python -I "$consumer\train_checkpointed.py" --dtype float64 --output "$consumer\checkpointed-float64.json"
```

The qualified pin is CPython 3.11.8 with PyTorch 2.9.1+cpu. Numerical work uses one Torch intra-op and one inter-op thread. Installed package/native-library and consumer-source identities accompany the receipt. The receipt checker, independent source review, numerical execution and exact-revision CI remain separate evidence.

## Sources and scope

The [PyTorch 2.9 checkpoint reference](https://docs.pytorch.org/docs/2.9/checkpoint.html) explains non-reentrant behavior and RNG preservation. The exact [v2.9.1 implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/checkpoint.py) shows its CPU state handling, metadata-only check and early-stop behavior. The [autograd note](https://docs.pytorch.org/docs/2.9/notes/autograd.html) explains saved tensors and packing hooks.

Chen and colleagues' [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174), section 4.1 and Algorithm 1, supplies the recomputation rationale. Its memory results are not measurements of this example.

This route is eager first-order CPU training. It excludes compilation, CUDA, custom-generator or worker replay, distributed execution, reentrant/selective checkpoint policies, higher-order checkpointing, pending accumulation recovery and inference export. In compiler mode, the RNG-preservation flag does not supply the same negative control. Existing [fresh-process training recovery](stochastic-resume.md) keeps its separate contract.

Implementation and verification use Codex assistance, consistent with [NOTICE](../NOTICE). The declared checks do not establish cross-platform replay, production readiness, security certification, employer qualification or operational defense experience.

