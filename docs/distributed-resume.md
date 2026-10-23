# Continue after a completed distributed update

`examples/resume_distributed.py` continues the fixed [accumulated Linux CPU model](distributed-accumulation.md) in fresh workers. Two ranks complete the first optimizer window, clear gradients, write a checkpoint and exit. A new two-rank cohort restores both model and momentum-SGD state, then performs the second window. The native pooling API and kernel stay unchanged.

This is a controlled handoff after a completed update. It does not recover a crashed collective or an unfinished `no_sync()` window. The fixture has no stochastic loader, dropout, scheduler or random state consumed during training.

## Preserve the next update

The model has five parameter blocks containing 25 scalars. Its checkpoint stores those named weights, the five corresponding momentum buffers, the exact SGD options, fixture and installed-source identities, and the cursor `next_window=1, next_microbatch=0`. Gradients must already be cleared. The producer refuses to checkpoint pending gradients.

Weights alone do not describe the next update. With momentum 0.8, the second update uses:

```text
new_momentum = 0.8 * restored_momentum + current_gradient
new_parameter = restored_parameter - 0.03 * new_momentum
```

The example restores a second model with the same weights and zero momentum as a deliberate error control. Its predictions and current gradients must match the correct continuation. Its new momentum and updated parameters must differ. Matching predictions would miss the error.

The checkpoint uses bounded JSON rather than pickle. The loader reads one bounded byte buffer, checks its expected SHA-256 digest, then validates duplicate keys, nonfinite numbers, exact fields and types, tensor names/order/shapes, optimizer options and cursor before constructing replacement state. Restoration copies into the existing parameters and preserves optimizer bindings. It does not authenticate the producer or make arbitrary external files safe.

## Check both ranks before DDP can repair one

[DDP construction](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/parallel/distributed.py) checks parameter shapes and synchronizes model state from rank 0. That is useful training behavior, but it could conceal a failed rank 1 model restore. Optimizer momentum is not synchronized by that constructor.

Each fresh worker therefore restores and records its complete weights, momentum and cursor before process-group initialization or DDP construction. The parent compares both ready records with the canonical checkpoint and releases an owned gate only after both match. A deliberately changed rank 1 momentum value must cause the parent to refuse the gate and reap both workers without initializing Gloo.

The program uses three sequential two-worker cohorts: uninterrupted two-window training; a one-window checkpoint producer; and a fresh continuation. Every producer worker must have exited and been reaped before continuation starts. Parent numerical comparisons run after workers exit. This ordering establishes a completed file handoff, without claiming power-loss durability or fault tolerance.

## Run the installed Linux consumer

Use the [pinned Linux source distribution route](linux-cpu.md): Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GCC 13 and C++17. Install the built wheel into a fresh acceptance environment outside the checkout. Copy both `resume_distributed.py` and its adjacent `train_distributed.py` helper from the checked source distribution into a separate consumer directory. Do not add the checkout to PYTHONPATH.

```bash
# python names the acceptance environment; wheel names its exact installed CPU wheel.
"$python" -I -B /tmp/ragged-consumer/resume_distributed.py --dtype float32 --directory /tmp/resume-f32 --output /tmp/resume-f32.json --wheel "$wheel"
"$python" -I -B /tmp/ragged-consumer/resume_distributed.py --dtype float64 --directory /tmp/resume-f64 --output /tmp/resume-f64.json --wheel "$wheel"
```

Use new directories/output paths and run dtypes serially. The example compares complete predictions, pooled vectors, raw input gradients, accumulated parameter gradients, parameters and momentum against uninterrupted DDP and global native/independent primitive models. Its fixed objective remains the complete-window group mean, with nine groups in the first window and five in the second. Raw input gradients remain local and carry the established world-size factor.

The Linux workflow runs both installed dtype consumers and the rank 1 readiness rejection. Full bounded arrays, checkpoint bytes, pre-DDP witnesses, source/wheel identities and worker lifecycle records accompany its exact revision artifacts. Serial checkpoint admission and mathematical tests are separate from actual Gloo execution, operator registration and sanitizer checks. A successful CI result is attached to its tested revision; the workflow definition alone is not an executed result.

The [module loading API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) uses `assign=False` to retain live parameters. The [optimizer loading API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html) matches parameter IDs by order, so fixed names, shapes and order are checked explicitly. [Python's JSON decoder hooks](https://docs.python.org/3.11/library/json.html) support duplicate-key and nonfinite-constant rejection. A digest checks byte identity, not authorship.

The route retains finite deadlines, explicit TCP loopback, two directly owned workers and their cleanup. Each worker sets one numerical intra-op/inter-op thread and a separate Gloo communication thread. This is not a claim of two total OS threads. Windows refuses before creating workers; the earlier Windows Gloo initialization failure remains adverse evidence.

Scope excludes pending-gradient resume, changing rank count, stochastic data replay, multi-node/GPU DDP, compile/export, mixed precision, malicious-file isolation, secure transport and a general training runtime. It establishes no performance, model-quality or operational-security advantage.
