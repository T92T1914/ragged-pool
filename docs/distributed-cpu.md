# One global objective across two CPU ranks

`examples/train_distributed.py` trains one fixed packed model through two local CPU processes using PyTorch DistributedDataParallel (DDP). Each process has its own model and SGD momentum. DDP averages their parameter gradients before both optimizers update. The Linux installed consumer passed two actual updates in float32 and float64 against global native and independent primitive models. A separate injected worker failure checked bounded cleanup.

This is a model caller of the existing native CPU operation. It adds no pooling API, native kernel, dispatcher registration or distributed runtime. Earlier device, derivative and saved-artifact contracts retain their own limits.

## Average groups rather than ranks

Rank zero has two groups and rank one has three in the first update. A mean loss on each rank followed by DDP's gradient average would give the two groups half the total weight and the three groups the other half. That optimizes an equal-rank objective, rather than the intended mean over all five groups.

Let S_r be a rank's squared-error sum, G the global group count and W=2. Each rank backpropagates:

```text
local_backward_loss = W / G * S_r
DDP parameter gradient = (1/W) * sum_r gradient(local_backward_loss)
                       = gradient(sum_r S_r / G)
```

For example, sums of 2 and 12 over two and three groups give global loss 14/5=2.8. Averaging the local means instead gives (2/2+12/3)/2=2.5. Matching a plausible prediction does not establish the correct training objective. This follows DDP's documented [gradient-averaging behavior](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html).

The consumer all-reduces the integer group count and a detached squared-error sum. The latter supplies a global metric without introducing another gradient path. DDP reduces parameter gradients, not leaf input gradients. Each worker's raw-input gradient is therefore compared to twice its corresponding slice of the global-mean oracle gradient.

## Empty groups and a zero-group rank

The fixed encoder/scorer/pool/head model has five parameter blocks and 25 scalars. Two changing packed updates use local group counts `[2,3]`, then `[0,4]`, with global counts 5 and 4. Empty and singleton groups are included. Raw width is 3, hidden width is 4 and seed 1811 fixes initialization and fixtures.

An empty group is still an example. Its pool is zero, but the head bias predicts a value and receives a target-bearing loss. A rank with no groups is different: offsets are `[0]`, predictions are empty and the local summed loss is a connected zero. It still runs forward, backward and every collective. All five parameter gradients remain connected, without `find_unused_parameters=True`. Its optimizer takes the same global update as the other rank.

SGD uses lr=0.03 and momentum=0.8, with dampening/weight decay/Nesterov disabled and foreach/fused/differentiable false. An independent recurrence checks the actual momentum and parameter update. Inputs and initial templates stay unchanged, gradients are cleared between steps, and model/optimizer parameter identities remain stable.

## Run the installed Linux example

The qualified route is Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GCC 13 and C++17. Follow the existing [Linux source-distribution build](linux-cpu.md) to produce the CPU wheel. Create a fresh acceptance environment outside the checkout, install that wheel normally, and copy the self-contained example to another directory. Do not add the checkout to `PYTHONPATH`.

```bash
python3.11 -m venv /tmp/ragged-ddp-accept
python=/tmp/ragged-ddp-accept/bin/python
"$python" -m pip --isolated install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
# Set wheel to the exact CPU wheel built from the checked source archive.
"$python" -m pip --isolated install "$wheel"
"$python" -m pip check
"$python" -I -B /tmp/ragged-ddp-consumer/train_distributed.py --dtype float32 --directory /tmp/ddp-f32 --output /tmp/ddp-f32.json --wheel "$wheel"
"$python" -I -B /tmp/ragged-ddp-consumer/train_distributed.py --dtype float64 --directory /tmp/ddp-f64 --output /tmp/ddp-f64.json --wheel "$wheel"
```

Use new scenario directories and output paths. Run the dtypes serially. Each invocation retains complete tensor receipts and worker/bootstrap logs. The parent runs its numerical oracles only after both workers exit. Each worker uses one Torch intra-op/inter-op setting and one Gloo communication thread. This is not a claim of two total OS threads.

The [backend registration API](https://docs.pytorch.org/docs/2.9/distributed.html#torch.distributed.Backend.register_backend) is experimental. A process-local factory supplies Gloo with an explicit `127.0.0.1` device, one communication thread and a 30-second timeout. Its `_Options`, `_devices` and `_threads` bindings are specific to the pinned PyTorch version. The ordinary Gloo initializer does not forward those options in [v2.9.1 source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/distributed_c10d.py). The published launcher additionally pins child TCP transport and disables lazy initialization, without changing the caller's global networking settings.

## Executed evidence and controls

The [initial qualification record](results/distributed-cpu.json) identifies the exact tested merge `26011453ccc3b1c158a2bc1d3cca33d243946606`, tree `7c5bbbd95c9dc5c853661de960076ea801ba9dea`, Linux run and native artifact. The job built from the checked source distribution, installed an ordinary wheel in a separate environment, passed 1,462 tests with two warnings, and executed both dtype consumers and the failure control. Complete raw JSON and logs were downloaded and their archive SHA256 verified. The public projection retains every comparison field and numerical precision while declaring omitted paths, process IDs and full tensor trajectories.

| Initial Linux comparison across both updates | float32 | float64 |
| --- | ---: | ---: |
| Rank-to-rank gradient, parameter and momentum difference | 0 | 0 |
| Largest worker difference against either global oracle | 3.725290298461914e-9 | 5.551115123125783e-17 |
| Final parameter drift from wrong local-mean training | 0.009621202945709229 | 0.01841184972153795 |

Native and independent primitive global models compare complete predictions, pools, global metric, input gradients, parameter gradients, updated parameters and momentum. Float32 uses rtol=2e-4/atol=2e-6 and float64 uses rtol=2e-10/atol=2e-12. The deliberate local-mean model starts with exactly the same first predictions and pools, then must diverge in gradients, momentum and both parameter updates. These are synthetic numerical comparisons, without model-quality or speed conclusions.

The final consumer additionally pins child TCP/lazy-init settings after the initial qualification. That record keeps its original source identity. Final-head and actual-main CI separately execute the published source, rather than relabeling the initial run. Mathematical fixtures are separate from real distributed execution, and operator registration checks remain separate from gradient evidence.

The injected rank-one failure happens before process-group initialization. CI requires exit1, its exact deliberate RuntimeError, no completed numerical oracles, and both directly owned nonzero workers reaped. The parent has a 120-second deadline and bounded terminate/kill waits. Successful workers destroy their group before exit. This qualifies one error path, without a promise of arbitrary descendant cleanup, crash recovery or fault tolerance.

## Windows result and limits

The local Windows CPU wheel compiled and installed successfully. Its ordinary contract/regression checks passed separately. The first distributed attempt failed before training with `makeDeviceForHostname(): unsupported gloo device`, even though Gloo availability reported true. Isolated explicit TCP and UV loopback device probes also failed, and all owned processes were reaped. That failed attempt remains retained. The current consumer refuses non-Linux before creating files, changing worker environment variables or launching children. No Windows DDP success is claimed.

This fixed example excludes multi-node/GPU DDP, stochastic distributed recovery, arbitrary models/optimizers, uneven iteration counts, compile/export, AMP, secure transport, hostile-file handling, fault tolerance, broad ABI/platform compatibility, performance and model quality. Loopback ownership and bounded cleanup are software-assurance practices, not an operational security qualification.
