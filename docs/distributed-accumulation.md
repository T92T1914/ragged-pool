# Accumulate groups before a distributed update

The optional `--accumulate` mode in `examples/train_distributed.py` splits each optimizer update into two microbatches on each of two Linux CPU ranks. It uses the same installed native pooling operation and fixed 25-scalar model as the [ordinary distributed example](distributed-cpu.md). It adds a training schedule, without changing the pooling API or native kernel.

## One objective for the complete window

An optimizer window is all the examples contributing to one update. The fixed schedule is:

| Window | Microbatch | Rank 0 groups | Rank 1 groups | Global microbatch groups |
| --- | --- | ---: | ---: | ---: |
| 0 | 0 | 2 | 3 | 5 |
| 0 | 1 | 0 | 4 | 4 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 4 | 0 | 4 |

The window totals are nine and five groups. Every rank makes the same two forward/backward calls, including a connected zero-group microbatch. An empty group remains an example with a target and head-bias loss. A zero-group microbatch has no examples. Each declared global microbatch contains at least one group; globally empty microbatches and windows are outside this fixed protocol.

Let S_r,m be a local squared-error sum, G the complete window's group count and W=2 ranks. Each microbatch backpropagates:

```text
local_backward_loss = W / G * S_r,m
DDP averaged accumulated gradient
    = (1/W) * sum_r sum_m gradient(W/G * S_r,m)
    = gradient(sum_r sum_m S_r,m / G)
```

There is no additional division by two. Nor does each microbatch get an equal vote regardless of its number of groups. For example, global microbatch sums of 2 and 12 over one and four groups give complete-window loss 14/5=2.8. Averaging their means gives (2/1+12/4)/2=2.5. Both calculations look reasonable until the desired objective is stated.

The consumer collectively establishes the two global counts before backward, clears gradients once, accumulates both contributions and takes one momentum-SGD step. Parameters and optimizer momentum stay unchanged through the microbatches. It checks two updates, so momentum continuity and gradient clearing between windows matter.

## Synchronization is a separate check

Both forward and backward of the first microbatch are inside [DDP.no_sync()](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync). The final forward/backward runs outside that context, synchronizing accumulated parameter gradients. Putting only backward inside the context would miss DDP's forward-time preparation.

Mathematical agreement alone cannot establish deferred communication. The example therefore runs a second, correctly normalized model that synchronizes every microbatch but still steps only once per window. Its optimizer trajectory must match the deferred route within the declared tolerance.

A public communication-hook wrapper records completed-backward deltas and delegates unchanged to PyTorch's [official default averaging hook](https://docs.pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook). Custom-hook buckets are not already divided by world size. Replacing that mean with a raw SUM would change the mathematics.

For this fixed same-dtype 25-scalar model, a successful deferred window requires hook deltas `[0,1]`; eager synchronization requires `[1,1]`. The recorded bucket contains 25 scalars. These counts concern gradient-bucket hook calls. They exclude count/metric collectives and do not measure packets, total network bytes, time or memory savings.

## Run the installed Linux consumer

Use the [pinned Linux build route](linux-cpu.md): Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GCC 13 and C++17. Build the wheel from the checked source distribution. Install it into a separate acceptance environment and copy `train_distributed.py` outside the checkout, without setting PYTHONPATH to the repository.

```bash
# python names the acceptance environment's interpreter; wheel names the exact CPU wheel.
"$python" -m pip --isolated install "$wheel"
"$python" -m pip check
"$python" -I -B /tmp/ragged-consumer/train_distributed.py --accumulate --dtype float32 --directory /tmp/accum-f32 --output /tmp/accum-f32.json --wheel "$wheel"
"$python" -I -B /tmp/ragged-consumer/train_distributed.py --accumulate --dtype float64 --directory /tmp/accum-f64 --output /tmp/accum-f64.json --wheel "$wheel"
```

Use new directories and output paths, and run dtypes serially. The default invocation without `--accumulate` retains the previous route. Windows refuses before worker or scenario-file creation; its preserved Gloo transport failure is not a Windows distributed pass.

The same two-worker lifecycle pins child TCP loopback, uses file rendezvous, a 30-second group timeout and a 120-second parent deadline, and reaps directly owned workers. Each worker uses one numerical intra-op/inter-op setting and one separate Gloo communication thread. Parent numerical oracles run after workers exit. This does not mean two total OS threads.

## What the receipts and tests establish

A successful run compares complete full-window native and independent primitive models with every worker's predictions, pooled vectors, raw gradients, final accumulated parameter gradients, updated parameters and momentum. Raw-input gradients are unreduced by DDP and compare to twice the matching global-oracle slices. Intermediate first-microbatch parameter gradients compare to independent local primitive contributions.

A third worker model deliberately optimizes equal microbatch means. Its first window starts with exactly matching predictions and pools, then must diverge in gradients, momentum and both optimizer updates. Serial mathematical fixtures independently reconstruct the model and manual SGD recurrence, and check dropped-first-gradient and premature-per-microbatch-update errors. Those fixtures do not initialize Gloo.

Full bounded JSON arrays, source/script/wheel identities, hook events and worker/bootstrap logs are retained. The shared receipt ceiling is explicitly 2 MiB, checked before writing; arrays are not truncated. The Linux workflow executes both dtype consumers and an injected rank-one pre-initialization failure, requiring the exact error, no numerical oracles and both directly owned workers reaped. Its uploaded evidence accompanies each exact PR/main run. Registration checks, serial mathematical checks, actual DDP and sanitizer campaigns are separate evidence.

This example assumes deterministic per-example model computation. It excludes batch normalization/dropout equivalence, arbitrary models, globally empty microbatches, uneven rank schedules, pending-gradient resume, compile/export, AMP, GPU/multi-node DDP, fault-tolerant recovery and secure transport. Public backend registration is experimental and explicit Gloo options remain private bindings in the pinned version. It makes no model-quality, speed, memory-saving, broad portability or operational-security claim.
