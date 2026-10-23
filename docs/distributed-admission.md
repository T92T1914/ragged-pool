# Agree before changing an update

`examples/admit_distributed.py` adds one shared numerical admission decision to the fixed [accumulated Linux CPU model](distributed-accumulation.md). Both responsive ranks must finish the final synchronized backward and agree that their parameter gradients and joint norms are finite before either clips gradients, reduces the metric or calls momentum-SGD. The native pooling API and kernel stay unchanged.

The existing [clipping example](distributed-clipping.md) checks local gradients before mutation. That protects its local tensors, but one rank can refuse while a healthy peer proceeds toward the following metric collective. Parent timeouts and cleanup bound that failure. They do not give the two ranks a common update decision.

## Reduce a decision, not another norm

DDP already holds the averaged gradient on both ranks when its final backward returns. The gate checks every scalar in the five fixed CPU gradient blocks with `math.isfinite`, then reads their joint L2 norm using [`get_total_norm`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.get_total_norm.html) with `error_if_nonfinite=False` and `foreach=False`. Neither probe mutates the gradients. A nonfinite norm also refuses the update, even when every gradient element is finite, because the norm calculation itself can overflow its dtype.

Every rank converts its local numerical result into one CPU `int64` scalar: zero for valid, one for invalid. All ranks then call synchronous [`all_reduce`](https://docs.pytorch.org/docs/2.9/distributed.html#torch.distributed.all_reduce) with `ReduceOp.SUM`. The result counts invalid ranks. Only zero admits the update. Expected numerical failure becomes a vote rather than an exception that prevents collective participation.

This is one admission collective per attempted gated window. It does not reduce the gradient payload or count the replicated norm twice. The correct clipping route still adds no norm collective, and its two backward hook deltas remain `[0, 1]`. These are explicit operation counts, not network traffic measurements.

An admitted window continues through the existing public `clip_grad_norm_` route, the metric reduction and SGD. This example uses its own clipping caps, 0.25 then 0.1, so both clips are active in the fixed fixture. The earlier clipping consumer retains its separate 0.25 then 2.0 schedule. Independent concatenated-vector clipping and manual SGD references use the same declared caps as the new consumer. The gate must leave that mathematical update unchanged.

## Refuse with real momentum to preserve

Each dtype runs three sequential two-worker cohorts. A normal gated cohort completes both optimizer windows. The other two cohorts first complete window zero normally, establishing actual momentum in all five blocks. After window one's final synchronized backward, rank 1 deliberately replaces the single `head.bias.grad[0]` scalar with NaN.

The timing is intentional. A NaN introduced before DDP's gradient reduction would generally spread through the replicated gradient. This is a synthetic fault after synchronization, used to test the agreement protocol at that exact stage. It is not a claim to recover ordinary loss overflow or arbitrary native failures.

In the coordinated-refusal cohort, local bad flags are zero on rank 0 and one on rank 1. Both observe the reduced count one and refuse. Neither clips, reduces the subsequent metric, calls SGD, clears gradients or starts another attempt. Parameters and momentum must remain exactly at their completed-window-zero state. All post-injection gradient bytes, objects, storage bindings and versions must remain unchanged by the gate.

The local-only control makes no admission collective. Its bad rank refuses, while its healthy rank locally admits and actually clips its gradient. That healthy gradient's content bytes must change. Both workers then deliberately stop before the metric collective or SGD. This finite control exposes different decisions and partial gradient mutation. It does not claim an observed collective hang or a partial optimizer update.

Expected refusals produce complete records and orderly worker returns, allowing both ranks to destroy their groups and write receipts before the parent reaps them. Unexpected import, process or Gloo errors retain the existing bounded owned-worker cleanup path. An unexpected failure cannot satisfy a declared numerical-refusal control.

## Preserve evidence containing NaN

Ordinary tensor equality cannot establish unchanged NaN values, and JSON has no standard NaN number. The example retains the finite pre-injection gradient arrays, the exact injected location and classification, and bounded raw gradient-byte representations before and after admission. It compares the actual bytes together with object/storage identities and versions. It does not serialize nonstandard JSON NaN literals or substitute a boolean assertion for the invalid tensor's representation.

Normal records keep full predictions, pooled vectors, raw input derivatives, pre/post clipping gradients, model parameters and momentum. Native and independent primitive full-window references run after owned workers have exited. Control pre-injection records must match the corresponding normal trajectory, making the fault and admission decision the relevant difference.

## Run the installed Linux consumer

Use the [pinned Linux source distribution route](linux-cpu.md): Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GCC 13 and C++17. Install the ordinary wheel into a fresh environment outside the checkout. Copy `admit_distributed.py`, `clip_distributed.py` and `train_distributed.py` from the checked source distribution into the same separate consumer directory. Do not add the checkout to PYTHONPATH.

```bash
# python names the acceptance environment; wheel names its exact installed CPU wheel.
"$python" -I -B /tmp/ragged-consumer/admit_distributed.py --dtype float32 --directory /tmp/admit-f32 --output /tmp/admit-f32.json --wheel "$wheel"
"$python" -I -B /tmp/ragged-consumer/admit_distributed.py --dtype float64 --directory /tmp/admit-f64 --output /tmp/admit-f64.json --wheel "$wheel"
```

Use new output paths and run dtypes serially. The Linux workflow retains the two complete installed dtype invocations and a separate injected pre-Gloo worker failure. Serial numerical/admission tests do not start process groups and remain separate from actual Gloo execution, operator registration and sanitizer checks. A workflow definition describes intended checks, not an executed result.

Each cohort owns two workers with finite deadlines, explicit TCP loopback and one numerical intra-op/inter-op thread per worker, plus a separate Gloo communication thread. The next cohort starts after the preceding workers are reaped. Parent numerical work runs afterward. This is not a claim of two total OS threads. Windows refuses before creating workers, and its earlier Gloo initialization failure remains adverse evidence.

The contract assumes the fixed model, optimizer, CPU representation and collective schedule, with both live ranks reaching this gate. It does not handle a dead or malicious rank, a failure during DDP backward, a hung collective, secure transport, mixed precision, sharded/multi-node/GPU training, automatic skip/retry or crash recovery. The fixture is synthetic. No model-quality, privacy, performance, operational-security or hiring claim follows from coordinated refusal.
