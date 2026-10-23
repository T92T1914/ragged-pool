# Clip the synchronized model gradient

`examples/clip_distributed.py` extends the fixed [accumulated Linux CPU example](distributed-accumulation.md) with one joint L2 clip of its five parameter-gradient blocks, containing 25 scalars. It clips after the last synchronized backward and before momentum-SGD. The native pooling API and kernel stay unchanged.

This policy differs from the [per-example clipping example](per-example-gradients.md), which clips individual examples before averaging. Clipping is nonlinear, so these operations generally produce different optimizer inputs. Neither example establishes differential privacy, adversarial robustness or better model quality.

## Count the replicated gradient once

The objective remains squared error divided by the complete window's group count. Each rank accumulates two local microbatches. The first backward uses `no_sync()` and the final backward synchronizes the accumulated parameter gradients. DDP averages those gradients across two ranks, so the existing world-size scaling preserves the global group mean.

After synchronization, both ranks already hold the same complete model gradient `g`. Each rank calculates its joint norm locally and applies the same coefficient:

```text
norm = sqrt(sum of squares across all five parameter-gradient blocks)
coefficient = min(1, cap / (norm + 1e-6))
clipped_gradient = coefficient * g
new_momentum = 0.8 * previous_momentum + clipped_gradient
new_parameter = previous_parameter - 0.03 * new_momentum
```

The route calls PyTorch 2.9.1's public [`clip_grad_norm_`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html), with L2 norm, `error_if_nonfinite=True` and `foreach=False`. Its return value is the norm before clipping. The coefficient follows the [pinned implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/utils/clip_grad.py), including the epsilon and gradient dtype. Even a norm exactly equal to the cap can receive a coefficient slightly below one. The wrapper requires a finite positive nonboolean cap and all five valid live gradients before mutation. It refuses missing, nonfinite, incompatible or overflowed gradient state.

The cap limits the current gradient, not the momentum buffer or the eventual parameter displacement. Existing momentum can exceed the cap. Parameter clipping also does not change raw input gradients already produced by backward. Those local derivatives retain the established world-size factor in comparisons with the full-window oracle.

The deliberate error route sums the squares of the already-replicated norm across both ranks before taking its square root. It therefore counts the same gradient twice and uses approximately `sqrt(2) * norm`. Its first predictions, raw input derivatives and unclipped parameter gradients must match the correct route exactly. Its clipping coefficient, clipped gradients, momentum and updated parameters must diverge. The consumer runs this alternate optimizer trajectory and checks its complete records.

## Run the installed Linux consumer

Use the [pinned Linux source distribution route](linux-cpu.md): Ubuntu 24.04 x86-64, CPython 3.11.8, PyTorch 2.9.1+cpu, GCC 13 and C++17. Install the ordinary wheel in a fresh environment outside the checkout. Copy `clip_distributed.py` and its adjacent `train_distributed.py` helper from the checked source distribution into a separate consumer directory. Do not add the checkout to PYTHONPATH.

```bash
# python names the acceptance environment; wheel names its installed CPU wheel.
"$python" -I -B /tmp/ragged-consumer/clip_distributed.py --dtype float32 --directory /tmp/clip-f32 --output /tmp/clip-f32.json --wheel "$wheel"
"$python" -I -B /tmp/ragged-consumer/clip_distributed.py --dtype float64 --directory /tmp/clip-f64 --output /tmp/clip-f64.json --wheel "$wheel"
```

Use new output paths and run dtypes serially. The two optimizer windows use nine and five groups. Caps 0.25 and 2.0 exercise one active and one inactive clip in this fixed fixture. Parent native and independent primitive oracles calculate clipping by concatenated-vector arithmetic and check the manual SGD recurrence. Complete pre/post gradients, predictions, pooled vectors, raw derivatives, parameters and momentum are retained for both ranks and references. The fixture is synthetic and seeded. It demonstrates integration, not accuracy on real data.

The Linux workflow runs both installed dtype consumers and a deliberate rank 1 failure. It retains complete bounded worker receipts and bootstrap/worker logs, source/wheel identities and cleanup records at the tested revision. Serial mathematical/admission tests remain separate from Gloo execution, operator registration and sanitizer checks. A workflow definition is not an executed result.

The correct route adds no norm collective. The error route adds one explicit scalar sum per window. Recorded gradient communication hook calls remain `[0, 1]` across the two microbatches. This describes the example's hooks and explicit norm operations, not all network traffic. Two directly owned workers use one numerical intra-op/inter-op thread each and a separate Gloo communication thread. Parent numerical work starts after workers have exited and been reaped. This is not a claim of two total OS threads.

Windows refuses before creating workers. The earlier Windows Gloo loopback initialization failure remains adverse evidence. Scope excludes sharded parameters/FSDP, multi-node or GPU DDP, mixed precision, compile/export of this consumer, stochastic recovery, malicious-peer isolation and a general trainer. No speed, security, platform portability or employer eligibility follows from this small numerical example.
