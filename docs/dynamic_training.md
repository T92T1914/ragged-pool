# Packed dynamic-shape training

`examples/train_dynamic.py` trains one small packed model with `torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=True)`. It calls `torch.compile` once and keeps the same model, parameter objects and SGD optimizer across five real updates while item count N and group count B vary. Raw features stay width 3, hidden features stay width 4 and dtype stays fixed within each run.

A single call to `torch.compile` does not mean that every later input uses one captured graph. The compiler may specialize and cache additional graphs. This example establishes the declared first-order training sequence on the executed pinned CPU environment, without a general graph-count or performance guarantee.

## One changing-shape trajectory

The deterministic encoder/scorer/pool/head has 25 scalar parameters in five blocks. The compiled forward includes the original packed `ragged_pool` operation. Loss construction, logging, reference checks and optimizer steps remain outside capture. Backward uses the existing native derivative registration.

The same initial model state and fixture sequence are used for compiled native, eager native and independently differentiated primitive PyTorch models:

| Update | N | B | Offsets |
| --- | --- | --- | --- |
| 1 | 9 | 5 | [0,2,2,5,7,9] |
| 2 | 12 | 6 | [0,3,5,5,8,10,12] |
| 3 | 11 | 4 | [0,1,5,7,11] |
| 4 | 11 | 4 | [0,3,3,8,11] |
| 5 | 9 | 5 | [0,1,4,6,6,9] |

This varies N and B, changes group boundaries at unchanged metadata, and returns to an earlier shape. Empty groups occur within the training sequence. At every step, the consumer compares pooled values, predictions, scalar loss, raw-input gradients, each parameter gradient and updated parameters. Every parameter block must receive an informative finite gradient and actually change in all three routes. Inputs and offsets retain byte/version checks.

The independent reference performs ordinary primitive tensor math. Its Python group boundaries are prepared outside capture. Compiling a reference that reads offsets with `tolist()` would test a different Python tracing problem instead of independently checking native pooling.

## What dynamic means here

PyTorch's [versioned compile contract](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html) says that `dynamic=True` attempts symbolic tracing upfront. Operations and guards can still force specialization. `fullgraph=True` rejects graph breaks in the captured forward. It does not certify that all future shapes share one graph.

The package's fake forward computes output metadata `[B,D]` from offset count and feature width. Fake backward describes gradients `[N,D]` and `[N]`. These metadata rules let tracing reason about shapes without reading numerical contents. Native execution still checks offsets, finite input ranges and original primal admission. A fake tensor result is not evidence that imaginary offset contents are valid.

Shape guards and native data checks serve different purposes. The direct compiler checks change valid partitions without changing shapes, reject invalid offsets/ranges after a valid capture and then recover on valid data. They also test original saved-tensor versions, unsupported primal metadata and ordinary strided/expanded upstream gradients. The [contract](contract.md) retains contiguous CPU float32/float64 primals, int64 offsets, original size/range bounds, fresh outputs and no input mutation or device transfers.

The backend is `aot_eager`. Its [exact PyTorch v2.9.1 implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_dynamo/backends/debugging.py) uses AOTAutograd with no-op compilers. This is graph and automatic-differentiation integration, without an optimized-code, Inductor, speed or memory-saving result.

## Zero and one need separate checks

The five optimizer steps use nondegenerate N/B sizes. Separate direct compiled-pool controls cover all seven feasible combinations of zero, one and larger counts. N=0/B=0 requires offsets [0]. N=0 with positive B produces zero pooled rows. Singleton groups have zero logit derivatives. These structural cases receive independent output/gradient checks.

An empty output must not be fed to an undeclared mean loss, which would yield NaN. The direct pool corner controls use an appropriate finite objective instead of pretending that the five-step model trained on a zero-group batch.

Exact [v2.9.1 symbolic-shape source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/fx/experimental/symbolic_shapes.py) specializes input sizes zero and one by default. Its ordinary symbolic size range starts at two. Thus the ordinary training family does not prove that zero/one inputs reuse its graph. Extra specializations are allowed and reported. No private production shape setting, cache-limit increase or error suppression is used.

Test-only capture instrumentation records Dynamo backend submissions while delegating to the pinned `aot_eager` backend. The callable-backend interface follows [PyTorch's custom backend contract](https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html). The underscored registry lookup is explicitly pinned test instrumentation. Submission counts are separate from the deciding installed consumer's public backend string. They are not automatically counts of backward graphs, guard failures, native kernel compilations or optimized code. Actual recorded counts belong to their executed revision and sequence.

## Run the installed consumer

Use the [ordinary pinned wheel installation route](../README.md#build-and-run), with a fresh isolated environment outside the checkout. Copy `examples/train_dynamic.py` into a separate consumer directory. It is standalone and imports the ordinary installed package under `-I`. Use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\train_dynamic.py" --dtype float32 --output "$consumer\dynamic-float32.json"
& $python -I "$consumer\train_dynamic.py" --dtype float64 --output "$consumer\dynamic-float64.json"
```

Receipts identify the installed package/native library, own script bytes, precision, compile options, single compile API call, five optimizer steps, shape sequence, comparison errors, actual updates, ownership and thread settings. They explicitly avoid a graph-reuse or performance claim. Mathematical checks stay separate from operator registration. Completed revision-linked local and CI records establish execution, not a workflow file alone.

This addition retains earlier fixed-shape packed/padded compilation and eager second-order, JVP, functional-map, HVP and damped-solve routes. It does not compile those transforms or broaden padded dynamic shapes, compiled double backward, export, GPU or other toolchains. Other models, stochastic forwards, changing dtype/feature widths and general shape portability are outside this declared consumer.

Implementation and verification used Codex assistance, consistent with [NOTICE](../NOTICE). The execution record establishes component behavior, without unaided-expertise, model-quality, robustness, privacy, security or employer qualification claims.
