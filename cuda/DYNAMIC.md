# Train through changing packed CUDA shapes

`examples/train_compiled.py --dynamic` keeps one compiled encoder/scorer/pool/head model and one SGD optimizer through changes in both packed item count N and group count B. The same consumer's default remains the two-update fixed-shape route. Raw feature width is3 and pooled width is4. Native pooling and the installed CUDA package are unchanged.

The five seeded updates exercise `(N,B)=(9,5),(12,6),(12,4),(11,4),(9,5)`. The middle transitions vary B at fixed N, then N at fixed B. The final update returns to the first shape with new raw inputs, targets and partitions. Empty and singleton groups occur within the batches. This is a synthetic integration example, without a model-quality or performance claim.

## Use the installed consumer

Use the qualified Windows x64, Python3.11.8, PyTorch2.9.1+cu128/runtime12.8 and RTX4090 environment described in the [CUDA guide](README.md). Install the accepted CUDA wheel normally into a fresh isolated environment. Copy `examples/train_cuda.py` and `examples/train_compiled.py` together outside the checkout. Keep NumPy2.2.6 and pytest8.4.2 for the full checks.

```powershell
& $acceptPython -I C:\work\cuda-consumer\train_compiled.py --dynamic --dtype float32 --output C:\work\cuda-consumer\dynamic-float32.json
& $acceptPython -I C:\work\cuda-consumer\train_compiled.py --dynamic --dtype float64 --output C:\work\cuda-consumer\dynamic-float64.json
```

Each dtype owns three persistent models and SGD optimizers: compiled native, eager native and independent per-row GPU primitives. Every step compares complete predictions, pooled values, loss, raw-input gradients, all five parameter gradients and updated parameters. It requires finite informative gradients and an actual change in each parameter block. Source-input bytes/versions/storage, initial state, separate route storage and model/parameter/optimizer identities remain checked. Receipt hashes distinguish the consumer, adjacent helper, installed Python package and loaded native payload.

The compiler call is `torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=True)`. The model forward is captured. Loss construction, independent comparisons, logging and SGD remain outside capture. Actual `CompiledFunctionBackward` outputs witness AOT differentiation. One compile API call does not promise one graph.

The [PyTorch2.9 dynamic-shape guide](https://docs.pytorch.org/docs/2.9/torch.compiler_dynamic_shapes.html) explains that dynamic=True attempts generalization and guards can still specialize or recompile. The [Dynamo concepts guide](https://docs.pytorch.org/docs/2.9/compile/programming_model.dynamo_core_concepts.html) describes tracing, guards and recompilation. `aot_eager` executes captured forward/backward graphs eagerly, without optimizing generated kernels. Zero and one dimensions have separate specialization behavior. They are outside this new model-training sequence, while the established eager and fixed-shape corner tests retain their own evidence.

## Contract and deciding checks

The native contract remains ordinary contiguous same-device CUDA float32/float64 values`[N,D]`, logits`[N]` and int64 offsets`[B+1]`. Bounds are N<=4096, B<=128 and D1..128. Offsets start at zero, end at N and are nondecreasing. Numerical primals and all upstream entries must be finite within`[-1000,1000]`. Differentiation is first order only. Native admission retains its explicit four-byte host status transfer and stream synchronization. Inputs stay on CUDA and are not mutated.

Separate tests compare changing-shape forward/first-gradient math with independent row-wise double-softmax arithmetic. Other tests reuse the exercised compiled callable for content refusal and valid recovery, saved-original mutation refusal, metadata admission, fresh outputs and genuine expanded/transposed/stepped/logical-negative upstreams. Registration checks and finite differences remain separate in the established suite.

Pinned callback tests inspect symbolic input/output metadata and native forward/private-backward targets. They record actual submissions for the exercised sequence. This diagnostic path is separate from the public backend-string consumer and is not a universal graph-count or compiler-cache guarantee. No private symbolic-shape setting, error suppression or enlarged cache is needed by the declared route.

## Evidence limits

The accepted runtime wheel and native payload are reused. A checked source archive supplies the changed consumer, tests and guide. A new source archive does not imply a new runtime wheel or native compilation. Historical NVCC12.8.93/MSVC19.44/C++17/sm89 provenance and native diagnostics remain their original cohorts. This changing-shape caller has no new native sanitizer result.

Exact executed source, installation, test, graph and training results belong in the accompanying revision-linked record. Source descriptions alone are not passing results. Hosted Windows/Linux CPU and CPU ASan checks are separate repository regression gates, not CUDA CI.

This qualification does not extend padded CUDA to changing outer shapes, export to dynamic shapes, higher derivatives, transforms, CUDA graphs, Inductor/Triton, mixed precision, distributed training, other platforms/architectures/ABIs, performance, peak memory or operational security. Source remains MIT licensed with the existing attribution notices.
