# Hand a trained CUDA model to another process

The two adjacent examples save a small trained encoder/scorer/pool/head model and run its inference graph in a fresh Python process. The consumer imports the installed CUDA package, but does not import the producer, training script or original model class. The saved graph retains one native pooling operation. It still needs that installed operation and the pinned PyTorch environment.

This is a trusted-local model handoff. An expected manifest digest identifies bytes relative to a trusted producer receipt. It does not authenticate a publisher, sandbox a model or make an unknown PT2 archive safe. [PyTorch's security policy](https://github.com/pytorch/pytorch/security/policy) treats models and custom operators as executable code.

## The finite contract

Use Windows x64, Python3.11.8, PyTorch2.9.1+cu128/runtime12.8 and the qualified RTX4090 route. Each artifact has one dtype, float32 or float64. The default fixed artifact accepts ordinary contiguous CUDA0 tensors: raw features `[9,3]` and int64 offsets `[6]`. Raw entries are finite within `[-6,6]`. Offsets satisfy the native contract, with five groups covering all nine items. Predictions have shape `[5]` and pooled features `[5,4]`. Empty and singleton groups are included. The caller supplies the intended device and dtype. Prediction does not move or coerce inputs.

The producer performs two real SGD updates at learning rate0.03, comparing the native model with independently differentiated GPU primitives. It then places a copy of the selected model in evaluation mode and calls `torch.export.export(..., strict=True)` under `no_grad`. The default keeps static shapes and schema1. The explicit dynamic mode below uses schema2. The [PyTorch2.9 export API](https://docs.pytorch.org/docs/2.9/export/api_reference.html) documents this route and warns that saved artifacts may not work in newer versions. Export here is separate from `torch.compile`, exported training, an optimizing backend or a standalone CUDA executable.

The artifact directory contains exactly `manifest.json` and `model.pt2`. The manifest records the source labels, environment, trained weights as a small independent numerical oracle, expected outputs and the program's size/digest. The graph already stores its own parameters, so a second weights archive is unnecessary. The loader checks bounded regular files, exact names, manifest identity, schema and environment before deserializing the verified program bytes. It then checks the loaded graph and every parameter against the declared state. These checks define this example's owned-file contract rather than a general serialization service.

## Run the installed examples

Use an ordinary companion wheel in a fresh isolated environment outside the checkout, following the [CUDA installation route](README.md#one-explicit-installation-route). Install the pinned test dependencies when running checks:

```powershell
& $acceptPython -m pip --isolated install numpy==2.2.6 pytest==8.4.2
```

Copy `export_cuda.py`, `predict_cuda.py` and `train_cuda.py` together from this source revision into an external consumer directory. Use a new artifact directory and receipt path for every run. The two commands below are separate process invocations. Repeat them with float64 and different paths for the other dtype.

```powershell
$consumer = 'C:\work\cuda-consumer'
$artifact = 'C:\work\cuda-owned-float32'
$revision = git -C $sourceCheckout rev-parse HEAD
$tree = git -C $sourceCheckout rev-parse 'HEAD^{tree}'
& $acceptPython -I "$consumer\export_cuda.py" --dtype float32 --directory $artifact --source-revision $revision --source-tree $tree --wheel $wheel --output "$consumer\producer-float32.json"
$produced = Get-Content -LiteralPath "$consumer\producer-float32.json" -Raw | ConvertFrom-Json
& $acceptPython -I "$consumer\predict_cuda.py" --directory $artifact --manifest-sha256 $produced.artifact.manifest_sha256 --output "$consumer\consumer-float32.json"
```

The predictor compares changed raw data and offset partitions against primitive GPU math and saved native canaries. A within-group permutation must preserve both outputs. It checks original input bytes/versions, parameters and fresh output storage. Separate controls test native offset refusal and recovery, fixed-shape guards and rejected artifact changes before deserialization. Mathematical agreement, graph inspection and file admission are distinct checks.

## One artifact for changing packed shapes

Add `--dynamic` to the producer to export independent item and group counts. This inference subset accepts raw `[N,3]` with `2 <= N <= 32` and int64 offsets `[B+1]` with `2 <= B <= 8`. Device remains CUDA0 and dtype remains the artifact's float32 or float64. Inputs are ordinary contiguous tensors without gradients or logical negative/conjugate views. Raw features remain finite in `[-6,6]`. Predictions have shape `[B]` and pooled features `[B,4]`. Empty groups produce zero pooled vectors, while their predictions may equal the trained head bias.

N and B are independent. Two items can occupy eight groups when six groups are empty. The producer captures an interior N9/B5 example with independent named `items` and `offset_count` dimensions. It saves one graph per dtype. The same loaded graph is then reused over changed counts, including the four N/B corners, new partitions and data at equal shapes, and a return to an earlier pair. Changing a boundary remains a runtime input rather than a new capture.

Use new paths for the dynamic artifact and receipts, then run the same predictor command. The loader selects the exact supported contract from the manifest schema rather than a consumer flag:

```powershell
$dynamicArtifact = 'C:\work\cuda-dynamic-float32'
& $acceptPython -I "$consumer\export_cuda.py" --dynamic --dtype float32 --directory $dynamicArtifact --source-revision $revision --source-tree $tree --wheel $wheel --output "$consumer\dynamic-producer-float32.json"
$produced = Get-Content -LiteralPath "$consumer\dynamic-producer-float32.json" -Raw | ConvertFrom-Json
& $acceptPython -I "$consumer\predict_cuda.py" --directory $dynamicArtifact --manifest-sha256 $produced.artifact.manifest_sha256 --output "$consumer\dynamic-consumer-float32.json"
```

The dynamic loader checks actual symbolic input/output metadata and exact range constraints after loading the verified bytes. It verifies the relationship between offsets length and output B, independent item/offset symbols, fixed widths, dtype/device, one native pool and no backward node. This pinned metadata inspection does not promise stable internal symbolic names or a cross-version format.

Explicit consumer admission and saved graph guards are separate. In PyTorch2.9.1, a printed `Dim(min=2)` range does not strictly exclude zero or one at runtime. The [exact upstream input check](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_export/utils.py#L397-L417) skips that lower-bound check when the minimum is at most two. This consumer explicitly refuses N0/1 before invoking the graph. Direct-module tests observe the actual guard behavior separately, including offset-count bounds. Raw numerical bounds are also wrapper admission: tanh can turn an infinite raw value into a finite encoded value, so native encoded-input validation alone cannot establish the raw-feature contract. Invalid offset contents still reach native validation and must be refused before indexed pooling work, followed by successful valid recovery.

Schema1 retains the fixed contract and six cases. Schema2 has its own exact contract, ranges and variable-B canaries. A schema change around a static program does not supply the required symbolic graph metadata. Neither schema admits unknown fields or treats a digest as permission to load untrusted code.

## Executed acceptance

The [local validation record](evidence/export-windows-sm89-20261021.json) identifies executed source cb5c8b565b1eb7603a4dfbb2093b624692866803 and its checked archive, reused wheel, native payload and source hashes. All seven listed checks passed. The installed suite reported256passed,4skipped and1warning in8.04seconds, including the previous175CUDA tests. The four skipped symlink fixtures could not be created with ordinary Windows privilege. PyTorch's PT2 loader emitted a nonwritable-buffer warning, which remains visible in the logs. Loaded inference state byte/version checks passed. Neither skipped checks nor the warning were hidden.

Separate producer and consumer invocations passed for both dtypes, with distinct process identities and no producer/training imports in the consumer. Each producer performed two matched updates and changed every parameter block in both routes. Captured and reloaded native outputs exactly matched eager native canaries in the six declared cases. Float32 comparisons with independent math were zero in this cohort. Float64 pooled inference differences were at most2.220446049250313e-16. These small numerical comparisons establish the exercised integration, without model-quality or general numerical-error claims.

Review caught a draft dtype-variable shadowing defect before execution. The first passing cohort then exposed a receipt-label bug: merging an adjacent helper's identity dictionary overwrote the producer's top-level script hash. The artifact manifest and consumer retained the correct producer identity. The producer now checks and removes the helper's generic label before merging its other fields. Final acceptance reran the corrected source. Literal byte checks also replaced numerical equality for immutability, with a signed-zero write that bypasses the version counter as a sensitive regression control. Earlier attempts remain preserved privately.

## Choices and limits

Strict export is explicit because the pinned API's default is False. The opaque custom operation keeps its runtime validation kernel and synchronization. Fake metadata permits capture without reading offsets. Changing offset contents must still affect execution after reload. A captured shape is not a substitute for content admission.

The existing accepted wheel and native binary can be reused because this increment changes examples, tests and documentation rather than runtime package code. Its original NVCC12.8.93/MSVC19.44/C++17 provenance remains attached to that payload. Reusing it is not a fresh compiler result. The source archive includes these new examples and tests through its existing manifest rules.

The dynamic route covers only its declared packed inference subset. Zero/one items or groups, dynamic padded export, CUDA1 or other devices, mixed precision, exported backward, graph capture, Inductor/Triton, cross-version serialization/ABI compatibility, performance, peak memory, model quality and hostile artifact safety remain outside it. Earlier eager sanitizer evidence retains its original scope. The export consumer has ordinary functional execution checks rather than a new instrumented native cohort.
