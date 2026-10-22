# Hand a trained CUDA model to another process

The two adjacent examples save a small trained encoder/scorer/pool/head model and run its inference graph in a fresh Python process. The consumer imports the installed CUDA package, but does not import the producer, training script or original model class. The saved graph retains one native pooling operation. It still needs that installed operation and the pinned PyTorch environment.

This is a trusted-local model handoff. An expected manifest digest identifies bytes relative to a trusted producer receipt. It does not authenticate a publisher, sandbox a model or make an unknown PT2 archive safe. [PyTorch's security policy](https://github.com/pytorch/pytorch/security/policy) treats models and custom operators as executable code.

## The finite contract

Use Windows x64, Python3.11.8, PyTorch2.9.1+cu128/runtime12.8 and the qualified RTX4090 route. Each artifact has one dtype, float32 or float64. Inputs are ordinary contiguous CUDA0 tensors: raw features `[9,3]` and int64 offsets `[6]`. Raw entries are finite within `[-6,6]`. Offsets satisfy the native contract, with five groups covering all nine items. Predictions have shape `[5]` and pooled features `[5,4]`. Empty and singleton groups are included. The caller supplies the intended device and dtype. Prediction does not move or coerce inputs.

The producer performs two real SGD updates at learning rate0.03, comparing the native model with independently differentiated GPU primitives. It then places a copy of the selected model in evaluation mode and calls `torch.export.export(..., strict=True)` under `no_grad`. Shapes remain static. The [PyTorch2.9 export API](https://docs.pytorch.org/docs/2.9/export/api_reference.html) documents this route and warns that saved artifacts may not work in newer versions. Export here is separate from `torch.compile`, exported training, an optimizing backend or a standalone CUDA executable.

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

## Choices and limits

Strict export is explicit because the pinned API's default is False. The opaque custom operation keeps its runtime validation kernel and synchronization. Fake metadata permits capture without reading offsets. Changing offset contents must still affect execution after reload. A captured shape is not a substitute for content admission.

The existing accepted wheel and native binary can be reused because this increment changes examples, tests and documentation rather than runtime package code. Its original NVCC12.8.93/MSVC19.44/C++17 provenance remains attached to that payload. Reusing it is not a fresh compiler result. The source archive includes these new examples and tests through its existing manifest rules.

This route does not qualify dynamic shapes, CUDA1 or other devices, mixed precision, exported backward, graph capture, Inductor/Triton, cross-version serialization/ABI compatibility, performance, peak memory, model quality or hostile artifact safety. Earlier eager sanitizer evidence retains its original scope. The export consumer has ordinary functional execution checks rather than a new instrumented native cohort.
