# Train, save and consume a padded model

The padded API can be used inside a small model that is trained, strictly exported and loaded in a separate Python process. `examples/export_padded.py` takes two SGD steps through the installed native package, saves the updated weights and a `torch.export` program, then checks saved predictions against the trained eager model and an independent per-row PyTorch reference. `examples/predict_padded.py` loads that owned artifact without importing the producer or the original model class.

This is a framework integration example. The two updates demonstrate that trained parameters survive serialization. They do not measure model quality, production usefulness, robustness or speed. The native numerical implementation and autograd registrations are unchanged.

## Input and output contract

Each dtype has a separate artifact. Raw input is a contiguous strided CPU float32 or float64 tensor `[B,T,3]`; mask is contiguous CPU bool `[B,T]`. The saved caller admits `2 <= B <= 32` and `2 <= T <= 128`, so its largest rectangle contains 4,096 slots. Selected raw features must be finite and within `[-6,6]`. Ignored entries may contain NaN or either infinity. The broader pooling library still has its own [padded contract](padded-batches.md).

The model clears ignored raw entries with `where` before the encoder, applies a linear encoder and tanh, computes learned logits, calls native padded pooling and applies a scalar prediction head. The output is `(predictions[B], pooled[B,4])`. An empty row has a zero pooled vector. Its prediction can equal the head bias, which need not be zero.

Mask contents remain runtime inputs. Cases compare holes, singletons, mixed empty rows, all-false masks, different B/T shapes and the rectangle's corners. The same selected items are also placed in different holes with more padding and changed ignored nonfinite values. Their pooled values and predictions must agree. Inputs retain their data and version counters.

## Exported graph and explicit admission

The producer requests `strict=True` explicitly and shares named B/T dimensions between raw inputs and mask. The exported graph must retain exactly one `ragged_pool.padded.default` forward node, with no other ragged-pool or backward operation. Compaction remains inside the existing opaque CPU operation. The archive stores a reference to that operation, so importing the installed `ragged_pool` package before loading supplies the native and Python registrations. It is not a standalone runtime or a cross-version ABI guarantee.

The predictor validates original metadata, shape bounds and selected raw contents before invoking the graph. These checks are separate from the exported program's guards. In pinned PyTorch 2.9.1, a printed `Dim(min=2)` range does not itself guarantee rejection of size zero or one. The example records actual direct size-one behavior while its supported consumer explicitly refuses those shapes. The [exact input-check implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_export/utils.py#L397-L417) explains that distinction. A test uses a spy graph to verify that rejected inputs never reach it.

## Owned artifact and reference

The directory contains exactly three bounded ordinary files: `manifest.json`, `weights.pt` and `model.pt2`. An expected manifest SHA256 comes from the producer receipt through a trusted route. The consumer checks the exact schema, declared dtype/contract, pinned environment and installed native identity, then verifies both payloads before deserialization. It loads the exact verified buffers, so reopening a changed path cannot substitute different bytes after verification. Tensor state uses `weights_only=True` with CPU placement and exact key/shape/dtype/finite checks.

The fresh consumer uses those admitted weights in a small functional reference: encoder and score arithmetic followed by independent per-row softmax and weighted reduction. That reference calls no custom pooling operation and imports no training class. Predictions and pooled values are checked against it and the producer's recorded expectations.

Only load artifacts produced from code and dependencies you trust. Hash agreement identifies bytes relative to a trusted receipt. It does not authenticate an unknown publisher or make hostile serialized files safe. This example retains the existing [packed artifact](exported-consumer.md) trust boundary rather than introducing a general model registry or service.

## Installed commands

Build and install the ordinary wheel using the [Windows](../README.md#build-and-run) or [Linux source archive](linux-cpu.md) route. Copy the complete `examples` directory outside the checkout. Keep the pinned environment outside the source tree and use new artifact/receipt paths. Supply the actual complete source revision and the wheel used for installation:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$examples = 'C:\work\ragged-consumer\examples'
$revision = 'FULL_40_CHARACTER_SOURCE_REVISION'
$wheel = 'C:\work\dist\ragged_pool-0.1.0-cp311-cp311-win_amd64.whl'
& $python -I "$examples\export_padded.py" --dtype float32 --directory 'C:\work\padded-f32' --source-revision $revision --wheel $wheel --output 'C:\work\padded-producer-f32.json'
$digest = (Get-Content 'C:\work\padded-producer-f32.json' -Raw | ConvertFrom-Json).manifest_sha256
& $python -I "$examples\predict_padded.py" --directory 'C:\work\padded-f32' --manifest-sha256 $digest --output 'C:\work\padded-predictor-f32.json'
```

Check each command's exit status. Run float64 in separate new paths with `--dtype float64`. The producer and predictor are separate processes, and all three CPU workflows execute both dtype routes alongside the earlier consumers. Actual completed job revisions establish execution. Workflow text alone does not.

The supported route remains CPython 3.11.8, PyTorch 2.9.1+cpu and the declared C++17 CPU build. This example adds inference consumption without changing packed second-order support, padded first-order autograd or compiled first-order acceptance. Padded second derivatives, compiled double backward, transforms, GPU execution and new performance claims remain outside it.
