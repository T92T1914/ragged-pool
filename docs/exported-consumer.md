# A trained model as an exported consumer

This example takes the existing synthetic presence classifier through training, a tensor-only checkpoint, strict graph export and inference in a fresh process. It makes one native CPU model reusable beyond its training process. The saved graph still needs an installed compatible ragged-pool library. It is not a standalone binary or a new inference runtime.

`export_sets.py` uses native attention with model seed 11, the existing 512 training bags and 128 validation bags, 200 Adam updates and earliest minimum validation BCE. It does not evaluate the held-out sets again or change the N2 comparison. The selected weights are saved as an ordinary state dictionary, explicitly loaded with `weights_only=True`, checked for keys, shapes, dtype and finite contents, then compared with the independent PyTorch reference. The shorter `--smoke` route performs 20 updates and is labeled separately.

The producer sets the model to evaluation mode, calls `torch.export.export(..., strict=True)` and saves the graph with `torch.export.save`. The graph retains one `ragged_pool::pool` operation. Its backward is not captured in this inference artifact. The consumer imports the installed package before `torch.export.load`, obtains the guarded `.module()` and runs it under `torch.inference_mode()`. It does not import the training script or reconstruct the training model class. PyTorch 2.9.1 prohibits calling `.eval()` on that loaded module, so evaluation mode is selected before export. [Pinned export APIs](https://docs.pytorch.org/docs/2.9/export.html), [versioned ExportedProgram implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/export/exported_program.py#L1298).

## Exported consumer contract

The model takes contiguous CPU float32 `values[N,4]` and contiguous CPU int64 `offsets[B+1]`. Its raw features must be finite and within `[-6,6]`, matching the synthetic generator. The input width is fixed at four, while the encoded native pool width is 16. Offsets start at zero, end at N and are nondecreasing. The exported route admits `0 <= N <= 4096` and `2 <= B <= 128`. Zero-length groups and an entirely empty item array within those group bounds are included.

The first strict-export feasibility attempt requested the eager operator's full group-count range and failed a generated constraint requiring at least three boundaries. The example therefore exports the narrower range explicitly and checks it at its consumer boundary. It does not create several specialized graphs to conceal that limitation. The existing eager operator still supports zero and one group. Printed symbolic ranges alone are not evidence that every zero/one size works. The executed shape cases determine coverage.

An empty group's pooled feature vector is zero, but the classifier head can add its learned bias. The expected classifier logit is therefore compared with the eager model, rather than assumed to be zero. Changed offsets at unchanged tensor shapes are tested because group membership remains runtime data.

## Artifact identity and trust

The output directory contains `weights.pt`, `model.pt2` and `manifest.json`. The manifest records their identities, the source and pinned environment, selected training checkpoint and deterministic prediction cases. The consumer requires the expected manifest SHA256 from its caller and verifies both artifact file hashes before graph deserialization. This catches accidental changes relative to that expected identity. The producer and consumer use new output paths and preserve failed attempts.

Only load trusted artifacts you produced or otherwise trust as programs. `torch.export.load` is not a restricted `weights_only` loader; the pinned PT2 implementation can use unrestricted loading internally. Hashing a file or replacing its adjacent manifest does not authenticate its publisher, and these checks are not a sandbox. The separate `weights_only=True` checkpoint route has a narrower loading policy without making an arbitrary `.pt2` safe. [PyTorch serialization guidance](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch-load-with-weights-only-true), [versioned PT2 loader](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/export/pt2_archive/_package.py), [PyTorch security policy](https://github.com/pytorch/pytorch/blob/v2.9.1/SECURITY.md#using-pytorch-securely).

This is one pinned CPU route, without cross-version artifact or ABI compatibility claims. The native operation remains opaque, and its runtime offset/content admission remains relevant after graph loading. The model's raw-input checks belong to the example consumer, outside the exported graph. Export success does not demonstrate optimized inference, deployment scale, or offensive/defensive security work.

## Execution record

The full producer and fresh consumer passed on October 20, 2026 using the examples at source [ae53f69](https://github.com/T92T1914/ragged-pool/tree/ae53f69dc4ec5bf51eb97c0936a1abb32c465d63). An ordinary Windows x64 wheel was installed into a fresh environment outside the checkout. It used CPython 3.11.8, PyTorch 2.9.1+cpu and the existing MSVC 19.44.35228/C++17 extension route. One execution worker and one intra-op/interop thread were used. The native implementation was unchanged.

The producer completed 200 Adam updates with seed 11, selected update 200 through validation BCE, and generated only training/validation data. Its restricted weight reload preserved every parameter exactly. The reloaded native model and saved/reloaded export matched the original selected model exactly on all eight declared cases. The independent PyTorch reference differed by at most `3.58e-7`. This is an artifact-consistency check, not another model-quality evaluation. Existing held-out results remain in the N2 study.

The separate consumer process loaded the exported graph from the installed package, imported no training module, passed all eight prediction cases and left every input unchanged. Its predictions matched the manifest's selected-model outputs exactly. Cases include the same values with changed offsets, zero items/two groups, one item/two groups, mixed empty groups, 128 groups and 4096 items. These observations qualify the narrower exported contract without upgrading it to arbitrary dynamic-shape support.

| Identity | Executed value |
| --- | --- |
| Manifest SHA256 | `43195303bb081a2b3dabb47448d43f57fa60068f6821c51bddb7861845a8edbd` |
| Exported graph | 17,952 bytes, SHA256 `21aa7f763ed7fee8f32341e5be907bebc75571b8b4e98559e0b90a864c09823f` |
| Tensor checkpoint | 2,905 bytes, SHA256 `4d2fa6ebb4ef767dd7bc191275c656c85f8eed24d2ed37a701f458ed36d551e6` |
| Installed wheel SHA256 | `3bb4367b0e95f528f468e6987beee43e6774dc0ca877b9d2ae38f9cb4aadf94c` |
| Loaded native library SHA256 | `528fd536ef5b64583b5e9d6b62bd347ac067dea270a5a57cf4118e9f46a2fb14` |

The [machine-readable producer and consumer record](results/export-roundtrip.json) preserves the executed values and source hashes with local paths replaced. The trained local `.pt2` is not committed as a generally portable artifact. CI creates and retains its own separately identified 20-update smoke artifact using that job's installed wheel. Its result is not the full 200-update run, and a new build need not produce identical archive bytes.

The installed test suite passed 214 cases, including restricted checkpoint failures, digest/schema/version rejection before graph deserialization, actual dynamic execution and direct loaded-graph rejection of malformed offsets. The first test attempt expected a `RuntimeError` for the saved small-group guard; PyTorch 2.9.1 correctly raised `AssertionError`. The corrected test asserts that exact guard message, without changing the implementation's rejection. Both the full-range export failure and the original test failure remain in the private execution record.

PyTorch emitted its existing prototype nested-tensor warning and a PT2 loader warning about a non-writable buffer. They were retained rather than suppressed. This route performs inference without writing those weights. It makes no writable-weight or arbitrary-artifact safety claim. The original random-weight probes establish API feasibility only; trained acceptance is supplied by the later producer and fresh consumer records above.
