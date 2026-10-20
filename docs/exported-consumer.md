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

The integrated producer, fresh consumer and final installed checks are pending. The earlier random-weight shape probes establish API feasibility only. Their failed and narrowed outcomes are retained in the local engineering record. This section will record actual trained-artifact acceptance when it has executed.
