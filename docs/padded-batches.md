# Padded batches and masking

The native operator consumes packed items and offsets. A model may instead already have one padded tensor and a mask. `ragged_pool_padded(values, logits, mask)` supplies that conversion without changing the C++ operation. `ragged_pool_padded_reference` is a readable pure PyTorch oracle that selects and pools each row independently.

## Contract

Values have shape `[B,T,D]`, logits `[B,T]` and mask `[B,T]`. All three inputs must be contiguous strided CPU tensors. Values and logits share float32 or float64 dtype; the mask is bool, with true selecting an item. B is from zero through 128, T from zero through 4096, D from 1 through 128, and `B*T <= 4096`. This bounds the raw padded slots, rather than only the selected count. Highly skewed groups that exceed the padded limit can use the original packed API.

Mask holes are allowed. Selected entries retain their within-row order. The implementation selects both values and logits before any softmax or multiplication, computes int64 row counts and cumulative offsets, then calls the native operator once. Selected values and logits must be finite and in `[-1000,1000]`. Ignored padding may contain NaN, infinity or arbitrary sentinels. An all-masked row yields a zero vector with zero input gradients. Empty batches and zero sequence width are included. The mask has no derivative.

Advanced indexing allocates packed storage. It does not mutate the padded tensors or move/cast them. This is an explicit adapter cost, not a zero-copy or performance claim. [PyTorch indexing semantics](https://docs.pytorch.org/docs/2.9/tensor_view.html) describe that copy behavior; the [2.9.1 indexing source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/IndexingUtils.h) and [nonzero ordering](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html) support the row-order construction. Gradients scatter back only into selected positions. Ignored entries receive exactly zero gradients.

The original padded values, logits and mask are saved for backward. Do not mutate them between forward and backward, including ignored slots. Backward repacks those originals and uses PyTorch's saved-tensor version checks. This tightens the earlier eager adapter's lifecycle: that version saved packed copies and did not necessarily reject a later change to original padded values or logits. The no-input-mutation promise describes the operation's behavior, rather than permitting callers to change saved tensors.

## Compiled training boundary

The public wrapper checks original metadata before calling an opaque Python CPU custom operation, `ragged_pool::padded`. That operation performs compaction and calls the existing C++ packed forward. Its registered first-order autograd formula calls a second opaque operation, `ragged_pool::_padded_backward`, which repacks unchanged inputs, calls the existing C++ backward and scatters the derivatives into fresh padded zeros. No second numerical derivative implementation is added.

Both operations have fake tensor implementations. Their visible outputs are `[B,D]` and `[B,T,D]`/`[B,T]`, determined entirely by input metadata. The selected item count remains inside each CPU implementation. Fake execution never reads the mask. This follows the supported [custom operation and autograd registration APIs](https://docs.pytorch.org/docs/2.9/library.html#torch.library.register_autograd), checked against the [2.9.1 implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_library/custom_ops.py). The raw dispatcher operations are internal integration points; callers should use the public wrapper for admission checks on original views before dispatcher materialization.

The declared path is `torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)` on the pinned CPU environment. It captures model forward and its autograd backward with fixed input shapes. The same compiled callable can accept different mask contents and selected counts at those shapes. This does not require enabling data-dependent output-shape capture. The surrounding model must still clear ignored raw padding before its encoder. The optimizer itself remains ordinary eager SGD.

This route demonstrates graph capture and autograd integration. The CPU implementations still execute Python packing and the existing native math at runtime. `aot_eager` does not establish optimized generated kernels, fusion through the opaque boundary or a speed improvement. The new registration adds dispatch and repacking costs that are not measured here.

## A worked group

Suppose one row contains values `[[2,0], [99,99], [0,4]]`, logits `[0,99,log(3)]` and mask `[true,false,true]`. The adapter selects `[[2,0],[0,4]]` and `[0,log(3)]`. Softmax weights are `[1/4,3/4]`, so the output is `[0.5,3]`. The middle slot contributes neither to normalization nor to either input gradient. A following all-false row repeats the same offset and returns `[0,0]`.

The reference computes each row separately and branches around empty selections. It does not share the adapter's offsets or call the native operation. A dense shortcut that applies softmax to an entirely `-inf` row requires its own empty-row handling; it is not a valid oracle by itself.

## Padding must be cleared before upstream arithmetic

The adapter can ignore a raw NaN that it never selects. It cannot repair invalid arithmetic that an earlier encoder/scorer has already placed in the autograd graph. A linear layer can have a finite masked forward result and still accumulate a parameter gradient involving `NaN * 0`. [PyTorch's autograd notes](https://docs.pytorch.org/docs/2.9/notes/autograd.html#division-by-zero-in-autograd) explain why masking after an invalid operation is insufficient.

The training example explicitly replaces ignored raw features with zero using `torch.where` before its encoder and scorer. It then passes encoded features, scores and the mask to the adapter. Its debugging counterexample records the unsafe ordering separately from the corrected model. This is an input-boundary lesson, not evidence of robustness to arbitrary model inputs.

## Installed consumer and checks

Build and install the wheel using the [Windows](../README.md#build-and-run) or [Linux source distribution](linux-cpu.md) route. Copy `examples/train_padded.py` outside the checkout, use the fresh acceptance environment and run each dtype in a separate process:

```powershell
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_padded.py' --dtype float32
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_padded.py' --dtype float64
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_padded.py' --dtype float32 --compiled
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_padded.py' --dtype float64 --compiled
```

The model uses seeded synthetic features and targets, an encoder, learned scores, the pooling layer and a prediction head. From identical initial states, native/reference routes compare predictions, raw-input and parameter gradients, and two SGD parameter updates. `--compiled` adds the compiled native model to those comparisons and exercises the same callable with changed mask contents, including all-false and all-true masks. Padding gradients must be zero, parameter gradients finite and inputs unchanged. Loss is an integration check, with no accuracy or model-quality result.

The adapter tests compare both input gradients with the independent reference, check finite differences on finite fixtures, and exercise holes, empty dimensions, numerical endpoints, ignored versus selected invalid data, unsupported metadata, ownership and upstream gradient views. Separate registration tests check both new operations' fake metadata and supported autograd registration. Private-backward AOT checks use detached inputs; a separate requires-grad check inspects its autograd registration, and an explicit derivative test confirms the first-order limit. These are not substitutes for the reference and finite differences.

Compiler tests count one captured forward graph across changing mask contents, confirm it contains the padded operation without exposed selection, execute backward with ordinary upstream views, and check runtime rejection of selected invalid contents. All three CPU workflows run the installed eager and compiled padded consumer in both dtypes. Use their exact source revision and completed result as execution evidence.

## Limits

The adapter's training route is first-order CPU with one declared fixed-shape `aot_eager` path. Changed input shapes may compile new graphs. A separate [trained padded export example](padded-export.md) supports its own bounded variable B/T inference envelope. Inductor optimization, vmap and padded second derivatives remain outside the declared routes. The original packed compiler, second-order and trained export routes keep their own scope. No GPU, speed advantage or cross-version compatibility is established.
