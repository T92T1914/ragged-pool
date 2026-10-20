# Padded batches and masking

The native operator consumes packed items and offsets. A model may instead already have one padded tensor and a mask. `ragged_pool_padded(values, logits, mask)` supplies that conversion without changing the C++ operation. `ragged_pool_padded_reference` is a readable pure PyTorch oracle that selects and pools each row independently.

## Contract

Values have shape `[B,T,D]`, logits `[B,T]` and mask `[B,T]`. All three inputs must be contiguous strided CPU tensors. Values and logits share float32 or float64 dtype; the mask is bool, with true selecting an item. B is from zero through 128, T from zero through 4096, D from 1 through 128, and `B*T <= 4096`. This bounds the raw padded slots, rather than only the selected count. Highly skewed groups that exceed the padded limit can use the original packed API.

Mask holes are allowed. Selected entries retain their within-row order. The implementation selects both values and logits before any softmax or multiplication, computes int64 row counts and cumulative offsets, then calls the native operator once. Selected values and logits must be finite and in `[-1000,1000]`. Ignored padding may contain NaN, infinity or arbitrary sentinels. An all-masked row yields a zero vector with zero input gradients. Empty batches and zero sequence width are included. The mask has no derivative.

Advanced indexing allocates packed storage. It does not mutate the padded tensors or move/cast them. This is an explicit adapter cost, not a zero-copy or performance claim. [PyTorch indexing semantics](https://docs.pytorch.org/docs/2.9/tensor_view.html) describe that copy behavior; the [2.9.1 indexing source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/IndexingUtils.h) and [nonzero ordering](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html) support the row-order construction. Gradients scatter back only into selected positions. Ignored entries receive exactly zero gradients.

The packed copies are the native operation's saved inputs. Do not assume that mutating the original padded values after forward produces the same saved-tensor version error as mutating the packed API's original values. Ordinary PyTorch indexing rules still apply, including to its saved mask. The no-input-mutation promise describes the operation's behavior.

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
```

The model uses seeded synthetic features and targets, an encoder, learned scores, the pooling layer and a prediction head. From identical initial states, native/reference routes compare predictions, raw-input and parameter gradients, and two SGD parameter updates. Padding gradients must be zero, parameter gradients finite and inputs unchanged. Loss is an integration check, with no accuracy or model-quality result.

The adapter tests compare both input gradients with the independent reference, check finite differences on finite fixtures, and exercise holes, empty dimensions, numerical endpoints, ignored versus selected invalid data, unsupported metadata, ownership and upstream gradient views. Existing operator registration checks remain separate. All three CPU workflows run the installed padded consumer. Use their exact source revision and completed result as execution evidence.

## Limits

This is an eager-only, first-order CPU adapter. Boolean selection produces a data-dependent packed length. It adds no custom operator, fake registration, compiler override or decomposition. The original packed `aot_eager` compile and trained export routes keep their declared scope. No padded compile/export, vmap, second derivative, GPU, speed advantage or cross-version compatibility is established by this addition.
