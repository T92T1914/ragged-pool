# Padded second-order training

The eager padded API supports differentiating through its first backward. A model can therefore train with a loss that depends on its raw input gradient while keeping native pooling in the training path. The adapter packs selected entries, calls the existing C++ second derivative and scatters the results into padded zeros. It adds no new native numerical formula.

The original [padded input contract](padded-batches.md) remains: contiguous strided CPU values `[B,T,D]`, logits `[B,T]` and bool mask `[B,T]`, matching float32/float64 values and logits, `B <= 128`, `T <= 4096`, `1 <= D <= 128` and `B*T <= 4096`. Only selected values/logits must be finite and within `[-1000,1000]`. Ignored nonfinite padding is selected out before numerical operations.

## Derivative and saved inputs

`ragged_pool::_padded_backward` saves original output cotangent `g`, values, logits and mask. Its registered derivative calls opaque CPU `ragged_pool::_padded_double_backward`. Given adjoints for the first value/logit gradients, that operation packs only selected entries and calls native `ragged_pool::_double_backward` once. It returns derivatives with respect to `g`, values and logits. Mask has no derivative. The derivative with respect to `g` preserves parameter paths through a trainable prediction head. The [packed derivative note](second-order.md) explains the underlying equations.

Fresh padded derivative outputs start at zero and receive only selected item derivatives. Empty rows and masked slots have zero derivatives. A singleton has zero second derivatives with respect to its value and logit, while its first-gradient adjoint can contribute to the derivative with respect to `g`. Ordinary strided upstream views, including noncontiguous, stepped, expanded zero-stride and logical negative views, are admitted or normalized internally.

Do not mutate saved original tensors before either backward, including ignored padding or mask cells. PyTorch saved-tensor version checks enforce that rule. Saving packed snapshots would hide those original changes and lose the existing ownership contract. Upstream magnitudes remain unbounded, so arbitrary upstream NaN/Inf or extreme values do not carry a numerical stability guarantee.

## A penalty that ignores padded width

`examples/train_padded_gradient_penalty.py` uses a seeded encoder, tanh, bias-free scorer, padded pool and scalar prediction head. It clears ignored raw entries before the encoder, then constructs the gradient of summed predictions with respect to raw features using `create_graph=True`. For the selected raw features it computes:

```text
R = sum(selected_raw_input_gradient ** 2)
    / (selected_item_count * raw_feature_width)
loss = MSE(predictions, targets) + 0.5 * R
```

The selected gradient is indexed before squaring. When the selection is empty, the example defines a differentiable zero penalty. This denominator is a deliberate choice for this objective. It avoids changing the regularizer merely because the same items occupy a wider padded tensor.

From identical initial parameters, native and independent per-row reference models compare pools, predictions, MSE, penalty, first/total raw gradients, every parameter gradient and two SGD updates in both dtypes. Nonempty training requires all five parameters to update. The same selected data, targets and initial parameters are also tested with moved holes, doubled padded width and changed ignored padding. Their two-step trajectories must agree within dtype tolerances. Ignored raw gradients must remain zero, gradients finite and input bytes/version counters unchanged.

The controls make two different mistakes visible. Detaching the penalty keeps its scalar value but removes its gradient contribution, so the resulting parameter gradients match MSE alone. The proper penalty must change scorer/head weight gradients. A separate dense-mean control divides by every padded raw feature, including zero gradient slots. Doubling padded width then halves that control's penalty at fixed selected data. Dense normalization is padding-sensitive for this chosen objective, not universally invalid.

For an independent scalar check, softmax weights `[1/4,3/4]`, head weight `2` and two selected scalar raw features give input gradients `[0.5,1.5]`. The selected-feature penalty contribution `0.5 * mean(input_gradient**2)` is `0.625`, and its head derivative is also `0.625`.

## Run the installed caller

Build and install a wheel using the [Windows](../README.md#build-and-run) or [Linux source archive](linux-cpu.md) route. Install into a fresh environment outside the checkout, copy the standalone example outside the checkout and use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$example = 'C:\work\ragged-consumer\train_padded_gradient_penalty.py'
& $python -I $example --dtype float32 --output 'C:\work\ragged-consumer\padded-penalty-float32.json'
if ($LASTEXITCODE -ne 0) { throw 'Installed float32 padded penalty caller failed' }
& $python -I $example --dtype float64 --output 'C:\work\ragged-consumer\padded-penalty-float64.json'
if ($LASTEXITCODE -ne 0) { throw 'Installed float64 padded penalty caller failed' }
```

The receipt declares `selected_raw_features` normalization and retains training, invariance and control results. All three CPU workflows invoke both dtypes and retain their receipts. Use an actual completed run and revision to establish execution. Workflow source alone is not a passed check.

## Checks and limits

Mathematical checks compare all three second-VJP outputs with independent PyTorch differentiation, float64 `gradgradcheck` and a hand-calculated example. They cover unused branches, empty/singleton rows, numerical endpoints, upstream views, ignored nonfinite padding, malformed metadata and saved original versions. Operator registration/fake/AOT checks are separate from those mathematical checks.

The eager derivative chain uses supported [autograd registration](https://docs.pytorch.org/docs/2.9/library.html#torch.library.register_autograd), with fake metadata for every training operation. Third derivatives are explicitly rejected. The existing `aot_eager`, `fullgraph=True`, `dynamic=False` training path remains first-order. Exact [PyTorch 2.9.1 AOT runtime](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_functorch/_aot_autograd/runtime_wrappers.py#L2135-L2156) rejects compiled double backward. The [saved padded model](padded-export.md) remains an inference caller.

This demonstrates a gradient-dependent loss and its parameter contribution. It does not establish improved accuracy, smoothness, adversarial robustness, operational security or speed. `torch.func`, vmap, JVP, other devices, new dtypes, broader input layouts and cross-version ABI compatibility remain outside the contract.
