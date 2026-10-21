# Train with per-example clipped gradients

`examples/train_clipped.py` uses one bag as one training example. It computes a separate parameter gradient for each bag, limits the length of each complete gradient vector, averages those vectors, and takes two SGD steps. The model keeps native padded pooling between its learned scorer and prediction head. This changes the optimizer's input without adding a package API or another native derivative.

The example uses ordinary padded forward evaluation followed by the experimental [`torch.autograd.grad(..., is_grads_batched=True)` API](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html). This batches backward products from an existing graph. Direct `torch.func.grad`, vmap and JVP through the native layer remain outside the contract. The existing first-order compiled path, eager second derivatives and exported inference callers keep their own limits.

## One gradient for each bag

The encoder, tanh, bias-free scorer and scalar prediction head operate independently on each bag. There is no batch normalization, dropout or cross-example objective. The model produces predictions `[B]` and unreduced squared losses `[B]`. The full-batch oracle used to check chunking selects each loss with identity cotangents:

```python
gradients = torch.autograd.grad(
    losses,
    tuple(model.parameters()),
    grad_outputs=torch.eye(B, dtype=losses.dtype, device="cpu"),
    is_grads_batched=True,
    create_graph=False,
    allow_unused=False,
)
```

For each parameter tensor, the result has shape `[B, *parameter.shape]`. Its row `i` is the gradient of example `i` with respect to that parameter. The example requires all five named parameter blocks to remain connected. It does not implement a general solution for arbitrary models or unused parameters. Unlike `.backward()`, this API returns gradients without accumulating them into parameter `.grad` fields.

The caller requires at least one example. An empty bag within a nonempty batch is valid. Its pooled features and encoder/scorer parameter contributions are zero. A scalar head bias can still receive a loss gradient because an empty bag can still have a target and a prediction. Including that gradient is part of the update.

## Clip the complete parameter vector

Let `G[i,p]` be example `i`'s gradient for parameter block `p`. For a declared positive clipping threshold `C`, the example computes:

```text
norm[i] = sqrt(sum over p and every element of G[i,p] ** 2)
scale[i] = 1                         if norm[i] == 0
           min(1, C / norm[i])       otherwise
mean_gradient[p] = mean over i of scale[i] * G[i,p]
```

The threshold is `1.0`. The norm joins all parameter blocks for an example. Clipping each block separately would define a different update. Zero-norm examples use scale one without division by zero or an added epsilon. Each batch example remains in the averaging denominator, including an empty bag or a zero-gradient example.

Native gradients retain the model's float32 or float64 dtype. The clipping helper converts gradients to float64 for the norm, scaling and averaging, then casts each mean once to the parameter dtype. Float32 rounding therefore happens when the final optimizer gradient is formed. This deliberate local arithmetic conversion does not move model inputs or parameters to another device or change the pooling contract. The means are assigned to `.grad` before SGD takes a step.

Two small vectors show why clipping happens before the mean. With gradients `[2,0]` and `[0,0.5]` and threshold one, the mean of clipped gradients is `[0.5,0.25]`. Their unmodified mean is `[1,0.25]`. Clipping that mean gives approximately `[0.9701,0.2425]`, a different result. A second distinction is the joint norm: blocks containing `1.2` and `1.6` have combined norm two, so joint clipping produces `0.6` and `0.8`. Clipping each scalar block independently would produce `1` and `1`.

## Chunk the backward selectors

The default `--aggregation backward-chunks` uses `chunked_clipped_gradients` with chunk size `K=2`. The helper runs one full forward evaluation, then constructs each selector directly with shape `[r,B]`, where `r <= min(K,B)`. Row `j` selects the global example `start+j`, including the final uneven chunk. It does not construct `eye(B)` and slice it. For five examples and K=2, selector shapes are `[2,5]`, `[2,5]` and `[1,5]`.

Each batched VJP returns `[r,*parameter.shape]` blocks. The helper clips each example's joint vector, adds its float64 clipped sums to the accumulators and releases those gradient blocks before the next VJP call. It retains the forward graph before the last call and frees saved backward state on the last, with `create_graph=False`. Parameters remain fixed across the chunks. The caller assigns the final means and takes one optimizer step only after the whole batch has been processed.

Divide the complete sums by global B once, then cast once to the parameter dtype. Averaging chunk means equally gives the wrong weight to an uneven tail. For already clipped scalar rows `[1,1,1,1,0]` and chunks `[2,2,1]`, the correct mean is `4/5`. The chunk means are `1,1,0`, whose unweighted mean is `2/3`. A focused executed control distinguishes these rules. Empty examples still count in B, including their possible head-bias contribution.

K must be a positive Python integer. Bool, zero, negative and nonintegral values are rejected. K greater than B is normalized to B, so `--chunk-size 5` or `--chunk-size 9` gives one full-size call for this fixture. K=1 exercises one example at a time. Every training step constructs a fresh graph. The helper preserves input bytes/version counters, parameters and existing `.grad` tensors or None.

The named intermediate bounds are `min(K,B)*B` selector entries and at most `min(K,B)*P` returned parameter-gradient entries across all blocks, with P=25 here. The full forward graph, saved activations, input snapshots, float64 temporaries and legacy batching fallback allocations remain. Norms/scales retain B entries. The validation command also runs full-gradient oracles. These dimensions establish a bounded helper interface, not a hard memory cap, measured peak RSS, throughput improvement or input microbatching.

## Run one forward graph per microbatch

Select `--aggregation forward-microbatches` to use `microbatched_clipped_gradients`. It slices raw values, mask and targets before each model call. Each microbatch has `r <= min(K,B)` examples, local losses `[r]`, a local identity selector `[r,r]` and returned gradients `[r,*parameter.shape]`. With B=5,K=2, actual forward rows are `[2,2,1]` and selector shapes are `[2,2],[2,2],[1,1]`.

Each fresh forward is followed by one batched VJP with `retain_graph=False`, `create_graph=False` and `allow_unused=False`. The helper detaches and jointly clips each example's complete parameter gradient, adds float64 sums, then discards its selectors, returned gradient blocks, local outputs and loss before starting the next forward. Parameters stay fixed. After all microbatches, divide by global B once and cast each mean once. One outer optimizer step follows. The final short microbatch, empty bags and zero-norm examples retain the same weight as every other example.

For the current encoder/tanh/scorer/pool/head model, a bag's prediction and pointwise squared loss do not depend on another bag. Splitting the forward therefore preserves the mathematical per-example derivatives. Batch-shape changes can still introduce floating point rounding differences, which are checked with the existing dtype tolerances. Both actual optimizer updates compare the new route with the unchanged retained-graph helper, full batched gradients, serial native gradients and an independent per-row PyTorch reference.

Sample independence is a required property of this fixed caller, not something the helper can prove for an arbitrary module. Batch normalization, dropout or another stochastic forward, mutable buffers, cross-example attention/objectives and parameter updates between microbatches can change the function. A small counterexample uses prediction `weight * mean(x)` for every example, weight one, x=[1,3], zero targets and C=100. A full forward predicts two for both examples, so each loss derivative is `2*2*2=8`. Singleton forwards give derivatives two and eighteen, whose mean is ten. The test requires this disagreement. It does not force a batch-coupled model to satisfy an invalid equivalence.

The distinction is also present in the original [microbatching discussion by Sohoni et al.](https://arxiv.org/pdf/1904.10631), introduction, and the [Opacus explanation of batch normalization](https://opacus.ai/tutorials/guide_to_module_validator). These sources inform the boundary. Their performance, memory and privacy results are not measurements or guarantees for this example. This route does not use activation checkpointing or recompute previously processed examples.

Tests observe actual forward/backward interleaving, local row contents, selectors, returned-gradient shapes and graph-retention options. Weak references check that previous Python outputs, losses, gradients and seeds are released before the next forward. A saved-tensor hook test packs detached tensors into weakly tracked holders and checks that those saved backward-state holders are released too. This is scoped lifetime evidence under the tested hook, separate from allocator or process-memory measurement.

Within this helper, a local forward graph contains at most min(K,B) examples, the selector has at most min(K,B) squared entries and returned parameter-gradient blocks at most min(K,B)*P entries across all blocks. P=25 here. Full input storage and byte snapshots, B-entry norm/scale diagnostics, float64 clipping temporaries and framework/native fallback allocations remain. The complete validation command still executes full-gradient oracles. No input streaming, hard memory cap, lower peak RSS, faster training or arbitrary-model support is claimed.

## Checks around the update

At each step the consumer compares every native batched per-example gradient with ordinary serial native gradients and an independent per-row PyTorch reference. The mean of unclipped rows must also match the gradient of mean loss. Native and reference models compare clipped means, updated parameters and predictions through two actual SGD steps in each supported dtype.

The fixed fixture must activate clipping. Controls require meaningful differences from both the unclipped update and clipping after the mean. Independent split-parameter vectors check the joint norm and zero-norm behavior. Chunk partitions include K=1, uneven tails, K=B and K>B, with real batched-VJP selector/gradient-shape and graph-retention checks. Two-step comparisons verify the optimizer result rather than only a final scalar loss. These checks establish the chosen update's semantics for the fixture, not that clipping improves a model.

Ignored raw padding is cleared before encoder arithmetic. Masking only the pool's output cannot undo a NaN that has already entered an upstream parameter-gradient calculation. Equivalent selected items are also represented with moved holes, added padded width and changed ignored nonfinite values. Those layouts must preserve the per-example gradients and two-step trajectory within dtype tolerances. Input bytes and version counters remain unchanged. The original [padded input contract](padded-batches.md) still applies at the pooling boundary.

## Run the installed consumer

Build and install a wheel using the [Windows route](../README.md#build-and-run) or [Linux source-distribution route](linux-cpu.md). Use a fresh environment outside the checkout, copy the standalone example outside the checkout, and choose new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$example = 'C:\work\ragged-consumer\train_clipped.py'
& $python -I $example --dtype float32 --aggregation forward-microbatches --chunk-size 2 --output 'C:\work\ragged-consumer\clipped-float32.json'
if ($LASTEXITCODE -ne 0) { throw 'Installed float32 clipped-gradient consumer failed' }
& $python -I $example --dtype float64 --aggregation forward-microbatches --chunk-size 2 --output 'C:\work\ragged-consumer\clipped-float64.json'
if ($LASTEXITCODE -ne 0) { throw 'Installed float64 clipped-gradient consumer failed' }
```

The selected JSON receipt declares method `autograd.grad.is_grads_batched`, aggregation method `microbatched_per_example_joint_clip`, requested/effective sizes, actual forward rows, local selector and gradient shapes, all-false graph-retention flags, global denominator, native and reference optimizer-step counts, retained-graph comparison errors, and control, invariance and empty-bag results. All three existing CPU workflows explicitly select this route in both dtypes and check the `[2,2,1]` forward partition before retaining receipts. A completed run at an identified revision establishes execution. Workflow source alone does not establish a pass.

Omit `--aggregation` or select `backward-chunks` to run the accepted retained-full-graph route. Its receipt keeps method `chunked_per_example_joint_clip` and selectors `[2,5],[2,5],[1,5]`. Both routes share the positive `--chunk-size` contract and global clipping rule. They differ in where forward evaluation occurs and how long saved graph state is retained.

## Batching limits and interpretation

PyTorch 2.9.1 implements this experimental API through a [legacy batching wrapper](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/autograd/__init__.py#L455-L468). Its [fallback](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/LegacyBatchedFallback.cpp#L235-L330) can execute slices separately and stack the results. This is not a dedicated batched native kernel. The full oracle still uses a `B * B` identity, while the chunk helper allocates each smaller selector directly. Keep this caller small. No throughput or process-memory advantage is claimed.

The preliminary pinned Windows CPU probe matched serial native gradients in float32 and float64. Enabling PyTorch's documented fallback diagnostics identified missing legacy batching rules for the padded backward operation and tanh backward. A separate direct `torch.func.grad` attempt failed because the generated library autograd Function does not provide the separate `setup_context` required by transforms. Those observations qualify the selected route. The diagnostic warnings are not a performance measurement, and switching to a different transform is not a demonstrated repair.

This example demonstrates per-example gradient handling and an actual clipped optimizer update. It adds no noise mechanism or privacy accountant and provides no differential privacy guarantee. It is not DP-SGD, an adversarial defense, a security control or evidence of improved prediction quality. Higher-order differentiation of this caller, arbitrary model batching, other devices and cross-version behavior are not promised.
