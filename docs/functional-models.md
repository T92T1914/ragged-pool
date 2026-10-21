# Functional gradients and bounded model maps

The existing `ragged_pool_forward_ad` bridge also supports packed eager `torch.func.grad`, `torch.func.vjp`, a single `torch.vmap` and `vmap(grad)` on the pinned CPU environment. Its name is retained from the first forward-AD addition. The original `ragged_pool` and padded APIs retain their separate contracts.

A gradient of a scalar loss tells an optimizer how each input or parameter affects that loss. A VJP propagates a supplied output cotangent back to the inputs. A map applies the same function to several lanes. Here those lanes are complete packed problems or model replicas, each containing its own variable-length groups. They are not automatically individual bags.

## Native boundaries

The bridge's ordinary backward now calls a private public-API `autograd.Function` around the existing native `_backward`. That Function has separate `forward` and `setup_context`; its ordinary backward reuses native `_double_backward`. This supplies a supported transform boundary around the derivative itself. Merely giving the outer pool Function a separate context was insufficient for functional reverse transforms.

Explicit `Function.vmap` rules select each mapped argument, evaluate the same native-backed Function serially per lane, and stack the results. The implementation does not claim parallel native execution or a batching speed advantage. It adds no C++ kernel, dispatcher operation, fake registration or dependency. Existing operator registration checks remain separate from mathematical gradient checks.

The route follows [PyTorch 2.9's custom Function transform guidance](https://docs.pytorch.org/docs/2.9/notes/extending.func.html) and [manual vmap contract](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.Function.vmap.html). Automatic rule generation is unsuitable for these opaque native implementations.

## Admitted maps

Each manual batching-rule invocation must have 1 through 8 mapped lanes. Every lane has the same logical `[N,D]` values, `[N]` logits and `[B+1]` offset shapes. Values, logits and offsets can each be shared or mapped. Offset contents may differ by lane while total N and B stay equal. Each selected lane must satisfy the original [packed contract](contract.md), including contiguous CPU float32/float64 values and logits, contiguous int64 offsets, finite bounds and valid group boundaries.

Per-invocation aggregate limits are `K*N <= 4096` and `K*B <= 128`. Shared values or offsets still represent K evaluations and are charged K times. Aggregate violations fail before native lane execution. Zero lanes and nested maps are rejected. PyTorch can split a larger outer map using `chunk_size`; each rule invocation then sees only its current chunk. A nine-lane unchunked call fails before native work, while chunk sizes one or four execute nine native lane calls across several bounded invocations. These limits do not bound the enclosing map's total work or process memory. An identity Function guard inside the manual rules preserves strides and detects an outer map before it can bypass these bounds.

Mapping over a nonleading dimension is admitted when the actual selected lanes remain contiguous. A dimension choice that leaves stepped primal lanes is rejected. The rule never calls `contiguous()` on primals to change that admission. A mapped output dimension can rearrange the freshly stacked result according to PyTorch's own `out_dims` behavior. Backward continues to normalize supported strided, expanded and logical negative upstream gradients within native code.

Shared inputs receive the sum of their lane contributions when differentiating a summed mapped objective. A mean objective divides those contributions by the number of lanes. This reduction is part of the caller's objective, not an automatic averaging rule in the pooling layer.

Single-call first-order JVP and ordinary second reverse derivatives remain available through their declared routes. The backward Function also supports one eager [loss HVP](hessian-vector-products.md) through `jvp(grad(scalar_loss))`, with fixed offsets and no map. Its tangent reuses native first and second derivatives and includes the output-cotangent direction. General nested reverse transforms, nested maps, JVP/map compositions, full Jacobian/Hessian helpers, compiled/exported transforms and padded transforms remain outside the declared routes. The single-map guard still rejects forward differentiation. Further HVP differentiation and the earlier excluded tangent/nested-forward compositions are rejected through their own guards.

## Train three small models

`examples/train_functional_models.py` uses three distinct seeded encoder/scorer/pool/head models. It stacks their parameter dictionaries into leaf tensors and evaluates one model function through [functional_call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html). `vmap(grad(single_model_loss))` produces one complete parameter gradient for each model. Two actual SGD updates are compared with serial original-native and independent primitive PyTorch models from matched initial states.

All three models share a packed raw input with empty, singleton and longer groups. This is an ensemble integration example. It does not compute one gradient per unequal-length bag or establish useful ensemble accuracy. After training, parameters stay fixed while summed and mean shared-input gradients are compared with unequal serial native and independent reference contributions. A control requires the sum and mean to differ.

After installing the ordinary wheel with pinned dependencies into a fresh environment outside the checkout, copy the example to a separate consumer directory. Use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\train_functional_models.py" --dtype float32 --output "$consumer\functional-models-float32.json"
& $python -I "$consumer\train_functional_models.py" --dtype float64 --output "$consumer\functional-models-float64.json"
```

Receipts identify the installed package and loaded native library, script bytes, precision, map shapes, actual update counts, comparison errors, input ownership and thread settings. These small synthetic results demonstrate framework integration. They do not establish accuracy, robustness, privacy, reduced memory, faster execution or arbitrary model semantics.

## Debugging the inner derivative boundary

Before this extension, functional `grad` on the forward-AD bridge failed with PyTorch's `setup_context` requirement. The outer Function already had that method. Its backward called a dispatcher operation whose generated autograd Function used a different context shape. Wrapping native backward with another supported Function fixes the actual inner boundary rather than bypassing framework dispatch.

A plain vmap loop can also conceal an admission error if it makes every selected primal contiguous. The identity guard deliberately preserves views, and tests require stepped lanes to remain rejected. Shared-gradient tests use unequal lane contributions so that an accidental mean cannot pass as the required sum.
