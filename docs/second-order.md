# Packed second-order differentiation

The packed eager CPU API can differentiate through its own first backward. This makes a small gradient-dependent training loss possible without replacing the native layer with the reference. The new C++ operation computes a vector-Jacobian product of the first backward. It does not construct or store a full Hessian matrix.

The original forward contract remains: contiguous CPU float32/float64 values `[N,D]` and logits `[N]`, int64 offsets, N <= 4096, B <= 128, D from 1 through 128 and finite values/logits in `[-1000,1000]`. Empty groups and singleton groups retain their exact structural behavior. Forward and first-backward numerical implementations are unchanged.

## Derivative and ownership

Within one group, let `p_i` be stable softmax weights, `y = sum(p_i*v_i)` and `g` the upstream feature gradient. The first backward returns `dv_i = p_i*g` and `dl_i = p_i*dot(g, v_i-y)`. Suppose another objective supplies adjoints `a_i` for `dv_i` and `b_i` for `dl_i`. Define:

```text
b_bar = sum_i p_i*b_i
H = sum_i p_i*a_i + sum_i p_i*(b_i-b_bar)*(v_i-y)

dS/dg   = H
dS/dv_i = p_i*(b_i-b_bar)*g
dS/dl_i = p_i*dot(g, a_i-H+(b_i-b_bar)*(v_i-y))
```

These are the three outputs of `ragged_pool::_double_backward`. Stable maximum-subtracted weights and the pooled vector are recomputed from saved inputs. Centered factors and float64 accumulation reduce avoidable cancellation before casting results to the input dtype. This is an implementation choice, not an error bound for every upstream magnitude.

The derivative with respect to `g` matters. A prediction head can make `g` depend on its trainable parameters. Returning only value/logit derivatives would lose part of that model's gradient. For a scalar example with weights `[1/4,3/4]` and prediction `w*y`, the penalty `0.5*sum((d prediction/d values)**2)` equals `1.25` at `w=2`, and its derivative with respect to `w` is also `1.25`. A unit test checks both values directly.

An empty group writes zero to its `dS/dg` row and has no item derivatives. A singleton writes `a` to that row and zero value/logit derivatives. Ordinary noncontiguous, sliced, negative-view and expanded zero-stride `a`, `b` and `g` are normalized internally with `resolve_neg().contiguous()`. This does not broaden the original forward-layout contract. Original forward metadata is checked before Negative dispatch materializes tensors. Unbounded or nonfinite upstreams can still overflow or propagate nonfinite results.

The registered first-backward context saves original `g`, values, logits and offsets. They must not change before second backward. PyTorch version checks enforce that rule. Outputs use fresh storage and none of the operations mutate inputs. Offsets never receive a gradient.

## An actual gradient-dependent loss

`examples/train_gradient_penalty.py` uses a seeded packed model with a learned scorer and prediction head. It computes:

```python
predictions, pooled = model(values, offsets, ragged_pool)
input_gradient, = torch.autograd.grad(
    predictions.sum(), (values,), create_graph=True, retain_graph=True,
)
loss = torch.nn.functional.mse_loss(predictions, targets)
loss = loss + 0.5 * input_gradient.square().mean()
loss.backward()
optimizer.step()
```

The derivative of the penalty requires differentiating through the input gradient. From identical initial states, native/reference models compare outputs, that first input gradient, total input and parameter gradients, and two SGD updates. Every trainable parameter must change in each update. The fixed fixture includes an empty group and a singleton. Inputs and offsets must remain unchanged.

The debugging control detaches the penalty scalar. Its numerical value remains the same, but its parameter gradients equal MSE alone. The proper penalty must change the scorer and head gradients, which catches a disconnected regularizer that merely appears in the printed loss. Neither control takes an optimizer step. This demonstrates integration and a derivative contribution. It does not measure improved accuracy, smoothness, adversarial robustness or production usefulness.

Build an ordinary wheel using the [Windows](../README.md#build-and-run) or [Linux archive](linux-cpu.md) route, install it in a fresh external environment, then copy the example outside the checkout:

```powershell
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_gradient_penalty.py' --dtype float32 --output 'C:\work\ragged-consumer\penalty-float32.json'
& 'C:\work\ragged-accept\Scripts\python.exe' -I 'C:\work\ragged-consumer\train_gradient_penalty.py' --dtype float64 --output 'C:\work\ragged-consumer\penalty-float64.json'
```

Use new receipt paths. All three CPU workflows run the installed example in both dtypes. Actual completed jobs and revisions establish execution, rather than the workflow file alone.

## Checks and limits

`tests/test_second_order.py` compares all three second-backward outputs with independently differentiated PyTorch composition, checks a hand-calculated mixed derivative, runs float64 `gradgradcheck`, compares Hessian-vector products and checks symmetry. It covers either unused output branch, empty/singleton groups, numerical and shape endpoints, upstream views, malformed inputs and saved-tensor versions. Operator registration/fake/AOT checks are separate from those mathematical checks. The last operation's AOT check uses detached inputs; a distinct requires-grad registration check inspects its explicit third-order refusal.

The API route uses supported [autograd registration](https://docs.pytorch.org/docs/2.9/library.html#torch.library.register_autograd), with one registered operation calling the next. The [exact 2.9.1 custom-op examples](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/testing/_internal/custom_op_db.py) and [autograd wrapper](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_library/autograd.py) informed that integration. The equations are established softmax calculus implemented for this package, not a copied research kernel or a new algorithm.

The [padded eager adapter](padded-second-order.md) reuses this native second derivative through its own opaque packing/scatter operation and a padding-invariant model caller. The existing `aot_eager` compiled training route remains first-order. Exact PyTorch 2.9.1 [AOT runtime code](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_functorch/_aot_autograd/runtime_wrappers.py) rejects differentiating its compiled backward again. Third order, `torch.func` transforms, vmap, GPU, new performance claims and cross-version ABI compatibility are excluded.
