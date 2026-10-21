# Contract and implementation choices

## Admitted inputs

All forward inputs are contiguous strided CPU tensors. `values` has shape `[N,D]`, `logits` has shape `[N]` and `offsets` has shape `[B+1]`. N may be zero and is at most 4096. D is from 1 through 128. B may be zero and is at most 128. When B is zero, N must also be zero because offsets are `[0]`.

Values and logits share float32 or float64 dtype. Offsets are int64, start at zero, end at N and never decrease. Duplicate offsets produce empty groups. These conditions also force every boundary to lie in `[0,N]` before native code uses it for indexing.

Values and logits are finite and in the inclusive range `[-1000,1000]`. NaN, either infinity and values outside that interval are rejected, even when they would belong to a singleton group. This is an intentionally bounded numerical contract, not a claim about every representable magnitude. Forward rejects unsupported inputs instead of copying them into an admitted device, dtype or layout.

The native kernels are serial and do not start worker threads. Each call allocates output and bounded temporary weight storage. Backward normalizes a strided CPU upstream gradient with `resolve_neg().contiguous()`, accepting transposes, slices and expanded zero strides. This internal backward copy does not broaden forward input support. No upstream magnitude guarantee is made for gradients that overflow the output dtype.

## Numerical rationale review

A publication review on October 20, 2026 used Consensus to locate Blanchard, Higham and Higham's [softmax error analysis](https://arxiv.org/html/1909.03469v1). The original introduction and shifting analysis support retaining maximum subtraction with explicit normalization. This later literature review did not design or measure the earlier implementation. It supplies context, not an error bound for this pooling layer. Tests and the finite numerical contract remain the implementation evidence.

## Mathematics

For a nonempty group, let `m = max(logits)`, `a_i = exp(logit_i - m)`, `w_i = a_i / sum(a)` and `y = sum_i w_i value_i`. Maximum subtraction prevents positive exponent overflow. With finite logits, at least one exponent is one, so the denominator is positive. Very small weights can underflow to zero at large logit differences.

For an upstream feature vector `g`, first order derivatives are:

```text
dValues_i = w_i * g
dLogits_i = w_i * dot(g, value_i - y)
```

Native backward recomputes weights and the pooled vector in float64 from saved inputs. It avoids differentiating a rounded float32 output as if that rounded value were an exact mathematical intermediate. The Python reference instead lets autograd differentiate built in softmax, multiplication and reduction. Tests compare these separately rounded paths with declared tolerances and include float64 finite differences.

An empty group writes zero and evaluates no softmax. It contributes no input gradient. The reference constructs a zero-valued dependency on the inputs even when every group is empty, allowing the same zero gradients through ordinary autograd. Offsets have no derivative.

## Framework integration

The native library defines `ragged_pool::pool`, `ragged_pool::_backward` and `ragged_pool::_double_backward` with `TORCH_LIBRARY` and CPU implementations. `torch.ops.load_library` loads their definitions before Python registers fake metadata and autograd. A small CPython entrypoint satisfies Windows extension linking but does not expose a second numerical API.

Forward saves values, logits and offsets through `ctx.save_for_backward`. PyTorch checks their versions before backward reads them. Mutating a saved tensor between these calls therefore raises the framework's version error. The operator returns fresh storage and never changes an input.

Python's backward callback calls the private registered backward operator. Differentiating that operation calls the native second backward, returning derivatives with respect to its upstream gradient, values and logits. All three operations have fake implementations, which only inspect metadata. Data-dependent offset and range validation stays in native execution. Shape/dtype admission in fake mode is not validation of imaginary tensor contents.

The native Negative-key handlers check original forward metadata before resolving logical negative views and entering CPU numerical code. This matters when a stepped, noncontiguous tensor also carries a negative bit. PyTorch's generic fallback can clone it into contiguous storage before a CPU admission check sees it. The original layout must be rejected rather than admitted through that materialization. Contiguous logical negative views retain their mathematical values without a device transfer or input mutation. A strided upstream gradient may still be normalized internally, including when it carries a negative bit. The compiled acceptance path uses ordinary contiguous tensors. Private `torch.ops` entrypoints remain implementation details, but their admission paths are tested directly.

Packed eager autograd supports second derivatives through the [declared native second-backward route](second-order.md). Its context saves the original upstream gradient, values, logits and offsets, with the same version-check requirement before the next derivative. The [padded eager adapter](padded-second-order.md) reuses that native derivative, saving its original upstream gradient, padded values, logits and mask. Both final operations explicitly reject third-order differentiation. Direct `vmap` on these original APIs, mixed precision, tensor subclasses and other device backends are outside the contract. The separate packed bridge supplies its declared [first-order JVP](forward-ad.md), [functional reverse/map routes](functional-models.md) and [forward-over-reverse loss HVP](hessian-vector-products.md). These do not broaden transforms on the original APIs.

The declared compile path is first-order `aot_eager` with `fullgraph=True`, exercised with concrete admitted shapes. Compiled double backward is excluded on the pinned framework. This is graph capture plus AOTAutograd, not an Inductor performance result. Invalid runtime offset contents must still fail inside the captured native call. Dynamic-shape generality beyond the executed registration checks is not promised.

## Packed forward AD

The separate `ragged_pool_forward_ad` bridge uses the public `torch.autograd.Function` interface with separate `forward` and `setup_context`, `save_for_forward`, `save_for_backward`, `jvp` and ordinary `backward`. Native forward validates the original primals. Its JVP validates floating tangents with matching shape, dtype and CPU strided layout and finite entries in `[-1000,1000]`. Noncontiguous, zero-stride and logical negative tangents can be normalized by the existing native derivative routine. Missing floating tangents become zero tensors. Offsets have no tangent.

For directions `a_i` in values and `b_i` in logits, let `b_bar = sum_i w_i b_i`. The derivative is `sum_i w_i a_i + sum_i w_i (b_i-b_bar)(value_i-y)`. Empty groups return zero tangents. A singleton returns its value direction and has no logit contribution. The first result of `_double_backward(a,b,zeros,values,logits,offsets)` already evaluates this expression. Its other two results are unused allocations, so this reuse is not an efficiency claim. No new custom operator or fake registration is introduced.

The bridge saves original primal tensors for version checks and calls native `_backward` for ordinary reverse differentiation. Public Function guards clone the four floating JVP inputs and reject nested forward differentiation before the opaque native call can discard an outer tangent. A guard on the result explicitly rejects reverse differentiation of that first-order tangent. These bounded copies add overhead and can normalize supplied views. The separately declared loss HVP below differentiates the backward boundary, rather than reversing the pool's first-order tangent. Other transform compositions are outside the declared routes. Compiled/exported forward AD, padded JVP and mapped JVP remain excluded.

## Packed functional reverse transforms and maps

The existing packed bridge wraps native `_backward` in a separate public Function with its own context and native second-backward rule. Its declared eager functional reverse routes are `torch.func.grad` and `vjp`. Manual `Function.vmap` rules on pool and backward select lanes and stack serial native results. They add no custom operation or fake registration.

Each batching-rule invocation admits 1 through 8 lanes with one common N/D/B shape envelope, shared or mapped arguments, `K*N <= 4096` and `K*B <= 128`. Shared inputs are charged K times per invocation. Caller chunking can produce a larger outer map through several invocations; no total enclosing-work or process-memory bound is promised. Every selected primal lane must satisfy original admission without normalization into compliance. Native backward still admits its supported upstream views. An identity Function guard preserves strides and rejects nested maps before they can bypass aggregate bounds. The [functional model guide](functional-models.md) defines supported dimensions, reductions, ownership and exclusions.

## Packed scalar-loss Hessian-vector products

The packed bridge admits one eager `torch.func.jvp(torch.func.grad(scalar_loss))` composition with fixed offsets. The backward Function saves its original output cotangent, values, logits and offsets for both backward and forward AD. For `B(g,x)=J(x)^T*g`, its tangent is `B(dg,x) + D_x B(g,x)[dx]`. Native `_backward` supplies the first term and the last two outputs of native `_double_backward` supply the second. A nonlinear surrounding loss generally needs both terms. This adds no native kernel, dispatcher operation or fake registration.

Values/logit directions retain the first-order JVP shape, dtype, CPU strided layout and finite `[-1000,1000]` policy, including directions induced by a model's parameters. Internal output cotangents and their directions retain native upstream metadata/view rules without a new magnitude bound. Extreme or nonfinite upstreams can overflow or propagate nonfinite results. Missing floating directions become zeros. Offsets never receive tangents or gradients. Inputs must remain unchanged while saved derivatives use them, and returned derivatives use fresh storage.

Public Function guards cover backward-JVP inputs and results before opaque calls can lose excluded higher tangents. Further differentiation of an HVP, including with respect to its supplied direction, is rejected. Third derivatives, nested forward derivatives, mapped HVPs, full Jacobian/Hessian helpers, padded transforms and compiled/exported transforms remain excluded. The [HVP guide](hessian-vector-products.md) gives the analytic case, consumer and independent mathematical checks. No full Hessian is materialized and no process-memory or speed improvement is claimed.

## Toolchain and source decisions

The selected build uses Python 3.11.8, PyTorch 2.9.1+cpu, MSVC 19.44 and `/std:c++17`, `/O2`, `/fp:precise`. `CppExtension` and `BuildExtension` add their standard PyTorch/Windows compiler and linker flags. The actual invocation belongs in the build log. There is no fast-math flag or GPU kernel.

This uses the ordinary ATen/LibTorch API, not the newer LibTorch Stable ABI API. Rebuild and requalify for a different PyTorch, compiler or Python version. The wheel tag does not expand the executed platform evidence.

The original sources informing these choices are [PyTorch's released CPU installation instructions](https://pytorch.org/get-started/previous-versions/), [the custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [2.9 extension documentation](https://docs.pytorch.org/docs/2.9/cpp_extension.html), [exact v2.9.1 C++17 build source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/cpp_extension.py), [2.9 library registration documentation](https://docs.pytorch.org/docs/2.9/library.html) and [2.9 compiler FAQ](https://docs.pytorch.org/docs/2.9/torch.compiler_faq.html). Context7 helped locate versioned APIs, with decisions checked against original documentation and exact release source. The private planning research motivated the project but supplies no measured result for this implementation.
