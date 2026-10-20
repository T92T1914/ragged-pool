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

The native library defines `ragged_pool::pool` and `ragged_pool::_backward` with `TORCH_LIBRARY` and CPU implementations. `torch.ops.load_library` loads their definitions before Python registers fake metadata and autograd. A small CPython entrypoint satisfies Windows extension linking but does not expose a second numerical API.

Forward saves values, logits and offsets through `ctx.save_for_backward`. PyTorch checks their versions before backward reads them. Mutating a saved tensor between these calls therefore raises the framework's version error. The operator returns fresh storage and never changes an input.

Python's backward callback calls the private registered backward operator. Both operations have fake implementations, which only inspect metadata. Data-dependent offset and range validation stays in native execution. Shape/dtype admission in fake mode is not validation of imaginary tensor contents.

PyTorch's dispatcher resolves logical negative views on CPU before raw-pointer access. Contiguous logical views therefore retain their mathematical values without a device transfer or input mutation. Tests compare these views with resolved tensors, including first order gradients. The compiled acceptance path uses ordinary contiguous tensors. Private `torch.ops` entrypoints are implementation details, not an additional supported user API.

Only first order autograd is supported. The backward operator has an autograd registration that explicitly rejects differentiation through its outputs. This avoids silently presenting an incomplete second derivative. `vmap`, mixed precision, tensor subclasses and other device backends are outside the contract.

The declared compile path is `aot_eager` with `fullgraph=True`, exercised with concrete admitted shapes. This is graph capture plus AOTAutograd, not an Inductor performance result. Invalid runtime offset contents must still fail inside the captured native call. Dynamic-shape generality beyond the executed registration checks is not promised.

## Toolchain and source decisions

The selected build uses Python 3.11.8, PyTorch 2.9.1+cpu, MSVC 19.44 and `/std:c++17`, `/O2`, `/fp:precise`. `CppExtension` and `BuildExtension` add their standard PyTorch/Windows compiler and linker flags. The actual invocation belongs in the build log. There is no fast-math flag or GPU kernel.

This uses the ordinary ATen/LibTorch API, not the newer LibTorch Stable ABI API. Rebuild and requalify for a different PyTorch, compiler or Python version. The wheel tag does not expand the executed platform evidence.

The original sources informing these choices are [PyTorch's released CPU installation instructions](https://pytorch.org/get-started/previous-versions/), [the custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [2.9 extension documentation](https://docs.pytorch.org/docs/2.9/cpp_extension.html), [exact v2.9.1 C++17 build source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/cpp_extension.py), [2.9 library registration documentation](https://docs.pytorch.org/docs/2.9/library.html) and [2.9 compiler FAQ](https://docs.pytorch.org/docs/2.9/torch.compiler_faq.html). Context7 helped locate versioned APIs, with decisions checked against original documentation and exact release source. The private planning research motivated the project but supplies no measured result for this implementation.
