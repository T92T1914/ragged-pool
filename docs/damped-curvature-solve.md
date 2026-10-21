# A bounded damped curvature solve

`examples/solve_curvature.py` consumes the packed eager HVP through one small conjugate-gradient (CG) solve. It first reuses the two ordinary SGD updates from `train_curvature.py`. It then fixes the trained parameters and solves a local linear system. The solution is a diagnostic vector. It is never applied to the model.

For scalar loss L at fixed parameters theta, let g be its gradient and H its Hessian. The example solves:

```text
A(p) = H*p + lambda*p
A(s) = b, where b = -g
```

Each H*p comes from the existing `jvp(grad(loss))` route through native pooling derivatives. Neither the solver nor the installed consumer constructs a dense Hessian. The example adds no pooling API, kernel, operator, derivative composition or optimizer.

## One fixed operator

The loss, parameters, template, raw inputs, targets and offsets stay fixed throughout each solve. Changing them would change A while CG is using its earlier products. Stochastic forwards, mutable buffers, batch-dependent objectives and solving while parameters update are outside this example.

The solver is detached numerical work. It owns its right-hand side, recurrence vectors, callback directions and returned products. Detaching breaks both reverse and forward AD tracking, while cloning supplies independent storage. The two operations have distinct purposes because `detach()` alone shares storage. See the versioned [detach](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html) and [clone](https://docs.pytorch.org/docs/2.9/generated/torch.clone.html) documentation. This consumes repeated first HVPs without differentiating through CG.

Parameter dictionaries pass through [functional_call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html), `grad` and [jvp](https://docs.pytorch.org/docs/2.9/generated/torch.func.jvp.html). The [HVP guide](hessian-vector-products.md) retains its separate derivative limits. In particular, actual values/logit directions reaching the pool must satisfy its finite `[-1000,1000]` admission. The solver never clamps or silently normalizes a direction to bypass that rule. An HVP admission error is an explicit unsuccessful solve.

## A hand-checkable system

Take H and lambda so that A is:

```text
H = [[2,1], [1,1]], lambda = 2
A = H + 2I = [[4,1], [1,3]]
b = [1,2]
s = [1/11, 7/11]
```

Multiplying A by s gives `[1,2]`. Starting from zero, exact arithmetic CG reaches this solution in two iterations. The implementation still has to pass its recomputed residual gate in the executed dtype. A fixed iteration count from exact arithmetic is not promised for a rounded general system.

The unpreconditioned recurrence starts with `s=0`, `r=b` and `p=r`. Each iteration computes `q=A(p)`, then `alpha=(r dot r)/(p dot q)`, updates s and r, and uses the ratio of successive squared residual norms to form the next direction. A zero right-hand side returns zero without a division, after one fresh callback at zero verifies its true residual.

## Admission, stopping and refusal

The example-local helper admits CPU strided float32/float64 flat vectors with 1 through 32 entries and 1 through 32 iterations. Damping is finite and nonnegative. Absolute and relative tolerances are finite and positive, with relative tolerance below one. Original right-hand-side contents and versions remain unchanged. The callback receives an owned direction, whose mutation is detected. Its result must match shape, dtype, CPU strided layout and finite-value requirements before the solver owns it.

Before dividing, the helper checks that `p dot A(p)` is finite, positive and sufficiently separated from numerical zero relative to the declared scale. A true Hessian can be indefinite. Adding a positive lambda does not automatically make it positive definite. Tests include a negative eigendirection for which positive damping remains insufficient. Positive encountered products do not certify the complete operator's spectrum.

The stopping threshold is `atol + rtol*norm(b)`. A small recursively updated residual is only a convergence candidate. Acceptance requires a fresh product at the returned solution and the same threshold:

```text
true_residual = b - A(s)
norm(true_residual) <= atol + rtol*norm(b)
```

Stopping norms use scaled `math.hypot` over the bounded CPU vector. Squaring first can erase a tiny nonzero residual or overflow before a square root. The scalar dot products accumulate in float64 and can still underflow or overflow, producing an unsuccessful numerical outcome rather than a full-range convergence guarantee. A regression retains a tiny identity-system case that the initial squared-norm implementation falsely accepted.

The final product counts toward a hard budget of `max_iterations + 1`, consisting of at most one product per iteration and one final check. A disagreement between recursive and true residuals refuses convergence. The solver does not retune damping, restart, expand the budget or quietly relax tolerances.

Nonpositive or unresolved curvature, nonfinite output/arithmetic, malformed callback output, callback `RuntimeError` including HVP admission errors, iteration exhaustion and true-residual disagreement produce structured unsuccessful outcomes. Other programming exceptions propagate to the caller. The returned detached iterate is diagnostic and does not make a refused solve successful. No solve outcome changes model parameters. The record distinguishes completed iterations, callback attempts, threshold and recursive/true residual norms. An unexecuted true residual remains null. These bounds do not limit arbitrary callback allocations or total process memory.

The fixed consumer uses these policies:

| Setting | float32 | float64 |
| --- | --- | --- |
| Damping lambda | 2 | 2 |
| Maximum iterations | 32 | 32 |
| Relative tolerance | 2e-5 | 1e-9 |
| Absolute tolerance | 1e-7 | 1e-12 |

These are declared settings for one tiny fixture, not an adaptive optimization policy or a generally useful damping recommendation.

## Installed consumer and checks

The consumer uses the existing 25-parameter encoder/scorer/pool/head and unchanged training helper. Two matched actual SGD steps compare bridge, original-native and independent primitive PyTorch models. Those are the only parameter updates. At the fixed post-training point, three matrix-free solves use bridge HVPs, original-native reverse-over-reverse HVPs and independent primitive HVPs. Complete solution vectors and parameter blocks are compared.

Independent operators then recompute residuals of the returned bridge solution. An omitted-H control uses only the damping term and must fail the same intended-system residual threshold. The example records fixed parameters, unchanged inputs, callback counts, ownership/version checks and unset parameter gradient fields. It makes no optimizer-improvement claim.

Unit checks retain analytic SPD, zero-right-hand-side, indefinite, nearzero-curvature, malformed/nonfinite callback, ownership, budget and misleading-recursive-residual cases. A tiny independently differentiated primitive dense Hessian and direct solve serve only as a test oracle for the 25-parameter model. This does not add a full-Hessian route to the custom bridge or construct a dense Hessian in the installed consumer. Mathematical checks remain separate from operator registration checks.

After installing the ordinary wheel with pinned dependencies into a fresh environment outside the checkout, copy `solve_curvature.py`, `damped_cg.py` and `train_curvature.py` together into a separate consumer directory. Explicit neighbor-file loading keeps package imports isolated under `-I`. Use new receipt paths:

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
& $python -I "$consumer\solve_curvature.py" --dtype float32 --output "$consumer\solve-float32.json"
& $python -I "$consumer\solve_curvature.py" --dtype float64 --output "$consumer\solve-float64.json"
```

Receipts identify the installed package and native library, own script and neighbor-helper digests, dtype, threads, two ordinary training updates and zero solve-time updates. Completed revision-linked jobs establish execution. A workflow declaration alone does not.

## Rationale and limits

[Shewchuk's 1994 guide](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf), sections 8-9 and 11.2, explains linear CG and the danger of a false zero in a recursively updated floating-point residual. The fresh residual gate addresses that specific acceptance concern. It supplies no speed or memory result for this package.

[Martens's 2010 paper](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf), sections 2 and 4, motivates curvature products and distinguishes fixed quadratic curvature from a changing nonlinear problem. Its positive-semidefinite Gauss-Newton setting and adaptive damping are different from this tiny fixed true-Hessian solve. Its training outcomes do not transfer here. The recurrence is implemented from established equations without copying research code.

This is a bounded numerical consumer, with no line search, adaptive damping, solve-direction update, differentiable solver, preconditioner or general optimizer. Successful residuals and refusal controls do not establish a positive-definiteness certificate, model quality, adversarial robustness, privacy, security, faster training, process-memory savings, GPU behavior or compatibility outside the executed pinned CPU routes.

