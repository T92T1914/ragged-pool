"""Deterministic native-boundary cases shared by pytest and the sanitizer campaign."""
import random

import torch

from ragged_pool import ragged_pool, ragged_pool_reference


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, label):
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if actual.dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-11}
    try:
        torch.testing.assert_close(actual, expected, **tolerance)
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error


def _tick(counters, name):
    counters[name] = counters.get(name, 0) + 1


def _snapshots(inputs):
    snapshots = []
    for tensor in inputs:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.layout == torch.strided and not tensor.is_nested:
            snapshots.append((tensor, tensor.detach().resolve_neg().clone(), tensor._version, tensor.stride(), tensor.is_neg()))
    return snapshots


def _unchanged(snapshots, counters):
    for tensor, before, version, stride, negative in snapshots:
        torch.testing.assert_close(tensor.detach().resolve_neg(), before, rtol=0, atol=0, equal_nan=True)
        _require(tensor._version == version and tensor.stride() == stride and tensor.is_neg() == negative,
                 "An operation changed an input's storage or view metadata")
    _tick(counters, "input_nonmutation")


def _reject(operation, inputs, label, category, counters):
    snapshots = _snapshots(inputs)
    try:
        operation(*inputs)
    except (RuntimeError, TypeError):
        _tick(counters, category)
    else:
        raise AssertionError(f"{label}: operation accepted an input outside the contract")
    _unchanged(snapshots, counters)


def _evaluate(values, logits, offsets, upstream):
    v = values.detach().clone().requires_grad_()
    l = logits.detach().clone().requires_grad_()
    output = torch.ops.ragged_pool.pool.default(v, l, offsets)
    gradients = torch.autograd.grad(output, (v, l), upstream)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _generate(seed, dtype, rng, generator):
    mode = (seed // 2) % 8
    width = rng.choice((1, 3, 7, 17))
    if mode == 0:
        lengths = []
    elif mode == 1:
        lengths = [0] * rng.randint(1, 8)
    elif mode == 2:
        lengths = [0, 1, 0, 1, 1, 0]
    elif mode == 3:
        lengths = [0, rng.randint(16, 96), 0]
    elif mode == 5:
        # Powers of two keep endpoint arithmetic exact in the float32 reference.
        lengths = [2, 0, 4, 1, 0]
    elif mode == 7:
        boundary = (seed // 16) % 3
        if boundary == 0:
            lengths, width = [0, 4096], 1
        elif boundary == 1:
            lengths, width = [0, 2, 1, 0], 128
        else:
            lengths = [rng.randint(0, 1) for _ in range(128)]
    else:
        lengths = [rng.randint(2, 12), 0, 1] + [rng.randint(0, 12) for _ in range(rng.randint(0, 5))]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    count = boundaries[-1]
    values = torch.randint(-32, 33, (count, width), generator=generator, device="cpu").to(dtype) / 4
    logits = torch.randint(-16, 17, (count,), generator=generator, device="cpu").to(dtype) / 4
    if mode == 5:
        values = torch.randint(0, 2, (count, width), generator=generator, device="cpu").to(dtype) * 2000 - 1000
        logits.fill_(rng.choice((-1000, 0, 1000)))
    elif mode == 6:
        logits = torch.randint(0, 2, (count,), generator=generator, device="cpu").to(dtype) * 2000 - 1000
    if width == 128:
        # All channels receive value gradients, while only four enter the score derivative.
        values[:, 4:] = 0
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cpu")
    return values, logits, offsets, lengths, mode


def _upstream(batches, width, dtype, kind, rng, generator):
    if kind == "expanded":
        return torch.tensor([[rng.choice((-2.0, 0.5, 1.5))]], dtype=dtype, device="cpu").expand(batches, width)
    if kind == "transposed":
        return (torch.randint(-16, 17, (width, batches), generator=generator, device="cpu").to(dtype) / 4).t()
    if kind == "negative":
        return torch._neg_view(torch.randint(-16, 17, (batches, width), generator=generator, device="cpu").to(dtype) / 4)
    row_step, column_step = rng.randint(2, 4), rng.randint(2, 4)
    backing = torch.randint(-16, 17, (batches * row_step + 1, width * column_step + 1),
                            generator=generator, device="cpu").to(dtype) / 4
    return backing[1:1 + batches * row_step:row_step, 1:1 + width * column_step:column_step]


def _differential(values, logits, offsets, upstream, counters):
    snapshots = _snapshots((values, logits, offsets, upstream))
    output, gradients = _evaluate(values, logits, offsets, upstream)
    v = values.detach().clone().requires_grad_()
    l = logits.detach().clone().requires_grad_()
    reference = ragged_pool_reference(v, l, offsets)
    reference_gradients = torch.autograd.grad(reference, (v, l), upstream)
    _close(output, reference.detach(), "reference forward")
    _tick(counters, "differential_forward")
    if values.dtype == torch.float32:
        # Score gradients can nearly cancel. Float32 composition itself lost
        # 3.14e-6 at seed 184, while native differed from double composition
        # by 4.57e-8. Cast the higher precision oracle to the public dtype.
        precise_v = values.double().requires_grad_()
        precise_l = logits.double().requires_grad_()
        precise_output = ragged_pool_reference(precise_v, precise_l, offsets)
        precise_gradients = torch.autograd.grad(
            precise_output, (precise_v, precise_l), upstream.resolve_neg().double()
        )
        reference_gradients = tuple(gradient.to(values.dtype) for gradient in precise_gradients)
        _tick(counters, "high_precision_gradient_oracle")
    for actual, expected in zip(gradients, reference_gradients):
        _close(actual, expected, "reference autograd")
    _tick(counters, "differential_autograd")
    private = torch.ops.ragged_pool._backward.default(upstream, values, logits, offsets)
    for actual, expected in zip(private, reference_gradients):
        _close(actual, expected, "direct private backward")
    _tick(counters, "differential_private_backward")
    public = ragged_pool(values, logits, offsets)
    _close(public, output, "public and direct forward")
    _unchanged(snapshots, counters)
    return output, gradients


def _metamorphic(values, logits, offsets, lengths, upstream, output, gradients, generator, rng, counters):
    if any(length > 1 for length in lengths):
        indices = [torch.randperm(end - start, generator=generator, device="cpu") + start
                   for start, end in zip(offsets.tolist(), offsets.tolist()[1:])]
        permutation = torch.cat(indices)
        permuted, permuted_gradients = _evaluate(values[permutation], logits[permutation], offsets, upstream)
        _close(permuted, output, "within-group permutation output")
        for actual, original in zip(permuted_gradients, gradients):
            _close(actual, original[permutation], "within-group permutation gradient rule")
        _tick(counters, "within_group_permutation")

    shifted = logits.clone()
    shifted_groups = 0
    for start, end in zip(offsets.tolist(), offsets.tolist()[1:]):
        if start == end:
            continue
        low = -1000 - float(logits[start:end].min())
        high = 1000 - float(logits[start:end].max())
        candidates = [amount for amount in (-128, -16, 16, 128) if low <= amount <= high]
        if candidates:
            shifted[start:end] += rng.choice(candidates)
            shifted_groups += 1
    if shifted_groups:
        shifted_output, shifted_gradients = _evaluate(values, shifted, offsets, upstream)
        _close(shifted_output, output, "safe group logit shift output")
        for actual, expected in zip(shifted_gradients, gradients):
            _close(actual, expected, "safe group logit shift gradient")
        _tick(counters, "safe_logit_shift")

    uniform_output, uniform_gradients = _evaluate(values, torch.zeros_like(logits), offsets, upstream)
    expected_output = torch.zeros_like(uniform_output)
    expected_values = torch.zeros_like(values)
    expected_logits = torch.zeros_like(logits)
    # The analytic oracle uses higher precision before casting to the public dtype.
    for batch, (start, end) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        if start == end:
            continue
        group = values[start:end].double()
        mean = group.mean(dim=0)
        grad = upstream[batch].resolve_neg().double()
        expected_output[batch] = mean.to(values.dtype)
        expected_values[start:end] = (grad / (end - start)).to(values.dtype)
        expected_logits[start:end] = (((group - mean) * grad).sum(dim=1) / (end - start)).to(logits.dtype)
    _close(uniform_output, expected_output, "uniform mean")
    _close(uniform_gradients[0], expected_values, "uniform value derivative")
    _close(uniform_gradients[1], expected_logits, "uniform score derivative")
    _tick(counters, "uniform_mean_analytic")


def _negative_nondense(tensor):
    shape = tuple(tensor.shape[:-1]) + (tensor.shape[-1] * 2,)
    backing = torch.zeros(shape, dtype=tensor.dtype, device="cpu")
    view = backing[..., ::2]
    view.copy_(-tensor)
    result = torch._neg_view(view)
    _require(result.is_neg() and not result.is_contiguous(), "Regression input must retain both negative and nondense state")
    return result


def _admission(dtype, rng, counters):
    values = torch.arange(6, dtype=dtype, device="cpu").reshape(3, 2)
    logits = torch.tensor([0.0, 0.5, -0.5], dtype=dtype, device="cpu")
    offsets = torch.tensor([0, 2, 2, 3], dtype=torch.int64, device="cpu")
    upstream = torch.ones(3, 2, dtype=dtype, device="cpu")
    pool = torch.ops.ragged_pool.pool.default
    backward = torch.ops.ragged_pool._backward.default
    valid = (values, logits, offsets)
    minimum, maximum = torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max
    bad_offsets = ([0, minimum, 3], [0, maximum, 3], [minimum, 0, 3], [0, 2, maximum],
                   [0, minimum, maximum, 3], [0, -rng.randint(1, 64), 3], [0, 3 + rng.randint(1, 64), 3])
    for boundaries in bad_offsets:
        invalid = (values, logits, torch.tensor(boundaries, dtype=torch.int64, device="cpu"))
        _reject(pool, invalid, f"direct offsets {boundaries}", "invalid_offsets_direct", counters)
        _reject(backward, (upstream,) + invalid, f"private offsets {boundaries}", "invalid_offsets_private", counters)

    malformed = [
        ("scalar values", (values[0, 0], logits, offsets)),
        ("three-dimensional values", (values.reshape(1, 3, 2), logits, offsets)),
        ("logits rank", (values, logits[:, None], offsets)),
        ("offsets rank", (values, logits, offsets[None])),
        ("complex values", (values.to(torch.complex64), logits.to(torch.complex64), offsets)),
        ("boolean offsets", (values, logits, offsets.bool())),
        ("expanded values", (values[:, :1].expand(3, 2), logits, offsets)),
        ("expanded logits", (values, logits[:1].expand(3), offsets)),
        ("empty offsets", (values, logits, offsets[:0])),
        ("non-Tensor argument", (None, logits, offsets)),
        ("nested values", (torch.nested.nested_tensor([[1.0, 2.0], [3.0]], dtype=dtype, device="cpu"), logits[:2], torch.tensor([0, 2], device="cpu"))),
    ]
    for index in range(3):
        invalid = list(valid)
        invalid[index] = torch.empty_like(invalid[index], device="meta")
        malformed.append((f"meta input {index}", tuple(invalid)))
    for index in (0, 1):
        invalid = [tensor.clone() for tensor in valid]
        invalid[index].view(-1)[rng.randrange(invalid[index].numel())] = rng.choice((float("nan"), float("inf"), 1001.0, -1001.0))
        malformed.append((f"invalid numerical content {index}", tuple(invalid)))
    for label, invalid in malformed:
        _reject(pool, invalid, f"direct {label}", "invalid_metadata_direct", counters)
        _reject(backward, (upstream,) + invalid, f"private {label}", "invalid_metadata_private", counters)

    for index in range(3):
        invalid = list(valid)
        invalid[index] = _negative_nondense(invalid[index])
        invalid = tuple(invalid)
        _reject(ragged_pool, invalid, f"public combined negative/nondense {index}", "negative_nondense_public", counters)
        _reject(pool, invalid, f"direct combined negative/nondense {index}", "negative_nondense_direct", counters)
        _reject(backward, (upstream,) + invalid, f"private combined negative/nondense {index}", "negative_nondense_private", counters)

    bad_gradients = (
        upstream[0, 0], upstream.flatten(), upstream[:, :, None], upstream[:0], upstream[:, :1],
        upstream.to(torch.complex64), upstream.long(), torch.empty_like(upstream, device="meta"),
        upstream.to_sparse(), torch.nested.nested_tensor([[1.0, 2.0], [3.0]], dtype=dtype, device="cpu"),
    )
    for index, grad in enumerate(bad_gradients):
        _reject(backward, (grad,) + valid, f"private malformed upstream {index}", "invalid_upstream_private", counters)


def run_case(seed: int, dtype=None):
    """Run one reproducible case and return category counts without changing thread settings."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    dtype = (torch.float32 if seed % 2 == 0 else torch.float64) if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("generated cases require float32 or float64")
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
    counters = {}
    try:
        values, logits, offsets, lengths, mode = _generate(seed, dtype, rng, generator)
        kind = ("expanded", "transposed", "sliced", "negative")[(seed // 16) % 4]
        upstream = _upstream(len(lengths), values.shape[1], dtype, kind, rng, generator)
        output, gradients = _differential(values, logits, offsets, upstream, counters)
        _tick(counters, f"upstream_{kind}")
        _tick(counters, "empty_batch" if not lengths else "empty_groups" if not values.shape[0] else "nonempty_batch")
        _metamorphic(values, logits, offsets, lengths, upstream, output, gradients, generator, rng, counters)
        _admission(dtype, rng, counters)
    except Exception as error:
        raise AssertionError(f"Generated case seed={seed}, dtype={dtype} failed: {error}") from error
    return {"seed": seed, "dtype": str(dtype), "mode": mode,
            "shape": {"N": values.shape[0], "D": values.shape[1], "B": len(lengths)}, "counters": counters}
