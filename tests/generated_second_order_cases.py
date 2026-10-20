"""Deterministic second-VJP cases for pytest and the installed ASan campaign."""
import random

import torch

from ragged_pool import ragged_pool, ragged_pool_padded


FROZEN_SEEDS = (0, 2, 4, 6, 8, 10, 12, 14, 24, 30, 40, 46, 56, 72, 88, 168)
_LAYOUTS = ("contiguous", "expanded", "sliced", "transposed", "negative", "negative_sliced")
_MODES = ("zero_batch", "empty_rows", "singletons", "mixed", "mixed_holes",
          "endpoints", "wide", "zero_steps")
REQUIRED_CATEGORIES = (
    "second_public_packed", "second_private_packed", "second_public_padded", "second_private_padded",
    "packed_second_vjp", "padded_second_vjp", "second_order_oracle", "input_nonmutation",
    "unused_values_branch", "unused_logits_branch", "ignored_nonfinite", "zero_ignored_derivatives",
    "within_group_permutation", "safe_logit_shift", "equivalent_padding", "packed_padded_equivalence",
    *("mode_" + name for name in _MODES), *("upstream_" + name for name in _LAYOUTS),
    *("nonzero_view_" + name for name in _LAYOUTS),
    "nonzero_branch_both", "nonzero_branch_values", "nonzero_branch_logits",
)


def _tick(counters, name):
    counters[name] = counters.get(name, 0) + 1


def assert_second_close(actual, expected, dtype, context="second VJP"):
    """Check every second output, labeling dg independently for fault controls."""
    if len(actual) != 3 or len(expected) != 3:
        raise AssertionError(f"{context}: expected dg, dvalues and dlogits")
    tolerance = {"rtol": 4e-5, "atol": 4e-6} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-11}
    for name, left, right in zip(("dg", "dvalues", "dlogits"), actual, expected):
        try:
            if not bool(torch.isfinite(left).all()):
                raise AssertionError("nonfinite result")
            torch.testing.assert_close(left, right, **tolerance)
        except AssertionError as error:
            raise AssertionError(f"{context} {name}: {error}") from error


def row_reference(values, logits, boundaries):
    """Differentiate each row directly, without the package's packing adapter."""
    padded = boundaries.dtype == torch.bool
    selected_values = values[boundaries] if padded else values
    selected_logits = logits[boundaries] if padded else logits
    # Read selected entries before arithmetic. Ignored NaN/Inf never enter zero.
    zero = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
    groups = []
    rows = range(boundaries.shape[0]) if padded else range(boundaries.numel() - 1)
    for row in rows:
        if padded:
            v, l = values[row][boundaries[row]], logits[row][boundaries[row]]
        else:
            start, end = int(boundaries[row]), int(boundaries[row + 1])
            v, l = values[start:end], logits[start:end]
        groups.append((torch.softmax(l, dim=0)[:, None] * v).sum(dim=0) if l.numel() else zero)
    return torch.stack(groups) if groups else zero.unsqueeze(0)[:0].clone()


def second_vjp(pool, a, b, g, values, logits, boundaries, branch="both", *, oracle=False):
    """Return (dg,dvalues,dlogits), preserving the native upstream views."""
    if branch not in ("both", "values", "logits"):
        raise ValueError("branch must be both, values or logits")
    upstream, v, l = variables = tuple(t.detach().requires_grad_() for t in (g, values, logits))
    if upstream.stride() != g.stride() or upstream.is_neg() != g.is_neg():
        raise AssertionError("The public chain lost the original g view")
    first = torch.autograd.grad(pool(v, l, boundaries), (v, l), upstream, create_graph=True)
    indices = (0, 1) if branch == "both" else (0,) if branch == "values" else (1,)
    outputs, cotangents = [], []
    if oracle:
        selected_v = v[boundaries] if boundaries.dtype == torch.bool else v
        selected_l = l[boundaries] if boundaries.dtype == torch.bool else l
        zero = upstream.sum() * 0 + selected_v.sum() * 0 + selected_l.sum() * 0
    for index in indices:
        output, cotangent = first[index], (a, b)[index]
        if oracle:
            if boundaries.dtype == torch.bool:
                output, cotangent = output[boundaries], cotangent[boundaries]
            # Empty reference derivatives can otherwise lose their connection.
            output = output + zero
        outputs.append(output)
        cotangents.append(cotangent)
    result = torch.autograd.grad(tuple(outputs), variables, tuple(cotangents), allow_unused=True)
    return tuple(torch.zeros_like(t) if gradient is None else gradient.detach()
                 for t, gradient in zip(variables, result))


def _view(tensor, kind):
    if kind == "expanded":
        scalar = tensor.reshape(-1)[0] if tensor.numel() else tensor.new_tensor(0.5)
        return scalar.reshape((1,) * tensor.dim()).expand(tensor.shape)
    if kind in ("sliced", "negative_sliced"):
        storage = tensor.new_empty((*tensor.shape, 2))
        storage[..., 0] = -tensor if kind == "negative_sliced" else tensor
        result = storage[..., 0]
        return torch._neg_view(result) if kind == "negative_sliced" else result
    if kind == "negative":
        return torch._neg_view(-tensor)
    if kind == "transposed" and tensor.dim() > 1:
        return tensor.transpose(0, -1).contiguous().transpose(0, -1)
    return tensor


def _padded(arguments, kind, *, extra=0, shift=1, zero_steps=False):
    a, b, g, values, logits, offsets = arguments
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    steps = 0 if zero_steps else 2 * max(lengths, default=1) + 2 + extra
    mask = torch.zeros((len(lengths), steps), dtype=torch.bool, device="cpu")
    for row, length in enumerate(lengths):
        mask[row, shift:shift + 2 * length:2] = True
    padded = []
    for name, tensor in zip(("a", "b", "values", "logits"), (a, b, values, logits)):
        # An expanded cotangent is constant. Other layouts also test ignored
        # nonfinite second upstreams, independently of ignored forward data.
        fill = (0.5 if name == "a" else -0.75) if kind == "expanded" and name in ("a", "b") else float("nan")
        target = tensor.new_full((*mask.shape, *tensor.shape[1:]), fill)
        target[mask] = tensor
        padded.append(_view(target, kind) if name in ("a", "b") else target)
    pa, pb, pv, pl = padded
    return pa, pb, g, pv, pl, mask


def make_case(seed, dtype=None):
    """Use local RNGs only; return packed and equivalent padded arguments."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    dtype = (torch.float32 if seed % 2 == 0 else torch.float64) if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("second-order cases require float32 or float64")
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
    mode = (seed // 2) % len(_MODES)
    lengths = ([], [0, 0, 0], [1, 0, 1, 1], [2, 0, 1, 3], [3, 1, 0, 2],
               [2, 0, 4], [2, 0, 3], [0, 0, 0])[mode]
    width = 1 if mode == 5 else 7 if mode == 6 else 3
    offsets = torch.tensor([0] + lengths, dtype=torch.int64, device="cpu").cumsum(0)
    count = sum(lengths)

    def numbers(shape):
        return torch.randint(-8, 9, shape, generator=generator, device="cpu").to(dtype) / 4

    values, logits = numbers((count, width)), numbers((count,))
    if mode == 5:
        values = torch.randint(0, 2, values.shape, generator=generator, device="cpu").to(dtype) * 2000 - 1000
        logits.fill_(rng.choice((-1000, 1000)))
    kind = _LAYOUTS[(seed // 16) % len(_LAYOUTS)]
    a, b, g = numbers(values.shape), numbers(logits.shape), numbers((len(lengths), width))
    if kind == "expanded":
        a.fill_(0.5)
        b.fill_(-0.75)
        g.fill_(0.25)
    a, b, g = (_view(t, kind) for t in (a, b, g))
    arguments = a, b, g, values, logits, offsets
    return {"arguments": arguments, "padded_arguments": _padded(arguments, kind, zero_steps=mode == 7),
            "mode": _MODES[mode], "layout": kind, "branch": ("both", "values", "logits")[(seed // 2) % 3],
            "lengths": lengths}


def _snapshots(arguments):
    return [(t, t.detach().resolve_neg().clone(), t._version, t.stride(), t.is_neg()) for t in arguments]


def _unchanged(snapshots, counters):
    for tensor, before, version, stride, negative in snapshots:
        torch.testing.assert_close(tensor.detach().resolve_neg(), before, rtol=0, atol=0, equal_nan=True)
        if tensor._version != version or tensor.stride() != stride or tensor.is_neg() != negative:
            raise AssertionError("Second-order evaluation changed input version or view metadata")
    _tick(counters, "input_nonmutation")


def _check_route(arguments, branch, counters):
    a, b, g, values, logits, boundaries = arguments
    padded = boundaries.dtype == torch.bool
    route = "padded" if padded else "packed"
    snapshots = _snapshots(arguments)
    pool = ragged_pool_padded if padded else ragged_pool
    actual = second_vjp(pool, *arguments, branch)
    # For float32, double per-row composition is the deciding numerical oracle.
    precise = tuple(t.double() if t.is_floating_point() else t for t in arguments)
    expected = tuple(t.to(values.dtype) for t in second_vjp(row_reference, *precise, branch, oracle=True))
    assert_second_close(actual, expected, values.dtype, "public " + route)
    _tick(counters, "second_public_" + route)
    private_a = torch.zeros_like(a) if branch == "logits" else a
    private_b = torch.zeros_like(b) if branch == "values" else b
    operation = torch.ops.ragged_pool._padded_double_backward.default if padded else torch.ops.ragged_pool._double_backward.default
    private = operation(private_a, private_b, g, values, logits, boundaries)
    assert_second_close(private, expected, values.dtype, "private " + route)
    _tick(counters, "second_private_" + route)
    _tick(counters, route + "_second_vjp")
    _tick(counters, "second_order_oracle")
    lengths = boundaries.sum(dim=1) if padded else boundaries[1:] - boundaries[:-1]
    if bool((lengths <= 1).all()):
        expected_dg = torch.zeros_like(g)
        if branch != "logits":
            if padded:
                expected_dg[lengths == 1] = a[boundaries]
            else:
                expected_dg[lengths == 1] = a[boundaries[:-1][lengths == 1]]
        for result in (actual, private):
            if bool(torch.count_nonzero(result[1])) or bool(torch.count_nonzero(result[2])):
                raise AssertionError("Empty/singleton input second derivatives must be zero")
            torch.testing.assert_close(result[0], expected_dg, rtol=0, atol=0)
        _tick(counters, "empty_singleton_analytic")
    if padded:
        for result in (actual, private):
            if bool(torch.count_nonzero(result[1][~boundaries])) or bool(torch.count_nonzero(result[2][~boundaries])):
                raise AssertionError("Ignored padded entries received nonzero second derivatives")
            if bool(torch.count_nonzero(result[0][~boundaries.any(dim=1)])):
                raise AssertionError("An empty padded row received a nonzero dg")
        _tick(counters, "zero_ignored_derivatives")
        if values[~boundaries].numel():
            _tick(counters, "ignored_nonfinite")
    _unchanged(snapshots, counters)
    return actual


def _relations(case, baseline, counters, generator):
    arguments, branch = case["arguments"], case["branch"]
    a, b, g, values, logits, offsets = arguments
    dtype = values.dtype
    if any(length > 1 for length in case["lengths"]):
        indices = [torch.randperm(end - start, generator=generator, device="cpu") + start
                   for start, end in zip(offsets.tolist(), offsets.tolist()[1:])]
        permutation = torch.cat(indices)
        changed = (a[permutation], b[permutation], g, values[permutation], logits[permutation], offsets)
        permuted = _check_route(changed, branch, counters)
        assert_second_close(permuted, (baseline[0], baseline[1][permutation], baseline[2][permutation]), dtype, "permutation")
        _tick(counters, "within_group_permutation")
    shifted = logits.clone()
    changed_groups = 0
    for start, end in zip(offsets.tolist(), offsets.tolist()[1:]):
        if start != end:
            amount = -16 if float(logits[start:end].max()) > 984 else 16
            shifted[start:end] += amount
            changed_groups += 1
    if changed_groups:
        changed = a, b, g, values, shifted, offsets
        assert_second_close(_check_route(changed, branch, counters), baseline, dtype, "safe group logit shift")
        _tick(counters, "safe_logit_shift")
    if case["mode"] != "zero_steps":
        relocated = _padded(arguments, case["layout"], extra=2, shift=2)
        result = _check_route(relocated, branch, counters)
        assert_second_close((result[0], result[1][relocated[-1]], result[2][relocated[-1]]), baseline, dtype, "equivalent padding")
        _tick(counters, "equivalent_padding")


def run_case(seed, dtype=None):
    """Run one bounded reproducible case without changing global RNGs or threads."""
    case = make_case(seed, dtype)
    counters = {}
    try:
        packed = _check_route(case["arguments"], case["branch"], counters)
        padded = _check_route(case["padded_arguments"], case["branch"], counters)
        mask = case["padded_arguments"][-1]
        actual_dtype = case["arguments"][3].dtype
        assert_second_close((padded[0], padded[1][mask], padded[2][mask]), packed, actual_dtype, "packed/padded")
        _tick(counters, "packed_padded_equivalence")
        _tick(counters, "mode_" + case["mode"])
        _tick(counters, "upstream_" + case["layout"])
        if any(bool(torch.count_nonzero(value)) for value in packed):
            _tick(counters, "nonzero_view_" + case["layout"])
            _tick(counters, "nonzero_branch_" + case["branch"])
        if case["branch"] != "both":
            _tick(counters, "unused_" + ("values" if case["branch"] == "logits" else "logits") + "_branch")
        generator = torch.Generator(device="cpu").manual_seed((seed + 104729) % (2 ** 63))
        _relations(case, packed, counters, generator)
    except Exception as error:
        raise AssertionError(f"Generated second-order case seed={seed}, dtype={case['arguments'][3].dtype}, "
                             f"mode={case['mode']}, branch={case['branch']}, view={case['layout']} failed: {error}") from error
    values = case["arguments"][3]
    return {"seed": seed, "dtype": str(values.dtype), "mode": case["mode"], "branch": case["branch"],
            "layout": case["layout"], "shape": {"N": values.shape[0], "D": values.shape[1],
                "B": len(case["lengths"]), "T": case["padded_arguments"][-1].shape[1]}, "counters": counters}
