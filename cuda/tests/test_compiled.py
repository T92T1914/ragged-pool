"""Pinned CUDA AOTAutograd checks, separate from derivative mathematics."""
import json

import pytest
import torch

from ragged_pool_cuda import ragged_pool


DTYPES = (torch.float32, torch.float64)
BOUNDARIES = (0, 2, 2, 5, 6)


@pytest.fixture(autouse=True)
def pinned_compiler():
    assert str(torch.__version__) == "2.9.1+cu128"
    assert torch.version.cuda == "12.8" and torch.cuda.is_available()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    assert not torch._dynamo.config.suppress_errors
    torch._dynamo.reset()
    yield
    torch.cuda.synchronize()
    torch._dynamo.reset()


def inputs(dtype=torch.float64, boundaries=BOUNDARIES):
    count = boundaries[-1]
    values = (torch.arange(count * 3, dtype=dtype, device="cuda").reshape(count, 3) / 11 - 0.7)
    logits = torch.linspace(-0.8, 0.6, count, dtype=dtype, device="cuda")
    return (values.requires_grad_(), logits.requires_grad_(),
            torch.tensor(boundaries, dtype=torch.int64, device="cuda"))


def primitive(values, logits, boundaries):
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        v, l = values[start:stop].double(), logits[start:stop].double()
        row = (v.sum(0) * 0 + l.sum() * 0 if start == stop
               else (l.softmax(0)[:, None] * v).sum(0))
        rows.append(row.to(values.dtype))
    return (torch.stack(rows) if rows else
            values[:0] + logits.sum() * 0)


def compare(pool, dtype=torch.float64, boundaries=BOUNDARIES, upstream=None):
    args = inputs(dtype, boundaries)
    originals = [x.detach().clone() for x in args]
    versions = [x._version for x in args]
    rv, rl = [x.detach().clone().requires_grad_() for x in args[:2]]
    observed = pool(*args)
    expected = primitive(rv, rl, boundaries)
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype == torch.float32
                 else {"rtol": 2e-10, "atol": 2e-12})
    torch.testing.assert_close(observed, expected, **tolerance)
    if upstream is None:
        actual_grads = torch.autograd.grad(observed.sum(), args[:2])
        wanted_grads = torch.autograd.grad(expected.sum(), (rv, rl))
    else:
        actual_grads = torch.autograd.grad(observed, args[:2], upstream)
        wanted_grads = torch.autograd.grad(expected, (rv, rl), upstream)
    for actual, wanted in zip(actual_grads, wanted_grads):
        torch.testing.assert_close(actual, wanted, **tolerance)
        assert bool(torch.isfinite(actual).all())
    for value, original, version in zip(args, originals, versions):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert value._version == version
    assert args[2].grad is None
    for row, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if start == stop:
            assert torch.count_nonzero(observed[row]) == 0
        elif stop - start == 1:
            assert actual_grads[1][start] == 0
    if not boundaries[-1]:
        assert torch.count_nonzero(observed) == 0
        assert all(torch.count_nonzero(grad) == 0 for grad in actual_grads)
    return observed.detach()


def compiled_pool():
    return torch.compile(ragged_pool, backend="aot_eager", fullgraph=True, dynamic=False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", ("pool", "backward"))
def test_registration_aot_dispatch_is_separate_from_gradient_checks(dtype, operation):
    values, logits, offsets = inputs(dtype)
    if operation == "pool":
        op = torch.ops.ragged_pool_cuda.pool.default
        args = values, logits, offsets
    else:
        op = torch.ops.ragged_pool_cuda._backward.default
        args = (torch.ones(4, 3, dtype=dtype, device="cuda"),
                values.detach(), logits.detach(), offsets)
    result = torch.library.opcheck(op, args, test_utils=("test_aot_dispatch_dynamic",))
    assert result == {"test_aot_dispatch_dynamic": "SUCCESS"}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boundaries", (BOUNDARIES, (0,), (0, 0, 0), (0, 0, 1, 1)))
def test_public_backend_string_outputs_and_first_gradients(dtype, boundaries):
    compare(compiled_pool(), dtype, boundaries)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinned_graph_callbacks_keep_native_forward_and_backward(dtype):
    # These private v2.9.1 callbacks inspect submissions, not optimized code.
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.backends.debugging import boxed_nop
    dynamo, forward, backward = [], [], []
    aot = lookup_backend("aot_eager")

    def capture(target):
        def compiler(graph, example_inputs):
            target.append([str(node.target) for node in graph.graph.nodes])
            return boxed_nop(graph, example_inputs)
        return compiler

    def backend(graph, example_inputs):
        dynamo.append([str(node.target) for node in graph.graph.nodes])
        return aot(graph, example_inputs, fw_compiler=capture(forward), bw_compiler=capture(backward))

    pool = torch.compile(ragged_pool, backend=backend, fullgraph=True, dynamic=False)
    first = compare(pool, dtype)
    second = compare(pool, dtype, (0, 0, 1, 4, 6))
    assert not torch.equal(first, second)
    assert len(dynamo) == len(forward) == len(backward) == 1
    assert "ragged_pool_cuda.pool.default" in forward[0]
    assert "ragged_pool_cuda._backward.default" in backward[0]
    print(json.dumps({"status": "passed", "dtype": str(dtype),
                      "scope": "pinned graph callbacks, separate from public backend-string execution",
                      "dynamo_submissions": len(dynamo), "aot_forward": forward, "aot_backward": backward}))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ("sum", "expanded", "transposed", "stepped", "negative"))
def test_compiled_backward_upstream_layouts(dtype, layout):
    pool = compiled_pool()
    if layout == "sum":
        compare(pool, dtype)
        return
    if layout == "expanded":
        upstream = torch.tensor(0.4, dtype=dtype, device="cuda").expand(4, 3)
        assert upstream.stride() == (0, 0)
    elif layout == "transposed":
        upstream = (torch.arange(12, dtype=dtype, device="cuda").reshape(3, 4) / 10).t()
        assert upstream.stride() == (1, 4) and not upstream.is_contiguous()
    elif layout == "stepped":
        upstream = (torch.arange(24, dtype=dtype, device="cuda").reshape(4, 6) / 10)[:, ::2]
        assert upstream.stride() == (6, 2) and not upstream.is_contiguous()
    else:
        upstream = torch._neg_view(torch.ones(4, 3, dtype=dtype, device="cuda"))
        assert upstream.is_neg()
    original, version = upstream.clone(), upstream._version
    compare(pool, dtype, upstream=upstream)
    torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    assert upstream._version == version


@pytest.mark.parametrize("changed", (0, 1, 2))
def test_compiled_saved_original_mutation_refused(changed):
    pool = compiled_pool()
    compare(pool)
    args = inputs()
    output = pool(*args)
    with torch.no_grad():
        args[changed].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
    compare(pool)


@pytest.mark.parametrize("case", ("start", "decreasing", "negative", "past_end", "values_nan",
                                  "values_range", "logits_inf", "logits_range"))
def test_compiled_runtime_content_refused_then_recovers(case):
    pool = compiled_pool()
    compare(pool)
    values, logits, offsets = inputs()
    with torch.no_grad():
        if case == "start":
            offsets[0] = 1
        elif case == "decreasing":
            offsets[2] = 1
        elif case == "negative":
            offsets[1] = -1
        elif case == "past_end":
            offsets[2] = 7
        elif case == "values_nan":
            values[0, 0] = float("nan")
        elif case == "values_range":
            values[0, 0] = 1001
        elif case == "logits_inf":
            logits[0] = float("inf")
        else:
            logits[0] = -1001
    with pytest.raises(RuntimeError):
        pool(values, logits, offsets)
    compare(pool)


def test_compiled_private_backward_refuses_nonfinite_upstream_then_recovers():
    backward = torch.compile(torch.ops.ragged_pool_cuda._backward.default,
                             backend="aot_eager", fullgraph=True, dynamic=False)
    values, logits, offsets = inputs()
    values, logits = values.detach(), logits.detach()
    grad = torch.ones(4, 3, dtype=values.dtype, device="cuda")
    before = backward(grad, values, logits, offsets)
    grad[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        backward(grad, values, logits, offsets)
    grad[0, 0] = 1
    after = backward(grad, values, logits, offsets)
    for first, second in zip(before, after):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_compiled_outputs_own_fresh_storage():
    pool = compiled_pool()
    values, logits, offsets = inputs()
    first, second = pool(values, logits, offsets), pool(values, logits, offsets)
    pointers = [x.untyped_storage().data_ptr() for x in (values, logits, offsets, first, second)]
    assert len(set(pointers)) == len(pointers)
