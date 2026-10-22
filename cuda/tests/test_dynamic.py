"""Changing packed CUDA shapes: mathematics and separate AOT graph evidence."""
import json

import pytest
import torch

from ragged_pool_cuda import ragged_pool


DTYPES = (torch.float32, torch.float64)
SEQUENCE = ((0, 2, 2, 5, 6), (0, 1, 4, 7), (0, 0, 1, 3, 3, 5, 8),
            (0, 3, 3, 6), (0, 0, 4, 5, 5, 8), (0, 0, 1, 4, 6))
METADATA_REFUSAL = (RuntimeError, ValueError, TypeError, NotImplementedError,
                    torch._dynamo.exc.Unsupported, torch._dynamo.exc.TorchRuntimeError)


@pytest.fixture(autouse=True)
def pinned_compiler():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128"
    assert torch.version.cuda == "12.8" and torch.cuda.is_available()
    torch.cuda.set_device(0)
    assert not torch._dynamo.config.suppress_errors
    torch._dynamo.reset()
    yield
    torch.cuda.synchronize(0)
    torch._dynamo.reset()


def inputs(dtype, boundaries):
    count = boundaries[-1]
    values = (torch.arange(count * 3, dtype=dtype, device="cuda:0").reshape(count, 3) / 17 - .6)
    logits = torch.linspace(-.8, .7, count, dtype=dtype, device="cuda:0")
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda:0")
    return values.requires_grad_(), logits.requires_grad_(), offsets


def primitive(values, logits, boundaries):
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        v, l = values[start:stop].double(), logits[start:stop].double()
        rows.append((v.sum(0) * 0 + l.sum() * 0 if start == stop else
                     (l.softmax(0)[:, None] * v).sum(0)).to(values.dtype))
    return torch.stack(rows)


def snapshot(tensors):
    return [(tensor.detach().resolve_neg().resolve_conj().contiguous().view(torch.uint8).clone(),
             tensor._version, tensor.stride(), tensor.untyped_storage()._cdata, tensor.is_neg())
            for tensor in tensors]


def unchanged(tensors, saved):
    for tensor, (data, version, stride, storage, negative) in zip(tensors, saved):
        assert torch.equal(tensor.detach().resolve_neg().resolve_conj().contiguous().view(torch.uint8), data)
        assert tensor._version == version and tensor.stride() == stride
        assert tensor.untyped_storage()._cdata == storage and tensor.is_neg() == negative


def tolerance(dtype):
    return {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-10, "atol": 2e-12}


def compare(pool, dtype, boundaries, upstream=None):
    args = inputs(dtype, boundaries)
    tensors = args if upstream is None else (*args, upstream)
    saved = snapshot(tensors)
    rv, rl = (tensor.detach().clone().requires_grad_() for tensor in args[:2])
    actual, expected = pool(*args), primitive(rv, rl, boundaries)
    assert actual.shape == (len(boundaries) - 1, 3)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    if upstream is None:
        gradients = torch.autograd.grad(actual.sum(), args[:2])
        wanted = torch.autograd.grad(expected.sum(), (rv, rl))
    else:
        gradients = torch.autograd.grad(actual, args[:2], upstream)
        wanted = torch.autograd.grad(expected, (rv, rl), upstream)
    for gradient, reference in zip(gradients, wanted):
        torch.testing.assert_close(gradient, reference, **tolerance(dtype))
        assert bool(torch.isfinite(gradient).all())
    for row, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if start == stop:
            assert torch.count_nonzero(actual[row]) == 0
        elif stop - start == 1:
            assert gradients[1][start] == 0
    unchanged(tensors, saved)
    assert args[2].grad is None
    owned = (*tensors, actual, *gradients)
    assert len({tensor.untyped_storage()._cdata for tensor in owned}) == len(owned)
    return actual.detach()


def compiled_pool():
    return torch.compile(ragged_pool, backend="aot_eager", fullgraph=True, dynamic=True)


def warm(pool, dtype):
    outputs = [compare(pool, dtype, boundaries) for boundaries in SEQUENCE]
    assert len({output.untyped_storage()._cdata for output in outputs}) == len(outputs)
    assert outputs[0].shape == outputs[-1].shape and not torch.equal(outputs[0], outputs[-1])
    return outputs


@pytest.mark.parametrize("dtype", DTYPES)
def test_public_dynamic_backend_matches_both_gradients_across_decoupled_shapes(dtype):
    shapes = {(boundary[-1], len(boundary) - 1) for boundary in SEQUENCE}
    assert any(n == other_n and b != other_b for n, b in shapes for other_n, other_b in shapes)
    assert any(b == other_b and n != other_n for n, b in shapes for other_n, other_b in shapes)
    warm(compiled_pool(), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_upstream_views_after_shape_changes_are_preserved(dtype):
    pool = compiled_pool()
    warm(pool, dtype)
    expanded = torch.tensor(.4, dtype=dtype, device="cuda:0").expand(4, 3)
    transposed = (torch.arange(12, dtype=dtype, device="cuda:0").reshape(3, 4) / 10).t()
    stepped = (torch.arange(24, dtype=dtype, device="cuda:0").reshape(4, 6) / 10)[:, ::2]
    negative = torch._neg_view(torch.ones(4, 3, dtype=dtype, device="cuda:0"))
    assert expanded.stride() == (0, 0)
    assert transposed.stride() == (1, 4) and not transposed.is_contiguous()
    assert stepped.stride() == (6, 2) and not stepped.is_contiguous()
    assert negative.is_neg()
    for upstream in (expanded, transposed, stepped, negative):
        compare(pool, dtype, SEQUENCE[0], upstream)


@pytest.mark.parametrize("dtype", DTYPES)
def test_runtime_content_admission_on_the_same_callable_then_valid_recovery(dtype):
    pool = compiled_pool()
    warm(pool, dtype)
    for fault in ("start", "decreasing", "end", "values_nan", "values_inf", "values_range",
                  "logits_nan", "logits_inf", "logits_range"):
        args = inputs(dtype, SEQUENCE[2])
        with torch.no_grad():
            if fault == "start": args[2][0] = 1
            elif fault == "decreasing": args[2][3] = 0
            elif fault == "end": args[2][-1] = SEQUENCE[2][-1] - 1
            else:
                target = args[0] if fault.startswith("values") else args[1]
                target.view(-1)[0] = (float("nan") if fault.endswith("nan") else
                                       float("inf") if fault.endswith("inf") else 1001.)
        with pytest.raises(RuntimeError):
            pool(*args)
        compare(pool, dtype, SEQUENCE[2])


@pytest.mark.parametrize("dtype", DTYPES)
def test_saved_original_mutation_refused_after_shape_changes_then_recovers(dtype):
    pool = compiled_pool()
    warm(pool, dtype)
    for changed in (0, 1, 2):
        args = inputs(dtype, SEQUENCE[0])
        output = pool(*args)
        with torch.no_grad(): args[changed].view(-1)[0].add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            output.sum().backward()
        compare(pool, dtype, SEQUENCE[-1])


def test_metadata_refusal_uses_bounded_allocations_and_preserves_recovery():
    pool, dtype = compiled_pool(), torch.float64
    warm(pool, dtype)
    for fault in ("cpu", "meta", "sparse", "values_rank", "logits_rank", "offsets_rank",
                  "logits_length", "offsets_dtype", "float16", "logits_dtype", "strided",
                  "N_limit", "B_limit", "D_zero", "D_limit"):
        args = list(inputs(dtype, SEQUENCE[0]))
        if fault == "cpu": args[0] = args[0].detach().cpu()
        elif fault == "meta": args[2] = torch.empty(5, dtype=torch.int64, device="meta")
        elif fault == "sparse": args[0] = args[0].to_sparse()
        elif fault == "values_rank": args[0] = args[0].reshape(2, 3, 3)
        elif fault == "logits_rank": args[1] = args[1][:, None]
        elif fault == "offsets_rank": args[2] = args[2][:, None]
        elif fault == "logits_length": args[1] = args[1][:-1]
        elif fault == "offsets_dtype": args[2] = args[2].int()
        elif fault == "float16": args[:2] = [tensor.half() for tensor in args[:2]]
        elif fault == "logits_dtype": args[1] = args[1].float()
        elif fault == "strided": args[0] = args[0].t().contiguous().t()
        elif fault == "N_limit":
            args = [torch.zeros(4097, 3, dtype=dtype, device="cuda:0"),
                    torch.zeros(4097, dtype=dtype, device="cuda:0"),
                    torch.tensor([0, 1, 4097], dtype=torch.int64, device="cuda:0")]
        elif fault == "B_limit": args[2] = torch.zeros(130, dtype=torch.int64, device="cuda:0")
        else:
            width = 0 if fault == "D_zero" else 129
            args[0] = torch.zeros(6, width, dtype=dtype, device="cuda:0")
        with pytest.raises(METADATA_REFUSAL):
            pool(*args)
        compare(pool, dtype, SEQUENCE[0])


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinned_aot_submissions_retain_native_targets_and_symbolic_dimensions(dtype):
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.backends.debugging import boxed_nop
    dynamo, forward, backward = [], [], []
    aot = lookup_backend("aot_eager")

    def capture(destination, target, value_index, offset_index):
        def compiler(graph, example_inputs):
            record = {"targets": [str(node.target) for node in graph.graph.nodes], "calls": []}
            for node in graph.graph.nodes:
                if node.op != "call_function" or str(node.target) != target:
                    continue
                values = node.args[value_index].meta["val"]
                offsets = node.args[offset_index].meta["val"]
                record["calls"].append({"values_shape": [str(size) for size in values.shape],
                    "offsets_shape": [str(size) for size in offsets.shape],
                    "N_symbolic": isinstance(values.shape[0], torch.SymInt),
                    "offset_length_symbolic": isinstance(offsets.shape[0], torch.SymInt)})
            destination.append(record)
            return boxed_nop(graph, example_inputs)
        return compiler

    def backend(graph, example_inputs):
        dynamo.append([str(node.target) for node in graph.graph.nodes])
        return aot(graph, example_inputs,
            fw_compiler=capture(forward, "ragged_pool_cuda.pool.default", 0, 2),
            bw_compiler=capture(backward, "ragged_pool_cuda._backward.default", 1, 3))

    pool = torch.compile(ragged_pool, backend=backend, fullgraph=True, dynamic=True)
    warm(pool, dtype)
    assert dynamo and forward and backward
    for submissions, target in ((forward, "ragged_pool_cuda.pool.default"),
                                (backward, "ragged_pool_cuda._backward.default")):
        assert any(target in record["targets"] for record in submissions)
        assert any(call["N_symbolic"] and call["offset_length_symbolic"]
                   for record in submissions for call in record["calls"])
    print(json.dumps({"status": "passed", "dtype": str(dtype),
        "scope": "Pinned callback metadata inspection, separate from public backend-string execution",
        "dynamo_submissions": len(dynamo), "aot_forward": forward, "aot_backward": backward,
        "universal_graph_count_claimed": False}))
