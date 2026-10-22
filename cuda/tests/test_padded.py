"""Bounded CUDA padded mathematics, admission and first-order registration."""
import json
import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from ragged_pool_cuda import ragged_pool_padded, ragged_pool_padded_reference


DTYPES = (torch.float32, torch.float64)
REFUSAL = (RuntimeError, ValueError, TypeError, NotImplementedError)


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128"
    assert torch.version.cuda == "12.8" and torch.cuda.is_available()
    torch.cuda.set_device(0)


def inputs(dtype=torch.float64, *, shape=(3, 4, 2), selection="holes", sentinels=False):
    batches, steps, width = shape
    values = torch.linspace(-0.8, 0.9, batches * steps * width, dtype=dtype,
                            device="cuda:0").reshape(shape)
    logits = torch.linspace(-0.4, 0.7, batches * steps, dtype=dtype,
                            device="cuda:0").reshape(batches, steps)
    mask = torch.ones((batches, steps), dtype=torch.bool, device="cuda:0")
    if selection == "empty":
        mask.zero_()
    elif selection == "holes":
        mask.zero_()
        if batches and steps:
            mask[0, ::2] = True
            if batches > 2:
                mask[2, min(1, steps - 1)] = True
    if sentinels:
        values[~mask], logits[~mask] = float("nan"), float("inf")
        if batches and steps > 1 and not bool(mask[0, 1]):
            values[0, 1], logits[0, 1] = 1001., -1001.
    return values.requires_grad_(), logits.requires_grad_(), mask


def tolerance(dtype):
    return {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-10, "atol": 2e-12}


def primitive(values, logits, mask):
    """Pool independent rows without offsets or custom operations."""
    rows = []
    for row in range(values.shape[0]):
        selected, scores = values[row][mask[row]].double(), logits[row][mask[row]].double()
        pooled = (selected.sum(0) * 0 + scores.sum() * 0 if not scores.numel()
                  else (scores.softmax(0)[:, None] * selected).sum(0))
        rows.append(pooled.to(values.dtype))
    return (torch.stack(rows) if rows else
            (values.sum(dim=(0, 1)) * 0 + logits.sum() * 0).unsqueeze(0)[:0])


def snapshot(tensors):
    return [(tensor.detach().view(torch.uint8).clone(), tensor._version) for tensor in tensors]


def unchanged(tensors, saved):
    for tensor, (data, version) in zip(tensors, saved):
        assert torch.equal(tensor.detach().view(torch.uint8), data)
        assert tensor._version == version


def compare(pool=ragged_pool_padded, dtype=torch.float64, *, selection="holes", shape=(3, 4, 2),
            sentinels=False, upstream=None):
    args = inputs(dtype, shape=shape, selection=selection, sentinels=sentinels)
    saved = snapshot(args)
    rv, rl = [x.detach().clone().requires_grad_() for x in args[:2]]
    actual, expected = pool(*args), primitive(rv, rl, args[2])
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    gradients = (torch.autograd.grad(actual.sum(), args[:2]) if upstream is None else
                 torch.autograd.grad(actual, args[:2], upstream))
    reference = (torch.autograd.grad(expected.sum(), (rv, rl)) if upstream is None else
                 torch.autograd.grad(expected, (rv, rl), upstream))
    for gradient, wanted in zip(gradients, reference):
        torch.testing.assert_close(gradient, wanted, **tolerance(dtype))
        assert bool(torch.isfinite(gradient).all())
        assert torch.count_nonzero(gradient[~args[2]]) == 0
    assert torch.count_nonzero(actual[~args[2].any(1)]) == 0
    unchanged(args, saved)
    assert args[2].grad is None
    return actual.detach()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pool", [ragged_pool_padded, ragged_pool_padded_reference])
def test_row_math_both_gradients_and_ignored_nonfinite_padding(dtype, pool):
    compare(pool, dtype, sentinels=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,selection", [((0, 0, 2), "all"), ((0, 4, 2), "all"),
    ((3, 0, 2), "empty"), ((3, 4, 2), "empty"), ((3, 4, 2), "all")])
def test_empty_dimensions_empty_rows_and_full_selection(dtype, shape, selection):
    compare(dtype=dtype, shape=shape, selection=selection, sentinels=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(128, 32, 1), (1, 4096, 1), (1, 1, 128)])
def test_admitted_boundaries_retain_independent_math(dtype, shape):
    compare(dtype=dtype, shape=shape, selection="all")


def test_analytic_weighted_singleton_and_empty_rows():
    values = torch.tensor([[[1., 2.], [99., 99.], [3., -2.]],
                           [[99., 99.], [7., 5.], [99., 99.]],
                           [[99., 99.], [99., 99.], [99., 99.]]], dtype=torch.float64, device="cuda:0", requires_grad=True)
    logits = torch.tensor([[0., 99., math.log(3)], [99., 0., 99.], [99., 99., 99.]],
                          dtype=torch.float64, device="cuda:0", requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, True, False], [False, False, False]], device="cuda:0")
    output = ragged_pool_padded(values, logits, mask)
    upstream = torch.tensor([[2., -3.], [.5, -1.], [9., 9.]], dtype=values.dtype, device=values.device)
    gv, gl = torch.autograd.grad(output, (values, logits), upstream)
    torch.testing.assert_close(output, output.new_tensor([[2.5, -1.], [7., 5.], [0., 0.]]), **tolerance(values.dtype))
    expected_v, expected_l = torch.zeros_like(values), torch.zeros_like(logits)
    expected_v[0, 0], expected_v[0, 2], expected_v[1, 1] = upstream[0] / 4, upstream[0] * .75, upstream[1]
    expected_l[0, 0], expected_l[0, 2] = -3., 3.
    torch.testing.assert_close(gv, expected_v, **tolerance(values.dtype))
    torch.testing.assert_close(gl, expected_l, **tolerance(values.dtype))


def test_bounded_double_precision_finite_differences():
    values, logits, mask = inputs(shape=(2, 2, 1), selection="all")
    assert torch.autograd.gradcheck(ragged_pool_padded, (values, logits, mask), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("fault", ["cpu", "meta", "sparse", "values_rank", "logits_shape", "mask_shape",
    "float16", "logits_dtype", "mask_dtype", "values_strided", "logits_strided", "mask_strided",
    "B_limit", "T_limit", "slots_limit", "D_zero", "D_limit"])
def test_metadata_refusal(fault):
    values, logits, mask = inputs()
    if fault == "cpu": values = values.detach().cpu()
    elif fault == "meta": mask = torch.empty(mask.shape, dtype=torch.bool, device="meta")
    elif fault == "sparse": values = values.to_sparse()
    elif fault == "values_rank": values = values.reshape(12, 2)
    elif fault == "logits_shape": logits = logits[:, :3].contiguous()
    elif fault == "mask_shape": mask = mask[:, :3].contiguous()
    elif fault == "float16": values, logits = values.half(), logits.half()
    elif fault == "logits_dtype": logits = logits.float()
    elif fault == "mask_dtype": mask = mask.long()
    elif fault == "values_strided": values = values.transpose(1, 2).contiguous().transpose(1, 2)
    elif fault == "logits_strided": logits = logits.t().contiguous().t()
    elif fault == "mask_strided": mask = mask.t().contiguous().t()
    else:
        shape = {"B_limit": (129, 0, 2), "T_limit": (0, 4097, 2), "slots_limit": (65, 64, 2),
                 "D_zero": (3, 4, 0), "D_limit": (3, 4, 129)}[fault]
        values, logits, mask = inputs(shape=shape)
    with pytest.raises(REFUSAL): ragged_pool_padded(values, logits, mask)


@pytest.mark.parametrize("which", [0, 1])
def test_original_negative_primal_is_refused_before_custom_dispatch(which):
    args = list(inputs())
    args[which] = torch._neg_view(args[which])
    assert args[which].is_neg()
    class NoPaddedDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            if operation is torch.ops.ragged_pool_cuda.padded.default:
                raise AssertionError("Original negative view reached padded dispatch")
            return operation(*args, **(kwargs or {}))
    with NoPaddedDispatch(), pytest.raises(REFUSAL): ragged_pool_padded(*args)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 1001., -1001.])
def test_selected_invalid_content_refused_and_valid_call_recovers(which, invalid):
    args = inputs()
    with torch.no_grad(): args[which][0, 0] = invalid
    with pytest.raises(REFUSAL): ragged_pool_padded(*args)
    compare()


@pytest.mark.parametrize("which", [0, 1, 2])
def test_saved_originals_include_ignored_slots_and_mask(which):
    args = inputs()
    output = ragged_pool_padded(*args)
    with torch.no_grad():
        if which == 2: args[which][0, 1] = True
        else: args[which][0, 1].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
    compare()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["expanded", "transposed", "stepped", "negative"])
def test_upstream_views_are_genuine_and_preserved(dtype, layout):
    if layout == "expanded":
        upstream = torch.tensor(.4, dtype=dtype, device="cuda:0").expand(3, 2)
        assert upstream.stride() == (0, 0)
    elif layout == "transposed":
        upstream = (torch.arange(6, dtype=dtype, device="cuda:0").reshape(2, 3) / 10).t()
        assert upstream.stride() == (1, 3) and not upstream.is_contiguous()
    elif layout == "stepped":
        upstream = (torch.arange(12, dtype=dtype, device="cuda:0").reshape(3, 4) / 10)[:, ::2]
        assert upstream.stride() == (4, 2) and not upstream.is_contiguous()
    else:
        upstream = torch._neg_view(torch.ones(3, 2, dtype=dtype, device="cuda:0"))
        assert upstream.is_neg()
    original, version = upstream.clone(), upstream._version
    compare(dtype=dtype, upstream=upstream)
    torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    assert upstream._version == version


def test_outputs_and_gradients_own_storage_and_second_order_is_refused():
    args = inputs()
    first, second = ragged_pool_padded(*args), ragged_pool_padded(*args)
    gradients = torch.autograd.grad(first.sum(), args[:2], create_graph=True)
    tensors = (*args, first, second, *gradients)
    assert len({tensor.untyped_storage()._cdata for tensor in tensors}) == len(tensors)
    assert len({id(tensor) for tensor in tensors}) == len(tensors)
    with pytest.raises(RuntimeError, match="first.order|second|order"):
        torch.autograd.grad(sum(value.square().sum() for value in gradients), args[:2])
    empty = inputs(shape=(0, 0, 2))
    one, two = ragged_pool_padded(*empty), ragged_pool_padded(*empty)
    assert one is not two and one.untyped_storage()._cdata != two.untyped_storage()._cdata


@pytest.mark.parametrize("fault", ["shape", "dtype", "cpu", "meta", "values_shape", "mask_dtype", "nan", "inf", "range"])
def test_private_backward_has_its_own_upstream_admission(fault):
    args = list(value.detach() for value in inputs())
    upstream = torch.ones(3, 2, dtype=args[0].dtype, device="cuda:0")
    if fault == "shape": upstream = upstream[:2]
    elif fault == "dtype": upstream = upstream.float()
    elif fault == "cpu": upstream = upstream.cpu()
    elif fault == "meta": upstream = torch.empty(3, 2, dtype=args[0].dtype, device="meta")
    elif fault == "values_shape": args[0] = args[0].reshape(12, 2)
    elif fault == "mask_dtype": args[2] = args[2].long()
    else: upstream[0, 0] = {"nan": float("nan"), "inf": float("inf"), "range": 1001.}[fault]
    with pytest.raises(REFUSAL): torch.ops.ragged_pool_cuda._padded_backward.default(upstream, *args)
    valid = tuple(value.detach() for value in inputs())
    result = torch.ops.ragged_pool_cuda._padded_backward.default(torch.ones(3, 2, dtype=valid[0].dtype, device="cuda:0"), *valid)
    assert all(bool(torch.isfinite(value).all()) for value in result)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", ["padded", "_padded_backward"])
def test_operator_registration_separate_from_mathematical_gradients(dtype, operation):
    args = inputs(dtype)
    op = getattr(torch.ops.ragged_pool_cuda, operation).default
    if operation == "_padded_backward":
        args = (torch.ones(3, 2, dtype=dtype, device="cuda:0"), *(value.detach() for value in args))
    result = torch.library.opcheck(op, args)
    assert all(value == "SUCCESS" for value in result.values())


def test_fake_metadata_for_forward_and_private_backward_without_reading_mask():
    with FakeTensorMode():
        values = torch.empty(3, 4, 2, dtype=torch.float64, device="cuda:0")
        logits = torch.empty(3, 4, dtype=torch.float64, device="cuda:0")
        mask = torch.empty(3, 4, dtype=torch.bool, device="cuda:0")
        output = torch.ops.ragged_pool_cuda.padded.default(values, logits, mask)
        gradients = torch.ops.ragged_pool_cuda._padded_backward.default(torch.empty_like(output), values, logits, mask)
        assert isinstance(output, FakeTensor) and output.shape == (3, 2)
        assert [tensor.shape for tensor in gradients] == [values.shape, logits.shape]
        with pytest.raises(REFUSAL): torch.ops.ragged_pool_cuda.padded.default(values, logits, mask.long())
        with pytest.raises(REFUSAL):
            torch.ops.ragged_pool_cuda._padded_backward.default(torch.empty(2, 2, dtype=values.dtype, device=values.device), values, logits, mask)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinned_graph_submissions_hide_compaction_and_keep_padded_backward(dtype):
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
    assert not torch._dynamo.config.suppress_errors
    torch._dynamo.reset()
    try:
        pool = torch.compile(ragged_pool_padded, backend=backend, fullgraph=True, dynamic=False)
        first, second = compare(pool, dtype), compare(pool, dtype, selection="all")
        assert not torch.equal(first, second)
        invalid = inputs(dtype)
        with torch.no_grad():
            invalid[0][0, 0] = float("inf")
        with pytest.raises(REFUSAL):
            pool(*invalid)
        compare(pool, dtype)
        assert len(dynamo) == len(forward) == len(backward) == 1
        assert "ragged_pool_cuda.padded.default" in forward[0]
        assert "ragged_pool_cuda._padded_backward.default" in backward[0]
        assert not any("nonzero" in target or target == "ragged_pool_cuda.pool.default"
                       for graph in forward + backward for target in graph)
        print(json.dumps({"status": "passed", "dtype": str(dtype), "scope": "pinned callback graph inspection",
                          "dynamo_submissions": len(dynamo), "aot_forward": forward, "aot_backward": backward}))
    finally:
        torch.cuda.synchronize(0)
        torch._dynamo.reset()
