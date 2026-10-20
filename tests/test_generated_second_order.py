"""Generated higher derivatives, an oracle fault control and small numerical checks."""
from collections import Counter
import importlib.util
import math
from pathlib import Path
import random

import pytest
import torch

from ragged_pool import ragged_pool, ragged_pool_padded


_spec = importlib.util.spec_from_file_location(
    'ragged_pool_generated_second_order',
    Path(__file__).with_name('generated_second_order_cases.py'),
)
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)


def _pool(route, values, logits, structure):
    return (ragged_pool if route == 'packed' else ragged_pool_padded)(values, logits, structure)


def _first_scalar(route, g, values, logits, structure, a, b):
    v, l = (tensor.detach().clone().requires_grad_() for tensor in (values, logits))
    first = torch.autograd.grad(_pool(route, v, l, structure), (v, l), g)
    return ((first[0] * a).sum() + (first[1] * b).sum()).detach()


def _native_second(route, g, values, logits, structure, a, b):
    upstream, v, l = (tensor.detach().clone().requires_grad_() for tensor in (g, values, logits))
    first = torch.autograd.grad(_pool(route, v, l, structure), (v, l), upstream, create_graph=True)
    return torch.autograd.grad(first, (upstream, v, l), (a, b))


@pytest.mark.parametrize('route', ['packed', 'padded'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('g_value', [0., 2.])
def test_checker_rejects_only_dropped_dg_on_known_nonzero_witness(route, dtype, g_value):
    # p=(1/4,3/4). The hand-computed dg=-2 does not depend on g.
    # At g=0 both other outputs vanish, so checking only those would miss it.
    values = torch.tensor([[1.], [3.]], dtype=dtype)
    logits = torch.tensor([0., math.log(3.)], dtype=dtype)
    a = torch.tensor([[4.], [-2.]], dtype=dtype)
    b = torch.tensor([3., -1.], dtype=dtype)
    structure = torch.tensor([0, 2])
    if route == 'padded':
        values, logits, a, b = (tensor.unsqueeze(0) for tensor in (values, logits, a, b))
        structure = torch.ones((1, 2), dtype=torch.bool)
    g = torch.tensor([[g_value]], dtype=dtype)
    expected = (torch.tensor([[-2.]], dtype=dtype),
                values.new_tensor([[.75 * g_value], [-.75 * g_value]]).reshape_as(values),
                logits.new_tensor([.375 * g_value, -.375 * g_value]).reshape_as(logits))
    actual = _native_second(route, g, values, logits, structure, a, b)
    _cases.assert_second_close(actual, expected, dtype, context='unfaulted witness')
    # The test owns this one corrupted tuple. Production registration and the
    # loaded native library are never replaced or patched.
    dropped_dg = (torch.zeros_like(actual[0]), *actual[1:])
    for left, right in zip(dropped_dg[1:], expected[1:]):
        torch.testing.assert_close(left, right, rtol=4e-5, atol=4e-6)
    with pytest.raises(AssertionError, match=r'\bdg\b'):
        _cases.assert_second_close(dropped_dg, expected, dtype, context='injected fault')


@pytest.mark.parametrize('route', ['packed', 'padded'])
@pytest.mark.parametrize('seed', [19, 731, 20261020])
@pytest.mark.parametrize('block', ['g', 'values', 'logits', 'joint'])
def test_tiny_float64_second_vjp_direction_matches_first_gradient_finite_difference(route, seed, block):
    # All numerical inputs own nonoverlapping storage and lie well inside the
    # admitted range. Views and endpoint inputs are checked by other tests.
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((5, 2), generator=generator, dtype=torch.float64) * .5
    logits = torch.randn((5,), generator=generator, dtype=torch.float64) * .25
    a = torch.randn(values.shape, generator=generator, dtype=values.dtype) * .5
    b = torch.randn(logits.shape, generator=generator, dtype=values.dtype) * .5
    structure = torch.tensor([0, 2, 2, 5])
    if route == 'padded':
        mask = torch.tensor([[True, False, True, False], [False] * 4,
                             [False, True, True, True]])
        padded = []
        for tensor in (values, logits, a, b):
            target = tensor.new_zeros((3, 4, *tensor.shape[1:]))
            target[mask] = tensor
            padded.append(target)
        values, logits, a, b = padded
        structure = mask
    g = torch.randn((3, 2), generator=generator, dtype=values.dtype) * .5
    variables = (g, values, logits)
    directions = []
    for name, tensor in zip(('g', 'values', 'logits'), variables):
        direction = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        direction = direction / max(1., float(torch.linalg.vector_norm(direction)))
        directions.append(direction if block in (name, 'joint') else torch.zeros_like(direction))
    second = _native_second(route, *variables, structure, a, b)
    analytical = sum((gradient * direction).sum() for gradient, direction in zip(second, directions))
    eps = 1e-5
    plus = tuple(tensor + eps * direction for tensor, direction in zip(variables, directions))
    minus = tuple(tensor - eps * direction for tensor, direction in zip(variables, directions))
    numerical = (_first_scalar(route, *plus, structure, a, b)
                 - _first_scalar(route, *minus, structure, a, b)) / (2 * eps)
    torch.testing.assert_close(analytical, numerical, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('seed', [0, 2, 4, 6, 8, 10, 12, 14, 24, 30, 40, 46, 56, 72, 88, 168])
def test_generated_second_order_seed_manifest(seed, dtype):
    report = _cases.run_case(seed, dtype)
    assert report['seed'] == seed
    for name in ('second_public_packed', 'second_private_packed',
                 'second_public_padded', 'second_private_padded'):
        assert report['counters'][name] > 0


def test_frozen_manifest_reaches_declared_categories_and_replays_without_global_rng_changes():
    counters = Counter()
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    for seed in _cases.FROZEN_SEEDS:
        for dtype in (torch.float32, torch.float64):
            report = _cases.run_case(seed, dtype)
            counters.update(report['counters'])
    # Compare the actual generated tensors as well as the returned counters.
    # Equal metadata alone would not establish that a seed replays its inputs.
    left, right = (_cases.make_case(88, torch.float64) for _ in range(2))
    for key in ('arguments', 'padded_arguments'):
        for first, second in zip(left[key], right[key]):
            torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
            assert first.stride() == second.stride() and first.is_neg() == second.is_neg()
    assert _cases.run_case(8, torch.float64) == _cases.run_case(8, torch.float64)
    assert _cases.make_case(8)['arguments'][3].dtype == torch.float32
    assert _cases.make_case(9)['arguments'][3].dtype == torch.float64
    assert python_state == random.getstate()
    assert torch.equal(torch_state, torch.random.get_rng_state())
    for name in _cases.REQUIRED_CATEGORIES:
        assert counters[name] > 0, name
