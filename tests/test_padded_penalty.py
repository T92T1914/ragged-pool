"""Independent normalization witnesses and the installed padded penalty caller."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def consumer():
    path = Path(__file__).resolve().parents[1] / 'examples/train_padded_gradient_penalty.py'
    spec = importlib.util.spec_from_file_location('padded_penalty_example', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('steps', [4, 8])
def test_selected_penalty_helper_keeps_known_value_and_head_derivative(steps):
    example = consumer()
    head = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    mask = torch.zeros((1, steps), dtype=torch.bool)
    mask[0, 0], mask[0, steps-1] = True, True
    coefficients = torch.zeros((1, steps, 1), dtype=torch.float64)
    coefficients[0, 0, 0], coefficients[0, steps-1, 0] = 0.25, 0.75
    gradient = torch.where(mask[..., None], head * coefficients, float('nan'))
    penalty = 0.5 * example.selected_gradient_penalty(gradient, mask)
    derivative, = torch.autograd.grad(penalty, (head,))
    torch.testing.assert_close(penalty, penalty.new_tensor(0.625), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(derivative, derivative.new_tensor(0.625), rtol=1e-12, atol=1e-12)


def test_all_false_penalty_is_differentiable_zero_without_reading_nan_padding():
    example = consumer()
    gradient = torch.full((3, 4, 2), float('nan'), dtype=torch.float64, requires_grad=True)
    penalty = example.selected_gradient_penalty(gradient, torch.zeros((3, 4), dtype=torch.bool))
    derivative, = torch.autograd.grad(penalty, (gradient,))
    assert penalty.item() == 0 and torch.count_nonzero(derivative) == 0
    assert bool(torch.isfinite(derivative).all())


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_full_installed_padded_penalty_caller(dtype):
    result = consumer().run(dtype)
    assert result['status'] == 'passed' and result['dtype'] == dtype
    assert result['optimizer_steps'] == result['reference_optimizer_steps'] == 2
    assert result['normalization'] == 'selected_raw_features'
    assert result['invariance']['status'] == result['controls']['status'] == result['all_empty']['status'] == 'passed'
