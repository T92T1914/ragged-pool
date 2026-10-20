"""Run one seeded synthetic training step through an installed ragged-pool wheel."""
import argparse
import copy
import json
import os
import sys

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_reference


class SetRegressor(nn.Module):
    def __init__(self, width):
        super().__init__()
        # A shared scalar bias would cancel inside every group's softmax.
        self.scorer = nn.Linear(width, 1, bias=False)
        self.head = nn.Linear(width, 1)

    def forward(self, values, offsets, pool):
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def maximum_difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def run(seed, dtype_name):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype_name)
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    lengths = [3, 0, 5, 1, 4, 2]
    offsets = torch.tensor([0, 3, 3, 8, 9, 13, 15], dtype=torch.int64, device="cpu")
    features = torch.randn(sum(lengths), 4, dtype=dtype, device="cpu")
    targets = torch.tensor([0.25, -0.1, 0.75, -0.5, 0.3, 0.1], dtype=dtype, device="cpu")
    native_model = SetRegressor(4).to(device="cpu", dtype=dtype)
    reference_model = copy.deepcopy(native_model)
    before = {name: parameter.detach().clone() for name, parameter in native_model.named_parameters()}
    native_values = features.clone().requires_grad_()
    reference_values = features.clone().requires_grad_()
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    native_predictions, native_pooled = native_model(native_values, offsets, ragged_pool)
    reference_predictions, reference_pooled = reference_model(reference_values, offsets, ragged_pool_reference)
    native_loss = nn.functional.mse_loss(native_predictions, targets)
    reference_loss = nn.functional.mse_loss(reference_predictions, targets)
    torch.testing.assert_close(native_pooled, reference_pooled, **tolerance)
    torch.testing.assert_close(native_predictions, reference_predictions, **tolerance)
    torch.testing.assert_close(native_loss, reference_loss, **tolerance)
    native_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(native_values.grad, reference_values.grad, **tolerance)
    gradient_differences = {}
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in native_model.named_parameters():
        reference_gradient = reference_parameters[name].grad
        if parameter.grad is None or reference_gradient is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or nonfinite gradient for {name}")
        torch.testing.assert_close(parameter.grad, reference_gradient, **tolerance)
        gradient_differences[name] = maximum_difference(parameter.grad, reference_gradient)
    if not torch.count_nonzero(native_model.scorer.weight.grad):
        raise AssertionError("The learned logits did not receive a gradient")
    native_optimizer.step()
    reference_optimizer.step()
    parameter_updates = {}
    update_differences = {}
    for name, parameter in native_model.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], **tolerance)
        parameter_updates[name] = maximum_difference(parameter, before[name])
        update_differences[name] = maximum_difference(parameter, reference_parameters[name])
    if parameter_updates["scorer.weight"] == 0 or parameter_updates["head.weight"] == 0:
        raise AssertionError("The optimizer must update both the scorer and the prediction head")
    return {
        "purpose": "Synthetic training integration, not a model quality or accuracy study",
        "seed": seed,
        "dtype": dtype_name,
        "device": "cpu",
        "torch_version": torch.__version__,
        "package_version": package.__version__,
        "package_path": package.__file__,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "group_lengths": lengths,
        "optimizer": "SGD",
        "optimizer_steps": 1,
        "native_loss": float(native_loss.detach()),
        "reference_loss": float(reference_loss.detach()),
        "pooled_max_abs_difference": maximum_difference(native_pooled, reference_pooled),
        "prediction_max_abs_difference": maximum_difference(native_predictions, reference_predictions),
        "input_gradient_max_abs_difference": maximum_difference(native_values.grad, reference_values.grad),
        "parameter_gradient_max_abs_differences": gradient_differences,
        "native_parameter_update_max_abs": parameter_updates,
        "updated_parameter_max_abs_differences": update_differences,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.seed, arguments.dtype), indent=2, allow_nan=False))
