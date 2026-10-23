"""Check one two-rank CPU DDP objective against complete global-batch training.

Run a copied script outside the checkout with the ordinary installed CPU wheel.
This consumer qualifies one Linux PyTorch 2.9.1+cpu route. It uses two owned
local Gloo workers and two momentum updates. It measures agreement, not training
quality, speed, fault tolerance or security.

The optional --accumulate protocol divides each update into two microbatches.
It checks the whole-window objective and observes gradient-bucket reductions
through the official averaging hook. Each global microbatch contains a group,
while a rank-local microbatch can contain none.
"""
import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile

import torch
from torch import nn


SEED = 1811
WORLD_SIZE = 2
DTYPES = {"float32": torch.float32, "float64": torch.float64}
BOUNDARIES = (((0, 2, 2), (0, 1, 3, 4)), ((0,), (0, 0, 1, 3, 6)))
ACCUM_BOUNDARIES = ((((0, 2, 2), (0, 1, 3, 4)),
                     ((0,), (0, 0, 1, 3, 6))),
                    (((0,), (0, 2)), ((0, 0, 1, 3, 5), (0,))))
PARAMETER_NAMES = ("encoder.weight", "encoder.bias", "scorer.weight",
                   "head.weight", "head.bias")
SGD_OPTIONS = {"lr": 0.03, "momentum": 0.8, "dampening": 0,
               "weight_decay": 0, "nesterov": False, "maximize": False,
               "foreach": False, "differentiable": False, "fused": False}
RECEIPT_LIMIT = 2 * 1024 * 1024
GROUP_TIMEOUT_SECONDS = 30
PARENT_TIMEOUT_SECONDS = 120


def primitive_pool(values, logits, boundaries):
    """Independent double-precision primitive with graph-connected empty results."""
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        selected_values = values[start:stop].to(torch.float64)
        selected_logits = logits[start:stop].to(torch.float64)
        if start == stop:
            row = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
        else:
            weights = torch.softmax(selected_logits, dim=0)
            row = (weights[:, None] * selected_values).sum(dim=0)
        rows.append(row.to(values.dtype))
    if not rows:
        return values[:0] + logits.sum() * 0
    return torch.stack(rows)


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, boundaries, route="native"):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        if route == "native":
            from ragged_pool import ragged_pool
            pooled = ragged_pool(values, logits, offsets)
        elif route == "primitive":
            pooled = primitive_pool(values, logits, boundaries)
        else:
            raise ValueError("Unknown pooling route")
        return self.head(pooled).squeeze(-1), pooled


def fixture(dtype_name):
    """Return the fixed initialization, two rank-local batch pairs, and byte digest."""
    if dtype_name not in DTYPES:
        raise ValueError("Expected float32 or float64")
    dtype = DTYPES[dtype_name]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        initial = SetRegressor().to(dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 100)
    batches, digest = [], sha256()
    for step_boundaries in BOUNDARIES:
        ranks = []
        for boundaries in step_boundaries:
            groups = len(boundaries) - 1
            raw = 0.25 * torch.randn(boundaries[-1], 3, dtype=dtype, generator=generator)
            targets = (0.4 * torch.randn(groups, dtype=dtype, generator=generator)
                       + torch.linspace(-0.3, 0.5, groups, dtype=dtype))
            batch = {"raw": raw, "targets": targets,
                     "offsets": torch.tensor(boundaries, dtype=torch.int64),
                     "boundaries": boundaries}
            for name in ("raw", "targets", "offsets"):
                value = batch[name]
                digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
                digest.update(value.numpy().tobytes())
            ranks.append(batch)
        batches.append(tuple(ranks))
    return initial, batches, digest.hexdigest()


def rank_batch(batches, step, rank):
    if type(step) is not int or step not in (0, 1):
        raise ValueError("Expected update index zero or one")
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Expected rank zero or one")
    return batches[step][rank]


def global_batch(batches, step):
    first, second = (rank_batch(batches, step, rank) for rank in range(WORLD_SIZE))
    boundaries = first["boundaries"] + tuple(
        first["boundaries"][-1] + value for value in second["boundaries"][1:])
    return {"raw": torch.cat((first["raw"], second["raw"])),
            "targets": torch.cat((first["targets"], second["targets"])),
            "offsets": torch.tensor(boundaries, dtype=torch.int64),
            "boundaries": boundaries}


def normalized_local_loss(local_sse, global_groups, world_size=WORLD_SIZE):
    """Compensate DDP's parameter-gradient mean for unequal local group counts."""
    if not isinstance(local_sse, torch.Tensor):
        raise TypeError("The local squared-error sum must be a tensor")
    if (local_sse.ndim != 0 or local_sse.device.type != "cpu"
            or local_sse.dtype not in DTYPES.values()
            or not bool(torch.isfinite(local_sse)) or float(local_sse.detach()) < 0):
        raise ValueError("Expected one finite nonnegative CPU float32/float64 sum")
    for value in (global_groups, world_size):
        if type(value) is not int or value <= 0:
            raise ValueError("Group count and world size must be positive integers")
    return local_sse * (world_size / global_groups)


def snapshot(tensors):
    return {name: {"bytes": value.detach().clone(), "version": value._version,
                   "pointer": value.data_ptr(), "shape": value.shape,
                   "stride": value.stride(), "device": value.device,
                   "dtype": value.dtype}
            for name, value in tensors.items()}


def unchanged(tensors, before):
    if tensors.keys() != before.keys():
        raise AssertionError("Saved input keys changed")
    for name, value in tensors.items():
        saved = before[name]
        torch.testing.assert_close(value, saved["bytes"], rtol=0, atol=0)
        if (value._version != saved["version"] or value.data_ptr() != saved["pointer"]
                or value.shape != saved["shape"] or value.stride() != saved["stride"]
                or value.device != saved["device"] or value.dtype != saved["dtype"]):
            raise AssertionError(f"Saved input ownership changed: {name}")


def copied(tensors):
    return {name: value.detach().clone() for name, value in tensors.items()}


def maximum_difference(first, second):
    return float((first - second).abs().max()) if first.numel() else 0.0


def compare(first, second, tolerance):
    if first.keys() != second.keys():
        raise AssertionError("Compared tensor keys differ")
    errors = {}
    for name in first:
        if not bool(torch.isfinite(first[name]).all()) or not bool(torch.isfinite(second[name]).all()):
            raise AssertionError("Compared tensors must be finite")
        torch.testing.assert_close(first[name], second[name], **tolerance)
        errors[name] = maximum_difference(first[name], second[name])
    return errors


def _parameter_bindings(model, optimizer):
    unwrapped = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    named = dict(unwrapped.named_parameters())
    if tuple(named) != PARAMETER_NAMES or sum(value.numel() for value in named.values()) != 25:
        raise AssertionError("Expected the declared five-block, 25-scalar model")
    if len(optimizer.param_groups) != 1:
        raise AssertionError("Expected one optimizer parameter group")
    if [id(value) for value in optimizer.param_groups[0]["params"]] != [id(value) for value in named.values()]:
        raise AssertionError("Optimizer bindings differ from the model")
    for name, value in named.items():
        if value.device.type != "cpu" or not value.is_contiguous():
            raise AssertionError(f"Expected contiguous CPU parameters: {name}")
    return named


def _step(model, optimizer, batch, route, loss_mode, distributed=False):
    named = _parameter_bindings(model, optimizer)
    bindings = {name: (id(value), value.data_ptr()) for name, value in named.items()}
    before_parameters = copied(named)
    before_momentum = {name: optimizer.state[value].get("momentum_buffer", torch.zeros_like(value)).detach().clone()
                       for name, value in named.items()}
    optimizer.zero_grad(set_to_none=True)
    if any(value.grad is not None for value in named.values()):
        raise AssertionError("Parameter gradients accumulated between updates")
    raw = batch["raw"].detach().clone().requires_grad_(True)
    inputs = {"raw": raw, "targets": batch["targets"], "offsets": batch["offsets"]}
    input_before = snapshot(inputs)
    predictions, pooled = model(raw, batch["offsets"], batch["boundaries"], route)
    local_sse = (predictions - batch["targets"]).square().sum()
    local_groups = batch["targets"].numel()
    global_groups = local_groups
    metric_sum = local_sse.detach().clone()
    if distributed:
        import torch.distributed as dist
        count = torch.tensor(local_groups, dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_groups = int(count)
        dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
    if loss_mode == "global_group_mean":
        loss = normalized_local_loss(local_sse, global_groups,
                                     WORLD_SIZE if distributed else 1)
    elif loss_mode == "local_mean_control" and distributed:
        loss = local_sse / local_groups if local_groups else local_sse * 0
    else:
        raise ValueError("Unknown objective mode")
    loss.backward()
    if raw.grad is None or any(value.grad is None for value in named.values()):
        raise AssertionError("Every input and parameter block must remain connected")
    gradients = {name: value.grad.detach().clone() for name, value in named.items()}
    raw_gradient = raw.grad.detach().clone()
    optimizer.step()
    after_parameters = copied(named)
    momentum = {name: optimizer.state[value]["momentum_buffer"].detach().clone()
                for name, value in named.items()}
    for name, value in named.items():
        expected_momentum = before_momentum[name] * SGD_OPTIONS["momentum"] + gradients[name]
        expected_parameter = before_parameters[name].clone()
        expected_parameter.add_(expected_momentum, alpha=-SGD_OPTIONS["lr"])
        torch.testing.assert_close(momentum[name], expected_momentum, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_parameter, rtol=0, atol=0)
        if (id(value), value.data_ptr()) != bindings[name]:
            raise AssertionError("An optimizer step replaced a parameter binding")
        if torch.equal(value, before_parameters[name]):
            raise AssertionError(f"A global update did not change parameter block: {name}")
    unchanged(inputs, input_before)
    _parameter_bindings(model, optimizer)
    return {"predictions": predictions.detach().clone(), "pooled": pooled.detach().clone(),
            "raw_gradients": raw_gradient, "parameter_gradients": gradients,
            "parameters": after_parameters, "momentum": momentum,
            "local_groups": local_groups, "global_groups": global_groups,
            "local_sse": local_sse.detach().clone(), "metric": metric_sum / global_groups,
            "local_backward_loss": loss.detach().clone(),
            "input_unchanged": True, "bindings_unchanged": True,
            "sgd_recurrence_exact": True, "all_parameter_blocks_changed": True}


def accumulation_fixture(dtype_name):
    """Build two fixed accumulation windows, each with two microbatches per rank."""
    if dtype_name not in DTYPES:
        raise ValueError("Expected float32 or float64")
    dtype = DTYPES[dtype_name]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        initial = SetRegressor().to(dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 200)
    windows, digest = [], sha256()
    for window, micro_boundaries in enumerate(ACCUM_BOUNDARIES):
        micros = []
        for micro, rank_boundaries in enumerate(micro_boundaries):
            ranks = []
            for rank, boundaries in enumerate(rank_boundaries):
                groups = len(boundaries) - 1
                raw = 0.25 * torch.randn(boundaries[-1], 3, dtype=dtype,
                                         generator=generator)
                targets = (0.4 * torch.randn(groups, dtype=dtype, generator=generator)
                           + torch.linspace(-0.3, 0.5, groups, dtype=dtype))
                batch = {"raw": raw, "targets": targets,
                         "offsets": torch.tensor(boundaries, dtype=torch.int64),
                         "boundaries": boundaries}
                for name in ("raw", "targets", "offsets"):
                    value = batch[name]
                    digest.update(json.dumps([window, micro, rank, name,
                                              list(value.shape), str(value.dtype)]).encode())
                    digest.update(value.numpy().tobytes())
                ranks.append(batch)
            micros.append(tuple(ranks))
        windows.append(tuple(micros))
    return initial, windows, digest.hexdigest()


def accumulation_rank_batch(batches, window, micro, rank):
    for label, value in (("window", window), ("microbatch", micro), ("rank", rank)):
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"Expected {label} index zero or one")
    return batches[window][micro][rank]


def accumulation_global_batch(batches, window):
    """Pack whole groups in microbatch-major order, then rank order."""
    parts = [accumulation_rank_batch(batches, window, micro, rank)
             for micro in range(2) for rank in range(WORLD_SIZE)]
    boundaries, raw_count = [0], 0
    for batch in parts:
        boundaries.extend(raw_count + value for value in batch["boundaries"][1:])
        raw_count += batch["raw"].shape[0]
    return {"raw": torch.cat([batch["raw"] for batch in parts]),
            "targets": torch.cat([batch["targets"] for batch in parts]),
            "offsets": torch.tensor(boundaries, dtype=torch.int64),
            "boundaries": tuple(boundaries)}


def accumulation_global_counts(batches, window):
    counts = tuple(sum(accumulation_rank_batch(batches, window, micro, rank)
                       ["targets"].numel() for rank in range(WORLD_SIZE))
                   for micro in range(2))
    return sum(counts), counts


def accumulation_loss(local_sse, global_window_groups, global_micro_groups,
                      mode, world_size=WORLD_SIZE):
    """Apply the whole-window mean or the deliberately different microbatch mean."""
    if mode in ("global_window_mean", "eager_sync"):
        return normalized_local_loss(local_sse, global_window_groups, world_size)
    if mode == "microbatch_mean_control":
        return normalized_local_loss(local_sse, global_micro_groups, world_size) / 2
    raise ValueError("Unknown accumulation objective mode")


def _counted_allreduce(state, bucket):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    state["events"].append({"window": state["window"], "microbatch": state["microbatch"],
                            "scalars": bucket.buffer().numel(),
                            "dtype": str(bucket.buffer().dtype)})
    # The official hook supplies the mean. A custom hook's bucket is not already
    # divided by world size, so an unscaled SUM would change this objective.
    return default_hooks.allreduce_hook(state["group"], bucket)


def _accumulation_step(model, optimizer, microbatches, window=0,
                       mode="global_window_mean", hook_state=None, route="native",
                       distributed=False, global_groups=None, micro_global_groups=None):
    """Accumulate exactly two connected backwards, then take one momentum update."""
    if len(microbatches) != 2:
        raise ValueError("Expected exactly two microbatches in one update")
    if type(window) is not int or window not in (0, 1):
        raise ValueError("Expected accumulation window index zero or one")
    if mode not in ("global_window_mean", "eager_sync", "microbatch_mean_control"):
        raise ValueError("Unknown accumulation objective mode")
    named = _parameter_bindings(model, optimizer)
    bindings = {name: (id(value), value.data_ptr()) for name, value in named.items()}
    before_parameters = copied(named)
    before_momentum = {name: optimizer.state[value].get(
        "momentum_buffer", torch.zeros_like(value)).detach().clone()
        for name, value in named.items()}
    had_momentum = {name: "momentum_buffer" in optimizer.state[value]
                    for name, value in named.items()}
    counts = torch.tensor([batch["targets"].numel() for batch in microbatches],
                          dtype=torch.int64)
    if distributed:
        import torch.distributed as dist
        if hook_state is None:
            raise ValueError("Distributed accumulation requires its communication observer")
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    observed_micro_counts = tuple(int(value) for value in counts)
    observed_groups = sum(observed_micro_counts)
    if (global_groups is not None and global_groups != observed_groups
            or micro_global_groups is not None
            and tuple(micro_global_groups) != observed_micro_counts):
        raise AssertionError("Supplied accumulation counts differ from the batches")
    if any(value <= 0 for value in observed_micro_counts):
        raise ValueError("Each declared global microbatch must contain a group")
    optimizer.zero_grad(set_to_none=True)
    if any(value.grad is not None for value in named.values()):
        raise AssertionError("Gradients survived the start of an accumulation window")
    micro_records, metric_sum = [], torch.zeros((), dtype=next(iter(named.values())).dtype)
    first_micro_gradients = None
    for micro, batch in enumerate(microbatches):
        raw = batch["raw"].detach().clone().requires_grad_(True)
        inputs = {"raw": raw, "targets": batch["targets"], "offsets": batch["offsets"]}
        input_before = snapshot(inputs)
        if hook_state is not None:
            hook_state.update({"window": window, "microbatch": micro})
        calls_before = len(hook_state["events"]) if hook_state is not None else 0
        context = (model.no_sync() if distributed and micro == 0 and mode != "eager_sync"
                   else nullcontext())
        # DDP decides whether to prepare reduction during forward. Both calls
        # belong inside no_sync for the first microbatch, including an empty one.
        with context:
            predictions, pooled = model(raw, batch["offsets"], batch["boundaries"], route)
            local_sse = (predictions - batch["targets"]).square().sum()
            loss = accumulation_loss(local_sse, observed_groups, observed_micro_counts[micro],
                                     mode, WORLD_SIZE if distributed else 1)
            loss.backward()
        if raw.grad is None or any(value.grad is None for value in named.values()):
            raise AssertionError("Every microbatch must retain every input and parameter connection")
        gradients = {name: value.grad.detach().clone() for name, value in named.items()}
        if micro == 0:
            first_micro_gradients = copied(gradients)
        for name, value in named.items():
            torch.testing.assert_close(value, before_parameters[name], rtol=0, atol=0)
            if (id(value), value.data_ptr()) != bindings[name]:
                raise AssertionError("A microbatch changed a parameter binding")
            if ("momentum_buffer" in optimizer.state[value]) != had_momentum[name]:
                raise AssertionError("A microbatch changed optimizer state before its update")
            if had_momentum[name]:
                torch.testing.assert_close(optimizer.state[value]["momentum_buffer"],
                                           before_momentum[name], rtol=0, atol=0)
        unchanged(inputs, input_before)
        calls_after = len(hook_state["events"]) if hook_state is not None else 0
        delta = calls_after - calls_before
        if distributed:
            expected_calls = 1 if micro == 1 or mode == "eager_sync" else 0
            if delta != expected_calls:
                raise AssertionError("The fixed 25-scalar model used an unexpected gradient-bucket reduction count")
            for event in hook_state["events"][calls_before:calls_after]:
                if event["scalars"] != 25 or event["dtype"] != str(raw.dtype):
                    raise AssertionError("The observed gradient bucket differs from this fixed model")
        metric_sum += local_sse.detach()
        micro_records.append({"predictions": predictions.detach().clone(),
                              "pooled": pooled.detach().clone(),
                              "raw_gradients": raw.grad.detach().clone(),
                              "parameter_gradients_after_backward": gradients,
                              "local_sse": local_sse.detach().clone(),
                              "local_backward_loss": loss.detach().clone(),
                              "local_groups": batch["targets"].numel(),
                              "global_groups": observed_micro_counts[micro],
                              "hook_calls": delta, "input_unchanged": True,
                              "parameters_and_momentum_unchanged_before_step": True})
    gradients = {name: value.grad.detach().clone() for name, value in named.items()}
    if distributed:
        dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
    optimizer.step()
    after_parameters = copied(named)
    momentum = {name: optimizer.state[value]["momentum_buffer"].detach().clone()
                for name, value in named.items()}
    for name, value in named.items():
        expected_momentum = before_momentum[name] * SGD_OPTIONS["momentum"] + gradients[name]
        expected_parameter = before_parameters[name].clone()
        expected_parameter.add_(expected_momentum, alpha=-SGD_OPTIONS["lr"])
        torch.testing.assert_close(momentum[name], expected_momentum, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_parameter, rtol=0, atol=0)
        if (id(value), value.data_ptr()) != bindings[name]:
            raise AssertionError("An optimizer step replaced a parameter binding")
        if torch.equal(value, before_parameters[name]):
            raise AssertionError(f"A window update did not change parameter block: {name}")
    _parameter_bindings(model, optimizer)
    return {"microbatches": micro_records,
            "first_micro_parameter_gradients": first_micro_gradients,
            "parameter_gradients": gradients, "parameters": after_parameters,
            "momentum": momentum, "global_groups": observed_groups,
            "global_micro_groups": observed_micro_counts,
            "local_groups": sum(batch["targets"].numel() for batch in microbatches),
            "metric": metric_sum / observed_groups,
            "optimizer_steps": 1, "zero_grad_calls": 1,
            "bindings_unchanged": True, "sgd_recurrence_exact": True,
            "all_parameter_blocks_changed": True}


def _accumulation_oracle(initial, batches, route):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), **SGD_OPTIONS)
    steps, first_micro = [], []
    for window in range(2):
        global_groups, _ = accumulation_global_counts(batches, window)
        contributions = []
        for rank in range(WORLD_SIZE):
            local = copy.deepcopy(model)
            batch = accumulation_rank_batch(batches, window, 0, rank)
            raw = batch["raw"].detach().clone().requires_grad_(True)
            predictions, pooled = local(raw, batch["offsets"], batch["boundaries"], "primitive")
            local_sse = (predictions - batch["targets"]).square().sum()
            # This is the local contribution before any DDP reduction.
            (local_sse * (WORLD_SIZE / global_groups)).backward()
            if raw.grad is None or any(value.grad is None for value in local.parameters()):
                raise AssertionError("The independent first microbatch lost a gradient connection")
            contributions.append({"parameter_gradients": {
                name: value.grad.detach().clone() for name, value in local.named_parameters()},
                "raw_gradients": raw.grad.detach().clone(),
                "predictions": predictions.detach().clone(), "pooled": pooled.detach().clone()})
        first_micro.append(contributions)
        steps.append(_step(model, optimizer, accumulation_global_batch(batches, window), route,
                           "global_group_mean"))
    return steps, first_micro


def _compare_accumulation(receipts, native, primitive, first_micro, dtype_name, batches):
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    exact = {"rtol": 0, "atol": 0}
    modes = ("global_window_mean", "eager_sync", "microbatch_mean_control")
    rank_errors, oracle_errors, worker_errors, local_errors, eager_errors = [], [], [], [], []
    for window in range(2):
        rank_record, oracle_record = {}, {}
        for mode in modes:
            rank_record[mode] = {field: compare(receipts[0]["routes"][mode][window][field],
                                                receipts[1]["routes"][mode][window][field], exact)
                                 for field in ("parameter_gradients", "parameters", "momentum")}
        for field in ("parameter_gradients", "parameters", "momentum"):
            oracle_record[field] = compare(native[window][field], primitive[window][field], tolerance)
        oracle_record["outputs"] = compare(
            {name: native[window][name] for name in ("predictions", "pooled", "raw_gradients", "metric")},
            {name: primitive[window][name] for name in ("predictions", "pooled", "raw_gradients", "metric")},
            tolerance)
        rank_errors.append(rank_record)
        oracle_errors.append(oracle_record)
        for rank in range(WORLD_SIZE):
            correct = receipts[rank]["routes"]["global_window_mean"][window]
            eager = receipts[rank]["routes"]["eager_sync"][window]
            eager_record = {"window": window, "rank": rank}
            for field in ("parameter_gradients", "parameters", "momentum"):
                eager_record[field] = compare(correct[field], eager[field], tolerance)
            eager_record["microbatches"] = [compare(
                {name: correct["microbatches"][micro][name]
                 for name in ("predictions", "pooled", "raw_gradients")},
                {name: eager["microbatches"][micro][name]
                 for name in ("predictions", "pooled", "raw_gradients")}, tolerance)
                for micro in range(2)]
            eager_errors.append(eager_record)
            local_errors.append({"window": window, "rank": rank,
                                 "parameter_gradients": compare(
                                     correct["first_micro_parameter_gradients"],
                                     first_micro[window][rank]["parameter_gradients"], tolerance),
                                 "outputs": compare({name: correct["microbatches"][0][name]
                                                     for name in ("raw_gradients", "predictions", "pooled")},
                                                    {name: first_micro[window][rank][name]
                                                     for name in ("raw_gradients", "predictions", "pooled")},
                                                    tolerance)})
        raw_start = group_start = 0
        for micro in range(2):
            for rank in range(WORLD_SIZE):
                batch = accumulation_rank_batch(batches, window, micro, rank)
                worker = receipts[rank]["routes"]["global_window_mean"][window]
                observed = worker["microbatches"][micro]
                raw_stop = raw_start + batch["raw"].shape[0]
                group_stop = group_start + batch["targets"].shape[0]
                if (observed["local_groups"] != batch["targets"].numel()
                        or worker["global_groups"] != native[window]["global_groups"]):
                    raise AssertionError("Accumulated worker group counts differ from the fixture")
                record = {"window": window, "microbatch": micro, "rank": rank}
                for label, oracle in (("native", native[window]), ("primitive", primitive[window])):
                    expected = {"predictions": oracle["predictions"][group_start:group_stop],
                                "pooled": oracle["pooled"][group_start:group_stop],
                                "raw_gradients": oracle["raw_gradients"][raw_start:raw_stop] * WORLD_SIZE}
                    record[label] = {"outputs": compare({name: observed[name] for name in expected},
                                                          expected, tolerance),
                                     "metric": compare({"metric": worker["metric"]},
                                                       {"metric": oracle["metric"]}, tolerance)}
                    for field in ("parameter_gradients", "parameters", "momentum"):
                        record[label][field] = compare(worker[field], oracle[field], tolerance)
                worker_errors.append(record)
                raw_start, group_start = raw_stop, group_stop
    controls = []
    threshold = 1e-8 if dtype_name == "float32" else 1e-11
    for rank, receipt in enumerate(receipts):
        correct = receipt["routes"]["global_window_mean"]
        wrong = receipt["routes"]["microbatch_mean_control"]
        first_forward = [compare({name: correct[0]["microbatches"][micro][name]
                                  for name in ("predictions", "pooled")},
                                 {name: wrong[0]["microbatches"][micro][name]
                                  for name in ("predictions", "pooled")}, exact)
                         for micro in range(2)]
        drift = {field: [max(maximum_difference(correct[window][field][name],
                                                wrong[window][field][name])
                              for name in PARAMETER_NAMES) for window in range(2)]
                 for field in ("parameter_gradients", "parameters", "momentum")}
        if not all(value > threshold for values in drift.values() for value in values):
            raise AssertionError("The microbatch-mean control did not change the optimizer trajectory")
        controls.append({"rank": rank, "first_window_forward_exact": first_forward,
                         "max_absolute_drift": drift, "required_drift_greater_than": threshold})
    return {"tolerance": tolerance, "raw_input_gradient_scale": WORLD_SIZE,
            "rank_equality": rank_errors, "native_vs_primitive": oracle_errors,
            "worker_vs_oracle": worker_errors, "first_micro_vs_local_primitive": local_errors,
            "eager_sync_vs_accumulation": eager_errors, "microbatch_mean_controls": controls}


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "values": value.tolist()}
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _tensor_value(value):
    if isinstance(value, dict) and set(value) == {"shape", "dtype", "values"}:
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}[value["dtype"]]
        return torch.tensor(value["values"], dtype=dtype).reshape(value["shape"])
    if isinstance(value, dict):
        return {name: _tensor_value(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_tensor_value(item) for item in value]
    return value


def _write_json(path, result):
    payload = (json.dumps(_json_value(result), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > RECEIPT_LIMIT:
        raise ValueError("The fixed consumer receipt exceeded its size bound")
    with Path(path).open("xb") as stream:
        stream.write(payload)


def _read_json(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & 0x400 or info.st_size > RECEIPT_LIMIT):
        raise ValueError("Expected a bounded ordinary local JSON receipt")
    payload = path.read_bytes()
    if len(payload) > RECEIPT_LIMIT:
        raise ValueError("The receipt grew beyond its bound")
    return _tensor_value(json.loads(payload))


def installed_identity(wheel=None):
    import ragged_pool
    installed = distribution("ragged-pool")
    imported = Path(ragged_pool.__file__).resolve(strict=True)
    prefix = Path(sys.prefix).resolve()
    direct = installed.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed CPU wheel")
    if (str(torch.__version__) != "2.9.1+cpu" or torch.version.cuda is not None
            or imported != Path(installed.locate_file("ragged_pool/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != ragged_pool.__version__):
        raise AssertionError("Expected the pinned ordinary installed PyTorch 2.9.1+cpu route")
    config_path = imported.parent / "_build_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from ragged_pool._build import validate_runtime
    validate_runtime(config, str(torch.__version__), torch.version.cuda)
    if config["variant"] != "cpu":
        raise AssertionError("Expected a CPU wheel build configuration")
    libraries = tuple(imported.parent.glob("_C*.pyd")) + tuple(imported.parent.glob("_C*.so"))
    if len(libraries) != 1:
        raise AssertionError("Expected exactly one native CPU library")
    native = libraries[0].resolve(strict=True)
    if native not in {Path(path).resolve() for path in torch.ops.loaded_libraries}:
        raise AssertionError("The installed native CPU library is not loaded")
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "package_version": installed.version, "package_path": str(imported),
              "native_library_path": str(native), "native_library_sha256": sha256(native.read_bytes()).hexdigest(),
              "build_config": config, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "intraop_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads()}
    if wheel is not None:
        wheel = Path(wheel).resolve(strict=True)
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise AssertionError("Wheel members must be unique")
            for name in names:
                if name.startswith("ragged_pool/") and not name.endswith("/"):
                    actual = Path(installed.locate_file(name)).resolve(strict=True)
                    if not actual.is_relative_to(prefix) or actual.read_bytes() != archive.read(name):
                        raise AssertionError(f"Installed package bytes differ from wheel: {name}")
        result.update({"wheel_path": str(wheel), "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
                       "wheel_filename": wheel.name})
    return result


def _worker(rank, dtype_name, directory, rendezvous, wheel, fail_rank, accumulate=False):
    # These are fresh interpreters. No shared parent tensors or work queues are used.
    log = Path(directory) / f"worker-{rank}.log"
    receipt = Path(directory) / f"worker-{rank}.json"
    initialized = False
    result = {"status": "error", "rank": rank, "pid": os.getpid(), "parent_pid": os.getppid()}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log.open("x", encoding="utf-8", buffering=1) as stream:
        sys.stdout = stream
        sys.stderr = stream
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            if rank == fail_rank:
                raise RuntimeError("Deliberate owned-worker failure before Gloo initialization")
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel
            if not dist.is_available() or not dist.is_gloo_available():
                raise RuntimeError("This pinned environment does not provide Gloo")
            identity = installed_identity(wheel)

            def loopback_gloo(store, process_rank, size, timeout):
                # register_backend is public. These Gloo options are pinned 2.9.1
                # bindings, used explicitly to avoid an unqualified default NIC.
                options = dist.ProcessGroupGloo._Options()
                options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1", lazy_init=False)]
                options._threads = 1
                options._timeout = timeout
                return dist.ProcessGroupGloo(store, process_rank, size, options)

            backend = "ragged_gloo_loopback"
            dist.Backend.register_backend(backend, loopback_gloo, devices=["cpu"])
            dist.init_process_group(backend, init_method=Path(rendezvous).resolve().as_uri(),
                                    rank=rank, world_size=WORLD_SIZE,
                                    timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS))
            initialized = True
            make_fixture = accumulation_fixture if accumulate else fixture
            initial, batches, fixture_digest = make_fixture(dtype_name)
            template_before = snapshot(dict(initial.named_parameters()))
            if accumulate:
                fixture_inputs = _accumulation_fixture_inputs(batches)
            else:
                fixture_inputs = {f"{step}-{peer}-{name}": value
                                  for step, ranks in enumerate(batches)
                                  for peer, batch in enumerate(ranks)
                                  for name, value in batch.items() if isinstance(value, torch.Tensor)}
            fixture_before = snapshot(fixture_inputs)
            routes, communication = {}, {}
            modes = (("global_window_mean", "eager_sync", "microbatch_mean_control") if accumulate
                     else ("global_group_mean", "local_mean_control"))
            for mode in modes:
                model = copy.deepcopy(initial)
                ddp = DistributedDataParallel(model, device_ids=None,
                                              find_unused_parameters=False)
                optimizer = torch.optim.SGD(model.parameters(), **SGD_OPTIONS)
                if accumulate:
                    hook_state = {"group": dist.group.WORLD, "events": [],
                                  "window": None, "microbatch": None}
                    ddp.register_comm_hook(hook_state, _counted_allreduce)
                    routes[mode] = [_accumulation_step(
                        ddp, optimizer,
                        [accumulation_rank_batch(batches, window, micro, rank) for micro in range(2)],
                        window=window, mode=mode, hook_state=hook_state, distributed=True)
                        for window in range(2)]
                    communication[mode] = hook_state["events"]
                else:
                    routes[mode] = [_step(ddp, optimizer, rank_batch(batches, step, rank),
                                         "native", mode, distributed=True) for step in range(2)]
                del ddp, optimizer, model
            unchanged(dict(initial.named_parameters()), template_before)
            unchanged(fixture_inputs, fixture_before)
            result.update({"status": "passed", "dtype": dtype_name, "world_size": WORLD_SIZE,
                           "backend": dist.get_backend(), "identity": identity,
                           "gloo_device_hostname": "127.0.0.1", "gloo_threads": 1,
                           "backend_options_scope": "PyTorch 2.9.1 ProcessGroupGloo._Options bindings",
                           "fixture_sha256": fixture_digest, "routes": routes,
                           "template_unchanged": True, "fixtures_unchanged": True})
            if accumulate:
                result.update({"accumulation": True, "communication_hooks": communication,
                               "communication_scope": "official averaging hook calls for the fixed 25-scalar bucket"})
        except BaseException as error:
            traceback.print_exc()
            result.update({"error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            try:
                if initialized:
                    dist.destroy_process_group()
                    result["process_group_destroyed"] = True
            finally:
                try:
                    _write_json(receipt, result)
                finally:
                    sys.stdout, sys.stderr = original_stdout, original_stderr


def _cleanup_owned(processes):
    actions = []
    for process in processes:
        if process.poll() is None:
            process.terminate()
            actions.append({"pid": process.pid, "action": "terminate"})
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        if process.poll() is None:
            process.kill()
            actions.append({"pid": process.pid, "action": "kill"})
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    return actions


def _launch_workers(dtype_name, directory, wheel, fail_rank, accumulate=False):
    rendezvous = directory / "gloo-rendezvous"
    if rendezvous.exists():
        raise ValueError("The task-owned rendezvous path must not exist")
    executable = sys.executable
    environment = dict(os.environ, GLOO_DEVICE_TRANSPORT="TCP", TORCH_GLOO_LAZY_INIT="0")
    creationflags = 0
    launch_route = "direct current interpreter"
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        if sys.prefix != sys.base_prefix:
            # CPython 3.11's venv redirector would create an additional process.
            # Match its multiprocessing direct-base routing while setting explicit
            # hidden-process flags. The child receipt verifies actual PID/prefix.
            executable = sys._base_executable
            environment["__PYVENV_LAUNCHER__"] = sys.executable
            launch_route = "direct base interpreter with current venv __PYVENV_LAUNCHER__"
    failure = None
    cleanup = []
    started = []
    try:
        for rank in range(WORLD_SIZE):
            arguments = (rank, dtype_name, str(directory), str(rendezvous), wheel, fail_rank, accumulate)
            code = ("import runpy; runpy.run_path(" + repr(str(Path(__file__).resolve()))
                    + ")['_worker'](*" + repr(arguments) + ")")
            with (directory / f"bootstrap-{rank}.log").open("xb") as stream:
                process = subprocess.Popen([executable, "-I", "-B", "-c", code],
                                           stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, env=environment,
                                           shell=False, creationflags=creationflags)
            started.append(process)
        deadline = time.monotonic() + PARENT_TIMEOUT_SECONDS
        while True:
            for process in started:
                exitcode = process.poll()
                if exitcode is not None and exitcode != 0:
                    failure = f"Owned worker {process.pid} exited with code {exitcode}"
                    break
            if failure or all(process.poll() is not None for process in started):
                break
            if time.monotonic() >= deadline:
                failure = "Owned workers exceeded the finite parent deadline"
                break
            time.sleep(0.05)
    except BaseException as error:
        failure = f"Parent launch or wait failed: {type(error).__name__}: {error}"
    finally:
        if failure:
            cleanup = _cleanup_owned(started)
        else:
            for process in started:
                process.wait(timeout=1)
    owned = [{"rank": rank, "pid": process.pid, "exitcode": process.poll(),
              "alive": process.poll() is None}
             for rank, process in enumerate(started)]
    lifecycle = {"owned_processes": owned, "cleanup_actions": cleanup,
                 "all_owned_workers_reaped": len(owned) == WORLD_SIZE
                 and all(not process["alive"] and process["exitcode"] is not None for process in owned),
                 "parent_deadline_seconds": PARENT_TIMEOUT_SECONDS,
                 "process_group_timeout_seconds": GROUP_TIMEOUT_SECONDS,
                 "launch_route": launch_route, "launcher_python": platform.python_version(),
                 "creationflags": creationflags, "shell": False,
                 "windows_hidden_and_below_normal": os.name == "nt",
                 "failure": failure}
    if failure or not lifecycle["all_owned_workers_reaped"]:
        return lifecycle, None
    receipts = [_read_json(directory / f"worker-{rank}.json") for rank in range(WORLD_SIZE)]
    for rank, result in enumerate(receipts):
        if (result["status"] != "passed" or result["rank"] != rank
                or result["pid"] != started[rank].pid or result["parent_pid"] != os.getpid()
                or not result["process_group_destroyed"]):
            raise AssertionError("Worker receipt does not match its directly owned process")
    return lifecycle, receipts


def _oracle(initial, batches, route):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), **SGD_OPTIONS)
    return [_step(model, optimizer, global_batch(batches, step), route,
                  "global_group_mean") for step in range(2)]


def _compare_results(receipts, native, primitive, dtype_name, batches):
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    exact = {"rtol": 0, "atol": 0}
    errors, rank_errors, oracle_errors = [], [], []
    for step in range(2):
        workers = [receipt["routes"]["global_group_mean"][step] for receipt in receipts]
        ranks, checks = {}, {}
        for field in ("parameter_gradients", "parameters", "momentum"):
            ranks[field] = compare(workers[0][field], workers[1][field], exact)
            checks[field] = compare(native[step][field], primitive[step][field], tolerance)
        checks["outputs"] = compare({name: native[step][name] for name in
                                     ("predictions", "pooled", "raw_gradients", "metric")},
                                    {name: primitive[step][name] for name in
                                     ("predictions", "pooled", "raw_gradients", "metric")}, tolerance)
        rank_errors.append(ranks)
        oracle_errors.append(checks)
        raw_start = group_start = 0
        for rank in range(WORLD_SIZE):
            worker = workers[rank]
            batch = rank_batch(batches, step, rank)
            raw_stop = raw_start + batch["raw"].shape[0]
            group_stop = group_start + batch["targets"].shape[0]
            if (worker["local_groups"] != len(batch["boundaries"]) - 1
                    or worker["global_groups"] != native[step]["global_groups"]):
                raise AssertionError("Worker group count differs from the fixed objective")
            record = {"step": step, "rank": rank}
            for label, oracle in (("native", native[step]), ("primitive", primitive[step])):
                expected = {"predictions": oracle["predictions"][group_start:group_stop],
                            "pooled": oracle["pooled"][group_start:group_stop],
                            "raw_gradients": oracle["raw_gradients"][raw_start:raw_stop] * WORLD_SIZE,
                            "metric": oracle["metric"]}
                record[label] = {"outputs": compare({name: worker[name] for name in expected},
                                                      expected, tolerance)}
                for field in ("parameter_gradients", "parameters", "momentum"):
                    record[label][field] = compare(worker[field], oracle[field], tolerance)
            errors.append(record)
            raw_start, group_start = raw_stop, group_stop
    controls = []
    threshold = 1e-8 if dtype_name == "float32" else 1e-11
    for rank, receipt in enumerate(receipts):
        correct = receipt["routes"]["global_group_mean"]
        control = receipt["routes"]["local_mean_control"]
        same_forward = compare({name: correct[0][name] for name in ("predictions", "pooled")},
                               {name: control[0][name] for name in ("predictions", "pooled")}, exact)
        drift = {}
        for field in ("parameter_gradients", "parameters", "momentum"):
            drift[field] = [max(maximum_difference(correct[step][field][name], control[step][field][name])
                               for name in PARAMETER_NAMES) for step in range(2)]
            if not all(value > threshold for value in drift[field]):
                raise AssertionError("The local-mean control did not expose a consequential objective error")
        controls.append({"rank": rank, "first_forward_exact": same_forward,
                         "max_absolute_drift": drift, "required_drift_greater_than": threshold})
    for step in range(2):
        for field in ("parameter_gradients", "parameters", "momentum"):
            compare(receipts[0]["routes"]["local_mean_control"][step][field],
                    receipts[1]["routes"]["local_mean_control"][step][field], exact)
    return {"tolerance": tolerance, "raw_input_gradient_scale": WORLD_SIZE,
            "worker_vs_oracle": errors, "rank_equality": rank_errors,
            "native_vs_primitive": oracle_errors, "local_mean_controls": controls}


def run(dtype_name, directory, wheel=None, fail_rank=None):
    if sys.platform != "linux":
        raise RuntimeError(
            "The two-rank Gloo consumer is qualified on Linux only; the tested "
            "Windows PyTorch 2.9.1+cpu wheel did not provide a usable loopback transport")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[name] = "1"
    lifecycle, receipts = _launch_workers(dtype_name, directory, wheel, fail_rank)
    if receipts is None:
        result = {"status": "failed", "dtype": dtype_name, "lifecycle": lifecycle,
                  "deliberate_failure_rank": fail_rank}
        _write_json(directory / "parent.json", result)
        return result
    # No numerical oracle runs while either owned worker is alive.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identity = installed_identity(wheel)
    initial, batches, fixture_digest = fixture(dtype_name)
    template_before = snapshot(dict(initial.named_parameters()))
    fixture_inputs = {f"{step}-{rank}-{name}": value
                      for step, ranks in enumerate(batches)
                      for rank, batch in enumerate(ranks)
                      for name, value in batch.items() if isinstance(value, torch.Tensor)}
    fixture_before = snapshot(fixture_inputs)
    native = _oracle(initial, batches, "native")
    primitive = _oracle(initial, batches, "primitive")
    unchanged(dict(initial.named_parameters()), template_before)
    unchanged(fixture_inputs, fixture_before)
    for receipt in receipts:
        if receipt["fixture_sha256"] != fixture_digest or receipt["identity"] != identity:
            raise AssertionError("Workers and parent used different fixtures or installed bytes")
    comparison = _compare_results(receipts, native, primitive, dtype_name, batches)
    result = {"status": "passed", "dtype": dtype_name, "seed": SEED,
              "world_size": WORLD_SIZE, "backend": "ragged_gloo_loopback", "device": "cpu",
              "transport_scope": "single host; Gloo device hostname 127.0.0.1; file rendezvous",
              "gloo_threads_per_worker": 1,
              "backend_options_scope": "public custom-backend registry; pinned PyTorch 2.9.1 Gloo _Options bindings",
              "global_group_counts": [5, 4], "local_group_counts": [[2, 3], [0, 4]],
              "parameter_names": PARAMETER_NAMES, "parameter_scalar_count": 25,
              "sgd_options": SGD_OPTIONS, "identity": identity,
              "fixture_sha256": fixture_digest, "lifecycle": lifecycle,
              "comparison": comparison, "workers": receipts,
              "oracles": {"native": native, "primitive": primitive},
              "template_unchanged": True, "fixtures_unchanged": True}
    _write_json(directory / "parent.json", result)
    return result


def _accumulation_fixture_inputs(batches):
    return {f"{window}-{micro}-{rank}-{name}": value
            for window, micros in enumerate(batches)
            for micro, ranks in enumerate(micros)
            for rank, batch in enumerate(ranks)
            for name, value in batch.items() if isinstance(value, torch.Tensor)}


def run_accumulation(dtype_name, directory, wheel=None, fail_rank=None):
    if sys.platform != "linux":
        raise RuntimeError(
            "The two-rank Gloo consumer is qualified on Linux only; the tested "
            "Windows PyTorch 2.9.1+cpu wheel did not provide a usable loopback transport")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[name] = "1"
    lifecycle, receipts = _launch_workers(dtype_name, directory, wheel, fail_rank, accumulate=True)
    if receipts is None:
        result = {"status": "failed", "dtype": dtype_name, "accumulation": True,
                  "lifecycle": lifecycle, "deliberate_failure_rank": fail_rank}
        _write_json(directory / "parent.json", result)
        return result
    # The complete global and local primitive oracles run after both workers exit.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identity = installed_identity(wheel)
    initial, batches, fixture_digest = accumulation_fixture(dtype_name)
    template_before = snapshot(dict(initial.named_parameters()))
    fixture_inputs = _accumulation_fixture_inputs(batches)
    fixture_before = snapshot(fixture_inputs)
    native, native_first_micro = _accumulation_oracle(initial, batches, "native")
    primitive, primitive_first_micro = _accumulation_oracle(initial, batches, "primitive")
    unchanged(dict(initial.named_parameters()), template_before)
    unchanged(fixture_inputs, fixture_before)
    for receipt in receipts:
        if (receipt["fixture_sha256"] != fixture_digest or receipt["identity"] != identity
                or receipt.get("accumulation") is not True):
            raise AssertionError("Accumulation workers and parent used different fixtures or installed bytes")
    comparison = _compare_accumulation(receipts, native, primitive, primitive_first_micro,
                                       dtype_name, batches)
    result = {"status": "passed", "dtype": dtype_name, "seed": SEED,
              "accumulation": True, "world_size": WORLD_SIZE, "device": "cpu",
              "backend": "ragged_gloo_loopback", "windows": 2, "microbatches_per_window": 2,
              "global_group_counts": [accumulation_global_counts(batches, window)[0]
                                      for window in range(2)],
              "global_micro_group_counts": [accumulation_global_counts(batches, window)[1]
                                            for window in range(2)],
              "local_micro_group_counts": [[[accumulation_rank_batch(batches, window, micro, rank)
                                               ["targets"].numel() for rank in range(WORLD_SIZE)]
                                             for micro in range(2)] for window in range(2)],
              "transport_scope": "single host; Gloo device hostname 127.0.0.1; file rendezvous",
              "gloo_threads_per_worker": 1,
              "backend_options_scope": "public custom-backend registry; pinned PyTorch 2.9.1 Gloo _Options bindings",
              "communication_scope": "official averaging hook calls for one fixed 25-scalar bucket; excludes count and metric collectives",
              "parameter_names": PARAMETER_NAMES, "parameter_scalar_count": 25,
              "sgd_options": SGD_OPTIONS, "identity": identity, "fixture_sha256": fixture_digest,
              "lifecycle": lifecycle, "comparison": comparison, "workers": receipts,
              "oracles": {"native": native, "primitive": primitive,
                          "local_first_micro_at_native_state": native_first_micro,
                          "local_first_micro_at_primitive_state": primitive_first_micro},
              "template_unchanged": True, "fixtures_unchanged": True}
    _write_json(directory / "parent.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--accumulate", action="store_true",
                        help="Check two microbatches per update, delayed gradient synchronization and whole-window normalization")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--fail-rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    for name in ("source_revision", "source_tree"):
        value = getattr(arguments, name)
        if value is not None and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a complete lowercase Git object ID")
    if (arguments.source_revision is None) != (arguments.source_tree is None):
        parser.error("Supply both source identity labels together")
    if arguments.output.exists():
        parser.error("The output path must be new")
    wheel = arguments.wheel or os.environ.get("RAGGED_WHEEL")
    wheel = str(Path(wheel).resolve(strict=True)) if wheel else None
    consumer = run_accumulation if arguments.accumulate else run
    result = consumer(arguments.dtype, arguments.directory, wheel, arguments.fail_rank)
    result["source_labels"] = {"revision": arguments.source_revision,
                                "tree": arguments.source_tree,
                                "meaning": "caller-supplied labels; executable script hash is independently recorded"}
    _write_json(arguments.output, result)
    printed = {"method": "torch.nn.parallel.DistributedDataParallel",
               "status": result["status"], "dtype": result["dtype"],
               "output": str(arguments.output.resolve()),
               "all_owned_workers_reaped": result["lifecycle"]["all_owned_workers_reaped"]}
    for name in ("world_size", "global_group_counts", "backend"):
        if name in result:
            printed[name] = result[name]
    print(json.dumps(printed, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
