"""Clip one replicated CPU model gradient after complete DDP accumulation.

Copy this file and train_distributed.py outside the checkout and use the ordinary
installed CPU wheel. Two owned Linux Gloo workers compare a whole-window clip
with a deliberate double count of the replicated norm. The example checks
optimizer inputs and updates, not model quality, speed or privacy.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import torch


_TRAINING = None


def _helper():
    global _TRAINING
    if _TRAINING is None:
        path = Path(__file__).resolve().with_name("train_distributed.py")
        spec = importlib.util.spec_from_file_location("ragged_ddp_clipping_helper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINING = module
    return _TRAINING


def _source_files():
    path = Path(__file__).resolve()
    return {"consumer_sha256": sha256(path.read_bytes()).hexdigest(),
            "helper_sha256": sha256(path.with_name("train_distributed.py").read_bytes()).hexdigest()}


def _manual_clip(gradients, cap):
    """Independent concatenated-vector arithmetic with the pinned API epsilon."""
    vector = torch.cat([value.reshape(-1) for value in gradients.values()])
    norm = torch.linalg.vector_norm(vector, ord=2)
    if not bool(torch.isfinite(norm)):
        raise AssertionError("The independent full-window gradient norm must be finite")
    coefficient = torch.clamp(cap / (norm + 1e-6), max=1.0)
    clipped = {name: value * coefficient for name, value in gradients.items()}
    return clipped, norm, coefficient


def _oracle(helper, initial, batches, route):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(model.parameters(), **helper.SGD_OPTIONS)
    steps, first_micro = [], []
    for window, cap in enumerate(helper.CLIP_CAPS):
        named = helper._parameter_bindings(model, optimizer)
        before_parameters = helper.copied(named)
        before_momentum = {name: optimizer.state[value].get(
            "momentum_buffer", torch.zeros_like(value)).detach().clone()
            for name, value in named.items()}
        global_batch = helper.accumulation_global_batch(batches, window)
        groups = global_batch["targets"].numel()
        local_records = []
        for rank in range(2):
            local = copy.deepcopy(model)
            batch = helper.accumulation_rank_batch(batches, window, 0, rank)
            raw = batch["raw"].detach().clone().requires_grad_(True)
            predictions, pooled = local(raw, batch["offsets"], batch["boundaries"], "primitive")
            ((predictions - batch["targets"]).square().sum() * (2 / groups)).backward()
            local_records.append({"parameter_gradients": {
                name: value.grad.detach().clone() for name, value in local.named_parameters()},
                "raw_gradients": raw.grad.detach().clone(),
                "predictions": predictions.detach().clone(), "pooled": pooled.detach().clone()})
        first_micro.append(local_records)
        optimizer.zero_grad(set_to_none=True)
        raw = global_batch["raw"].detach().clone().requires_grad_(True)
        inputs = {"raw": raw, "targets": global_batch["targets"], "offsets": global_batch["offsets"]}
        before_inputs = helper.snapshot(inputs)
        predictions, pooled = model(raw, global_batch["offsets"], global_batch["boundaries"], route)
        loss = (predictions - global_batch["targets"]).square().sum() / groups
        loss.backward()
        if raw.grad is None or any(value.grad is None for value in named.values()):
            raise AssertionError("The full-window oracle lost a gradient connection")
        pre_gradients = {name: value.grad.detach().clone() for name, value in named.items()}
        clipped, norm, coefficient = _manual_clip(pre_gradients, cap)
        for name, value in named.items():
            value.grad.copy_(clipped[name])
        optimizer.step()
        parameters = helper.copied(named)
        momentum = {name: optimizer.state[value]["momentum_buffer"].detach().clone()
                    for name, value in named.items()}
        for name in named:
            expected_buffer = before_momentum[name] * helper.SGD_OPTIONS["momentum"] + clipped[name]
            expected_parameter = before_parameters[name].clone()
            expected_parameter.add_(expected_buffer, alpha=-helper.SGD_OPTIONS["lr"])
            torch.testing.assert_close(momentum[name], expected_buffer, rtol=0, atol=0)
            torch.testing.assert_close(parameters[name], expected_parameter, rtol=0, atol=0)
        helper.unchanged(inputs, before_inputs)
        steps.append({"predictions": predictions.detach().clone(), "pooled": pooled.detach().clone(),
                      "raw_gradients": raw.grad.detach().clone(), "metric": loss.detach().clone(),
                      "pre_clip_gradients": pre_gradients, "parameter_gradients": clipped,
                      "pre_clip_norm": norm, "coefficient": coefficient,
                      "parameters": parameters, "momentum": momentum, "global_groups": groups,
                      "sgd_recurrence_exact": True, "inputs_unchanged": True})
    return steps, first_micro


def _compare_results(helper, receipts, native, primitive, first_micro, batches, dtype_name):
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    exact = {"rtol": 0, "atol": 0}
    rank_errors, oracle_errors, worker_errors, local_errors = [], [], [], []
    for window in range(2):
        ranks = {}
        for mode in ("global", "replica_norm_control"):
            first, second = [receipt["routes"][mode][window] for receipt in receipts]
            ranks[mode] = {field: helper.compare(first[field], second[field], exact)
                           for field in ("parameter_gradients", "parameters", "momentum")}
            for field in ("pre_clip_gradients", "post_clip_gradients"):
                ranks[mode][field] = helper.compare(first["clipping"][field], second["clipping"][field], exact)
            ranks[mode]["norms"] = helper.compare(
                {name: first["clipping"][name] for name in
                 ("pre_clip_norm", "coefficient_norm", "coefficient", "post_clip_norm")},
                {name: second["clipping"][name] for name in
                 ("pre_clip_norm", "coefficient_norm", "coefficient", "post_clip_norm")}, exact)
            for worker in (first, second):
                clip = worker["clipping"]
                expected_collectives = 1 if mode == "replica_norm_control" else 0
                if (clip["norm_scalar_collectives"] != expected_collectives
                        or [micro["hook_calls"] for micro in worker["microbatches"]] != [0, 1]):
                    raise AssertionError("Clipping or gradient synchronization used an unexpected collective count")
                expected_norm = (clip["pre_clip_norm"].square() * 2).sqrt() if expected_collectives else clip["pre_clip_norm"]
                torch.testing.assert_close(clip["coefficient_norm"], expected_norm, **tolerance)
        rank_errors.append(ranks)
        checks = {field: helper.compare(native[window][field], primitive[window][field], tolerance)
                  for field in ("pre_clip_gradients", "parameter_gradients", "parameters", "momentum")}
        checks["outputs"] = helper.compare(
            {name: native[window][name] for name in
             ("predictions", "pooled", "raw_gradients", "metric", "pre_clip_norm", "coefficient")},
            {name: primitive[window][name] for name in
             ("predictions", "pooled", "raw_gradients", "metric", "pre_clip_norm", "coefficient")}, tolerance)
        oracle_errors.append(checks)
        raw_start = group_start = 0
        for micro in range(2):
            for rank in range(2):
                batch = helper.accumulation_rank_batch(batches, window, micro, rank)
                worker = receipts[rank]["routes"]["global"][window]
                observed = worker["microbatches"][micro]
                raw_stop = raw_start + batch["raw"].shape[0]
                group_stop = group_start + batch["targets"].numel()
                record = {"window": window, "microbatch": micro, "rank": rank}
                for label, oracle in (("native", native[window]), ("primitive", primitive[window])):
                    expected = {"predictions": oracle["predictions"][group_start:group_stop],
                                "pooled": oracle["pooled"][group_start:group_stop],
                                "raw_gradients": oracle["raw_gradients"][raw_start:raw_stop] * 2}
                    record[label] = {"outputs": helper.compare({name: observed[name] for name in expected},
                                                                expected, tolerance),
                                     "metric": helper.compare({"metric": worker["metric"]},
                                                              {"metric": oracle["metric"]}, tolerance),
                                     "pre_clip_gradients": helper.compare(
                                         worker["clipping"]["pre_clip_gradients"], oracle["pre_clip_gradients"], tolerance),
                                     "norm_and_coefficient": helper.compare(
                                         {"pre_clip_norm": worker["clipping"]["pre_clip_norm"],
                                          "coefficient": worker["clipping"]["coefficient"]},
                                         {"pre_clip_norm": oracle["pre_clip_norm"],
                                          "coefficient": oracle["coefficient"]}, tolerance)}
                    for field in ("parameter_gradients", "parameters", "momentum"):
                        record[label][field] = helper.compare(worker[field], oracle[field], tolerance)
                worker_errors.append(record)
                if micro == 0:
                    local_errors.append({"window": window, "rank": rank,
                                         "comparison": helper.compare(
                                             worker["first_micro_parameter_gradients"],
                                             first_micro[window][rank]["parameter_gradients"], tolerance)})
                raw_start, group_start = raw_stop, group_stop
    controls, threshold = [], 1e-8 if dtype_name == "float32" else 1e-11
    for rank, receipt in enumerate(receipts):
        correct, wrong = receipt["routes"]["global"], receipt["routes"]["replica_norm_control"]
        identical = [helper.compare(
            {name: correct[0]["microbatches"][micro][name]
             for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
            {name: wrong[0]["microbatches"][micro][name]
             for name in ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")}, exact)
            for micro in range(2)]
        same_gradient = helper.compare(correct[0]["clipping"]["pre_clip_gradients"],
                                        wrong[0]["clipping"]["pre_clip_gradients"], exact)
        coefficient_drift = helper.maximum_difference(correct[0]["clipping"]["coefficient"],
                                                        wrong[0]["clipping"]["coefficient"])
        drift = {field: [max(helper.maximum_difference(correct[window][field][name], wrong[window][field][name])
                              for name in helper.PARAMETER_NAMES) for window in range(2)]
                 for field in ("parameter_gradients", "parameters", "momentum")}
        if coefficient_drift <= threshold or not all(value > threshold for values in drift.values() for value in values):
            raise AssertionError("The duplicate-replica norm control did not change the optimizer trajectory")
        coefficients = [float(record["clipping"]["coefficient"]) for record in correct]
        control_coefficients = [float(record["clipping"]["coefficient"]) for record in wrong]
        if not (coefficients[0] < 1 and coefficients[1] == 1
                and control_coefficients[0] < 1 and control_coefficients[1] == 1):
            raise AssertionError("The fixed clipping schedule did not exercise one active and one inactive window")
        controls.append({"rank": rank, "first_window_forward_and_raw_gradients_exact": identical,
                         "first_window_pre_clip_gradients_exact": same_gradient,
                         "first_coefficient_absolute_drift": coefficient_drift,
                         "max_absolute_drift": drift, "required_drift_greater_than": threshold,
                         "correct_active_windows": [True, False],
                         "replica_control_active_windows": [True, False],
                         "correct_coefficients": coefficients,
                         "replica_control_coefficients": control_coefficients})
    return {"tolerance": tolerance, "raw_input_gradient_scale": 2,
            "rank_equality": rank_errors, "native_vs_primitive": oracle_errors,
            "worker_vs_oracle": worker_errors, "first_micro_vs_local_primitive": local_errors,
            "duplicate_replica_norm_controls": controls}


def run(dtype_name, directory, wheel=None, fail_rank=None):
    if sys.platform != "linux":
        raise RuntimeError(
            "The two-rank Gloo consumer is qualified on Linux only; the tested "
            "Windows PyTorch 2.9.1+cpu wheel did not provide a usable loopback transport")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[name] = "1"
    helper = _helper()
    lifecycle, receipts = helper._launch_workers(dtype_name, directory, wheel, fail_rank,
                                                 accumulate=True, clipping=True)
    if receipts is None:
        result = {"status": "failed", "dtype": dtype_name, "lifecycle": lifecycle,
                  "deliberate_failure_rank": fail_rank, "source_files": _source_files()}
        helper._write_json(directory / "parent.json", result)
        return result
    # No oracle computes while either directly owned training worker is alive.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identity = helper.installed_identity(wheel)
    identity["clip_consumer_sha256"] = _source_files()["consumer_sha256"]
    initial, batches, fixture_digest = helper.accumulation_fixture(dtype_name)
    initial_before = helper.snapshot(dict(initial.named_parameters()))
    inputs = helper._accumulation_fixture_inputs(batches)
    inputs_before = helper.snapshot(inputs)
    for receipt in receipts:
        if (receipt["identity"] != identity or receipt["fixture_sha256"] != fixture_digest
                or receipt.get("global_clipping") is not True):
            raise AssertionError("Clipping workers and parent used different installed bytes, source or fixtures")
    native, native_first = _oracle(helper, initial, batches, "native")
    primitive, primitive_first = _oracle(helper, initial, batches, "primitive")
    comparison = _compare_results(helper, receipts, native, primitive, primitive_first, batches, dtype_name)
    helper.unchanged(dict(initial.named_parameters()), initial_before)
    helper.unchanged(inputs, inputs_before)
    result = {"status": "passed", "dtype": dtype_name, "seed": helper.SEED,
              "global_clipping": True, "world_size": 2, "device": "cpu", "backend": "ragged_gloo_loopback",
              "clipping_caps": helper.CLIP_CAPS, "clipping_epsilon": 1e-6,
              "clipping_route": "clip_grad_norm_ after final DDP backward; L2; error_if_nonfinite=True; foreach=False",
              "norm_scope": "one joint replicated 25-scalar parameter-gradient vector, counted once",
              "negative_control": "sum already-replicated squared norms across two ranks before scaling",
              "transport_scope": "single host; TCP Gloo device 127.0.0.1; file rendezvous",
              "gloo_threads_per_worker": 1, "global_group_counts": [9, 5],
              "global_micro_group_counts": [[5, 4], [1, 4]],
              "identity": identity, "source_files": _source_files(), "fixture_sha256": fixture_digest,
              "sgd_options": helper.SGD_OPTIONS, "lifecycle": lifecycle, "workers": receipts,
              "comparison": comparison,
              "oracles": {"native": native, "primitive": primitive,
                          "local_first_micro_at_native_state": native_first,
                          "local_first_micro_at_primitive_state": primitive_first},
              "template_unchanged": True, "fixtures_unchanged": True}
    helper._write_json(directory / "parent.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
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
    result = run(arguments.dtype, arguments.directory, wheel, arguments.fail_rank)
    result["source_labels"] = {"revision": arguments.source_revision, "tree": arguments.source_tree,
                                "meaning": "caller-supplied labels; consumer and helper hashes are independently recorded"}
    _helper()._write_json(arguments.output, result)
    print(json.dumps({"method": "global parameter-gradient clipping after CPU DDP accumulation",
                      "status": result["status"], "dtype": result["dtype"],
                      "all_owned_workers_reaped": result["lifecycle"]["all_owned_workers_reaped"],
                      "output": str(arguments.output.resolve())}, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
