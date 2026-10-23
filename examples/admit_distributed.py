"""Check one numerical admission boundary after Linux CPU DDP synchronization.

Copy this file, clip_distributed.py and train_distributed.py outside the checkout.
Two responsive ranks vote on finite gradient elements and the joint norm before
clipping or SGD. A deliberate local-only control clips on the healthy rank and
then stops before any metric collective or optimizer step. No recovery follows.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

import torch


_CLIPPING = None


def _clipping():
    global _CLIPPING
    if _CLIPPING is None:
        path = Path(__file__).resolve().with_name("clip_distributed.py")
        spec = importlib.util.spec_from_file_location("ragged_ddp_admission_clipping", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLIPPING = module
    return _CLIPPING


def _source_files():
    directory = Path(__file__).resolve().parent
    return {"admit_consumer_sha256": sha256((directory / "admit_distributed.py").read_bytes()).hexdigest(),
            "clip_consumer_sha256": sha256((directory / "clip_distributed.py").read_bytes()).hexdigest(),
            "helper_sha256": sha256((directory / "train_distributed.py").read_bytes()).hexdigest()}


def _oracle(helper, initial, batches, route):
    """Use full-window derivatives and the independent clipping arithmetic."""
    model = copy.deepcopy(initial)
    opt = torch.optim.SGD(model.parameters(), **helper.SGD_OPTIONS)
    records = []
    for window, cap in enumerate(helper.ADMISSION_CAPS):
        named = helper._parameter_bindings(model, opt)
        before_parameters = helper.copied(named)
        before_momentum = {name: opt.state[value].get("momentum_buffer", torch.zeros_like(value)).detach().clone()
                           for name, value in named.items()}
        opt.zero_grad(set_to_none=True)
        batch = helper.accumulation_global_batch(batches, window)
        raw = batch["raw"].detach().clone().requires_grad_(True)
        inputs = {"raw": raw, "targets": batch["targets"], "offsets": batch["offsets"]}
        before_inputs = helper.snapshot(inputs)
        prediction, pooled = model(raw, batch["offsets"], batch["boundaries"], route)
        loss = (prediction - batch["targets"]).square().sum() / batch["targets"].numel()
        loss.backward()
        gradients = {name: value.grad.detach().clone() for name, value in named.items()}
        clipped, norm, coefficient = _clipping()._manual_clip(gradients, cap)
        for name, value in named.items():
            value.grad.copy_(clipped[name])
        opt.step()
        parameters = helper.copied(named)
        momentum = {name: opt.state[value]["momentum_buffer"].detach().clone() for name, value in named.items()}
        for name in named:
            expected_buffer = before_momentum[name] * helper.SGD_OPTIONS["momentum"] + clipped[name]
            expected_parameter = before_parameters[name].clone()
            expected_parameter.add_(expected_buffer, alpha=-helper.SGD_OPTIONS["lr"])
            torch.testing.assert_close(momentum[name], expected_buffer, rtol=0, atol=0)
            torch.testing.assert_close(parameters[name], expected_parameter, rtol=0, atol=0)
        helper.unchanged(inputs, before_inputs)
        records.append({"predictions": prediction.detach().clone(), "pooled": pooled.detach().clone(),
                        "raw_gradients": raw.grad.detach().clone(), "metric": loss.detach().clone(),
                        "pre_clip_gradients": gradients, "parameter_gradients": clipped,
                        "pre_clip_norm": norm, "coefficient": coefficient,
                        "parameters": parameters, "momentum": momentum,
                        "sgd_recurrence_exact": True, "inputs_unchanged": True})
    return records


def _compare_prefix(helper, first, second):
    exact = {"rtol": 0, "atol": 0}
    result = {field: helper.compare(first[field], second[field], exact)
              for field in ("parameter_gradients", "parameters", "momentum")}
    result["metric"] = helper.compare({"metric": first["metric"]}, {"metric": second["metric"]}, exact)
    result["microbatches"] = [_compare_micro(helper, first["microbatches"][micro],
                                               second["microbatches"][micro]) for micro in range(2)]
    return result


def _compare_micro(helper, first, second):
    result = helper.compare({name: first[name] for name in
                             ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
                            {name: second[name] for name in
                             ("predictions", "pooled", "raw_gradients", "local_sse", "local_backward_loss")},
                            {"rtol": 0, "atol": 0})
    result["parameter_gradients_after_backward"] = helper.compare(
        first["parameter_gradients_after_backward"], second["parameter_gradients_after_backward"],
        {"rtol": 0, "atol": 0})
    return result


def _normal_comparison(helper, workers, native, primitive, batches, dtype_name):
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    rank_checks, oracle_checks, worker_checks = [], [], []
    for window in range(2):
        records = [worker["routes"]["global"][window] for worker in workers]
        rank_checks.append({field: helper.compare(records[0][field], records[1][field], {"rtol": 0, "atol": 0})
                            for field in ("parameter_gradients", "parameters", "momentum")})
        checks = {field: helper.compare(native[window][field], primitive[window][field], tolerance)
                  for field in ("pre_clip_gradients", "parameter_gradients", "parameters", "momentum")}
        checks["outputs"] = helper.compare(
            {name: native[window][name] for name in
             ("predictions", "pooled", "raw_gradients", "metric", "pre_clip_norm", "coefficient")},
            {name: primitive[window][name] for name in
             ("predictions", "pooled", "raw_gradients", "metric", "pre_clip_norm", "coefficient")}, tolerance)
        oracle_checks.append(checks)
        raw_start = group_start = 0
        for micro in range(2):
            for rank in range(2):
                batch = helper.accumulation_rank_batch(batches, window, micro, rank)
                worker = records[rank]
                observed = worker["microbatches"][micro]
                raw_stop = raw_start + batch["raw"].shape[0]
                group_stop = group_start + batch["targets"].numel()
                record = {"window": window, "microbatch": micro, "rank": rank}
                for label, oracle in (("native", native[window]), ("primitive", primitive[window])):
                    expected = {"predictions": oracle["predictions"][group_start:group_stop],
                                "pooled": oracle["pooled"][group_start:group_stop],
                                "raw_gradients": oracle["raw_gradients"][raw_start:raw_stop] * 2,
                                "metric": oracle["metric"]}
                    record[label] = {"outputs": helper.compare(
                        {name: worker["metric"] if name == "metric" else observed[name] for name in expected},
                        expected, tolerance),
                        "pre_clip_gradients": helper.compare(worker["clipping"]["pre_clip_gradients"],
                                                              oracle["pre_clip_gradients"], tolerance),
                        "norm_and_coefficient": helper.compare(
                            {"pre_clip_norm": worker["clipping"]["pre_clip_norm"],
                             "coefficient": worker["clipping"]["coefficient"]},
                            {"pre_clip_norm": oracle["pre_clip_norm"], "coefficient": oracle["coefficient"]}, tolerance)}
                    for field in ("parameter_gradients", "parameters", "momentum"):
                        record[label][field] = helper.compare(worker[field], oracle[field], tolerance)
                worker_checks.append(record)
                raw_start, group_start = raw_stop, group_stop
        for record in records:
            gate = record["admission"]
            if (record["status"] != "updated" or not gate["admitted"] or gate["local_bad"] != 0
                    or gate["bad_count"] != 0 or gate["admission_scalar_collectives"] != 1
                    or not gate["finite_elements"] or not gate["joint_norm_finite"]
                    or gate["joint_norm_kind"] != "finite" or gate["nonfinite_elements"]
                    or gate["joint_norm_value"] != float(record["clipping"]["pre_clip_norm"])
                    or record["clipping"]["norm_scalar_collectives"] != 0
                    or record["gradient_bytes_before_gate"] != record["gradient_bytes_after_gate"]
                    or record["training_state_before_gate"] != record["training_state_after_gate"]
                    or [value["hook_calls"] for value in record["microbatches"]] != [0, 1]
                    or not float(record["clipping"]["coefficient"]) < 1):
                raise AssertionError("The normal cohort did not preserve observational admission and active global clipping")
    return {"tolerance": tolerance, "raw_input_gradient_scale": 2,
            "rank_equality": rank_checks, "native_vs_primitive": oracle_checks,
            "worker_vs_oracle": worker_checks}


def _verify_injection(helper, terminal, rank, dtype_name):
    injection = {"name": "head.bias", "flat_index": 0, "kind": "nan"} if rank == 1 else None
    if terminal["injection"] != injection:
        raise AssertionError("The nonfinite control has the wrong injection attribution")
    before = terminal["gradient_bytes_before_gate"]
    for name in helper.PARAMETER_NAMES:
        gradient = terminal["pre_injection_parameter_gradients"][name]
        expected = gradient.detach().view(torch.uint8).numpy().tobytes().hex()
        actual = before[name]
        if actual["shape"] != list(gradient.shape) or actual["dtype"] != str(gradient.dtype):
            raise AssertionError("Gradient byte metadata differs from its finite pre-injection tensor")
        if rank == 1 and name == "head.bias":
            payload = bytes.fromhex(actual["hex"])
            code = "f" if dtype_name == "float32" else "d"
            if len(payload) != struct.calcsize(code) or not math.isnan(struct.unpack("=" + code, payload)[0]):
                raise AssertionError("The single declared gradient scalar was not actually NaN")
            if actual["hex"] == expected:
                raise AssertionError("The deliberate NaN did not change its scalar bytes")
        elif actual["hex"] != expected:
            raise AssertionError("The injection changed an undeclared gradient scalar")


def _control_comparison(helper, normal, workers, scenario, dtype_name):
    results = []
    exact = {"rtol": 0, "atol": 0}
    for rank, worker in enumerate(workers):
        prefix, terminal = worker["routes"]["global"]
        reference = normal[rank]["routes"]["global"]
        prefix_check = _compare_prefix(helper, prefix, reference[0])
        before_check = helper.compare(terminal["pre_injection_parameter_gradients"],
                                      reference[1]["pre_injection_parameter_gradients"], exact)
        output_checks = [_compare_micro(helper, terminal["microbatches"][micro],
                                         reference[1]["microbatches"][micro]) for micro in range(2)]
        state_check = {field: helper.compare(terminal[field], reference[0][field], exact)
                       for field in ("parameters", "momentum")}
        _verify_injection(helper, terminal, rank, dtype_name)
        gate = terminal["admission"]
        expected_bad = int(rank == 1)
        expected_collectives = int(scenario == "coordinated")
        expected_count = 1 if scenario == "coordinated" else expected_bad
        expected_admitted = scenario == "local_only" and rank == 0
        if (gate["local_bad"] != expected_bad or gate["bad_count"] != expected_count
                or gate["admission_scalar_collectives"] != expected_collectives
                or gate["admitted"] is not expected_admitted or terminal["optimizer_steps"] != 0
                or terminal["post_gate_metric_collectives"] != 0 or terminal["norm_scalar_collectives"] != 0
                or worker["completed_optimizer_steps"] != 1
                or terminal["gradient_bytes_before_gate"] != terminal["gradient_bytes_after_gate"]
                or terminal["training_state_before_gate"] != terminal["training_state_after_gate"]
                or terminal["training_state_before_gate"] != terminal["training_state_at_stop"]):
            raise AssertionError("The numerical control did not stop at its declared admission boundary")
        expected_invalid = [{"name": "head.bias", "flat_index": 0, "kind": "nan"}] if rank == 1 else []
        if (gate["finite_elements"] is not (rank == 0) or gate["joint_norm_finite"] is not (rank == 0)
                or gate["nonfinite_elements"] != expected_invalid
                or gate["joint_norm_kind"] != ("nan" if rank == 1 else "finite")
                or rank == 1 and gate["joint_norm_value"] is not None):
            raise AssertionError("The numerical refusal was not caused by the one declared post-sync NaN")
        expected_status = "local_only_stopped" if expected_admitted else "refused"
        if terminal["status"] != expected_status or worker["status"] != expected_status:
            raise AssertionError("The numerical control recorded the wrong terminal worker status")
        before_bytes, after_bytes = terminal["gradient_bytes_before_gate"], terminal["gradient_bytes_at_stop"]
        changed = [name for name in helper.PARAMETER_NAMES if before_bytes[name]["hex"] != after_bytes[name]["hex"]]
        if expected_admitted:
            if terminal["clip_calls"] != 1 or not changed or not float(terminal["clipping"]["coefficient"]) < 1:
                raise AssertionError("The healthy local-only rank did not actually change gradient bytes through clipping")
            for name in helper.PARAMETER_NAMES:
                if any(before_bytes[name][field] != after_bytes[name][field]
                       for field in ("shape", "dtype", "object_id", "storage_pointer")):
                    raise AssertionError("The local-only clip changed gradient ownership")
        elif terminal["clip_calls"] != 0 or before_bytes != after_bytes or "clipping" in terminal:
            raise AssertionError("A refused rank changed gradient bytes, bindings or versions")
        results.append({"rank": rank, "prefix_matches_normal": prefix_check,
                        "pre_injection_gradients_match_normal": before_check,
                        "pre_injection_outputs_match_normal": output_checks,
                        "parameters_and_momentum_match_completed_window0": state_check,
                        "actual_post_injection_local_bad": expected_bad,
                        "admitted": expected_admitted, "gradient_content_changed_blocks": changed,
                        "optimizer_steps_after_gate": 0})
    return results


def run(dtype_name, directory, wheel=None, fail_rank=None):
    if sys.platform != "linux":
        raise RuntimeError(
            "The two-rank Gloo consumer is qualified on Linux only; the tested "
            "Windows PyTorch 2.9.1+cpu wheel did not provide a usable loopback transport")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    helper = _clipping()._helper()
    cohorts = {}
    for scenario in ("normal", "coordinated", "local_only"):
        path = directory / scenario
        path.mkdir(exist_ok=False)
        lifecycle, receipts = helper._launch_workers(
            dtype_name, path, wheel, fail_rank if scenario == "normal" else None,
            accumulate=True, clipping=True, admission=scenario)
        cohorts[scenario] = {"lifecycle": lifecycle, "workers": receipts}
        helper._write_json(path / "parent-cohort.json", cohorts[scenario])
        if receipts is None:
            result = {"status": "failed", "dtype": dtype_name, "cohorts": cohorts,
                      "deliberate_failure_rank": fail_rank, "source_files": _source_files()}
            helper._write_json(directory / "parent.json", result)
            return result
    # All numerical comparison work follows the three separately reaped cohorts.
    identity = helper.installed_identity(wheel)
    source_files = _source_files()
    identity.update({name: source_files[name] for name in ("clip_consumer_sha256", "admit_consumer_sha256")})
    initial, batches, fixture_digest = helper.accumulation_fixture(dtype_name)
    before_initial = helper.snapshot(dict(initial.named_parameters()))
    inputs = helper._accumulation_fixture_inputs(batches)
    before_inputs = helper.snapshot(inputs)
    for scenario, cohort in cohorts.items():
        for worker in cohort["workers"]:
            if (worker["identity"] != identity or worker["fixture_sha256"] != fixture_digest
                    or worker["admission_scenario"] != scenario):
                raise AssertionError("An admission cohort used different installed bytes, source or fixtures")
        if not cohort["lifecycle"]["all_owned_workers_reaped"] or any(
                item["exitcode"] != 0 or item["alive"] for item in cohort["lifecycle"]["owned_processes"]):
            raise AssertionError("A numerical admission cohort did not exit and reap normally")
    native = _oracle(helper, initial, batches, "native")
    primitive = _oracle(helper, initial, batches, "primitive")
    normal = cohorts["normal"]["workers"]
    if any(worker["completed_optimizer_steps"] != 2 for worker in normal):
        raise AssertionError("The normal admission cohort did not complete both optimizer steps")
    comparison = {"normal": _normal_comparison(helper, normal, native, primitive, batches, dtype_name),
                  "coordinated_refusal": _control_comparison(
                      helper, normal, cohorts["coordinated"]["workers"], "coordinated", dtype_name),
                  "local_only_control": _control_comparison(
                      helper, normal, cohorts["local_only"]["workers"], "local_only", dtype_name)}
    helper.unchanged(dict(initial.named_parameters()), before_initial)
    helper.unchanged(inputs, before_inputs)
    result = {"status": "passed", "dtype": dtype_name, "seed": helper.SEED, "world_size": 2,
              "device": "cpu", "backend": "ragged_gloo_loopback", "admission_caps": helper.ADMISSION_CAPS,
              "admission_scope": "finite elements and joint norm after final synchronized backward; two responsive ranks",
              "refusal_scope": "stop before clipping, metric reduction or SGD; no skip or recovery",
              "local_control_scope": "healthy rank actually clips, then stops before metric reduction or SGD",
              "source_files": source_files, "identity": identity, "fixture_sha256": fixture_digest,
              "cohorts": cohorts, "comparison": comparison, "oracles": {"native": native, "primitive": primitive},
              "sgd_options": helper.SGD_OPTIONS, "global_group_counts": [9, 5],
              "gloo_threads_per_worker": 1, "numerical_threads_per_worker": 1,
              "transport_scope": "single host; TCP Gloo device 127.0.0.1; unique file rendezvous per cohort",
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
                                "meaning": "caller-supplied labels; all three adjacent source hashes are independently recorded"}
    _clipping()._helper()._write_json(arguments.output, result)
    print(json.dumps({"method": "post-DDP numerical admission before clipping and SGD",
                      "status": result["status"], "dtype": result["dtype"],
                      "all_owned_workers_reaped": all(value["lifecycle"]["all_owned_workers_reaped"]
                                                       for value in result["cohorts"].values()),
                      "output": str(arguments.output.resolve())}, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
