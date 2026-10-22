"""Independent malformed-completion controls for the stochastic CI receipt gate."""
import copy
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "check_stochastic_receipt.py"
SPEC = importlib.util.spec_from_file_location("stochastic_receipt_checks", SOURCE)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def receipt(dtype="float64"):
    identity = {"package_path": "/accept/ragged_pool/__init__.py", "package_version": "0.1.0",
                "environment_prefix": "/accept", "native_library_path": "/accept/ragged_pool/_C.so",
                "native_library_sha256": "b" * 64, "script_sha256": "a" * 64,
                "native_library_loaded": True}
    threshold = 1e-7 if dtype == "float32" else 1e-10
    result = {"status": "passed", "method": "state_dict.stochastic_packed_training_resume",
              "dtype": dtype, "device": "cpu", "dropout_probability": 0.25,
              "training_mode": True, "optimizer_steps": 7, "producer_steps": 2,
              "resumed_steps": 5, "checkpoint_cursor": 2, "checkpoint_epoch": 0,
              "order_regenerations": 1, "distinct_processes": True,
              "same_native_exact": True, "same_native_max_abs_error": 0,
              "primitive_comparisons_passed": True, "primitive_max_abs_error": 1e-12,
              "comparison_tolerance": ({"rtol": 2e-4, "atol": 2e-6} if dtype == "float32"
                                       else {"rtol": 2e-10, "atol": 2e-12}),
              "independent_momentum_recurrence": True, "checkpoint_unchanged": True,
              "input_sources_unchanged": True, "initial_parameters_unchanged": True,
              "loaded_state_storage_independent": True, "installed_package": True,
              "threads": 1, "interop_threads": 1, "python_version": "3.11.8",
              "torch_version": "2.9.1+cpu", "checkpoint_bytes": 16000,
              "checkpoint_byte_limit": 131072, "checkpoint_sha256": "c" * 64,
              "process_ids": {"parent": 11, "producer": 22, "resumer": 33},
              "child_installed_identities": {"producer": dict(identity), "resumer": dict(identity)},
              "controls": {name: {"diverged": True, "divergence_threshold": threshold,
                                   "final_parameter_max_abs_error": 0.01}
                           for name in CHECKER.CONTROLS}, **identity}
    result["steps"] = [{"update": i + 1, "epoch": i // 5,
                        "cursor_before": 5 if i == 5 else i % 5,
                        "cursor_after": i % 5 + 1, "batch_id": i % 5,
                        "default_rng_before_sha256": f"{i:064x}",
                        "default_rng_after_sha256": f"{i + 1:064x}",
                        "order_rng_sha256": "d" * 64 if i < 5 else "e" * 64,
                        "N": 2, "B": 1, "offsets": [0, 2], "loss": 0.25}
                       for i in range(7)]
    return result


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complete_receipt_with_two_actual_order_prefixes_is_admitted(dtype):
    result = CHECKER.check(receipt(dtype), dtype, "a" * 64, "b" * 64)
    assert result == {"status": "passed", "dtype": dtype, "updates": 7,
                      "fresh_processes": 3, "epoch_boundary_exercised": True}


@pytest.mark.parametrize("fault", [
    "step_bool", "same_process", "native_error", "primitive_nan", "source_identity",
    "child_identity", "oversized_checkpoint", "incomplete_control", "control_no_drift",
    "cursor_skip", "duplicate_batch", "missing_rng", "stalled_dropout",
    "discontinuous_dropout", "shuffle_not_advanced", "shuffle_advanced_early", "invalid_offsets",
    "missing_step", "relaxed_tolerance", "training_false",
])
def test_incomplete_or_contradictory_receipt_is_rejected_without_mutation(fault):
    bad = receipt()
    if fault == "step_bool": bad["producer_steps"] = True
    elif fault == "same_process": bad["process_ids"]["resumer"] = 22
    elif fault == "native_error": bad["same_native_max_abs_error"] = 1e-12
    elif fault == "primitive_nan": bad["primitive_max_abs_error"] = float("nan")
    elif fault == "source_identity": bad["script_sha256"] = "f" * 64
    elif fault == "child_identity": bad["child_installed_identities"]["resumer"]["native_library_sha256"] = "f" * 64
    elif fault == "oversized_checkpoint": bad["checkpoint_bytes"] = 131073
    elif fault == "incomplete_control": bad["controls"].pop("reset_order_generator")
    elif fault == "control_no_drift": bad["controls"]["reset_dropout_rng"]["final_parameter_max_abs_error"] = 0
    elif fault == "cursor_skip": bad["steps"][2]["cursor_before"] = 3
    elif fault == "duplicate_batch": bad["steps"][2]["batch_id"] = 1
    elif fault == "missing_rng": bad["steps"][4].pop("default_rng_after_sha256")
    elif fault == "stalled_dropout": bad["steps"][4]["default_rng_after_sha256"] = bad["steps"][4]["default_rng_before_sha256"]
    elif fault == "discontinuous_dropout": bad["steps"][4]["default_rng_before_sha256"] = "f" * 64
    elif fault == "shuffle_not_advanced": bad["steps"][5]["order_rng_sha256"] = "d" * 64
    elif fault == "shuffle_advanced_early": bad["steps"][3]["order_rng_sha256"] = "f" * 64
    elif fault == "invalid_offsets": bad["steps"][4]["offsets"] = [2, 0]
    elif fault == "missing_step": bad["steps"].pop()
    elif fault == "relaxed_tolerance": bad["comparison_tolerance"]["atol"] = 1
    else: bad["training_mode"] = False
    before = copy.deepcopy(bad)
    with pytest.raises(ValueError):
        CHECKER.check(bad, "float64", "a" * 64, "b" * 64)
    # NaN needs a representation comparison, since NaN is unequal to itself.
    assert repr(bad) == repr(before)
