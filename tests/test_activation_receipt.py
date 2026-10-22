"""Independent literal receipts exercise the release gate, without numerical code."""
import copy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/check_activation_receipt.py"
spec = importlib.util.spec_from_file_location("activation_receipt_gate", path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def valid(dtype="float32"):
    size = 4 if dtype == "float32" else 8
    event = {"shape": [2, 3], "dtype": "torch." + dtype, "numel": 6,
             "element_size": size, "logical_bytes": 6 * size}
    direct = {"events": [copy.deepcopy(event), copy.deepcopy(event)], "count": 2, "logical_bytes": 12 * size}
    checkpointed = {"events": [copy.deepcopy(event)], "count": 1, "logical_bytes": 6 * size}
    receipt = {"status": "passed", "method": "checkpoint.non_reentrant.packed_training",
               "dtype": dtype, "device": "cpu", "use_reentrant": False, "preserve_rng_state": True,
               "early_stop": False, "determinism_check": "default", "dropout_probability": 0.25,
               "dropout_seed": 2211, "optimizer": "SGD", "learning_rate": 0.03, "momentum": 0.8,
               "optimizer_steps": 2, "scalar_parameters": 25, "parameter_tensors": 5,
               "independent_momentum_recurrence": True, "same_native_exact": True,
               "primitive_comparisons_passed": True, "inputs_unchanged": True,
               "parameter_ownership_unchanged": True, "installed_package": True, "native_library_loaded": True,
               "threads": 1, "interop_threads": 1, "python_version": "3.11.8", "torch_version": "2.9.1+cpu",
               "saved_accounting": "original_forward_pack_hook_events", "memory_performance_claim": False,
               "comparison_tolerance": ({"rtol": 2e-4, "atol": 2e-6} if dtype == "float32"
                                        else {"rtol": 2e-10, "atol": 2e-12}),
               "same_native_max_abs_error": 0.0, "primitive_max_abs_error": 1e-12,
               "script_sha256": "a" * 64, "native_library_sha256": "b" * 64,
               "train_dynamic_sha256": "c" * 64,
               "controls": {"without_rng_preservation": {"first_forward_exact": True, "diverged": True,
                   "rng_diverged": True, "divergence_threshold": 1e-7 if dtype == "float32" else 1e-10,
                   "gradient_max_abs_error": 0.02, "final_parameter_max_abs_error": 0.01},
                   "dropout_disabled": {"dropout_probability": 0.0, "same_native_exact": True,
                                        "max_abs_error": 0.0, "rng_unchanged": True}}, "steps": []}
    for index, offsets in enumerate(([0, 2, 2, 5, 7, 9], [0, 3, 5, 5, 8, 10, 12])):
        receipt["steps"].append({"update": index + 1, "N": offsets[-1], "B": len(offsets) - 1,
            "offsets": offsets, "loss": 0.2, "native_exact_max_abs_error": 0.0, "primitive_max_abs_error": 1e-12,
            "checkpoint_scope_events": ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"],
            "saved_forward": {"direct": copy.deepcopy(direct), "checkpointed": copy.deepcopy(checkpointed)},
            "default_rng_before_sha256": str(index + 1) * 64,
            "default_rng_after_forward_sha256": str(index + 2) * 64,
            "default_rng_after_backward_sha256": str(index + 2) * 64})
    return receipt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_literal_positive(dtype):
    assert gate.check(valid(dtype), dtype, "a" * 64, "b" * 64, "c" * 64)["status"] == "passed"


MUTATIONS = [
    (("use_reentrant",), True), (("optimizer_steps",), True),
    (("same_native_max_abs_error",), 1e-8), (("primitive_max_abs_error",), float("nan")),
    (("script_sha256",), "d" * 64), (("train_dynamic_sha256",), "d" * 64),
    (("native_library_sha256",), "d" * 64), (("memory_performance_claim",), True),
    (("steps", 0, "checkpoint_scope_events"), ["forward_enter", "forward_exit"]),
    (("steps", 0, "saved_forward", "direct", "count"), 99),
    (("steps", 0, "saved_forward", "direct", "logical_bytes"), 999),
    (("steps", 0, "saved_forward", "direct", "events", 0, "logical_bytes"), 999),
    (("steps", 0, "saved_forward", "direct", "events", 0, "element_size"), 8),
    (("steps", 0, "saved_forward", "direct", "events", 0, "shape"), [3, 3]),
    (("steps", 0, "default_rng_after_backward_sha256"), "f" * 64),
    (("steps", 1, "default_rng_before_sha256"), "f" * 64),
    (("steps", 1, "offsets"), [0, 3, 5, 5, 8, 10, 13]),
    (("controls", "without_rng_preservation", "first_forward_exact"), False),
    (("controls", "without_rng_preservation", "rng_diverged"), False),
    (("controls", "without_rng_preservation", "gradient_max_abs_error"), 0),
    (("controls", "without_rng_preservation", "final_parameter_max_abs_error"), 0),
    (("controls", "dropout_disabled", "rng_unchanged"), False),
]


@pytest.mark.parametrize("keys,value", MUTATIONS)
def test_invalid_receipt_refuses(keys, value):
    receipt = valid()
    target = receipt
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError):
        gate.check(receipt, "float32", "a" * 64, "b" * 64, "c" * 64)


def test_no_payload_reduction_refuses():
    receipt = valid()
    receipt["steps"][0]["saved_forward"]["checkpointed"] = copy.deepcopy(
        receipt["steps"][0]["saved_forward"]["direct"])
    with pytest.raises(ValueError, match="did not decrease"):
        gate.check(receipt, "float32", "a" * 64, "b" * 64, "c" * 64)
