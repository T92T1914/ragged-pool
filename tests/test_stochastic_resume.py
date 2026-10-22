"""State admission and fixed stochastic consumer trajectory controls."""
import copy
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "resume_stochastic.py"


def example():
    spec = importlib.util.spec_from_file_location("stochastic_resume_tests", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(dtype_name="float64"):
    module = example()
    initial, batches, tolerance = module.fixture(dtype_name)
    metadata = module.metadata(dtype_name, initial, batches)
    model, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(model)
    torch.manual_seed(module.DROPOUT_SEED)
    module.run_steps(model, opt, batches, order, stop=2)
    checkpoint = module.make_checkpoint(model, opt, order, metadata)
    return module, initial, batches, tolerance, metadata, checkpoint


def exact(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert list(first) == list(second)
        for key in first:
            exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            exact(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_actual_dropout_feeds_both_values_and_scores(dtype_name):
    module = example()
    model, batches, _ = module.fixture(dtype_name)
    assert sum(value.numel() for value in model.parameters()) == 25
    torch.manual_seed(module.DROPOUT_SEED)
    with patch.object(module.F, "dropout", wraps=module.F.dropout) as dropout:
        predictions, pooled = model(batches[0]["raw"], batches[0]["offsets"])
    assert dropout.call_count == 1
    assert dropout.call_args.kwargs == {"p": 0.25, "training": True}
    evidence = model.last_dropout
    keep = evidence["keep"].bool()
    assert bool(keep.any()) and not bool(keep.all())
    assert not bool((evidence["encoded"] == 0).any())
    exact(evidence["dropped"], torch.where(keep, evidence["encoded"] * (4 / 3),
                                          torch.zeros_like(evidence["encoded"])))
    expected_logits = torch.nn.functional.linear(evidence["dropped"], model.scorer.weight).squeeze(-1)
    exact(evidence["logits"], expected_logits.detach())
    expected_pool = module.dynamic.primitive_pool(batches[0]["python_boundaries"])(
        evidence["dropped"], expected_logits, batches[0]["offsets"])
    tolerance = module.fixture(dtype_name)[2]
    torch.testing.assert_close(pooled, expected_pool, **tolerance)
    torch.testing.assert_close(predictions, model.head(expected_pool).squeeze(-1), **tolerance)
    model.eval()
    with pytest.raises(ValueError, match="training mode"):
        model(batches[0]["raw"], batches[0]["offsets"])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_valid_restore_is_exact_owned_and_crosses_the_next_permutation(dtype_name):
    module, initial, batches, tolerance, metadata, checkpoint = prepared(dtype_name)
    original = copy.deepcopy(checkpoint)
    checkpoint_versions = [value._version for value in module._tensors(checkpoint)]
    full, primitive = copy.deepcopy(initial), copy.deepcopy(initial)
    torch.manual_seed(module.DROPOUT_SEED)
    full_records = module.run_steps(full, module.optimizer(full), batches, module.BatchOrder())
    torch.manual_seed(module.DROPOUT_SEED)
    primitive_records = module.run_steps(primitive, module.optimizer(primitive), batches,
                                         module.BatchOrder(), reference=True)
    module._compare_tree(full_records, primitive_records, tolerance)
    target, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(target)
    target.eval()
    parameter_ids = tuple(id(value) for value in target.parameters())
    torch.manual_seed(77)
    module.restore(checkpoint, target, opt, order, metadata)
    assert target.training is True
    exact(torch.get_rng_state(), checkpoint["cpu_rng"])
    exact(order.state_dict(), checkpoint["order"])
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    resumed = module.run_steps(target, opt, batches, order, start=2)
    exact(resumed, full_records[2:])
    assert [row["batch_id"] for row in full_records[:5]] == checkpoint["order"]["permutation"]
    assert [row["order_after"]["epoch"] for row in resumed] == [0, 0, 0, 1, 1]
    assert [row["order_after"]["cursor"] for row in resumed] == [3, 4, 5, 1, 2]
    exact(torch.get_rng_state(), full_records[-1]["rng_after"])
    exact(order.state_dict(), full_records[-1]["order_after"])
    saved_storage = {value.untyped_storage().data_ptr() for value in module._tensors(checkpoint)}
    target_tensors = {"model": dict(target.named_parameters()), "optimizer": opt.state_dict(),
                      "cpu_rng": torch.get_rng_state(), "order": order.state_dict()}
    assert all(value.untyped_storage().data_ptr() not in saved_storage
               for value in module._tensors(target_tensors))
    exact(checkpoint, original)
    assert [value._version for value in module._tensors(checkpoint)] == checkpoint_versions


@pytest.mark.parametrize("case", [
    "cpu_type", "cpu_dtype", "cpu_length", "cpu_view", "cpu_stride", "cpu_semantics",
    "order_rng_type", "order_rng_dtype", "order_rng_length", "order_rng_semantics",
    "order_rng_wrong_valid", "permutation_type", "permutation_duplicate", "permutation_bounds",
    "permutation_bool", "permutation_wrong_valid", "cursor_bool", "cursor_repeat", "cursor_skip",
    "epoch_bool", "epoch_wrong", "step_bool", "step_repeat", "step_skip", "training_false",
    "training_integer", "metadata_extra",
])
def test_invalid_state_refuses_before_model_optimizer_mode_sampler_or_rng_changes(case):
    module, initial, _, _, metadata, checkpoint = prepared()
    bad = copy.deepcopy(checkpoint)
    if case.startswith("cpu_") or case.startswith("order_rng_"):
        target = bad if case.startswith("cpu_") else bad["order"]
        key = "cpu_rng" if case.startswith("cpu_") else "generator"
        kind = case.removeprefix("cpu_").removeprefix("order_rng_")
        original = target[key]
        if kind == "type": target[key] = original.tolist()
        elif kind == "dtype": target[key] = original.to(torch.int64)
        elif kind == "length": target[key] = original[:-1].clone()
        elif kind == "view":
            target[key] = torch.zeros(original.numel() + 1, dtype=torch.uint8)[1:]
        elif kind == "stride": target[key] = torch.zeros(original.numel() * 2, dtype=torch.uint8)[::2]
        elif kind == "semantics": target[key].zero_()
        else: target[key] = torch.Generator(device="cpu").manual_seed(999).get_state()
    elif case == "permutation_type": bad["order"]["permutation"] = tuple(bad["order"]["permutation"])
    elif case == "permutation_duplicate": bad["order"]["permutation"][0] = bad["order"]["permutation"][1]
    elif case == "permutation_bounds": bad["order"]["permutation"][0] = 5
    elif case == "permutation_bool": bad["order"]["permutation"][0] = True
    elif case == "permutation_wrong_valid": bad["order"]["permutation"].reverse()
    elif case.startswith("cursor_"):
        bad["order"]["cursor"] = {"cursor_bool": True, "cursor_repeat": 1, "cursor_skip": 3}[case]
    elif case == "epoch_bool": bad["order"]["epoch"] = False
    elif case == "epoch_wrong": bad["order"]["epoch"] = 1
    elif case.startswith("step_"):
        bad["metadata"]["next_update"] = {"step_bool": True, "step_repeat": 1, "step_skip": 3}[case]
    elif case == "training_false": bad["training"] = False
    elif case == "training_integer": bad["training"] = 1
    else: bad["metadata"]["unknown"] = 0
    # A late RNG refusal must not apply otherwise admissible changed weights.
    next(iter(bad["model"].values())).add_(0.125)
    target, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(target)
    target.eval()
    for parameter in target.parameters():
        parameter.grad = torch.full_like(parameter, 0.5)
    torch.manual_seed(333)
    before_model, before_opt = copy.deepcopy(target.state_dict()), copy.deepcopy(opt.state_dict())
    before_order, before_rng = copy.deepcopy(order.state_dict()), torch.get_rng_state().clone()
    gradients = [parameter.grad.clone() for parameter in target.parameters()]
    versions = [parameter._version for parameter in target.parameters()]
    with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as model_load:
        with patch.object(opt, "load_state_dict", wraps=opt.load_state_dict) as opt_load:
            with patch.object(order, "apply_validated", wraps=order.apply_validated) as order_load:
                with patch.object(torch, "set_rng_state", wraps=torch.set_rng_state) as rng_load:
                    with pytest.raises((ValueError, RuntimeError, TypeError)):
                        module.restore(bad, target, opt, order, metadata)
                    assert model_load.call_count == opt_load.call_count == order_load.call_count == rng_load.call_count == 0
    exact(target.state_dict(), before_model)
    exact(opt.state_dict(), before_opt)
    exact(order.state_dict(), before_order)
    exact(torch.get_rng_state(), before_rng)
    assert target.training is False
    assert [parameter._version for parameter in target.parameters()] == versions
    for parameter, gradient in zip(target.parameters(), gradients):
        exact(parameter.grad, gradient)
    module.restore(checkpoint, target, opt, order, metadata)
    assert target.training and all(parameter.grad is None for parameter in target.parameters())


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_each_update_matches_an_independent_momentum_recurrence(dtype_name):
    module = example()
    model, batches, tolerance = module.fixture(dtype_name)
    opt, order, previous = module.optimizer(model), module.BatchOrder(), {}
    torch.manual_seed(module.DROPOUT_SEED)
    for index in range(7):
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        record = module.step(model, opt, batches, order, index)
        for name, parameter in model.named_parameters():
            gradient = record["parameter_gradients"][name]
            buffer = gradient.clone() if index == 0 else 0.8 * previous[name] + gradient
            torch.testing.assert_close(opt.state[parameter]["momentum_buffer"], buffer, **tolerance)
            torch.testing.assert_close(parameter, before[name] - 0.03 * buffer, **tolerance)
            previous[name] = buffer.clone()


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_live_fault_controls_diverge_without_admitting_bad_checkpoints(dtype_name):
    module, initial, batches, _, metadata, checkpoint = prepared(dtype_name)
    full = copy.deepcopy(initial)
    torch.manual_seed(module.DROPOUT_SEED)
    records = module.run_steps(full, module.optimizer(full), batches, module.BatchOrder())
    original = copy.deepcopy(checkpoint)
    controls = module.controls(initial, batches, metadata, checkpoint, records)
    assert set(controls) == {"reset_dropout_rng", "omit_dropout_rng", "reset_order_generator",
                             "reset_optimizer", "repeat_cursor", "skip_cursor"}
    for name, result in controls.items():
        assert result["final_parameter_max_abs_error"] > result["divergence_threshold"], name
        assert result["live_state_bypass_after_valid_restore"] is True
    suffix = [row["batch_id"] for row in records[2:]]
    assert controls["reset_order_generator"]["batch_ids"][:3] == suffix[:3]
    assert controls["reset_order_generator"]["batch_ids"][3:] != suffix[3:]
    exact(checkpoint, original)


@pytest.mark.parametrize("case", ["wrong_digest", "corrupt", "oversize", "directory"])
def test_file_identity_is_checked_before_deserialization(tmp_path, case):
    module, initial, _, _, metadata, checkpoint = prepared()
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    payload = stream.getvalue()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    if case == "wrong_digest": digest = "0" * 64
    elif case == "corrupt": path.write_bytes(payload[:-9])
    elif case == "oversize": path.write_bytes(b"x" * (128 * 1024 + 1))
    else: path = tmp_path
    target, order = copy.deepcopy(initial), module.BatchOrder()
    opt = module.optimizer(target)
    before_rng, before_order = torch.get_rng_state().clone(), order.state_dict()
    before_model = copy.deepcopy(target.state_dict())
    with patch.object(torch, "load", wraps=torch.load) as load:
        with pytest.raises((ValueError, RuntimeError, TypeError, OSError)):
            module.load_checkpoint(path, digest, target, opt, order, metadata)
        assert load.call_count == 0
    exact(target.state_dict(), before_model)
    exact(torch.get_rng_state(), before_rng)
    exact(order.state_dict(), before_order)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_fresh_process_consumer_exact_replay_and_receipt(tmp_path, dtype_name):
    module = example()
    receipt = module.run(dtype_name, tmp_path / "artifacts")
    assert receipt["status"] == "passed"
    assert receipt["same_native_exact"] and receipt["same_native_max_abs_error"] == 0
    assert receipt["primitive_comparisons_passed"]
    assert receipt["producer_steps"] == 2 and receipt["resumed_steps"] == 5
    assert receipt["optimizer_steps"] == 7 and receipt["order_regenerations"] == 1
    assert len(set(receipt["process_ids"].values())) == 3
    assert [row["epoch"] for row in receipt["steps"]] == [0, 0, 0, 0, 0, 1, 1]
    for row in receipt["steps"]:
        assert row["default_rng_before_sha256"] != row["default_rng_after_sha256"]
    assert receipt["checkpoint_unchanged"] and receipt["loaded_state_storage_independent"]
    payload = (tmp_path / "artifacts" / "checkpoint.pt").read_bytes()
    assert len(payload) == receipt["checkpoint_bytes"] <= receipt["checkpoint_byte_limit"]
    assert sha256(payload).hexdigest() == receipt["checkpoint_sha256"]
