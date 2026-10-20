"""Exercise the strict model artifact and its narrower inference contract."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch


def load_example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ragged_pool_export_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


producer = load_example("export_sets")
consumer = producer.predict
study = producer.learn


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    """Train two fixture steps, then use the real producer and serialization path."""
    model = study.make_model(11, "native")
    split = study.make_split(836, 16, 1, 3)
    values, offsets, labels = study.pack_batch(split, range(16))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(values, offsets)
        torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels).backward()
        optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    directory = tmp_path_factory.mktemp("strict-export") / "artifact"
    strict_arguments, weights_arguments = [], []
    original_export, original_load = torch.export.export, torch.load

    def record_export(*args, **kwargs):
        strict_arguments.append(kwargs.get("strict"))
        return original_export(*args, **kwargs)

    def record_load(*args, **kwargs):
        if args and isinstance(args[0], (str, Path)) and Path(args[0]).name == "weights.pt":
            weights_arguments.append(dict(kwargs))
        return original_load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.export, "export", record_export)
        patch.setattr(torch, "load", record_load)
        receipt = producer.create_artifact(
            model, directory, source_revision="0" * 40, wheel_sha256="0" * 64,
            training={"mode": "test_fixture", "seed": 11, "updates": 2,
                      "selected_update": 2, "data": {}, "validation": {}},
        )
    assert strict_arguments == [True]
    assert weights_arguments == [{"weights_only": True, "map_location": "cpu"}]
    predictor = consumer.load_artifact(directory, receipt["manifest_sha256"])
    return {"directory": directory, "receipt": receipt, "model": model,
            "predictor": predictor, "state": dict(model.state_dict())}


def test_strict_trained_export_keeps_native_forward_and_reloaded_parameters(trained_artifact):
    artifact = trained_artifact
    program = artifact["predictor"].exported_program
    operations = [str(node.target) for node in program.graph_module.graph.nodes if node.op == "call_function"]
    assert operations.count("ragged_pool.pool.default") == 1
    assert not any("backward" in operation.lower() for operation in operations)
    assert set(program.state_dict) == set(artifact["state"])
    for name, value in artifact["state"].items():
        torch.testing.assert_close(program.state_dict[name], value, rtol=0, atol=0)
    assert artifact["receipt"]["graph"] == {
        "pool_operator": "ragged_pool::pool", "pool_count": 1, "backward_count": 0,
    }


def valid_case(name):
    if name == "empty_items":
        values, boundaries = torch.empty(0, 4), [0, 0, 0]
    elif name == "one_item":
        values, boundaries = torch.tensor([[0.5, -1.0, 2.0, 0.25]]), [0, 0, 1]
    elif name == "maximum_items_groups":
        values = torch.linspace(-2, 2, 4096 * 4).reshape(4096, 4)
        boundaries = list(range(0, 4097, 32))
    else:
        values = torch.linspace(-1, 1, 28).reshape(7, 4)
        boundaries = [0, 2, 2, 3, 7, 7]
    return values.contiguous(), torch.tensor(boundaries, dtype=torch.int64)


@pytest.mark.parametrize("case", ["empty_items", "one_item", "maximum_items_groups", "mixed_empty"])
def test_saved_dynamic_graph_runs_declared_boundaries(trained_artifact, case):
    values, offsets = valid_case(case)
    with torch.inference_mode():
        expected = trained_artifact["model"](values, offsets)
        actual = trained_artifact["predictor"](values, offsets)
        direct = trained_artifact["predictor"].module(values, offsets)
    assert actual.shape == (offsets.numel() - 1,) and bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(direct, expected, rtol=2e-5, atol=2e-6)


def test_reloaded_graph_uses_new_contents_and_offsets(trained_artifact):
    values, offsets = valid_case("mixed_empty")
    changed_values = (values * -0.5 + 0.75).contiguous()
    changed_offsets = torch.tensor([0, 1, 4, 4, 6, 7], dtype=torch.int64)
    outputs = []
    with torch.inference_mode():
        for features, boundaries in ((values, offsets), (changed_values, offsets), (values, changed_offsets)):
            actual = trained_artifact["predictor"].module(features, boundaries)
            expected = trained_artifact["model"](features, boundaries)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            outputs.append(actual)
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])


@pytest.mark.parametrize("offsets", [[0, 1, 3], [0, 3, 2, 4]], ids=["wrong-final", "decreasing"])
def test_direct_loaded_graph_keeps_native_offset_admission(trained_artifact, offsets):
    values = torch.zeros(4, 4)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="offsets"):
        trained_artifact["predictor"].module(values, torch.tensor(offsets, dtype=torch.int64))


@pytest.mark.parametrize("groups", [0, 1])
def test_export_subset_rejects_small_group_counts_with_default_guards(trained_artifact, groups):
    values = torch.empty(0, 4) if groups == 0 else torch.zeros(1, 4)
    offsets = torch.tensor([0] if groups == 0 else [0, 1], dtype=torch.int64)
    with pytest.raises((RuntimeError, ValueError)):
        trained_artifact["predictor"](values, offsets)
    # The eager operator accepts these counts. A direct exported call must still
    # reject them through the saved offset-count guard, without the wrapper.
    with torch.inference_mode(), pytest.raises(AssertionError, match=r"Guard failed: offsets\.size\(\)\[0\] >= 3"):
        trained_artifact["predictor"].module(values, offsets)


def test_consumer_rejects_original_bad_inputs_before_graph_call(trained_artifact, monkeypatch):
    values, offsets = torch.zeros(4, 4), torch.tensor([0, 2, 4])
    bad_inputs = [
        (torch.zeros(4097, 4), torch.tensor([0, 0, 4097])),
        (torch.zeros(4, 5), offsets),
        (values.double(), offsets),
        (values, offsets.to(torch.int32)),
        (torch.zeros(4, 4), torch.zeros(130, dtype=torch.int64)),
        (values, torch.tensor([0, 3, 2, 4])),
        (values, torch.tensor([0, 2, 3])),
        (torch._neg_view(-torch.zeros(8, 4))[::2], offsets),
        (values, torch._neg_view(-torch.tensor([0, 0, 2, 0, 4, 0]))[::2]),
    ]
    for bad_value in (float("nan"), float("inf"), -float("inf"), 6.01):
        features = values.clone()
        features[0, 0] = bad_value
        bad_inputs.append((features, offsets))

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Invalid raw input reached the exported graph")

    monkeypatch.setattr(trained_artifact["predictor"], "module", must_not_execute)
    for features, boundaries in bad_inputs:
        with pytest.raises((RuntimeError, ValueError)):
            trained_artifact["predictor"](features, boundaries)


@pytest.mark.parametrize("case", ["keys", "shape", "dtype", "nonfinite", "not_tensor", "bool"])
def test_restricted_weight_reload_rejects_safe_malformed_mappings(case, trained_artifact, tmp_path, monkeypatch):
    state = {name: value.detach().clone() for name, value in trained_artifact["state"].items()}
    name = next(iter(state))
    if case == "keys":
        del state[name]
    elif case == "shape":
        state[name] = state[name].reshape(-1)
    elif case == "dtype":
        state[name] = state[name].double()
    elif case == "nonfinite":
        state[name].view(-1)[0] = float("nan")
    elif case == "not_tensor":
        state[name] = 0
    else:
        state[name] = state[name].to(torch.bool)
    path = tmp_path / "safe-malformed-weights.pt"
    torch.save(state, path)
    calls = []
    original = torch.load

    def restricted_load(*args, **kwargs):
        calls.append(dict(kwargs))
        assert kwargs.get("weights_only") is True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", restricted_load)
    with pytest.raises(ValueError):
        producer.read_weights(path, trained_artifact["state"])
    assert calls == [{"weights_only": True, "map_location": "cpu"}]


@pytest.mark.parametrize("case", ["manifest_digest", "program_bytes", "weights_bytes", "schema", "environment", "duplicate_key"])
def test_artifact_rejections_precede_deserialization(case, trained_artifact, tmp_path, monkeypatch):
    directory = tmp_path / "artifact"
    shutil.copytree(trained_artifact["directory"], directory)
    manifest_path = directory / "manifest.json"
    expected_digest = trained_artifact["receipt"]["manifest_sha256"]
    if case == "manifest_digest":
        manifest_path.write_bytes(b"This is deliberately invalid JSON")
    elif case in ("program_bytes", "weights_bytes"):
        path = directory / ("model.pt2" if case == "program_bytes" else "weights.pt")
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        if case == "duplicate_key":
            text = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(text.replace("{", '{"schema_version": 1,', 1), encoding="utf-8")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if case == "schema":
                manifest["schema_version"] = 999
            else:
                manifest["environment"]["torch_version"] = "0.0.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("Rejected artifact reached torch.export.load")

    monkeypatch.setattr(torch.export, "load", must_not_deserialize)
    if case == "manifest_digest":
        def must_not_parse(*args, **kwargs):
            raise AssertionError("Unverified manifest was parsed")

        monkeypatch.setattr(consumer.json, "loads", must_not_parse)
    with pytest.raises((RuntimeError, ValueError)):
        consumer.load_artifact(directory, expected_digest)
