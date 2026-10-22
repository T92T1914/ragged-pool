"""Bounded changing-shape CUDA inference, with separate admission controls."""
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch


def load_example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dynamic_export_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


consumer = load_example("predict_cuda")


@pytest.fixture(scope="module", autouse=True)
def pinned_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128"
    assert torch.version.cuda == "12.8" and torch.cuda.is_available()
    assert not torch._dynamo.config.suppress_errors
    torch.cuda.set_device(0)
    torch._dynamo.reset()
    yield
    torch.cuda.synchronize(0)
    torch._dynamo.reset()


@pytest.fixture(scope="module", params=["float32", "float64"])
def artifact(request, tmp_path_factory):
    producer = load_example("export_cuda")
    model, training = producer.train_model(request.param)
    directory = tmp_path_factory.mktemp("cuda-dynamic-export") / request.param
    receipt = producer.create_artifact(model, directory, dtype_name=request.param,
        source_revision="0" * 40, source_tree="1" * 40, training=training, dynamic=True)
    predictor = consumer.load_artifact(directory, receipt["manifest_sha256"])
    return {"directory": directory, "receipt": receipt, "model": model,
            "dtype": request.param, "predictor": predictor}


def cases(artifact):
    return consumer.make_cases(artifact["dtype"], dynamic=True)


def base_case(artifact):
    return next(case for case in cases(artifact) if case["name"] == "base")


def snapshot(tensors):
    return [(value.detach().contiguous().view(torch.uint8).clone(), value._version,
             value.untyped_storage()._cdata, value.stride(), value.storage_offset())
            for value in tensors]


def unchanged(tensors, saved):
    for value, (data, version, storage, strides, offset) in zip(tensors, saved):
        assert torch.equal(value.detach().contiguous().view(torch.uint8), data)
        assert (value._version, value.untyped_storage()._cdata, value.stride(), value.storage_offset()) == (
            version, storage, strides, offset)


def copy_artifact(artifact, tmp_path):
    directory = tmp_path / "artifact"
    shutil.copytree(artifact["directory"], directory)
    return directory


def rewrite_manifest(directory, mutate):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def test_dynamic_graph_contract_and_trained_state_are_retained(artifact):
    predictor = artifact["predictor"]
    assert predictor.manifest["schema_version"] == 2
    assert predictor.manifest["contract"] == consumer.contract(artifact["dtype"], dynamic=True)
    assert consumer.graph_metadata(predictor.exported_program, dynamic=True) == consumer.graph_contract(True)
    assert artifact["receipt"]["graph"] == predictor.manifest["graph"]
    training = predictor.manifest["training"]
    assert training["optimizer_steps"] == training["reference_optimizer_steps"] == 2
    assert training["all_parameter_blocks_changed"]
    for name, trained in artifact["model"].state_dict().items():
        saved = predictor.exported_program.state_dict[name]
        assert torch.equal(saved.detach().view(torch.uint8), trained.detach().view(torch.uint8))
        assert saved.untyped_storage()._cdata != trained.untyped_storage()._cdata
    shapes = {(case["raw"].shape[0], len(case["boundaries"]) - 1) for case in cases(artifact)}
    assert {(2, 2), (2, 8), (32, 2), (32, 8)} <= shapes
    assert any(n == n2 and b != b2 for n, b in shapes for n2, b2 in shapes)
    assert any(b == b2 and n != n2 for n, b in shapes for n2, b2 in shapes)


def test_all_dynamic_outputs_match_primitive_canaries_and_preserve_ownership(artifact):
    predictor = artifact["predictor"]
    state = [*predictor.module.state_dict().values(), *predictor.state.values()]
    state_before, previous, outputs = snapshot(state), [], {}
    declared = predictor.manifest["expected"]
    fixtures = cases(artifact)
    assert len(fixtures) == len(declared) == len(consumer.DYNAMIC_CASE_SPECS)
    for case, canary in zip(fixtures, declared):
        raw, offsets = case["raw"], case["offsets"]
        before = snapshot((raw, offsets))
        result = predictor(raw, offsets)
        with torch.inference_mode():
            primitive = consumer.reference(raw, case["boundaries"], predictor.state)
        assert result[0].shape == (offsets.numel() - 1,)
        assert result[1].shape == (offsets.numel() - 1, 4)
        for label, value, expected in zip(("predictions", "pooled"), result, primitive):
            saved = torch.tensor(canary[label], dtype=raw.dtype, device=raw.device)
            torch.testing.assert_close(value, expected, **consumer.tolerance(raw.dtype))
            torch.testing.assert_close(value, saved, **consumer.tolerance(raw.dtype))
            assert value.is_inference() and not value.requires_grad and value.grad_fn is None
        occupied = [raw, offsets, *state, *previous, *result]
        assert len({value.untyped_storage()._cdata for value in occupied}) == len(occupied)
        empty = [row for row, (start, stop) in enumerate(zip(case["boundaries"], case["boundaries"][1:]))
                 if start == stop]
        assert torch.count_nonzero(result[1][empty]) == 0
        unchanged((raw, offsets), before)
        previous.extend(result)
        outputs[case["name"]] = result
    unchanged(state, state_before)
    assert not torch.equal(outputs["base"][1], outputs["changed_raw"][1])
    assert not torch.equal(outputs["base"][1], outputs["changed_partition"][1])
    assert not torch.equal(outputs["base"][1], outputs["return_pair"][1])
    for original, permuted in zip(outputs["base"], outputs["within_group_permutation"]):
        torch.testing.assert_close(original, permuted, **consumer.tolerance(artifact["dtype"]))


@pytest.mark.parametrize("fault", ["N0", "N1", "N33", "B0", "B1", "B9", "width", "rank",
    "offset_rank", "cpu", "meta", "raw_strided", "offset_strided", "offset_dtype", "dtype",
    "half", "sparse", "requires_grad", "negative_raw", "negative_offsets", "nan", "inf", "limit"])
def test_dynamic_original_input_refusal_precedes_graph_and_valid_recovery(artifact, fault, monkeypatch):
    case = base_case(artifact)
    raw, offsets = case["raw"], case["offsets"]
    if fault.startswith("N"):
        raw = torch.zeros(int(fault[1:]), 3, dtype=raw.dtype, device="cuda:0")
    elif fault.startswith("B"):
        offsets = torch.tensor([0] * int(fault[1:]) + [9], dtype=torch.int64, device="cuda:0")
    elif fault == "width": raw = torch.zeros(9, 4, dtype=raw.dtype, device="cuda:0")
    elif fault == "rank": raw = raw.reshape(3, 3, 3)
    elif fault == "offset_rank": offsets = offsets[:, None]
    elif fault == "cpu": raw = torch.zeros(9, 3, dtype=raw.dtype)
    elif fault == "meta": offsets = torch.empty(6, dtype=torch.int64, device="meta")
    elif fault == "raw_strided":
        raw = raw.t().contiguous().t()
        assert not raw.is_contiguous()
    elif fault == "offset_strided":
        offsets = torch.zeros(1, dtype=torch.int64, device="cuda:0").expand(6)
        assert offsets.stride() == (0,)
    elif fault == "offset_dtype": offsets = offsets.int()
    elif fault == "dtype": raw = raw.to(torch.float64 if raw.dtype == torch.float32 else torch.float32)
    elif fault == "half": raw = raw.half()
    elif fault == "sparse": raw = raw.to_sparse()
    elif fault == "requires_grad": raw.requires_grad_()
    elif fault == "negative_raw":
        raw = torch._neg_view((-raw).contiguous())
        assert raw.is_neg()
    elif fault == "negative_offsets":
        offsets = torch._neg_view(-offsets)
        assert offsets.is_neg()
    else: raw[0, 0] = {"nan": float("nan"), "inf": float("inf"), "limit": 6.01}[fault]

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Rejected original input reached the exported graph")

    with monkeypatch.context() as patch:
        patch.setattr(artifact["predictor"], "module", must_not_execute)
        with pytest.raises(ValueError):
            artifact["predictor"](raw, offsets)
    valid = base_case(artifact)
    actual = artifact["predictor"](valid["raw"], valid["offsets"])
    with torch.inference_mode():
        expected = consumer.reference(valid["raw"], valid["boundaries"], artifact["predictor"].state)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, **consumer.tolerance(artifact["dtype"]))


@pytest.mark.parametrize("bounds", [(1, 2, 2, 5, 7, 9), (0, 2, 1, 5, 7, 9),
                                    (0, 2, 2, 5, 7, 8), (0, 2, 2, 5, 10, 9)])
def test_saved_dynamic_graph_keeps_native_offset_refusal_and_recovery(artifact, bounds):
    predictor, case = artifact["predictor"], base_case(artifact)
    bad = torch.tensor(bounds, dtype=torch.int64, device="cuda:0")
    inputs = (case["raw"], bad)
    before = snapshot(inputs)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="offset"):
        predictor.module(*inputs)
    with pytest.raises(RuntimeError, match="offset"):
        predictor(*inputs)
    unchanged(inputs, before)
    result = predictor(case["raw"], case["offsets"])
    with torch.inference_mode():
        expected = consumer.reference(case["raw"], case["boundaries"], predictor.state)
    for value, reference in zip(result, expected):
        torch.testing.assert_close(value, reference, **consumer.tolerance(artifact["dtype"]))


@pytest.mark.parametrize("fault", ["N33", "B0", "B1", "B9", "width"])
def test_direct_saved_graph_enforces_upper_offsets_and_static_width_guards(artifact, fault):
    case = base_case(artifact)
    raw, offsets = case["raw"], case["offsets"]
    if fault == "N33":
        raw = torch.zeros(33, 3, dtype=raw.dtype, device="cuda:0")
        offsets = torch.tensor([0, 2, 2, 5, 7, 33], dtype=torch.int64, device="cuda:0")
    elif fault == "B0":
        raw = torch.zeros(0, 3, dtype=raw.dtype, device="cuda:0")
        offsets = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    elif fault == "width": raw = torch.zeros(9, 4, dtype=raw.dtype, device="cuda:0")
    else: offsets = torch.tensor([0] * int(fault[1:]) + [9], dtype=torch.int64, device="cuda:0")
    with torch.inference_mode(), pytest.raises((RuntimeError, AssertionError),
                                               match="Guard failed: |Expected input.*shape"):
        artifact["predictor"].module(raw, offsets)


@pytest.mark.parametrize("count", [0, 1])
def test_direct_module_min_two_exception_is_distinct_from_predictor_admission(artifact, count):
    # Pinned PyTorch skips lower-bound checks when min <= 2. The predictor still
    # rejects these shapes, while this direct module exercises native corners.
    predictor = artifact["predictor"]
    raw = torch.zeros(count, 3, dtype=consumer.DTYPES[artifact["dtype"]], device="cuda:0")
    boundaries = (0, 0, count)
    offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda:0")
    with pytest.raises(ValueError): predictor(raw, offsets)
    with torch.inference_mode():
        actual = predictor.module(raw, offsets)
        expected = consumer.reference(raw, boundaries, predictor.state)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, **consumer.tolerance(artifact["dtype"]))


@pytest.mark.parametrize("fault", ["schema_bool", "schema_unknown", "static_schema", "mode",
    "item_range", "group_range_bool", "graph_range", "coupled_graph", "canary_shape"])
def test_dynamic_schema_mode_and_range_refusal_precedes_deserialization(artifact, fault, tmp_path, monkeypatch):
    directory = copy_artifact(artifact, tmp_path)

    def mutate(manifest):
        if fault == "schema_bool": manifest["schema_version"] = True
        elif fault == "schema_unknown": manifest["schema_version"] = 3
        elif fault == "static_schema": manifest["schema_version"] = 1
        elif fault == "mode": manifest["contract"]["shape_mode"] = "fixed"
        elif fault == "item_range": manifest["contract"]["item_count_range"] = [2, 33]
        elif fault == "group_range_bool": manifest["contract"]["group_count_range"] = [2, True]
        elif fault == "graph_range": manifest["graph"]["range_constraints"]["O"] = [2, 9]
        elif fault == "coupled_graph": manifest["graph"]["input_shapes"]["offsets"] = ["N"]
        else: manifest["expected"][0]["pooled"] = []

    digest = rewrite_manifest(directory, mutate)

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("Rejected dynamic manifest reached deserialization")

    monkeypatch.setattr(consumer.torch.export, "load", must_not_deserialize)
    with pytest.raises(ValueError): consumer.load_artifact(directory, digest)


@pytest.mark.parametrize("fault", ["static", "items_range", "offset_range"])
def test_actual_loaded_graph_must_match_dynamic_manifest_before_predictor_construction(
        artifact, fault, tmp_path, monkeypatch):
    case = base_case(artifact)
    shapes = None if fault == "static" else (
        {0: torch.export.Dim("items", min=2, max=31 if fault == "items_range" else 32)},
        {0: torch.export.Dim("offset_count", min=3, max=8 if fault == "offset_range" else 9)})
    with torch.no_grad():
        alternative = torch.export.export(copy.deepcopy(artifact["model"]).eval(),
            (case["raw"], case["offsets"]), dynamic_shapes=shapes, strict=True)
    directory = copy_artifact(artifact, tmp_path)
    path = directory / "model.pt2"
    torch.export.save(alternative, path)
    digest = rewrite_manifest(directory, lambda manifest: manifest["files"].__setitem__(
        "model.pt2", {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}))

    def must_not_construct(*args, **kwargs):
        raise AssertionError("A mismatched loaded graph reached predictor construction")

    monkeypatch.setattr(consumer, "Predictor", must_not_construct)
    with pytest.raises(ValueError): consumer.load_artifact(directory, digest)


def test_changed_dynamic_canary_is_detected_by_consumer(artifact, tmp_path):
    directory = copy_artifact(artifact, tmp_path)
    digest = rewrite_manifest(directory, lambda manifest: manifest["expected"][-1]["predictions"].__setitem__(0, 55.0))
    predictor = consumer.load_artifact(directory, digest)
    with pytest.raises(AssertionError): consumer.exercise(predictor)
