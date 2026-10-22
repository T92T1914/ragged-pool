"""Check the owned fixed-shape CUDA export and its separate admission boundaries."""
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch


def load_example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_export_test_{name}", path)
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
    torch.cuda.set_device(0)


@pytest.fixture(scope="module", params=["float32", "float64"])
def artifact(request, tmp_path_factory):
    producer = load_example("export_cuda")
    model, training = producer.train_model(request.param)
    directory = tmp_path_factory.mktemp("cuda-export") / request.param
    receipt = producer.create_artifact(
        model, directory, dtype_name=request.param, source_revision="0" * 40,
        source_tree="1" * 40, training=training)
    predictor = consumer.load_artifact(directory, receipt["manifest_sha256"])
    return {"directory": directory, "receipt": receipt, "model": model,
            "dtype": request.param, "predictor": predictor}


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


def test_strict_saved_graph_keeps_native_forward_and_updated_parameters(artifact):
    predictor = artifact["predictor"]
    assert consumer.graph_metadata(predictor.exported_program) == {
        "pool_operator": "ragged_pool_cuda::pool", "pool_count": 1, "backward_count": 0}
    assert predictor.manifest["training"]["optimizer_steps"] == 2
    assert predictor.manifest["training"]["reference_optimizer_steps"] == 2
    assert predictor.manifest["training"]["all_parameter_blocks_changed"] is True
    assert set(predictor.exported_program.state_dict) == set(consumer.STATE_SHAPES)
    for name, value in artifact["model"].state_dict().items():
        torch.testing.assert_close(predictor.exported_program.state_dict[name], value, rtol=0, atol=0)


def test_inference_matches_primitive_canary_and_unchanged_state(artifact):
    results = consumer.exercise(artifact["predictor"])
    assert [case["name"] for case in results] == list(consumer.CASE_NAMES)
    assert all(value >= 0 for case in results for key, value in case.items() if key != "name")


def test_new_contents_partitions_permutations_and_empty_groups(artifact):
    predictor = artifact["predictor"]
    results = []
    for case in consumer.make_cases(artifact["dtype"]):
        actual = predictor(case["raw"], case["offsets"])
        with torch.inference_mode():
            expected = consumer.reference(case["raw"], case["boundaries"], predictor.state)
        for observed, independent in zip(actual, expected):
            torch.testing.assert_close(observed, independent, **consumer.tolerance(artifact["dtype"]))
        results.append(actual)
    assert not torch.equal(results[0][1], results[1][1])
    assert not torch.equal(results[0][1], results[2][1])
    for actual, permuted in zip(results[0], results[-1]):
        torch.testing.assert_close(actual, permuted, **consumer.tolerance(artifact["dtype"]))
    assert torch.count_nonzero(results[4][1][[0, 1, 3, 4]]) == 0


@pytest.mark.parametrize("fault", ["cpu", "meta", "raw_shape", "offset_shape", "raw_strided",
                                   "offset_strided", "offset_dtype", "raw_dtype", "negative_view",
                                   "nan", "positive_inf", "negative_inf", "raw_limit"])
def test_original_input_refusal_precedes_graph(artifact, fault, monkeypatch):
    case = consumer.make_cases(artifact["dtype"])[0]
    raw, offsets = case["raw"], case["offsets"]
    if fault == "cpu":
        raw = torch.zeros(9, 3, dtype=raw.dtype)
    elif fault == "meta":
        offsets = torch.empty(6, dtype=torch.int64, device="meta")
    elif fault == "raw_shape":
        raw = raw[:8]
    elif fault == "offset_shape":
        offsets = offsets[:5]
    elif fault == "raw_strided":
        raw = raw.t().contiguous().t()
    elif fault == "offset_strided":
        offsets = torch.zeros(1, dtype=torch.int64, device="cuda:0").expand(6)
    elif fault == "offset_dtype":
        offsets = offsets.int()
    elif fault == "raw_dtype":
        raw = raw.to(torch.float64 if raw.dtype == torch.float32 else torch.float32)
    elif fault == "negative_view":
        raw = torch._neg_view((-raw).contiguous())
    else:
        raw[0, 0] = {"nan": float("nan"), "positive_inf": float("inf"),
                     "negative_inf": -float("inf"), "raw_limit": 6.01}[fault]

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Rejected input reached the exported graph")

    monkeypatch.setattr(artifact["predictor"], "module", must_not_execute)
    with pytest.raises(ValueError):
        artifact["predictor"](raw, offsets)


@pytest.mark.parametrize("bounds", [(0, 2, 1, 5, 7, 9), (0, 2, 2, 5, 7, 8),
                                    (1, 2, 2, 5, 7, 9), (0, 2, 2, 5, 10, 9)])
def test_direct_saved_graph_keeps_native_offset_refusal_and_recovers(artifact, bounds):
    predictor = artifact["predictor"]
    case = consumer.make_cases(artifact["dtype"])[0]
    bad = torch.tensor(bounds, dtype=torch.int64, device="cuda:0")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="offset"):
        predictor.module(case["raw"], bad)
    with pytest.raises(RuntimeError, match="offset"):
        predictor(case["raw"], bad)
    result = predictor(case["raw"], case["offsets"])
    with torch.inference_mode():
        expected = consumer.reference(case["raw"], case["boundaries"], predictor.state)
    for actual, primitive in zip(result, expected):
        torch.testing.assert_close(actual, primitive, **consumer.tolerance(artifact["dtype"]))


@pytest.mark.parametrize("which", ["raw", "offsets"])
def test_direct_saved_graph_has_fixed_shape_guards(artifact, which):
    case = consumer.make_cases(artifact["dtype"])[0]
    raw = case["raw"][:8] if which == "raw" else case["raw"]
    offsets = case["offsets"][:5] if which == "offsets" else case["offsets"]
    with torch.inference_mode(), pytest.raises((RuntimeError, AssertionError), match="shape|size|Expected|Guard"):
        artifact["predictor"].module(raw, offsets)


@pytest.mark.parametrize("fault", ["manifest_digest", "program_digest", "manifest_oversize", "program_oversize",
                                   "missing", "extra", "duplicate_key", "schema", "environment", "training",
                                   "contract_type", "weights_shape", "weights_nonfinite", "canary_shape"])
def test_rejected_artifact_stops_before_deserialization(artifact, fault, tmp_path, monkeypatch):
    directory = copy_artifact(artifact, tmp_path)
    expected = artifact["receipt"]["manifest_sha256"]
    if fault == "manifest_digest":
        (directory / "manifest.json").write_bytes(b"unverified invalid JSON")
    elif fault == "program_digest":
        with (directory / "model.pt2").open("ab") as stream:
            stream.write(b"changed")
    elif fault == "manifest_oversize":
        (directory / "manifest.json").write_bytes(b" " * (consumer.MANIFEST_LIMIT + 1))
    elif fault == "program_oversize":
        (directory / "model.pt2").write_bytes(b" " * (consumer.PROGRAM_LIMIT + 1))
    elif fault == "missing":
        (directory / "model.pt2").unlink()
    elif fault == "extra":
        (directory / "unexpected.txt").write_text("extra", encoding="utf-8")
    elif fault == "duplicate_key":
        path = directory / "manifest.json"
        path.write_text(path.read_text(encoding="utf-8").replace("{", '{"schema_version": 1,', 1), encoding="utf-8")
        expected = sha256(path.read_bytes()).hexdigest()
    else:
        def change(manifest):
            if fault == "schema":
                manifest["schema_version"] = True
            elif fault == "environment":
                manifest["environment"]["torch_version"] = "0.0.0"
            elif fault == "training":
                manifest["training"]["optimizer_steps"] = 0
            elif fault == "contract_type":
                manifest["contract"]["contiguous"] = 1
            elif fault == "weights_shape":
                manifest["weights"]["encoder.bias"] = [0.0]
            elif fault == "weights_nonfinite":
                manifest["weights"]["encoder.bias"][0] = True
            else:
                manifest["expected"][0]["pooled"] = []
        expected = rewrite_manifest(directory, change)

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("Rejected artifact reached torch.export.load")

    monkeypatch.setattr(consumer.torch.export, "load", must_not_deserialize)
    if fault == "manifest_digest":
        def must_not_parse(*args, **kwargs):
            raise AssertionError("Unverified manifest reached JSON parsing")
        monkeypatch.setattr(consumer.json, "loads", must_not_parse)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        consumer.load_artifact(directory, expected)


@pytest.mark.parametrize("which", ["file", "directory"])
def test_artifact_symlinks_are_refused_if_host_allows_creation(artifact, which, tmp_path, monkeypatch):
    directory = copy_artifact(artifact, tmp_path)
    if which == "file":
        target = tmp_path / "real-program.pt2"
        (directory / "model.pt2").rename(target)
        link = directory / "model.pt2"
    else:
        target, link = directory, tmp_path / "linked-artifact"
    try:
        link.symlink_to(target, target_is_directory=which == "directory")
    except OSError as error:
        pytest.skip(f"Host does not allow ordinary symlink creation: {error}")

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("Symlink artifact reached deserialization")

    monkeypatch.setattr(consumer.torch.export, "load", must_not_deserialize)
    with pytest.raises(ValueError):
        consumer.load_artifact(link if which == "directory" else directory,
                               artifact["receipt"]["manifest_sha256"])


def test_loader_deserializes_the_exact_verified_bytes_once(artifact, tmp_path, monkeypatch):
    directory = copy_artifact(artifact, tmp_path)
    path = directory / "model.pt2"
    verified, original, calls = path.read_bytes(), torch.export.load, []

    def change_after_verification(buffer, *args, **kwargs):
        assert isinstance(buffer, io.BytesIO)
        calls.append(buffer.getvalue())
        path.write_bytes(b"changed after verification")
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(consumer.torch.export, "load", change_after_verification)
    predictor = consumer.load_artifact(directory, artifact["receipt"]["manifest_sha256"])
    assert calls == [verified]
    case = consumer.make_cases(artifact["dtype"])[0]
    actual = predictor(case["raw"], case["offsets"])
    expected = artifact["predictor"](case["raw"], case["offsets"])
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_declared_weights_must_equal_loaded_state(artifact, tmp_path):
    directory = copy_artifact(artifact, tmp_path)
    expected = rewrite_manifest(directory, lambda manifest: manifest["weights"]["head.bias"].__setitem__(0, 3.5))
    with pytest.raises(AssertionError):
        consumer.load_artifact(directory, expected)


def test_changed_canary_is_detected_by_independent_consumer(artifact, tmp_path):
    directory = copy_artifact(artifact, tmp_path)
    expected = rewrite_manifest(directory, lambda manifest: manifest["expected"][0]["predictions"].__setitem__(0, 55.0))
    predictor = consumer.load_artifact(directory, expected)
    with pytest.raises(AssertionError):
        consumer.exercise(predictor)


def test_graph_admission_is_separate_from_mathematical_comparisons():
    for targets in ([torch.ops.ragged_pool_cuda._backward.default],
                    [torch.ops.ragged_pool_cuda.pool.default, torch.ops.ragged_pool_cuda.pool.default], []):
        nodes = [SimpleNamespace(op="call_function", target=target) for target in targets]
        exported = SimpleNamespace(graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes)))
        with pytest.raises(ValueError):
            consumer.graph_metadata(exported)
