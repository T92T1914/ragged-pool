"""Trained padded serialization, explicit admission and verified-byte loading."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch


def load_example(name):
    path = Path(__file__).resolve().parents[1] / 'examples' / (name + '.py')
    spec = importlib.util.spec_from_file_location('n8_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


producer = load_example('export_padded')
consumer = producer.predict


@pytest.fixture(scope='module', params=['float32', 'float64'])
def artifact(request, tmp_path_factory):
    dtype_name = request.param
    model, training = producer.train_model(dtype_name)
    directory = tmp_path_factory.mktemp('padded-export-' + dtype_name) / 'artifact'
    strict_calls, weight_calls = [], []
    original_export, original_load = torch.export.export, torch.load

    def export(*args, **kwargs):
        strict_calls.append(kwargs.get('strict'))
        return original_export(*args, **kwargs)

    def load(*args, **kwargs):
        if Path(sys._getframe(1).f_code.co_filename).name in ('export_padded.py', 'predict_padded.py'):
            weight_calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.export, 'export', export)
        patch.setattr(torch, 'load', load)
        receipt = producer.create_artifact(
            model, directory, dtype_name=dtype_name,
            source_revision='0' * 40, wheel_sha256='0' * 64, training=training,
        )
    assert strict_calls == [True]
    assert weight_calls and all(x.get('weights_only') is True and x.get('map_location') == 'cpu'
                                for x in weight_calls)
    predictor = consumer.load_artifact(directory, receipt['manifest_sha256'])
    return {'dtype': dtype_name, 'directory': directory, 'receipt': receipt,
            'model': model, 'training': training, 'predictor': predictor}


def test_updated_parameters_survive_strict_export_and_saved_reload(artifact):
    trained, predictor = artifact['model'].state_dict(), artifact['predictor']
    assert artifact['training']['optimizer_steps'] == 2
    assert artifact['training']['status'] == 'passed'
    assert set(trained) == set(predictor.state)
    assert set(trained) == set(predictor.exported_program.state_dict)
    for name, value in trained.items():
        torch.testing.assert_close(predictor.state[name], value, rtol=0, atol=0)
        torch.testing.assert_close(predictor.exported_program.state_dict[name], value, rtol=0, atol=0)
    graph = consumer.graph_metadata(predictor.exported_program)
    assert graph['pool_count'] == 1 and graph['backward_count'] == 0
    assert graph['pool_operator'] == 'ragged_pool::padded'


def test_variable_shapes_masks_and_padding_match_functional_reference(artifact):
    predictor = artifact['predictor']
    tolerance = {'rtol': 2e-5, 'atol': 2e-6} if artifact['dtype'] == 'float32' else {'rtol': 1e-10, 'atol': 1e-12}
    shapes = set()
    for case in consumer.make_cases(artifact['dtype']):
        raw, mask = case['raw'], case['mask']
        snapshots = [(x.clone(), x._version) for x in (raw, mask)]
        actual = predictor(raw, mask)
        expected = consumer.functional_reference(raw, mask, predictor.state)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, **tolerance)
            assert bool(torch.isfinite(left).all())
        assert torch.count_nonzero(actual[1][~mask.any(dim=1)]) == 0
        for tensor, (before, version) in zip((raw, mask), snapshots):
            torch.testing.assert_close(tensor, before, rtol=0, atol=0, equal_nan=True)
            assert tensor._version == version
        shapes.add(tuple(raw.shape[:2]))
    assert {(2, 2), (2, 128), (32, 2), (32, 128)} <= shapes


def test_consumer_checks_its_envelope_before_invoking_the_graph(artifact):
    predictor = consumer.load_artifact(artifact['directory'], artifact['receipt']['manifest_sha256'])
    dtype = getattr(torch, artifact['dtype'])
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError('Rejected input reached the saved graph')

    predictor.module = forbidden
    for b, t in ((0, 4), (1, 4), (33, 4), (4, 0), (4, 1), (4, 129)):
        with pytest.raises((ValueError, RuntimeError), match='B|T|envelope|shape|range|bounds'):
            predictor(torch.zeros(b, t, 3, dtype=dtype), torch.ones(b, t, dtype=torch.bool))
    assert not calls


@pytest.mark.parametrize('fault', ['keys', 'shape', 'dtype', 'nonfinite'])
def test_invalid_tensor_state_is_rejected_before_graph_load(artifact, fault, tmp_path, monkeypatch):
    directory = tmp_path / 'bad-state'
    shutil.copytree(artifact['directory'], directory)
    state = {key: value.clone() for key, value in artifact['predictor'].state.items()}
    if fault == 'keys':
        state.pop('encoder.bias')
    elif fault == 'shape':
        state['encoder.bias'] = state['encoder.bias'][:1].clone()
    elif fault == 'dtype':
        state['encoder.bias'] = state['encoder.bias'].to(torch.int64)
    else:
        state['encoder.bias'][0] = float('nan')
    path = directory / 'weights.pt'
    torch.save(state, path)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['weights.pt'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'size_bytes': path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest))
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(torch.export, 'load', lambda *args: calls.append(args))
    with pytest.raises((ValueError, RuntimeError)):
        consumer.load_artifact(directory, expected)
    assert not calls


def test_loader_uses_exact_verified_buffers_after_files_change(artifact, tmp_path, monkeypatch):
    directory = tmp_path / 'verified-buffers'
    shutil.copytree(artifact['directory'], directory)
    program_bytes = (directory / 'model.pt2').read_bytes()
    state_bytes = (directory / 'weights.pt').read_bytes()
    original_graph_load, original_state_load = torch.export.load, torch.load
    seen = []

    def state_load(buffer, *args, **kwargs):
        if Path(sys._getframe(1).f_code.co_filename).name == 'predict_padded.py':
            assert isinstance(buffer, io.BytesIO) and buffer.getvalue() == state_bytes
            assert kwargs.get('weights_only') is True and kwargs.get('map_location') == 'cpu'
            (directory / 'weights.pt').write_bytes(b'changed after verification')
            seen.append('weights')
        return original_state_load(buffer, *args, **kwargs)

    def graph_load(buffer, *args, **kwargs):
        assert isinstance(buffer, io.BytesIO) and buffer.getvalue() == program_bytes
        (directory / 'model.pt2').write_bytes(b'changed after verification')
        seen.append('graph')
        return original_graph_load(buffer, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', state_load)
    monkeypatch.setattr(torch.export, 'load', graph_load)
    predictor = consumer.load_artifact(directory, artifact['receipt']['manifest_sha256'])
    assert set(seen) == {'weights', 'graph'}
    case = consumer.make_cases(artifact['dtype'])[0]
    expected = consumer.functional_reference(case['raw'], case['mask'], predictor.state)
    for actual, reference in zip(predictor(case['raw'], case['mask']), expected):
        torch.testing.assert_close(actual, reference)


def test_fresh_predictor_needs_no_training_script_or_original_model_class(artifact, tmp_path):
    output = tmp_path / 'fresh-predictions.json'
    script = Path(__file__).resolve().parents[1] / 'examples' / 'predict_padded.py'
    process = subprocess.run([sys.executable, '-I', str(script), '--directory', str(artifact['directory']),
                              '--manifest-sha256', artifact['receipt']['manifest_sha256'],
                              '--output', str(output)], cwd=tmp_path, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert process.returncode == 0, process.stdout
    receipt = json.loads(output.read_text())
    assert receipt['status'] == 'passed'
    assert receipt['training_modules_loaded'] == []
    assert len(receipt['cases']) == len(consumer.make_cases(artifact['dtype']))


def test_exported_size_one_guard_is_separate_from_consumer_admission(artifact):
    predictor = artifact['predictor']
    dtype = getattr(torch, artifact['dtype'])
    # Pinned 2.9.1 skips lower-bound checks at min<=2. This is a positive
    # witness of the guard gap, while the supported consumer still refuses it.
    for b, t in ((1, 4), (4, 1)):
        raw, mask = torch.zeros(b, t, 3, dtype=dtype), torch.ones(b, t, dtype=torch.bool)
        with torch.inference_mode():
            predictions, pooled = predictor.module(raw, mask)
        assert predictions.shape == (b,) and pooled.shape == (b, 4)
        with pytest.raises((ValueError, RuntimeError)):
            predictor(raw, mask)


@pytest.mark.parametrize('bad', ['raw_dtype', 'mask_dtype', 'mask_shape', 'width', 'raw_stride',
                                 'mask_stride', 'meta', 'sparse', 'nan', 'inf', 'range'])
def test_bad_original_inputs_cannot_reach_saved_graph(artifact, bad):
    predictor = consumer.load_artifact(artifact['directory'], artifact['receipt']['manifest_sha256'])
    dtype = getattr(torch, artifact['dtype'])
    raw, mask = torch.zeros(4, 6, 3, dtype=dtype), torch.ones(4, 6, dtype=torch.bool)
    if bad == 'raw_dtype':
        raw = raw.to(torch.float32 if dtype == torch.float64 else torch.float64)
    elif bad == 'mask_dtype':
        mask = mask.to(torch.int64)
    elif bad == 'mask_shape':
        mask = mask[:, :-1].contiguous()
    elif bad == 'width':
        raw = torch.zeros(4, 6, 4, dtype=dtype)
    elif bad == 'raw_stride':
        raw = torch.zeros(4, 6, 6, dtype=dtype)[..., ::2]
    elif bad == 'mask_stride':
        mask = torch.ones(4, 12, dtype=torch.bool)[:, ::2]
    elif bad == 'meta':
        raw = torch.empty(4, 6, 3, dtype=dtype, device='meta')
    elif bad == 'sparse':
        raw = raw.to_sparse()
    else:
        raw[0, 0, 0] = {'nan': float('nan'), 'inf': float('inf'), 'range': 6.01}[bad]
    calls = []
    predictor.module = lambda *args: calls.append(args)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        predictor(raw, mask)
    assert not calls


def test_ignored_invalid_raw_content_remains_admitted(artifact):
    predictor = artifact['predictor']
    dtype = getattr(torch, artifact['dtype'])
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, 1] = True
    raw = torch.zeros(4, 6, 3, dtype=dtype)
    baseline = predictor(raw, mask)
    for ignored in (float('nan'), float('inf'), -float('inf'), 1e30):
        changed = raw.clone()
        changed[~mask] = ignored
        for left, right in zip(predictor(changed, mask), baseline):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_same_selected_items_survive_extra_padding_and_moved_holes(artifact):
    predictor = artifact['predictor']
    dtype = getattr(torch, artifact['dtype'])
    tolerance = {'rtol': 2e-5, 'atol': 2e-6} if dtype == torch.float32 else {'rtol': 1e-10, 'atol': 1e-12}
    active = torch.linspace(-0.8, 0.9, 18, dtype=dtype).reshape(6, 3)
    outputs = []
    for t, positions, padding in ((4, ([0, 3], [], [2], [0, 1, 3]), float('nan')),
                                  (9, ([2, 8], [], [6], [2, 4, 8]), float('inf'))):
        raw = torch.full((4, t, 3), padding, dtype=dtype)
        mask = torch.zeros(4, t, dtype=torch.bool)
        cursor = 0
        for row, selected in enumerate(positions):
            raw[row, selected] = active[cursor:cursor + len(selected)]
            mask[row, selected] = True
            cursor += len(selected)
        outputs.append(predictor(raw, mask))
        permutation = torch.arange(t - 1, -1, -1)
        permuted = predictor(raw[:, permutation].contiguous(), mask[:, permutation].contiguous())
        for actual, expected in zip(permuted, outputs[-1]):
            torch.testing.assert_close(actual, expected, **tolerance)
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize('fault', ['manifest', 'program', 'weights', 'extra_file', 'duplicate_key',
                                   'schema', 'environment', 'dtype', 'size', 'training'])
def test_rejected_bundle_is_stopped_before_either_deserializer(artifact, fault, tmp_path, monkeypatch):
    directory = tmp_path / 'bundle'
    shutil.copytree(artifact['directory'], directory)
    expected = artifact['receipt']['manifest_sha256']
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    if fault == 'manifest':
        path.write_bytes(path.read_bytes() + b' ')
    elif fault in ('program', 'weights'):
        target = directory / ('model.pt2' if fault == 'program' else 'weights.pt')
        target.write_bytes(target.read_bytes() + b'changed')
    elif fault == 'extra_file':
        (directory / 'unexpected.txt').write_text('unexpected')
    else:
        if fault == 'duplicate_key':
            text = path.read_text().replace('{', '{"schema_version":1,', 1)
        else:
            if fault == 'schema':
                manifest['schema_version'] = 2
            elif fault == 'environment':
                manifest['environment']['native_library_sha256'] = '0' * 64
            elif fault == 'dtype':
                manifest['dtype'] = 'float16'
            elif fault == 'size':
                manifest['files']['model.pt2']['size_bytes'] += 1
            elif fault == 'training':
                manifest['training']['optimizer_steps'] = 0
            text = json.dumps(manifest)
        path.write_text(text)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Rejected bytes reached a deserializer')

    monkeypatch.setattr(torch.export, 'load', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        consumer.load_artifact(directory, expected)
    assert not calls
