"""State admission and trajectory controls for the example-local checkpoint."""
import copy
from hashlib import sha256
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'examples' / 'resume_training.py'


def example():
    spec = importlib.util.spec_from_file_location('resume_training_tests', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(dtype_name='float64'):
    module = example()
    initial, batches, tolerance = module.fixture(dtype_name)
    model = copy.deepcopy(initial)
    opt = module.optimizer(model)
    metadata = module.metadata(dtype_name, initial, batches)
    module.run_steps(model, opt, batches, start=0, stop=2)
    checkpoint = module.make_checkpoint(model, opt, metadata)
    return module, initial, batches, tolerance, model, opt, metadata, checkpoint


def exact(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
        assert first.dtype == second.dtype and first.shape == second.shape
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second): exact(a, b)
    else:
        assert first == second


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value: yield from tensors(item)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_restored_native_suffix_is_exact_and_checkpoint_storage_is_owned(dtype_name):
    module, initial, batches, _, trained, opt, metadata, checkpoint = prepared(dtype_name)
    payload_before = copy.deepcopy(checkpoint)
    payload_versions = [t._version for t in tensors(checkpoint)]
    full = copy.deepcopy(initial)
    full_opt = module.optimizer(full)
    full_steps = module.run_steps(full, full_opt, batches)
    target, target_opt = copy.deepcopy(initial), None
    target_opt = module.optimizer(target)
    parameter_ids = tuple(id(p) for p in target.parameters())
    module.restore(checkpoint, target, target_opt, metadata)
    assert tuple(id(p) for p in target.parameters()) == parameter_ids
    exact(target.state_dict(), trained.state_dict())
    exact(target_opt.state_dict(), opt.state_dict())
    payload_storage = {t.untyped_storage().data_ptr() for t in tensors(checkpoint)}
    restored = list(target.parameters()) + [s['momentum_buffer'] for s in target_opt.state.values()]
    assert all(t.untyped_storage().data_ptr() not in payload_storage for t in restored)
    suffix = module.run_steps(target, target_opt, batches, start=2, stop=5)
    exact(suffix, full_steps[2:])
    exact(target.state_dict(), full.state_dict())
    exact(target_opt.state_dict(), full_opt.state_dict())
    exact(checkpoint, payload_before)
    assert [t._version for t in tensors(checkpoint)] == payload_versions


@pytest.mark.parametrize('case', [
    'cursor_bool', 'cursor_repeat', 'cursor_skip', 'metadata_extra', 'metadata_missing',
    'model_missing', 'model_extra', 'model_shape', 'model_dtype', 'model_nonfinite',
    'model_grad', 'model_view', 'model_stride', 'optimizer_missing', 'optimizer_extra',
    'missing_buffer', 'extra_buffer', 'momentum_shape', 'momentum_dtype', 'momentum_nan',
    'momentum_view', 'momentum_grad', 'id_bool', 'id_reordered', 'id_missing',
    'state_key_bool', 'extra_group', 'option_momentum', 'option_lr', 'option_extra',
])
def test_bad_checkpoint_refused_before_any_target_state_application(case):
    module, initial, batches, _, _, _, metadata, checkpoint = prepared()
    bad = copy.deepcopy(checkpoint)
    names = list(bad['model'])
    parameter = names[0]
    group = bad['optimizer']['param_groups'][0]
    buffer = bad['optimizer']['state'][0]['momentum_buffer']
    if case.startswith('cursor_'):
        # The exact metadata cursor key is deliberately inspected by contract.
        cursor = next(k for k in bad['metadata'] if k in {'completed_steps', 'next_step', 'next_update', 'cursor'})
        bad['metadata'][cursor] = {'cursor_bool': True, 'cursor_repeat': 1, 'cursor_skip': 3}[case]
    elif case == 'metadata_extra': bad['metadata']['unknown'] = 0
    elif case == 'metadata_missing': bad['metadata'].pop(next(iter(bad['metadata'])))
    elif case == 'model_missing': bad['model'].pop(parameter)
    elif case == 'model_extra': bad['model']['unknown'] = torch.zeros(1)
    elif case == 'model_shape': bad['model'][parameter] = bad['model'][parameter].reshape(-1)
    elif case == 'model_dtype': bad['model'][parameter] = bad['model'][parameter].float()
    elif case == 'model_nonfinite': bad['model'][parameter].view(-1)[0] = float('inf')
    elif case == 'model_grad': bad['model'][parameter].requires_grad_()
    elif case == 'model_view':
        original = bad['model'][parameter]
        bad['model'][parameter] = torch.zeros(original.numel() + 1, dtype=original.dtype)[1:].reshape(original.shape)
    elif case == 'model_stride': bad['model'][parameter] = bad['model'][parameter].t().contiguous().t()
    elif case == 'optimizer_missing': bad['optimizer'].pop('state')
    elif case == 'optimizer_extra': bad['optimizer']['unknown'] = 0
    elif case == 'missing_buffer': bad['optimizer']['state'][0].pop('momentum_buffer')
    elif case == 'extra_buffer': bad['optimizer']['state'][0]['unknown'] = torch.zeros(1)
    elif case == 'momentum_shape': bad['optimizer']['state'][0]['momentum_buffer'] = buffer.reshape(-1)
    elif case == 'momentum_dtype': bad['optimizer']['state'][0]['momentum_buffer'] = buffer.float()
    elif case == 'momentum_nan': buffer.view(-1)[0] = float('nan')
    elif case == 'momentum_view':
        bad['optimizer']['state'][0]['momentum_buffer'] = torch.zeros(buffer.numel() + 1, dtype=buffer.dtype)[:buffer.numel()].reshape(buffer.shape)
    elif case == 'momentum_grad': buffer.requires_grad_()
    elif case == 'id_bool': group['params'][0] = False
    elif case == 'id_reordered': group['params'][2:4] = group['params'][2:4][::-1]
    elif case == 'id_missing': group['params'].pop()
    elif case == 'state_key_bool': bad['optimizer']['state'][False] = bad['optimizer']['state'].pop(0)
    elif case == 'extra_group': bad['optimizer']['param_groups'].append(copy.deepcopy(group))
    elif case == 'option_momentum': group['momentum'] = 0.0
    elif case == 'option_lr': group['lr'] = True
    else: group['unknown'] = 0
    target = copy.deepcopy(initial)
    target_opt = module.optimizer(target)
    # A late optimizer error must also refuse an otherwise admissible changed model.
    next(iter(bad['model'].values())).detach().add_(0.125)
    before_model, before_opt = copy.deepcopy(target.state_dict()), copy.deepcopy(target_opt.state_dict())
    ids = tuple(id(p) for p in target.parameters())
    versions = tuple(p._version for p in target.parameters())
    with patch.object(target, 'load_state_dict', wraps=target.load_state_dict) as model_load:
        with patch.object(target_opt, 'load_state_dict', wraps=target_opt.load_state_dict) as opt_load:
            with pytest.raises((ValueError, RuntimeError, TypeError)):
                module.restore(bad, target, target_opt, metadata)
            assert model_load.call_count == opt_load.call_count == 0
    exact(target.state_dict(), before_model)
    exact(target_opt.state_dict(), before_opt)
    assert tuple(id(p) for p in target.parameters()) == ids
    assert tuple(p._version for p in target.parameters()) == versions
    assert all(p.grad is None for p in target.parameters())
    module.restore(checkpoint, target, target_opt, metadata)


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_momentum_and_cursor_controls_have_decisive_parameter_drift(dtype_name):
    module, initial, batches, tolerance, _, _, metadata, checkpoint = prepared(dtype_name)
    full = copy.deepcopy(initial)
    full_opt = module.optimizer(full)
    module.run_steps(full, full_opt, batches)
    drifts = []
    for case in ('optimizer_reset', 'seed_restart', 'repeat_cursor', 'skip_cursor'):
        target = copy.deepcopy(initial)
        opt = module.optimizer(target)
        if case == 'optimizer_reset': target.load_state_dict(checkpoint['model'])
        elif case != 'seed_restart': module.restore(checkpoint, target, opt, metadata)
        start, stop = (1, 4) if case == 'repeat_cursor' else ((3, 5) if case == 'skip_cursor' else (2, 5))
        module.run_steps(target, opt, batches, start=start, stop=stop)
        drift = max(float((a - b).abs().max().detach()) for a, b in zip(target.parameters(), full.parameters()))
        assert drift > (1e-7 if dtype_name == 'float32' else 1e-10), case
        drifts.append(drift)
    assert len(drifts) == 4


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_optimizer_steps_match_an_independently_written_momentum_recurrence(dtype_name):
    module = example()
    model, batches, tolerance = module.fixture(dtype_name)
    opt = module.optimizer(model)
    previous_buffers = {}
    for index, batch in enumerate(batches):
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        record = module.step(model, opt, batch, index)
        for name, parameter in model.named_parameters():
            gradient = record['parameter_gradients'][name].clone()
            expected_buffer = gradient if index == 0 else 0.8 * previous_buffers[name] + gradient
            torch.testing.assert_close(opt.state[parameter]['momentum_buffer'], expected_buffer, **tolerance)
            torch.testing.assert_close(parameter, before[name] - 0.03 * expected_buffer, **tolerance)
            if index == 0:
                torch.testing.assert_close(opt.state[parameter]['momentum_buffer'], gradient, rtol=0, atol=0)
            previous_buffers[name] = expected_buffer.clone()


@pytest.mark.parametrize('case', ['wrong_digest', 'corrupt', 'oversize', 'directory'])
def test_file_admission_precedes_loading_and_target_mutation(tmp_path, case):
    module, initial, _, _, _, _, metadata, checkpoint = prepared()
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    data = stream.getvalue()
    path = tmp_path / 'state.pt'
    path.write_bytes(data)
    expected = sha256(data).hexdigest()
    if case == 'wrong_digest': expected = '0' * 64
    elif case == 'corrupt': path.write_bytes(data[:-9])
    elif case == 'oversize': path.write_bytes(b'x' * (128 * 1024 + 1))
    else:
        path = tmp_path
    target, opt = copy.deepcopy(initial), None
    opt = module.optimizer(target)
    before = copy.deepcopy(target.state_dict())
    with patch.object(torch, 'load', wraps=torch.load) as load:
        with pytest.raises((ValueError, RuntimeError, TypeError, OSError)):
            module.load_checkpoint(path, expected, target, opt, metadata)
        assert load.call_count == 0
    exact(target.state_dict(), before)
