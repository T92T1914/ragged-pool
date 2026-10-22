"""Train the installed native CUDA layer, then measure one input sensitivity.

Two matched SGD updates qualify ordinary reverse mode. The trained model's
input JVP is checked against primitive GPU automatic differentiation, native
centered differences and a native reverse-mode adjoint identity.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch
import ragged_pool_cuda.forward_ad as forward_module
from ragged_pool_cuda.forward_ad import ragged_pool_forward_ad


_PATH = Path(__file__).resolve().with_name('train_cuda.py')
_SPEC = importlib.util.spec_from_file_location('_forward_cuda_eager_helper', _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError('The adjacent train_cuda.py helper is required')
eager = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eager)


class SetRegressor(eager.SetRegressor):
    def forward(self, raw, offsets, boundaries=None):
        if boundaries is not None:
            return super().forward(raw, offsets, boundaries)
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = ragged_pool_forward_ad(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def clone_parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def run(dtype_name, device_index=0, source_revision=None, source_tree=None):
    if device_index != 0:
        raise ValueError('This consumer qualifies CUDA device0 only')
    for label in (source_revision, source_tree):
        if label is not None and not re.fullmatch('[0-9a-f]{40}', label):
            raise ValueError('Source labels must be full Git identities')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == '2.9.1+cu128' and torch.version.cuda == '12.8'
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    identity = eager.installed_identity()
    identity['cuda_helper_script_sha256'] = identity.pop('script_sha256')
    module_path = Path(forward_module.__file__).resolve(strict=True)
    assert module_path.is_relative_to(Path(identity['environment_prefix']))
    identity['forward_ad_module_sha256'] = sha256(module_path.read_bytes()).hexdigest()
    identity['script_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    initial, batches, fixture_hash = eager.fixture(dtype_name, device)
    initial_parameters = clone_parameters(initial)
    models = {'forward_ad': SetRegressor().to(device=device, dtype=eager.DTYPES[dtype_name]),
              'native': copy.deepcopy(initial), 'primitive': copy.deepcopy(initial)}
    models['forward_ad'].load_state_dict(initial.state_dict())
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03, momentum=0, foreach=False, fused=False)
                  for name, model in models.items()}
    ownership = {name: (id(model), id(optimizers[name]), tuple(id(x) for x in model.parameters()))
                 for name, model in models.items()}
    tolerance = dict(rtol=2e-4, atol=2e-6) if dtype_name == 'float32' else dict(rtol=2e-10, atol=2e-12)
    steps = []
    for batch in batches:
        before = eager.snapshot({key: batch[key] for key in ('raw', 'offsets', 'targets')})
        outputs, gradients = {}, {}
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            raw = batch['raw'].detach().clone().requires_grad_()
            outputs[name], gradients[name] = eager.evaluate(model, raw, batch, name == 'primitive')
        comparisons = {name: {'outputs': eager.compare(outputs['forward_ad'], outputs[name], tolerance),
                             'parameter_gradients': eager.compare(gradients['forward_ad'], gradients[name], tolerance)}
                       for name in ('native', 'primitive')}
        old = clone_parameters(models['forward_ad'])
        for optimizer in optimizers.values(): optimizer.step()
        states = {name: clone_parameters(model) for name, model in models.items()}
        for name in comparisons:
            comparisons[name]['updated_parameters'] = eager.compare(states['forward_ad'], states[name], tolerance)
        changes = {key: eager.difference(old[key], value) for key, value in states['forward_ad'].items()}
        assert all(value > 0 for value in changes.values())
        for name, model in models.items():
            assert ownership[name] == (id(model), id(optimizers[name]), tuple(id(x) for x in model.parameters()))
        eager.unchanged({key: batch[key] for key in ('raw', 'offsets', 'targets')}, before)
        steps.append({'comparisons': comparisons, 'parameter_update_max_abs': changes})
    eager.compare(clone_parameters(initial), initial_parameters, dict(rtol=0, atol=0))
    batch = batches[-1]
    raw, offsets = batch['raw'], batch['offsets']
    direction = torch.linspace(-0.25, 0.4, raw.numel(), dtype=raw.dtype, device=device).reshape_as(raw)
    model = models['forward_ad']
    parameter_before = clone_parameters(model)
    gradient_objects = {name: value.grad for name, value in model.named_parameters()}
    gradient_before = {name: value.clone() for name, value in gradient_objects.items()}
    inputs_before = eager.snapshot({'raw': raw, 'offsets': offsets, 'direction': direction})
    # This query asks only for first-order forward sensitivity, not its reverse.
    with torch.no_grad():
        primal, tangent = torch.func.jvp(lambda x: model(x, offsets), (raw,), (direction,))
        expected, expected_t = torch.func.jvp(lambda x: models['primitive'](x, offsets, batch['boundaries']), (raw,), (direction,))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(raw, direction)
            explicit = tuple(torch.autograd.forward_ad.unpack_dual(x) for x in model(dual, offsets))
        h = 1e-3 if dtype_name == 'float32' else 1e-5
        plus, minus = models['native'](raw + h * direction, offsets), models['native'](raw - h * direction, offsets)
        centered = tuple((a - b) / (2 * h) for a, b in zip(plus, minus))
    primal_errors = eager.compare(dict(zip(('predictions', 'pooled'), primal)), dict(zip(('predictions', 'pooled'), expected)), tolerance)
    tangent_errors = eager.compare(dict(zip(('predictions', 'pooled'), tangent)), dict(zip(('predictions', 'pooled'), expected_t)), tolerance)
    explicit_errors = eager.compare(dict(zip(('predictions', 'pooled'), tangent)), dict(zip(('predictions', 'pooled'), (x.tangent for x in explicit))), tolerance)
    difference_tolerance = dict(rtol=1e-2, atol=1e-4) if dtype_name == 'float32' else dict(rtol=2e-7, atol=2e-9)
    difference_errors = eager.compare(dict(zip(('predictions', 'pooled'), tangent)), dict(zip(('predictions', 'pooled'), centered)), difference_tolerance)
    native_raw = raw.detach().clone().requires_grad_()
    native_outputs = models['native'](native_raw, offsets)
    cotangents = tuple(torch.linspace(-0.6, 0.8, value.numel(), dtype=raw.dtype, device=device).reshape_as(value) for value in native_outputs)
    reverse = torch.autograd.grad(native_outputs, native_raw, grad_outputs=cotangents)[0]
    lhs = sum((t.double() * g).sum() for t, g in zip(tangent, cotangents))
    rhs = (reverse.double() * direction).sum()
    torch.testing.assert_close(lhs, rhs, **tolerance)
    assert any(float(value.abs().max()) > 0 for value in tangent)
    empty = [i for i, (a, b) in enumerate(zip(batch['boundaries'], batch['boundaries'][1:])) if a == b]
    assert torch.equal(tangent[0][empty], torch.zeros_like(tangent[0][empty]))
    assert torch.equal(tangent[1][empty], torch.zeros_like(tangent[1][empty]))
    eager.compare(clone_parameters(model), parameter_before, dict(rtol=0, atol=0))
    for name, parameter in model.named_parameters():
        assert parameter.grad is gradient_objects[name]
        torch.testing.assert_close(parameter.grad, gradient_before[name], rtol=0, atol=0)
    eager.unchanged({'raw': raw, 'offsets': offsets, 'direction': direction}, inputs_before)
    assert eager.installed_identity()['native_library_sha256'] == identity['native_library_sha256']
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {'status': 'passed', 'method': 'native.cuda.first_order_jvp.trained_input_sensitivity',
            'dtype': dtype_name, 'device': str(device), 'device_name': properties.name,
            'compute_capability': [properties.major, properties.minor], 'python_version': sys.version.split()[0],
            'torch_version': str(torch.__version__), 'cuda_runtime_version': torch.version.cuda,
            'source_revision': source_revision, 'source_tree': source_tree, 'seed': eager.SEED,
            'fixture_sha256': fixture_hash, 'optimizer': 'SGD', 'learning_rate': 0.03,
            'momentum': 0, 'foreach': False, 'fused': False, 'optimizer_steps_per_route': 2, 'steps': steps,
            'sensitivity': {'primal_max_abs': primal_errors, 'primitive_jvp_max_abs': tangent_errors,
                            'explicit_dual_max_abs': explicit_errors, 'centered_difference_max_abs': difference_errors,
                            'difference_step': h, 'adjoint_absolute_difference': float((lhs-rhs).abs()),
                            'empty_prediction_and_pool_tangents_zero': True},
            'parameter_and_grad_fields_unchanged_during_sensitivity': True, 'inputs_unchanged': True,
            'initial_parameters_unchanged': True, 'model_optimizer_ownership_unchanged': True,
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=tuple(eager.DTYPES), default='float32')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--source-revision')
    parser.add_argument('--source-tree')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()): parser.error('Use a new output receipt')
    result = run(args.dtype, args.device, args.source_revision, args.source_tree)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream: stream.write(text)
    print(text, end='')


if __name__ == '__main__': main()
