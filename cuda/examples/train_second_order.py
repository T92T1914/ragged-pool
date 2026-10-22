"""Train an installed CUDA pool with an input-gradient-dependent objective.

Two SGD updates compare the full second-order route to independent GPU math.
A detached penalty has the same scalar value but loses its parameter gradient.
This is a small integration fixture, without a robustness or model-quality claim.
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
import ragged_pool_cuda.second_order as second_module
from ragged_pool_cuda import ragged_pool
from ragged_pool_cuda.second_order import ragged_pool_second_order


_PATH = Path(__file__).resolve().with_name('train_cuda.py')
_SPEC = importlib.util.spec_from_file_location('_second_cuda_eager_helper', _PATH)
if _SPEC is None or _SPEC.loader is None: raise ImportError('The adjacent train_cuda.py helper is required')
eager = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eager)


class SetRegressor(eager.SetRegressor):
    def forward(self, raw, offsets, boundaries, route):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        if route == 'primitive': pooled = eager.primitive_pool(values, logits, boundaries)
        elif route == 'second': pooled = ragged_pool_second_order(values, logits, offsets)
        elif route == 'ordinary': pooled = ragged_pool(values, logits, offsets)
        else: raise ValueError('Unknown pool route')
        return self.head(pooled).squeeze(-1).tanh(), pooled


def parameters(model): return {key: value.detach().clone() for key, value in model.named_parameters()}


def evaluate(model, batch, route, mode='proper'):
    if mode not in ('proper','detached','mse_only'): raise ValueError('Unknown penalty mode')
    model.zero_grad(set_to_none=True)
    raw = batch['raw'].detach().clone().requires_grad_()
    before = eager.snapshot({'raw':raw, 'offsets':batch['offsets'], 'targets':batch['targets']})
    predictions, pooled = model(raw, batch['offsets'], batch['boundaries'], route)
    mse = (predictions - batch['targets']).square().mean()
    first = torch.autograd.grad(predictions.sum(), raw, create_graph=route != 'ordinary', retain_graph=True)[0]
    penalty = first.square().mean()
    loss = mse if mode == 'mse_only' else mse + 0.5 * (penalty.detach() if mode == 'detached' else penalty)
    loss.backward()
    assert raw.grad is not None and all(p.grad is not None for p in model.parameters())
    gradients = {key:value.grad.detach().clone() for key,value in model.named_parameters()}
    outputs = {'predictions':predictions.detach().clone(), 'pooled':pooled.detach().clone(), 'mse':mse.detach().clone(),
               'input_gradient':first.detach().clone(), 'penalty':penalty.detach().clone(), 'loss':loss.detach().clone(),
               'raw_total_gradient':raw.grad.detach().clone()}
    assert all(bool(torch.isfinite(value).all()) for value in (*outputs.values(), *gradients.values()))
    empty = [i for i,(a,b) in enumerate(zip(batch['boundaries'],batch['boundaries'][1:])) if a == b]
    assert torch.equal(pooled[empty], torch.zeros_like(pooled[empty]))
    eager.unchanged({'raw':raw, 'offsets':batch['offsets'], 'targets':batch['targets']}, before)
    return outputs, gradients


def omitted_cotangent_control(device):
    values = torch.tensor([[0.4]], dtype=torch.float64, device=device, requires_grad=True)
    logits = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
    offsets = torch.tensor([0,1], dtype=torch.int64, device=device)
    y = ragged_pool_second_order(values, logits, offsets)
    first = torch.autograd.grad(0.5*y.square().sum(), values, create_graph=True)[0]
    full = torch.autograd.grad(first.sum(), values)[0]
    fixed = torch.ops.ragged_pool_cuda._double_backward.default(torch.ones_like(values),torch.zeros_like(logits),y.detach(),values.detach(),logits.detach(),offsets)[1]
    torch.testing.assert_close(full, torch.ones_like(values), rtol=0, atol=0)
    assert torch.equal(fixed, torch.zeros_like(values))
    return {'full_second_derivative':float(full), 'omitted_dg_result':float(fixed), 'difference':float((full-fixed).abs())}


def run(dtype_name, device_index=0, source_revision=None, source_tree=None):
    if device_index != 0: raise ValueError('This consumer qualifies CUDA0 only')
    for label in (source_revision,source_tree):
        if label is not None and not re.fullmatch('[0-9a-f]{40}',label): raise ValueError('Use full Git source labels')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    assert str(torch.__version__) == '2.9.1+cu128' and torch.version.cuda == '12.8' and torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    identity = eager.installed_identity()
    identity['cuda_helper_script_sha256'] = identity.pop('script_sha256')
    module = Path(second_module.__file__).resolve(strict=True)
    assert module.is_relative_to(Path(identity['environment_prefix']))
    identity['second_order_module_sha256'] = sha256(module.read_bytes()).hexdigest()
    identity['script_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    seeded, batches, fixture_hash = eager.fixture(dtype_name,device)
    initial = SetRegressor().to(device=device,dtype=eager.DTYPES[dtype_name])
    initial.load_state_dict(seeded.state_dict())
    initial_before = parameters(initial)
    models = {name:copy.deepcopy(initial) for name in ('second','primitive')}
    optimizers = {name:torch.optim.SGD(model.parameters(),lr=0.03,momentum=0,foreach=False,fused=False) for name,model in models.items()}
    ownership = {name:(id(model),id(optimizers[name]),tuple(id(x) for x in model.parameters())) for name,model in models.items()}
    tolerance = dict(rtol=3e-4,atol=3e-6) if dtype_name == 'float32' else dict(rtol=3e-10,atol=3e-12)
    steps = []
    for batch in batches:
        before = eager.snapshot({key:batch[key] for key in ('raw','offsets','targets')})
        # Controls begin at this step's exact parameters but never update them.
        detached, mse_only = copy.deepcopy(models['second']), copy.deepcopy(models['second'])
        outputs, gradients = {}, {}
        for name,model in models.items(): outputs[name],gradients[name] = evaluate(model,batch,name)
        control_output, control_grad = evaluate(detached,batch,'second','detached')
        mse_output, mse_grad = evaluate(mse_only,batch,'ordinary','mse_only')
        errors = {'outputs':eager.compare(outputs['second'],outputs['primitive'],tolerance),
                  'parameter_gradients':eager.compare(gradients['second'],gradients['primitive'],tolerance)}
        torch.testing.assert_close(control_output['loss'],outputs['second']['loss'],rtol=0,atol=0)
        torch.testing.assert_close(control_output['penalty'],outputs['second']['penalty'],rtol=0,atol=0)
        detached_mse = eager.compare(control_grad,mse_grad,tolerance)
        contribution = {key:eager.difference(value,control_grad[key]) for key,value in gradients['second'].items()}
        assert contribution['scorer.weight'] > 1e-8 and contribution['head.weight'] > 1e-8
        old = parameters(models['second'])
        for optimizer in optimizers.values(): optimizer.step()
        states = {name:parameters(model) for name,model in models.items()}
        errors['updated_parameters'] = eager.compare(states['second'],states['primitive'],tolerance)
        changes = {key:eager.difference(old[key],value) for key,value in states['second'].items()}
        assert all(value > 0 for value in changes.values())
        for name,model in models.items():
            assert ownership[name] == (id(model),id(optimizers[name]),tuple(id(x) for x in model.parameters()))
        eager.unchanged({key:batch[key] for key in ('raw','offsets','targets')},before)
        steps.append({'comparisons':errors, 'loss':float(outputs['second']['loss']), 'mse':float(outputs['second']['mse']),
                      'penalty':float(outputs['second']['penalty']), 'detached_scalar_loss_equal':True,
                      'detached_vs_mse_parameter_gradient_max_abs':detached_mse,
                      'penalty_parameter_gradient_max_abs':contribution, 'parameter_update_max_abs':changes})
    eager.compare(parameters(initial),initial_before,dict(rtol=0,atol=0))
    control = omitted_cotangent_control(device)
    assert eager.installed_identity()['native_library_sha256'] == identity['native_library_sha256']
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {'status':'passed', 'method':'native.cuda.eager.second_reverse.input_gradient_penalty',
            'dtype':dtype_name, 'device':str(device), 'device_name':properties.name,
            'compute_capability':[properties.major,properties.minor], 'python_version':sys.version.split()[0],
            'torch_version':str(torch.__version__), 'cuda_runtime_version':torch.version.cuda,
            'source_revision':source_revision, 'source_tree':source_tree, 'seed':eager.SEED, 'fixture_sha256':fixture_hash,
            'optimizer':'SGD', 'learning_rate':0.03, 'momentum':0, 'foreach':False, 'fused':False,
            'penalty_weight':0.5, 'prediction_head':'tanh(linear(pool))', 'optimizer_steps_per_route':2,
            'parameter_tensors':5, 'scalar_parameters':25, 'comparison_tolerance':tolerance, 'steps':steps,
            'omitted_cotangent_control':control, 'inputs_unchanged':True, 'initial_parameters_unchanged':True,
            'model_optimizer_ownership_unchanged':True, 'threads':torch.get_num_threads(),
            'interop_threads':torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype',choices=tuple(eager.DTYPES),default='float32')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--source-revision'); parser.add_argument('--source-tree'); parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()): parser.error('Use a new receipt path')
    text = json.dumps(run(args.dtype,args.device,args.source_revision,args.source_tree),indent=2,allow_nan=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream: stream.write(text)
    print(text,end='')


if __name__ == '__main__': main()
