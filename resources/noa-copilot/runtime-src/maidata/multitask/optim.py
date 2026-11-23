"""AdamW or Muon with internal AdamW for embeddings, heads and gates.

Packed experts are orthogonalized as independent matrices, never flattened
across experts. No optimizer touches a parameter with grad=None.
"""
import math
import torch


def zeropower(update, steps=5):
    x = update.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        x = 3.4445*x + (-4.775*a + 2.0315*(a@a)) @ x
    return x.transpose(-2, -1) if transposed else x


class MuonAdamW(torch.optim.Optimizer):
    """A single serializable optimizer containing both update algorithms."""
    def __init__(self, groups, *, weight_decay=0.01, betas=(0.9,0.95)):
        super().__init__(groups, dict(lr=2e-4, weight_decay=weight_decay,
                                     betas=tuple(betas), momentum=0.95))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                p.mul_(1-lr*wd)
                if group['algorithm']=='muon':
                    if not state:
                        state['momentum_buffer']=torch.zeros_like(p)
                    buf=state['momentum_buffer']
                    buf.lerp_(g,1-group['momentum'])
                    direction=g.lerp(buf,group['momentum'])
                    # A convolution is one output-channel x receptive-field
                    # matrix. Packed experts already carry a separate leading
                    # matrix dimension and must retain that independent batch.
                    matrix=direction.flatten(1) if group.get('flatten_convolution') else direction
                    update=zeropower(matrix).reshape_as(p)
                    aspect=(matrix.size(-1)/matrix.size(-2) if group.get('matrix_transposed') else
                            matrix.size(-2)/matrix.size(-1))
                    scale=math.sqrt(max(1.,aspect))
                    p.add_(update.to(p.dtype),alpha=-lr*scale)
                else:
                    b1,b2=group['betas']
                    if not state:
                        state.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                    state['step']+=1
                    state['exp_avg'].lerp_(g,1-b1)
                    state['exp_avg_sq'].mul_(b2).addcmul_(g,g,value=1-b2)
                    denom=state['exp_avg_sq'].sqrt()/(1-b2**state['step'])**0.5+1e-8
                    p.addcdiv_(state['exp_avg'],denom,value=-lr/(1-b1**state['step']))
        return loss


def make_optimizer(model,config):
    groups={}
    excluded=set()
    no_decay=set()
    convolution=set()
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):
            convolution.add(id(module.weight))
        if isinstance(module,torch.nn.Embedding):
            excluded.update(id(p) for p in module.parameters())
            no_decay.update(id(p) for p in module.parameters())
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm,
                               torch.nn.GroupNorm, torch.nn.modules.batchnorm._NormBase)):
            no_decay.update(id(p) for p in module.parameters(recurse=False))
            excluded.update(id(p) for p in module.parameters(recurse=False))
        if 'heads.' in name or 'rhythm_head' in name or 'moment_type_head' in name or '.router.' in name:
            excluded.update(id(p) for p in module.parameters(recurse=False))
    for name,p in model.named_parameters():
        algorithm='muon' if (config['optimizer']=='muon' and p.ndim>=2 and not name.endswith('bias') and id(p) not in excluded
                              and not any(x in name for x in ('heads.','rhythm_head','moment_type_head','gates.','gate_','router.'))) else 'adamw'
        role=('macro' if name.startswith(('encoder.mert_projection.', 'encoder.muq_projection.',
                                          'encoder.gate_macro.')) else
              'yourmt3' if name.startswith('encoder.yourmt3.') else
              'encoder' if name.startswith('encoder.') else 'decoder')
        transposed=name.endswith(('shared_up','shared_down'))
        flatten=algorithm=='muon' and id(p) in convolution
        decay=p.ndim>=2 and not name.endswith('bias') and id(p) not in no_decay
        groups.setdefault((algorithm,role,transposed,decay,flatten),[]).append(p)
    # Both algorithms use the same scheduler factor for a role. Rescale Muon's
    # coefficient so lr * weight_decay matches AdamW at every scheduled step.
    packed=[dict(params=params,algorithm=alg,role=role,matrix_transposed=transposed,
                 flatten_convolution=flatten,
                 lr=config['muon_learning_rate'] if alg=='muon' else config['learning_rate'],
                 weight_decay=(config['weight_decay'] *
                     (config['learning_rate']/config['muon_learning_rate'] if alg=='muon' else 1.)) if decay else 0.)
            for (alg,role,transposed,decay,flatten),params in groups.items()]
    if config['optimizer']=='muon':
        return MuonAdamW(packed,weight_decay=config['weight_decay'],betas=config['betas'])
    return torch.optim.AdamW(packed,lr=config['learning_rate'],betas=tuple(config['betas']),
                             weight_decay=config['weight_decay'])


def make_scheduler(optimizer,config,steps_by_epoch):
    total=sum(steps_by_epoch[:config['scheduler_epochs']])
    thaw=sum(steps_by_epoch[:config['encoder_freeze_epochs']])
    def factor(step,role):
        warm=config['warmup_steps']
        value=min(1.,(step+1)/max(warm,1)) if step<warm else (
            0.5*(1+math.cos(math.pi*min(1.,(step-warm)/max(total-warm,1)))))
        if role=='encoder':
            if step<thaw:
                return 0.
            if config['encoder_warmup_steps']:
                value*=min(1.,(step-thaw+1)/config['encoder_warmup_steps'])
        elif role=='yourmt3':
            unfreeze=max(thaw,config['warmup_steps'])
            if step<unfreeze:
                return 0.
            ramp=max(1,config.get('encoder_warmup_steps') or 500)
            value*=min(1.,(step-unfreeze+1)/ramp)*config['yourmt3_lr_scale']
        return value
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
        [lambda step,role=g['role']:factor(step,role) for g in optimizer.param_groups])
