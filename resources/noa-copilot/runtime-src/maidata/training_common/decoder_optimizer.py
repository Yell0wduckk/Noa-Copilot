"""AdamW state handling and precision helpers for the joint recipe."""

from contextlib import nullcontext


import torch
from maidata.training_common.optimizer import adamw_betas

PATH_KEYS = tuple('decoder_train_'+k for k in ('checkpoint','init_checkpoint','p0_cache','local_p0_cache','dp_cache','gt_plan_cache','window_cache','train_plan','validation_plan','chart_cache',
    'physical_cache','audio_cache','mel_root','source_root','geometry','slide_data','route_distances','output'))

def optimizer(parameters, config):
    parameters = [p for p in parameters if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=config['decoder_train_learning_rate'],
                             betas=adamw_betas(config, 'decoder_train'),
                             weight_decay=config['decoder_train_weight_decay'],
                             fused=config['decoder_train_fused_optimizer'] and parameters[0].device.type == 'cuda')

def restore_optimizer(opt, state):
    # The current implementation selects the kernel, while the checkpoint owns
    # learning rates, moments and steps. Set flags BEFORE load_state_dict so
    # PyTorch places fused step tensors on the parameter device correctly.
    groups = [dict(saved) for saved in state['param_groups']]
    for group, current in zip(groups, opt.param_groups):
        group['fused'] = current['fused']
        if current['fused']:
            group['foreach'] = False
    opt.load_state_dict(dict(state, param_groups=groups))

def autocast(device, config):
    from maidata.nn.precision import enable_tf32
    enable_tf32(device)
    return (torch.autocast('cuda',dtype=torch.bfloat16)
            if torch.device(device).type == 'cuda' and config['decoder_train_precision'] == 'bf16' else nullcontext())
