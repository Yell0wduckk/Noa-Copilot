"""One CE pass, shared song encoding, full logical-batch normalization."""
from contextlib import nullcontext
from functools import lru_cache
import torch
from maidata.multitask.batching import (
    cross_entropy,
    logical_batch,
    logical_split_batch,
    moment_type_loss,
    rhythm_loss,
)

def autocast(config,device):
    return (torch.autocast(device_type='cuda',dtype=torch.bfloat16)
            if device.type=='cuda' and config['precision']=='bfloat16' else nullcontext())

@lru_cache(maxsize=2)
def _live_frontends(mert, muq, device):
    from maidata.multitask.frontends import load_mert, load_muq
    return load_mert(mert,device),load_muq(muq,device)


def _branch_ce(logits, batch, targets, mask, config, tm_cfg, tm_enabled, post_warmup, components):
    return cross_entropy(
        logits, targets, mask, batch['legal_mask'],
        stage=config['stage'], task_ids=batch['task_ids'],
        current_slots=batch.get('current_slots'), pseudo=batch.get('pseudo'),
        previous_pitches=batch['previous_pitches'],
        target_pseudo=batch.get('target_pseudo'), target_groups=batch['target_groups'],
        target_velocity_db=batch['target_velocity_db'],
        velocity_sigma=config['velocity_sigma'], components=components,
        moment_only=batch.get('moment_only') if tm_enabled else None,
        residual_neg_mask=batch.get('residual_neg_mask') if tm_enabled else None,
        residual_neg_weight=float(tm_cfg.get('residual_neg_weight', 1.)),
        ll_ignore_threshold=float(tm_cfg.get('ll_ignore_threshold', 0.)),
        ll_ignore_weight=float(tm_cfg.get('ll_ignore_weight', .1)),
        apply_ll_ignore=bool(tm_enabled and post_warmup),
    )


def step_batch(model,groups,optimizer,scheduler,config):
    device=next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    count=sum(int(c['encoded'].loss_mask[1:].sum()) for g in groups for c in g['charts'])
    frozen=any(g.get('role')=='encoder' and g['lr']==0 for g in optimizer.param_groups)
    top_enabled=not frozen and scheduler.last_epoch>=config['warmup_steps']
    if model.encoder.yourmt3.freeze_top_block == top_enabled:
        model.encoder.yourmt3.set_top_trainable(top_enabled)
    macro_provider = None
    if model.config.macro_enabled:
        from maidata.multitask.audio import fast_resample
        from maidata.multitask.frontends import extract_macro
        opt = config['synthetic']

        def macro_provider(wave):
            mert, muq = _live_frontends(opt.get('mert_model','m-a-p/MERT-v1-330M'),
                                        opt.get('muq_model','OpenMuQ/MuQ-large-msd-iter'), device)
            with autocast(config,device):
                return extract_macro(fast_resample(wave,44100,24000), mert, muq,
                                     opt.get('frontend_batch',4), output_device=device)
    tm_cfg=config.get('tag_moment') or {}
    tm_enabled=bool(tm_cfg.get('enabled',False))
    post_warmup=scheduler.last_epoch>=config['warmup_steps']
    components={}
    if tm_enabled:
        audio_batch, full_pack, tag_pack = logical_split_batch(
            groups, config['stage'], device, model.encoder.frontend, macro_provider
        )
    else:
        batch, targets, mask = logical_batch(
            groups, config['stage'], device, model.encoder.frontend, macro_provider
        )
        audio_batch, full_pack, tag_pack = batch, (batch, targets, mask), None

    with autocast(config,device):
        streams=model.encode_batch(audio_batch)
        audio_kv=model.prepare_audio_kv(streams)
        if full_pack is not None and tag_pack is not None:
            # Length-homogeneous Branch B (long full charts) + Branch A (short
            # moment_only charts) sharing one song encode & one audio_kv, with
            # zero cross-branch padding bloat.
            fb, ft, fm = full_pack
            full_comp = {}
            full_logits = model.decode(fb, streams, audio_kv=audio_kv)
            full_loss = _branch_ce(full_logits, fb, ft, fm, config, tm_cfg, tm_enabled, post_warmup, full_comp)
            rhythm = config.get('rhythm_loss_weight', .2) * rhythm_loss(model, fb, streams, components=components)
            mtype = 0.
            if float(tm_cfg.get('moment_type_loss_weight', 0.)) > 0.:
                mtype = float(tm_cfg['moment_type_loss_weight']) * moment_type_loss(
                    model, fb,
                    ll_ignore_threshold=float(tm_cfg.get('type_ll_ignore_threshold', .85)),
                    ll_ignore_weight=float(tm_cfg.get('ll_ignore_weight', .1)),
                    apply_ll_ignore=bool(post_warmup),
                    components=components,
                )
            tb, tt, tm_mask = tag_pack
            tag_comp = {}
            tag_logits = model.decode(tb, streams, audio_kv=audio_kv)
            tag_loss = _branch_ce(tag_logits, tb, tt, tm_mask, config, tm_cfg, tm_enabled, post_warmup, tag_comp)
            loss = full_loss + tag_loss
            total_sup = full_comp['supervision_count'] + tag_comp['supervision_count']
            components['main_ce'] = loss.detach()
            components['supervision_count'] = total_sup
            components['uncond_ce'] = full_comp.get(
                'uncond_ce', full_loss.detach() / full_comp['supervision_count'].clamp_min(1)
            )
            if 'tag_moment_ce' in tag_comp:
                components['tag_moment_ce'] = tag_comp['tag_moment_ce']
        else:
            active_pack = full_pack if full_pack is not None else tag_pack
            ab, at, am = active_pack
            logits = model.decode(ab, streams, audio_kv=audio_kv)
            loss = _branch_ce(logits, ab, at, am, config, tm_cfg, tm_enabled, post_warmup, components)
            rhythm = config.get('rhythm_loss_weight', .2) * rhythm_loss(model, ab, streams, components=components)
            mtype = 0.
            if tm_enabled and full_pack is not None and float(tm_cfg.get('moment_type_loss_weight', 0.)) > 0.:
                mtype = float(tm_cfg['moment_type_loss_weight']) * moment_type_loss(
                    model, ab,
                    ll_ignore_threshold=float(tm_cfg.get('type_ll_ignore_threshold', .85)),
                    ll_ignore_weight=float(tm_cfg.get('ll_ignore_weight', .1)),
                    apply_ll_ignore=bool(post_warmup),
                    components=components,
                )
        support=components['supervision_count'].clamp_min(1)
        normalized=loss/support+rhythm+mtype
    normalized.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
    optimizer.step()
    scheduler.step()
    return dict(ce=float(loss.detach()/support),tokens=count,grad_norm=float(norm),
                **{k:float(v) for k,v in components.items()})
