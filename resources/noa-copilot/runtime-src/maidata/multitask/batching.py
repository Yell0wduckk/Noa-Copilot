"""Pack a complete logical update, sharing audio through chart indices."""

import torch
from . import TASKS
from .codec import TaskVocabulary


_MIDI_VOCAB = TaskVocabulary('midi')


def charts_batch(charts, stage, device, audio_indices=None):
    values = [chart['encoded'].model_inputs() for chart in charts]
    lengths = torch.tensor([len(row['targets']) for row in values])
    width = int(lengths.max())
    vocab_width = max(row["legal_mask"].shape[-1] for row in values)
    all_keys = list(dict.fromkeys(key for row in values for key in row.keys()))
    cpu = {}
    for key in all_keys:
        first = next(row[key] for row in values if key in row)
        if key == 'protected_spans':
            cpu[key] = [row.get(key, first[:0]) for row in values]
            continue
        shape = (len(values), width, *(first.shape[1:]))
        if key == 'legal_mask':
            shape = (len(values), width, vocab_width)
        target = first.new_full(shape, -1 if key == 'previous_pitches' else 0)
        for i, row in enumerate(values):
            if key not in row:
                continue
            item = row[key]
            if key == 'legal_mask':
                target[i, :len(item), :item.shape[-1]] = item
                target[i, len(item):, 0] = True
            else:
                target[i, :len(item)] = item
        cpu[key] = target
    cpu.update(task_ids=torch.tensor([TASKS.index(c['task']) for c in charts]),
               pseudo=torch.tensor([bool(c.get('record',{}).get('pseudo',c['metadata'].get('pseudo',False)))
                                    for c in charts]),
               moment_only=torch.tensor([bool(c['metadata'].get('moment_only', False)) for c in charts],
                                        dtype=torch.bool),
               audio_indices=torch.tensor(audio_indices if audio_indices is not None else [0]*len(charts)),
               lengths=lengths)
    batch = {key: value.to(device, non_blocking=True) for key, value in cpu.items()
             if isinstance(value, torch.Tensor) and key not in ('targets', 'loss_mask')}
    # Build layouts from original CPU metadata, never copy GPU times back.
    batch.update(protected_spans=cpu['protected_spans'], _cpu=cpu,
                 metadata=[c['metadata'] for c in charts], stage=stage)
    return batch, cpu['targets'].to(device), cpu['loss_mask'].to(device)


def chart_batch(chart, stage, device):
    batch, targets, mask = charts_batch([chart], stage, device)
    return batch, targets[0], mask[0]


def audio_group_batch(groups, device, frontend=None, macro_provider=None):
    from .audio import LogMelFrontend
    frontend = frontend if frontend is not None else LogMelFrontend().to(device)
    waves = [group['waveform'] for group in groups]
    samples = torch.tensor([len(wave) for wave in waves])
    padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device, non_blocking=True)
    speeds = torch.tensor([group.get('speed', 1.) for group in groups], dtype=torch.float64)
    features, lengths = frontend.batch(padded, samples, speeds)
    for i, group in enumerate(groups):
        if macro_provider is not None and group.get('synthetic') and 'mert' not in group:
            native = macro_provider(padded[i, :len(waves[i])])
            group.update(mert=native['mert'], muq=native['muq'],
                         macro_origins=(native['mert_origin'], native['muq_origin']))
    audio_meta = [group['charts'][0]['metadata'] if group.get('charts') else {} for group in groups]
    out = dict(
        features=features,
        audio_lengths=lengths,
        waveforms=padded,
        sample_lengths=samples,
        speeds=speeds,
        audio_metadata=audio_meta,
    )
    if any('mert' in g for g in groups):
        for key in ('mert', 'muq'):
            values = [g[key] for g in groups]
            if any(v.device != values[0].device for v in values):
                values = [v.to(device, non_blocking=True) for v in values]
            out[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(device, non_blocking=True)
        out['macro_lengths'] = [(len(g['mert']), len(g['muq'])) for g in groups]
        out['macro_origins'] = [g['macro_origins'] for g in groups]
    return out


def logical_batch(groups, stage, device, frontend=None, macro_provider=None):
    audio_out = audio_group_batch(groups, device, frontend, macro_provider)
    charts, audio_indices = [], []
    for i, group in enumerate(groups):
        charts.extend(group['charts'])
        audio_indices.extend([i] * len(group['charts']))
    batch, targets, mask = charts_batch(charts, stage, device, audio_indices)
    batch.update(audio_out)
    return batch, targets, mask


def logical_split_batch(groups, stage, device, frontend=None, macro_provider=None):
    """Pack full charts and short moment_only charts separately, sharing one audio batch."""
    audio_out = audio_group_batch(groups, device, frontend, macro_provider)
    full_charts, full_audio_idx = [], []
    tag_charts, tag_audio_idx = [], []
    for i, group in enumerate(groups):
        for chart in group['charts']:
            if bool(chart['metadata'].get('moment_only', False)):
                tag_charts.append(chart)
                tag_audio_idx.append(i)
            else:
                full_charts.append(chart)
                full_audio_idx.append(i)
    full_pack = None
    if full_charts:
        fb, ft, fm = charts_batch(full_charts, stage, device, full_audio_idx)
        fb['audio_metadata'] = audio_out['audio_metadata']
        full_pack = (fb, ft, fm)
    tag_pack = None
    if tag_charts:
        tb, tt, tm = charts_batch(tag_charts, stage, device, tag_audio_idx)
        tb['audio_metadata'] = audio_out['audio_metadata']
        tag_pack = (tb, tt, tm)
    return audio_out, full_pack, tag_pack


def _note_on_targets(target, supervised, safe, slots, previous):
    """Build a local Gaussian band, keeping only legal pitches in the GT direction."""
    start = _MIDI_VOCAB.offsets['NOTE_ON']
    pitch = (target - start).clamp(0, 127)
    rows = supervised & (target >= start) & (target < start+128) & (slots >= 4) & (slots <= 7)
    offsets = torch.arange(-5, 6, device=target.device)
    candidates = pitch[..., None] + offsets
    ids = start + candidates.clamp(0, 127)
    sigma = torch.where(slots == 5, 2.5, 1.)
    radius = torch.where(slots == 5, 5, 2)
    support = (offsets.abs() <= radius[..., None]) & (candidates >= 0) & (candidates < 128)
    if previous is not None:
        reference = previous[..., None]
        direction = (pitch[..., None] - reference).sign()
        support &= (reference < 0) | ((candidates - reference).sign() == direction)
    support &= safe.gather(-1, ids)
    weights = torch.exp(-offsets.square() / (2 * sigma[..., None].square())) * support
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-20)
    return rows, ids, weights


def _midi_soft_loss(z, denominator, target, full, safe, mask, task_ids,
                    note_targets, pseudo, target_pseudo,
                    target_groups=None, target_velocity_db=None, velocity_sigma=3.):
    device = z.device
    midi = (task_ids.to(device)[:, None] == TASKS.index('midi')) & mask
    groups = torch.zeros_like(target) if target_groups is None else target_groups.to(device)
    is_pseudo = torch.zeros_like(target, dtype=torch.bool)
    if pseudo is not None:
        is_pseudo |= pseudo.to(device)[:, None]
    if target_pseudo is not None:
        is_pseudo |= target_pseudo.to(device)
    for family in ('SLOT_PITCHED', 'SLOT_UNPITCHED'):
        start = _MIDI_VOCAB.offsets[family]
        local = (target - start).clamp(0, 3)
        section = z[..., start:start+4]
        bits = groups
        if family == 'SLOT_PITCHED':
            bits = torch.where((bits == 0) & is_pseudo & (local >= 2), 12, bits)
        member = ((bits[..., None] >> torch.arange(4, device=device)) & 1).bool()
        member &= safe[..., start:start+4]
        rows = midi & (target >= start) & (target < start+4) & member.any(-1)
        numerator = _subset_logsumexp(section, member)
        full = torch.where(rows, denominator - numerator, full)
    if note_targets is not None:
        rows, ids, weights = note_targets
        expected = (weights * z.gather(-1, ids).masked_fill(weights == 0, 0.)).sum(-1)
        full = torch.where(rows, denominator-expected, full)
    if target_velocity_db is not None:
        start = _MIDI_VOCAB.offsets['VELOCITY']
        db = target_velocity_db.to(device).float()
        # Unknown/missing measurement falls back to its hard known band.
        rows = midi & (target > start) & (target < start+5) & torch.isfinite(db)
        db = db.nan_to_num()
        edges = z.new_tensor([-torch.inf, -26., -16., -6., torch.inf])
        cdf = .5 * (1 + torch.erf((edges - db[..., None]) / (velocity_sigma * 2**.5)))
        q = (cdf[..., 1:] - cdf[..., :-1]) * safe[..., start+1:start+5]
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-20)
        expected = (q * z[..., start+1:start+5].masked_fill(q == 0, 0.)).sum(-1)
        full = torch.where(rows, denominator - expected, full)
    return full


def _subset_logsumexp(values, subset):
    """Unused empty subsets have a finite constant and zero gradient."""
    selected = values.masked_fill(~subset, -torch.inf)
    selected[..., 0] = selected[..., 0].masked_fill(~subset.any(-1), 0.)
    return torch.logsumexp(selected, -1)


def cross_entropy(logits, targets, mask, legal_mask, *, stage=1,
                  task_ids=None, current_slots=None, pseudo=None,
                  target_pseudo=None, previous_pitches=None, components=None,
                  target_groups=None, target_velocity_db=None, velocity_sigma=3.,
                  moment_only=None, residual_neg_mask=None, residual_neg_weight=1.,
                  ll_ignore_threshold=0., ll_ignore_weight=.1, apply_ll_ignore=False):
    """One legal flat softmax per supervised token, with MIDI soft labels."""
    if targets.ndim == 1:
        targets, mask = targets[None], mask[None]
    if legal_mask.ndim == 2:
        legal_mask = legal_mask[None]
    logits = logits.float()
    legal = legal_mask[..., :logits.shape[-1]]
    if legal.shape[-1] < logits.shape[-1]:
        legal = torch.nn.functional.pad(legal, (0, logits.shape[-1]-legal.shape[-1]))
    safe = legal & mask[..., None]
    safe[..., 0] |= ~mask
    target = targets.masked_fill(~mask, 0)
    note_targets = None
    if task_ids is not None and current_slots is not None:
        midi = task_ids.to(logits.device)[:, None] == TASKS.index('midi')
        previous = None if previous_pitches is None else previous_pitches.to(logits.device).reshape_as(target)
        note_targets = _note_on_targets(target, mask & midi, safe,
                                        current_slots.to(logits.device).reshape_as(target), previous)
    z = logits.masked_fill(~safe, -torch.inf)
    # Mixed no-history rows already contain log probabilities and -inf for
    # illegal candidates. Ignored carry/padding rows need a finite dummy.
    z[..., 0] = z[..., 0].masked_fill(~mask, 0.)
    selected = z.gather(-1, target[..., None]).squeeze(-1)
    denominator = torch.logsumexp(z, -1)
    full = denominator - selected
    if task_ids is None:
        if components is not None:
            components['supervision_count'] = mask.sum().detach()
        return full.masked_fill(~mask, 0).sum()
    if target_pseudo is not None and target_pseudo.ndim==1:
        target_pseudo=target_pseudo[None]
    full=_midi_soft_loss(z,denominator,target,full,safe,mask,task_ids,
                         note_targets,pseudo,target_pseudo,
                         target_groups,target_velocity_db,velocity_sigma)
    if moment_only is not None and bool(moment_only.any()):
        tag_rows = moment_only.to(logits.device).reshape(-1, 1) & mask
        neg_beat = tag_rows & (target >= 1) & (target <= 16)
        weights = mask.float()
        in_res = torch.zeros_like(neg_beat)
        if residual_neg_mask is not None:
            res = residual_neg_mask.to(logits.device)
            if res.ndim == 1:
                res = res[None]
            in_res = neg_beat & res
            weights = torch.where(in_res, weights * float(residual_neg_weight), weights)
        if apply_ll_ignore and ll_ignore_threshold > 0.:
            moment_probs = (z[..., 17:113].detach() - denominator.detach()[..., None]).exp()
            peak_prob = moment_probs.amax(dim=-1)
            ll_hit = neg_beat & (peak_prob > float(ll_ignore_threshold))
            ll_scale = torch.where(in_res, torch.zeros_like(weights), torch.full_like(weights, float(ll_ignore_weight)))
            weights = torch.where(ll_hit, weights * ll_scale, weights)
        weighted = full.masked_fill(~mask, 0.) * weights
        main = weighted.sum()
        support = weights.sum().detach()
        if components is not None:
            components.update(
                main_ce=main.detach(),
                supervision_count=support,
                tag_moment_ce=(weighted.masked_fill(~tag_rows, 0.).sum() /
                               weights.masked_fill(~tag_rows, 0.).sum().clamp_min(1.)).detach(),
                uncond_ce=(weighted.masked_fill(tag_rows, 0.).sum() /
                           weights.masked_fill(tag_rows, 0.).sum().clamp_min(1.)).detach(),
            )
        return main
    main = full.masked_fill(~mask, 0).sum()
    if components is not None:
        components.update(main_ce=main.detach(), supervision_count=mask.sum().detach())
    return main


def moment_type_loss(model, batch: dict, *, ll_ignore_threshold: float = .85,
                     ll_ignore_weight: float = .1, apply_ll_ignore: bool = False,
                     components: dict | None = None) -> torch.Tensor:
    """Unconditioned MOMENT -> 9-coarse + 31-fine multi-label BCE readout."""
    from .timbre_tags import NUM_COARSE_TAGS
    F = torch.nn.functional
    dec_hidden = getattr(model, 'last_decoder_hidden', None)
    if dec_hidden is None or not hasattr(model, 'moment_type_head'):
        zero = (dec_hidden.sum() * 0.0) if dec_hidden is not None else next(model.parameters()).sum() * 0.0
        if components is not None:
            components.update(moment_type_loss=zero.detach(), moment_coarse_loss=zero.detach(),
                              moment_fine_loss=zero.detach(), supervised_moments=0)
        return zero
    device = dec_hidden.device
    targets = batch.get('moment_type_targets')
    weights = batch.get('moment_type_weights')
    if targets is None or weights is None:
        zero = dec_hidden.sum() * 0.0
        if components is not None:
            components.update(moment_type_loss=zero.detach(), moment_coarse_loss=zero.detach(),
                              moment_fine_loss=zero.detach(), supervised_moments=0)
        return zero

    targets = targets.to(device=device, dtype=torch.float32)
    weights = weights.to(device=device, dtype=torch.float32)
    slot6_union = batch.get('moment_slot6_union')
    slot6_union = (torch.zeros(targets.shape[:2], device=device, dtype=torch.bool)
                   if slot6_union is None else slot6_union.to(device=device, dtype=torch.bool))

    # Strict double lock: never supervise moment_type_head on tag-conditioned rows.
    if 'moment_only' in batch:
        uncond_chart = ~batch['moment_only'].to(device=device, dtype=torch.bool)[:, None]
        weights = weights * uncond_chart.float()[..., None]
        slot6_union = slot6_union & uncond_chart

    active = (weights.sum(dim=-1) > 0) | slot6_union
    if not bool(active.any()):
        zero = dec_hidden.sum() * 0.0
        if components is not None:
            components.update(moment_type_loss=zero.detach(), moment_coarse_loss=zero.detach(),
                              moment_fine_loss=zero.detach(), supervised_moments=0)
        return zero

    logits = model.moment_type_head(dec_hidden[active]).float()
    tgt = targets[active]
    w = weights[active]
    u6 = slot6_union[active]

    if apply_ll_ignore and ll_ignore_threshold > 0.:
        probs = torch.sigmoid(logits.detach())
        ll_neg = (tgt < 0.5) & (probs > float(ll_ignore_threshold))
        ll_scale = torch.where(w < 0.99, torch.zeros_like(w), torch.full_like(w, float(ll_ignore_weight)))
        w = torch.where(ll_neg, w * ll_scale, w)

    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none') * w
    c_w = w[:, :NUM_COARSE_TAGS].sum().clamp_min(1.0)
    f_w = w[:, NUM_COARSE_TAGS:].sum().clamp_min(1.0)
    coarse_loss = bce[:, :NUM_COARSE_TAGS].sum() / c_w
    fine_loss = bce[:, NUM_COARSE_TAGS:].sum() / f_w
    union_loss = logits.sum() * 0.0
    if bool(u6.any()):
        union_loss = -F.logsigmoid(torch.logsumexp(logits[u6, 6:8], dim=-1)).mean()
    total = coarse_loss + fine_loss + union_loss
    if components is not None:
        components.update(
            moment_type_loss=total.detach(),
            moment_coarse_loss=coarse_loss.detach(),
            moment_fine_loss=fine_loss.detach(),
            supervised_moments=int(active.sum().item()),
        )
    return total


def shift_tolerant_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
                       tolerance: int = 3) -> torch.Tensor:
    """Beat This! shift-tolerant BCE via temporal max-pooling on positive peaks."""
    if logits.ndim == 2:
        logits = logits[:, None, :]
    if targets.ndim == 2:
        targets = targets[:, None, :]
    logits = logits.float()
    targets = targets.float()
    valid_mask = valid.to(device=logits.device, dtype=torch.bool)[:, None, :].expand_as(logits)
    if not valid_mask.any():
        return logits.sum() * 0.0
    tol = int(tolerance)
    kernel = 2 * tol + 1
    masked_logits = logits.masked_fill(~valid_mask, -1e4)
    pooled_logits = torch.nn.functional.max_pool1d(
        masked_logits, kernel_size=kernel, stride=1, padding=tol
    )
    masked_targets = targets.masked_fill(~valid_mask, 0.0)
    dilated_targets = torch.nn.functional.max_pool1d(
        masked_targets, kernel_size=kernel, stride=1, padding=tol
    )
    pos_mask = (targets > 0.5) & valid_mask
    neg_mask = (dilated_targets < 0.5) & valid_mask
    loss = logits.sum() * 0.0
    if pos_mask.any():
        pos_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pooled_logits, torch.ones_like(pooled_logits), reduction='none'
        )
        loss = loss + pos_bce.masked_fill(~pos_mask, 0.0).sum() / pos_mask.sum().clamp_min(1)
    if neg_mask.any():
        neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction='none'
        )
        loss = loss + neg_bce.masked_fill(~neg_mask, 0.0).sum() / neg_mask.sum().clamp_min(1)
    return 0.5 * loss


def rhythm_loss(model, batch: dict, streams, *, components: dict | None = None) -> torch.Tensor:
    """Joint Encoder frame-level + Decoder beat-token rhythm supervision."""
    from .grid import BeatGrid
    F = torch.nn.functional
    device = streams.p8.device
    if not hasattr(model, 'encoder_rhythm_head'):
        zero = streams.p8.sum() * 0.0
        if components is not None:
            components.update(rhythm_loss=zero.detach(), beat_downbeat_loss=zero.detach(),
                              bpm_loss=zero.detach(), meter_loss=zero.detach())
        return zero

    b_audio, num_frames, _ = streams.p8.shape
    enc_out = model.encoder_rhythm_head(streams.p8).float()
    valid_frames = torch.arange(num_frames, device=device)[None, :] < streams.p_lengths.to(device)[:, None]

    cpu = batch.get('_cpu', batch)
    metadata_list = batch.get('metadata') or [{}] * len(batch['token_ids'])
    if isinstance(metadata_list, dict):
        metadata_list = [{k: (v[i] if isinstance(v, (torch.Tensor, list, tuple)) else v)
                          for k, v in metadata_list.items()} for i in range(len(batch['token_ids']))]
    audio_indices = torch.as_tensor(cpu.get('audio_indices', torch.arange(len(metadata_list)))).tolist()
    preset_audio_meta = batch.get('audio_metadata')
    audio_meta = [dict(preset_audio_meta[i]) if preset_audio_meta and i < len(preset_audio_meta) else {}
                  for i in range(b_audio)]
    for chart_idx, a_idx in enumerate(audio_indices):
        if 0 <= a_idx < b_audio:
            m = metadata_list[chart_idx]
            if not audio_meta[a_idx] or m.get('meter_segments') or m.get('bpm_segments') or m.get('measure_multipliers'):
                audio_meta[a_idx] = m

    frame_targets = [BeatGrid(m).frame_rhythm_targets(num_frames, device=device) for m in audio_meta]
    beat_tgt = torch.stack([t['beat_target'] for t in frame_targets], dim=0)
    downbeat_tgt = torch.stack([t['downbeat_target'] for t in frame_targets], dim=0)
    log_bpm_tgt = torch.stack([t['log_bpm_target'] for t in frame_targets], dim=0)
    stop_tgt = torch.stack([t['stop_target'] for t in frame_targets], dim=0)
    num_tgt = torch.stack([t['num_target'] for t in frame_targets], dim=0)
    den_tgt = torch.stack([t['den_target'] for t in frame_targets], dim=0)
    has_meter_audio = torch.tensor([bool(t['has_meter']) for t in frame_targets], device=device, dtype=torch.bool)[:, None]
    has_tempo_audio = torch.tensor([bool(t['has_tempo'] or t['has_meter']) for t in frame_targets], device=device, dtype=torch.bool)[:, None]

    valid_beat_frames = valid_frames & has_tempo_audio
    valid_meter_frames = valid_frames & has_meter_audio
    enc_beat_loss = (
        shift_tolerant_bce(enc_out[:, :, 0:1].transpose(1, 2), beat_tgt[:, None, :], valid_beat_frames, tolerance=3) +
        shift_tolerant_bce(enc_out[:, :, 1:2].transpose(1, 2), downbeat_tgt[:, None, :], valid_meter_frames, tolerance=3)
    ) * 0.5

    not_stop = valid_beat_frames & (stop_tgt < 0.5)
    enc_bpm_loss = enc_out.sum() * 0.0
    if not_stop.any():
        enc_bpm_loss = enc_bpm_loss + F.smooth_l1_loss(enc_out[:, :, 2][not_stop], log_bpm_tgt[not_stop], beta=0.05)
    if valid_beat_frames.any():
        enc_bpm_loss = enc_bpm_loss + 0.2 * F.binary_cross_entropy_with_logits(
            enc_out[:, :, 3][valid_beat_frames], stop_tgt[valid_beat_frames]
        )

    enc_meter_loss = enc_out.sum() * 0.0
    if valid_meter_frames.any():
        enc_meter_loss = (F.cross_entropy(enc_out[:, :, 4:20][valid_meter_frames], num_tgt[valid_meter_frames]) +
                          F.cross_entropy(enc_out[:, :, 20:24][valid_meter_frames], den_tgt[valid_meter_frames]))

    dec_downbeat_loss = enc_out.sum() * 0.0
    dec_bpm_loss = enc_out.sum() * 0.0
    dec_meter_loss = enc_out.sum() * 0.0
    dec_hidden = getattr(model, 'last_decoder_hidden', None)
    if dec_hidden is not None and hasattr(model, 'decoder_rhythm_head'):
        tokens = batch['token_ids'].to(device)
        b_charts, seq_len = tokens.shape
        lengths = torch.as_tensor(cpu.get('lengths', torch.full((b_charts,), seq_len)), device=device)
        valid_tok = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
        if 'onset_seconds' in batch:
            valid_tok = valid_tok & (batch['onset_seconds'].to(device) >= -1e-6)
        is_moment_only = batch.get('moment_only')
        if is_moment_only is not None:
            valid_tok = valid_tok & (~is_moment_only.to(device)[:, None])
        beat_mask = valid_tok & (tokens >= 1) & (tokens <= 16)
        if beat_mask.any():
            causal_beats = batch['causal_beats'].to(device)
            active_charts = [c for c in range(b_charts) if bool(beat_mask[c].any())]
            gt_downbeats, gt_log_bpms, gt_bibs, gt_nums, gt_dens = [], [], [], [], []
            chart_has_meter = [False] * b_charts
            chart_has_tempo = [False] * b_charts
            zero_f = torch.zeros(seq_len, device=device, dtype=torch.float32)
            zero_l = torch.zeros(seq_len, device=device, dtype=torch.long)
            per_chart = {}
            for c in active_charts:
                grid_c = BeatGrid(metadata_list[c])
                cb = causal_beats[c]
                bi = grid_c.bar_info(cb)
                ti = grid_c.tempo_info(cb)
                per_chart[c] = (bi['is_downbeat'], torch.log2(ti['bpm'] / 120.0),
                                bi['beat_in_bar'], bi['num_id'], bi['den_id'])
                chart_has_meter[c] = bool(bi['has_meter'])
                chart_has_tempo[c] = bool(ti['has_tempo'] or bi['has_meter'])
            for c in range(b_charts):
                db_t, bpm_t, bib_t, num_t, den_t = per_chart.get(c, (zero_f, zero_f, zero_l, zero_l, zero_l))
                gt_downbeats.append(db_t)
                gt_log_bpms.append(bpm_t)
                gt_bibs.append(bib_t)
                gt_nums.append(num_t)
                gt_dens.append(den_t)
            gt_downbeat = torch.stack(gt_downbeats, dim=0)[beat_mask]
            gt_log_bpm = torch.stack(gt_log_bpms, dim=0)[beat_mask]
            gt_bib = torch.stack(gt_bibs, dim=0)[beat_mask]
            gt_num = torch.stack(gt_nums, dim=0)[beat_mask]
            gt_den = torch.stack(gt_dens, dim=0)[beat_mask]

            phase_active = getattr(model.conditioner, 'last_phase_active', None)
            if phase_active is not None and len(phase_active) == b_charts:
                chart_w = torch.where(phase_active.to(device), 0.2, 1.0)
            else:
                chart_w = torch.ones(b_charts, device=device)
            meter_c = torch.tensor(chart_has_meter, device=device, dtype=torch.float32)
            tempo_c = torch.tensor(chart_has_tempo, device=device, dtype=torch.float32)
            tok_w_meter = (chart_w * meter_c)[:, None].expand(b_charts, seq_len)[beat_mask]
            tok_w_tempo = (chart_w * tempo_c)[:, None].expand(b_charts, seq_len)[beat_mask]

            dec_out = model.decoder_rhythm_head(dec_hidden[beat_mask]).float()
            if tok_w_tempo.sum() > 0:
                w_t = tok_w_tempo / tok_w_tempo.sum().clamp_min(1e-8)
                dec_bpm_loss = (F.smooth_l1_loss(
                    dec_out[:, 1], gt_log_bpm, beta=0.05, reduction='none') * w_t).sum()
            if tok_w_meter.sum() > 0:
                w_m = tok_w_meter / tok_w_meter.sum().clamp_min(1e-8)
                dec_downbeat_loss = (F.binary_cross_entropy_with_logits(
                    dec_out[:, 0], gt_downbeat, reduction='none') * w_m).sum()
                dec_meter_loss = ((
                    F.cross_entropy(dec_out[:, 2:18], gt_bib, reduction='none') +
                    F.cross_entropy(dec_out[:, 18:34], gt_num, reduction='none') +
                    F.cross_entropy(dec_out[:, 34:38], gt_den, reduction='none')
                ) * w_m).sum()

    beat_downbeat_loss = enc_beat_loss + dec_downbeat_loss
    bpm_loss = enc_bpm_loss + dec_bpm_loss
    meter_loss = enc_meter_loss + dec_meter_loss
    total = beat_downbeat_loss + bpm_loss + 0.5 * meter_loss
    if components is not None:
        components.update(
            rhythm_loss=total.detach(),
            beat_downbeat_loss=beat_downbeat_loss.detach(),
            bpm_loss=bpm_loss.detach(),
            meter_loss=meter_loss.detach(),
        )
    return total
