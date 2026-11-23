"""Joint softmax over bounded high-resolution neighborhoods and optional G0."""

import torch
from torch.nn import functional as F


def prepare_joint(query_beats, query_valid, key_beats, key_valid, radii, *, query_width=32):
    """CPU rectangles. A None radius is reserved for the low-resolution G0 bank."""
    if query_beats.device.type != 'cpu' or any(b.device.type != 'cpu' for b in key_beats):
        raise ValueError('continuous attention layouts must be prepared on CPU')
    batch, length = query_beats.shape
    width = min(query_width, length)
    if width < 1:
        raise ValueError('continuous attention requires a nonempty padded query axis')
    owners, queries, validity = [], [], []
    selected_keys = [[] for _ in key_beats]
    for row in range(batch):
        sources = [v[row].nonzero().flatten() for v in key_valid]
        coordinates = [b[row, s].float().contiguous() for b, s in zip(key_beats, sources)]
        for start in range(0, length, width):
            q = torch.arange(start, min(start + width, length))
            valid = query_valid[row, q]
            if not valid.any():
                continue
            positions = query_beats[row, q[valid]].float()
            owners.append(row)
            queries.append(q[torch.arange(width) % len(q)] + row * length)
            validity.append(F.pad(valid, (0, width-len(q))))
            for bank, (source, coords, radius, beats) in enumerate(zip(sources, coordinates, radii, key_beats)):
                if radius is None:
                    chosen = source
                elif len(source):
                    margin = torch.finfo(torch.float32).eps * (positions.abs() + radius + 1) * 4
                    lo = int(torch.searchsorted(coords, positions-radius-margin).min())
                    hi = int(torch.searchsorted(coords, positions+radius+margin, right=True).max())
                    chosen = source[lo:hi]
                else:
                    chosen = source
                selected_keys[bank].append(chosen + row * beats.shape[1])
    query = torch.stack(queries) if queries else torch.empty((0, width), dtype=torch.long)
    valid = torch.stack(validity) if validity else torch.empty((0, width), dtype=torch.bool)
    keys, masks = [], []
    for beats, candidates in zip(key_beats, selected_keys):
        bank_width = max(1, max((len(k) for k in candidates), default=0))
        rows, bank_valid = [], []
        for row, candidate in zip(owners, candidates):
            # Padding reads are distributed across the owning song rather than
            # accumulating zero-gradient atomics on a common global index.
            spare = row * beats.shape[1] + torch.arange(bank_width-len(candidate)) % beats.shape[1]
            rows.append(torch.cat((candidate, spare)))
            bank_valid.append(torch.arange(bank_width) < len(candidate))
        keys.append(torch.stack(rows) if rows else torch.empty((0, bank_width), dtype=torch.long))
        masks.append(torch.stack(bank_valid) if rows else torch.empty((0, bank_width), dtype=torch.bool))
    selected = valid.flatten().nonzero().flatten()
    return dict(query=query, valid=valid, keys=tuple(keys), key_valid=tuple(masks),
                selected=selected, output=query.flatten()[selected], radii=tuple(radii),
                dense=bool(query_valid.all()) and length % width == 0)


def _allowed(layout, query_beats, key_beats):
    query = query_beats.float().flatten()[layout['query']]
    allowed = []
    for indices, valid, beats, radius in zip(layout['keys'], layout['key_valid'], key_beats, layout['radii']):
        mask = layout['valid'][..., None] & valid[:, None]
        if radius is not None:
            keys = beats.float().flatten()[indices]
            mask = mask & ((query[..., None] - keys[:, None]).abs() <= radius)
        allowed.append(mask)
    return torch.cat(allowed, dim=-1)


def with_joint_mask(layout, query_beats, key_beats):
    """Exact FP32 device coordinates decide edges once per stage, not per block."""
    if layout.get('shared_global', False):
        local = layout['local']
        return dict(layout, local=dict(local, allowed=_allowed(local, query_beats, key_beats[1:])),
                    global_allowed=layout['query_valid'][..., None] & layout['global_valid'][:, None])
    allowed = _allowed(layout, query_beats, key_beats)
    usable = allowed.any(-1, keepdim=True)
    safe = allowed.clone()
    safe[..., :1] |= ~usable
    return dict(layout, allowed=allowed, usable=usable, safe_allowed=safe)


def prepare_slot_joint(query_beats, query_valid, global_valid, local_beats, local_valid,
                       radius, *, query_width=24):
    """Keep G0 song-owned; only the bounded L bank needs rectangular gathers."""
    return dict(shared_global=True, query_valid=query_valid, global_valid=global_valid,
                local=prepare_joint(query_beats, query_valid, (local_beats,), (local_valid,),
                                    (radius,), query_width=query_width))


def _gather(value, index):
    flat = value.transpose(1, 2).flatten(0, 1)
    return flat.index_select(0, index.flatten()).unflatten(0, index.shape).transpose(1, 2)


def _restore(value, layout, batch, length, *, fill=0):
    """Restore query rectangles, also for scalar maxima and denominators."""
    if layout['dense']:
        return value.transpose(1, 2).reshape(batch, length, value.shape[1],
                                             *value.shape[3:]).transpose(1, 2)
    selected = value.transpose(1, 2).flatten(0, 1).index_select(0, layout['selected'])
    output = value.new_full((batch*length, *selected.shape[1:]), fill)
    output = output.index_copy(0, layout['output'], selected)
    return output.unflatten(0, (batch, length)).transpose(1, 2)


def _slot_attention(query, key_values, layout, dropout_p):
    """One softmax denominator over G0 and L, without rectangular G0 copies.

    G0 logits and value reductions run once on each song. Only L is gathered
    in small rectangles. Explicit FP32 normalization trades fused S SDPA for
    avoiding repeated G0 K/V and the wider L candidate unions; throughput
    still requires measurement on the target device.
    """
    local = layout['local']
    batch, _, length, depth = query.shape
    if not local['query'].shape[0]:
        return query * 0 + sum((k.sum()+v.sum()) * 0 for k, v in key_values)
    (global_key, global_value), (local_key, local_value) = key_values
    work_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    # Preserve the caller's GEMM/autocast dtype. Only the shared normalization
    # needs FP32; converting every gathered K/V to FP32 would double its storage.
    global_dot = query @ global_key.transpose(-1, -2)
    local_dot = _gather(query, local['query']) @ _gather(local_key, local['keys'][0]).transpose(-1, -2)
    with torch.autocast(device_type=query.device.type, enabled=False):
        global_scores = global_dot.to(work_dtype) / depth**.5
        global_scores = global_scores.masked_fill(~layout['global_allowed'][:, None], -torch.inf)
        local_scores = local_dot.to(work_dtype) / depth**.5
        local_scores = local_scores.masked_fill(~local['allowed'][:, None], -torch.inf)

        # A shared maximum and denominator are algebraically one joint softmax.
        # Empty rows use maximum zero, so exp(-inf) is zero and never inf-inf.
        maximum = torch.maximum(global_scores.amax(-1),
            _restore(local_scores.amax(-1), local, batch, length, fill=-torch.inf))
        maximum = maximum.masked_fill(torch.isneginf(maximum), 0).detach()
        global_mass = (global_scores-maximum[..., None]).exp()
        local_mass = (local_scores-_gather(maximum, local['query'])[..., None]).exp()
        denominator = (global_mass.sum(-1) +
                       _restore(local_mass.sum(-1), local, batch, length)).clamp_min(1e-20)
        global_weight = F.dropout(global_mass/denominator[..., None], p=dropout_p, training=True)
        local_weight = F.dropout(local_mass/_gather(denominator, local['query'])[..., None],
                                 p=dropout_p, training=True)
    result = global_weight.to(global_value.dtype) @ global_value
    local_result = local_weight.to(local_value.dtype) @ _gather(local_value, local['keys'][0])
    local_result = _restore(local_result, local, batch, length)
    if result.dtype in (torch.float16, torch.bfloat16):
        return (result.float() + local_result.float()).to(result.dtype)
    return result + local_result


def joint_attention(query, key_values, layout, dropout_p):
    """Every query attends to the concatenated permitted banks in one softmax."""
    if layout.get('shared_global', False):
        return _slot_attention(query, key_values, layout, dropout_p)
    batch, heads, length, depth = query.shape
    if not layout['query'].shape[0]:
        return query * 0 + sum((k.sum()+v.sum()) * 0 for k, v in key_values)

    keys = torch.cat([_gather(k, index) for (k, _), index in zip(key_values, layout['keys'])], dim=2)
    values = torch.cat([_gather(v, index) for (_, v), index in zip(key_values, layout['keys'])], dim=2)
    # Padding queries can occur within otherwise nonempty rectangles. Give
    # those rows one finite softmax entry, then clear their output explicitly;
    # this keeps even a naive backend's backward free of all--inf softmax NaNs.
    attended = F.scaled_dot_product_attention(_gather(query, layout['query']), keys, values,
        attn_mask=layout['safe_allowed'][:, None], dropout_p=dropout_p)
    attended = attended.masked_fill(~layout['usable'][:, None], 0)
    return _restore(attended, layout, batch, length)
