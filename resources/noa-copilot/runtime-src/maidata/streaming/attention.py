"""Local rectangular SDPA; no full chart-by-chart attention matrix."""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch.nn import functional as F
from maidata.nn.position import apply_rope
from maidata.decoder.types import KV
from maidata.decoder.projections import linear_parameters
from maidata.nn.varlen_attention import can_use_varlen_flash, varlen_flash_attention
from .profiling import section, measured_call

QUERY_WIDTH = 64
SELF_BEFORE = 8.
AUDIO_BEFORE, AUDIO_AFTER = 8., 4.
_compiled_bucket = ContextVar('streaming_compiled_bucket',default=None)


def layouts(causal, cross, audio_beats, audio_valid=None, query_width=QUERY_WIDTH, *, integer_boundaries=False,
            self_before=SELF_BEFORE, audio_before=AUDIO_BEFORE, audio_after=AUDIO_AFTER,
            types=None, vocab_ids=None, stage1_pretrain=False, self_integer_boundaries=None):
    """Offline CPU geometry; self mask uses onsets, RoPE uses token positions.

    Audio uses [q-8,q+4), or [floor(q)-8,floor(q)+4) for integer boundaries:
    exactly 96 frames on the 8/frame beat lattice.
    An inclusive upper endpoint would incorrectly select 97 frames.
    """
    causal, cross, audio_beats = (x.detach().cpu() for x in (causal,cross,audio_beats))
    moment_q = moment_k = None
    if stage1_pretrain and types is not None and vocab_ids is not None:
        from .moment_conditioning import moment_query_mask, moment_key_mask
        t_cpu = types.detach().cpu().reshape(-1)
        v_cpu = vocab_ids.detach().cpu().reshape(-1)
        moment_q = moment_query_mask(t_cpu, v_cpu)
        moment_k = moment_key_mask(t_cpu, v_cpu)
    # Causal positions are tick96 onsets. Compare integer ticks so the exact
    # 8-beat boundary cannot move with float32 cancellation at weak onsets.
    ticks = (causal*96).round().long()
    self_int = integer_boundaries and not stage1_pretrain if self_integer_boundaries is None else self_integer_boundaries
    lower = torch.div(ticks,96,rounding_mode="floor")*96 if self_int else ticks
    if integer_boundaries: cross=cross.floor()
    self_blocks, audio_blocks = [], []
    for lo in range(0,len(causal),query_width):
        hi = min(lo+query_width,len(causal))
        q = ticks[lo:hi]
        low = lower[lo:hi]-int(96*self_before)
        first = int(torch.searchsorted(ticks,low.min()))
        indices = torch.arange(max(1,first),hi)
        indices = torch.cat((torch.zeros(1,dtype=torch.long),indices))
        mask = ((ticks[indices][None] >= low[:,None]) &
                (ticks[indices][None] <= q[:,None]) & (indices[None] <= torch.arange(lo,hi)[:,None]))
        mask[:,0] = True
        if moment_k is not None:
            mask = mask & (~moment_k[indices][None, :])
            mask[:, 0] = True
            mq_local = moment_q[lo:hi]
            if mq_local.any():
                mask[mq_local, 1:] = False
                mask[mq_local, 0] = True
        self_blocks.append(dict(start=lo,stop=hi,indices=indices,allowed=mask))
        q = cross[lo:hi]
        a = int(torch.searchsorted(audio_beats,q.min()-audio_before))
        b = int(torch.searchsorted(audio_beats,q.max()+audio_after))
        keys = audio_beats[a:b]
        mask = (keys[None] >= q[:,None]-audio_before) & (keys[None] < q[:,None]+audio_after)
        if audio_valid is not None: mask &= audio_valid.cpu()[None,a:b]
        audio_blocks.append(dict(start=lo,stop=hi,lo=a,hi=b,allowed=mask))
    def pack(blocks,audio=False):
        length = len(causal)
        width = max((b['hi']-b['lo'] if audio else len(b['indices']) for b in blocks),default=0)
        indices = torch.zeros(len(blocks),width,dtype=torch.long)
        allowed = torch.zeros(len(blocks),query_width,width,dtype=torch.bool)
        for i,b in enumerate(blocks):
            keys = torch.arange(b['lo'],b['hi']) if audio else b['indices']
            indices[i,:len(keys)] = keys
            allowed[i,:b['stop']-b['start'],:len(keys)] = b['allowed']
        return dict(indices=indices,allowed=allowed,length=length,query_width=query_width)
    return dict(self_layout=pack(self_blocks),audio_layout=pack(audio_blocks,True))


def attend(query,key,value,mask,dropout=0.):
    if key.shape[2] == 0: return query*0
    has_visible = mask.any(-1,keepdim=True)
    safe = mask.clone()
    safe[...,0] |= ~has_visible.squeeze(-1)
    expanded = safe[:,None] if safe.ndim == 3 else safe[None,None]
    visible = has_visible[:,None] if has_visible.ndim == 3 else has_visible[None,None]
    # Dense fused SDPA requires equal batch dimensions. Explicit zero-stride
    # views preserve the shared song bank and its summed backward adjoint.
    if key.shape[0]==1 and query.shape[0]>1:key=key.expand(query.shape[0],-1,-1,-1)
    if value.shape[0]==1 and query.shape[0]>1:value=value.expand(query.shape[0],-1,-1,-1)
    y = F.scaled_dot_product_attention(query,key,value,attn_mask=expanded,dropout_p=dropout)
    return y.masked_fill(~visible,0)


def rectangular(module,query,kv,blocks,*,self_attention=False,prepared=None):
    if blocks.get('full_song') or blocks.get('ragged_audio'):
        return full_song_attention(module,query,kv,blocks,prepared=prepared)
    if 'buckets' in blocks:
        return bucketed(module,query,kv,blocks,self_attention=self_attention)
    # One gather and one batched SDPA for all local blocks of this chart.
    ids,mask = blocks['indices'],blocks['allowed']
    count,width = ids.shape
    qwidth,length = blocks['query_width'],blocks['length']
    heads,dim = query.shape[1],query.shape[-1]
    if 'query_indices' in blocks:
        q = query[0].index_select(1,blocks['query_indices'].flatten())
    else: q = F.pad(query,(0,0,0,count*qwidth-length))[0]
    q = q.reshape(heads,count,qwidth,dim).transpose(0,1)
    if width:
        if 'bank_rows' in blocks:
            # Gather local keys directly; do not flatten/copy whole audio banks.
            k,v = (x[blocks['bank_rows'],:,blocks['bank_columns']].permute(0,2,1,3)
                   for x in (kv.key,kv.value))
        else:
            k,v = (x[0].index_select(1,ids.flatten()).reshape(heads,count,width,dim).transpose(0,1)
                   for x in (kv.key,kv.value))
        if not self_attention: mask = mask & kv.valid.flatten().index_select(0,ids.flatten()).reshape(count,1,width)
        has_visible = mask.any(-1,keepdim=True)
        safe = mask.clone()
        safe[...,0] |= ~has_visible.squeeze(-1)
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None],
                    dropout_p=module.dropout if module.training else 0.)
        y = y.masked_fill(~has_visible[:,None],0)
    else: y = q*0
    y = y.transpose(1,2).reshape(1,count*qwidth,heads*dim)
    y = y.index_select(1,blocks['output_indices']) if 'output_indices' in blocks else y[:,:length]
    return module.output(y)


def query_visibility(layout, device, valid=None):
    """Rows with at least one permitted key, for ragged and rectangular layouts."""
    if 'visible_queries' in layout:
        visible = torch.zeros(layout['length'],device=device,dtype=torch.bool)
        visible[layout['visible_queries']] = True
        return visible
    if 'buckets' in layout:
        rows = [bucket['allowed'].any(-1).flatten() for bucket in layout['buckets']]
        visible = torch.cat(rows) if rows else torch.zeros(1,device=device,dtype=torch.bool)
    else:
        allowed = layout['allowed']
        if valid is not None:
            allowed = allowed & valid.flatten().index_select(0, layout['indices'].flatten()).reshape(
                allowed.shape[0], 1, allowed.shape[2])
        visible = allowed.any(-1).flatten()
    return (visible.index_select(0,layout['output_indices']) if 'output_indices' in layout
            else visible[:layout['length']])


def full_song_attention(module, query, kv, layout, *, prepared=None):
    """Gather each ragged segment once; share key sets across all its queries."""
    heads, dim = query.shape[1], query.shape[-1]
    qflat = query.transpose(1,2).reshape(-1,heads,dim)
    qi, ki = layout['varlen_queries'], layout['varlen_keys']
    q = qflat.index_select(0,qi)
    if prepared is not None and 'kv' in prepared:
        k,v = prepared['kv']
    else:
        kflat, vflat = (t.transpose(1,2).reshape(-1,heads,dim) for t in (kv.key,kv.value))
        k,v = (_RowGather.apply(t,ki) for t in (kflat,vflat))
        if prepared is not None: prepared['kv'] = k,v
    dropout = module.dropout if module.training else 0.
    if not layout['q_lengths']:
        y = qflat*0 + (k.sum()+v.sum())*0
    else:
        if layout["verified_valid"] and can_use_varlen_flash(q,k,v,dropout_p=dropout):
            full = varlen_flash_attention(q,k,v,layout['cu_q'],layout['cu_k'],
                layout['max_q'],layout['max_k'],dropout_p=dropout,is_causal=False,scale=dim**-.5)
        else:
            parts=[]
            validity = None if layout['verified_valid'] else kv.valid.flatten().index_select(0,ki).split(layout['k_lengths'])
            for i,(qs,ks,vs) in enumerate(zip(q.split(layout['q_lengths']),k.split(layout['k_lengths']),v.split(layout['k_lengths']))):
                # A segment shares one key set, so even the legacy runtime
                # validity fallback needs only a broadcast 1D key mask.
                mask = None if validity is None else validity[i][None,None,None,:]
                out=F.scaled_dot_product_attention(qs.transpose(0,1)[None],ks.transpose(0,1)[None],
                                                   vs.transpose(0,1)[None],attn_mask=mask,dropout_p=dropout)
                parts.append(out[0].transpose(0,1))
            full=torch.cat(parts)
        y=torch.zeros_like(qflat).index_copy(0,qi,full)
    return module.output(y.reshape(1,layout['length'],heads*dim))



@dataclass(frozen=True)
class RollingKV:
    key: torch.Tensor
    value: torch.Tensor


def self_kv(module,x,beats,*,tables=None):
    raw = module.split(module.key(x))
    key = apply_rope(raw,beats,module.base,module.scale,tables=tables)
    return key,module.split(module.value(x)),raw


def self_qkv(module, x, beats, *, tables=None):
    """One GEMM and joint Q/K rotation; registered weights unchanged."""
    with section('self_qkv_rope'):
        projected = F.linear(x, *linear_parameters((module.query, module.key, module.value)))
        q, k, v = (module.split(t) for t in projected.chunk(3, dim=-1))
        raw = torch.cat((q, k), dim=0)
        joint_tables = tuple(torch.cat((t,t),0) for t in tables) if tables is not None and tables[0].shape[0] > 1 else tables
        rotated = apply_rope(raw, beats.repeat(2,1) if beats.shape[0] > 1 else beats, module.base, module.scale, tables=joint_tables)
        q, k = rotated.chunk(2, dim=0)
        return q, k, v, raw.chunk(2, dim=0)[1]


def bucketed(module, query, kv, layout, *, self_attention=False):
    """Gather compact whole head rows once, then run fixed-width SDPA buckets."""
    prefix = 'self' if self_attention else 'audio'
    dropout = module.dropout if module.training else 0.
    heads, dim = query.shape[1], query.shape[-1]
    with section(prefix+'_gather'):
        qflat = query.transpose(1, 2).reshape(-1, heads, dim)
        kflat, vflat = (t.transpose(1, 2).reshape(-1, heads, dim).contiguous()
                        for t in (kv.key, kv.value))
    flash = (not self_attention and layout['verified_valid'] and layout['max_k'] > 0
             and can_use_varlen_flash(qflat, kflat, vflat, dropout_p=dropout))
    y = _bucket_values(qflat, kflat, vflat, kv.valid,
                      layout['masked_layout'] if flash else layout, dropout, self_attention)
    if flash:
        with section('audio_gather'):
            qi = layout['varlen_queries']; ki = layout['varlen_keys']
            q = qflat.index_select(0, qi)
            k, v = (_RowGather.apply(t,ki) for t in (kflat, vflat))
        with section('audio_sdpa'):
            full = varlen_flash_attention(q, k, v, layout['cu_q'], layout['cu_k'],
                layout['max_q'], layout['max_k'], dropout_p=dropout,
                is_causal=False, scale=dim**-.5)
        y = y.index_copy(0, qi, full)
    return module.output(y.reshape(1, layout['length'], heads*dim))


def _bucket_values(qflat, kflat, vflat, valid, layout, dropout, self_attention):
    prefix = 'self' if self_attention else 'audio'
    heads, dim = qflat.shape[-2:]
    q,k,v = measured_call(prefix+'_gather_rows', _gather_rows, qflat,kflat,vflat,
                           layout['query_indices'],layout['indices'])
    if not layout['buckets']:
        return qflat*0 + (k.sum()+v.sum())*0
    qs = q.split([b['count']*b['query_width'] for b in layout['buckets']])
    ks, vs = (t.split([b['count']*b['key_width'] for b in layout['buckets']]) for t in (k, v))
    validity = None if self_attention else valid.flatten().index_select(0, layout['indices']).split(
        [b['count']*b['key_width'] for b in layout['buckets']])
    outputs = []
    with section(prefix+'_sdpa'):
        for i, (b, q, k, v) in enumerate(zip(layout['buckets'], qs, ks, vs)):
            count, qw, kw = b['count'], b['query_width'], b['key_width']
            q, k, v = (t.reshape(count, width, heads, dim).transpose(1, 2)
                       for t, width in ((q, qw), (k, kw), (v, kw)))
            mask = b['allowed']
            if validity is not None: mask = mask & validity[i].reshape(count, 1, kw)
            kernel = _compiled_bucket.get() or _masked_sdpa
            y = measured_call(prefix+'_sdpa_kernel', kernel, q,k,v,mask,dropout)
            outputs.append(y.transpose(1, 2).reshape(-1, heads, dim))
    y = torch.cat(outputs).index_select(0, layout['output_indices'])
    if not self_attention:
        y = torch.zeros_like(qflat).index_copy(0, layout['visible_queries'],
                                              y.index_select(0, layout['visible_queries']))
    return y


def _gather_rows(q,k,v,qi,ki):
    return q.index_select(0,qi),_RowGather.apply(k,ki),_RowGather.apply(v,ki)


def _masked_sdpa(q,k,v,mask,dropout):
    has_visible = mask.any(-1,keepdim=True)
    safe = mask.clone(); safe[...,0] |= ~has_visible.squeeze(-1)
    y = F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None],dropout_p=dropout)
    return y.masked_fill(~has_visible[:,None],0)


@contextmanager
def compiled_buckets(*, backend='inductor'):
    """Explicit diagnostic opt-in; never change the checkpoint recipe or startup.

    Specialize query/key widths but allow the number of blocks to vary. Measure
    cold compile, recompiles and warm execution before retaining on a workload.
    """
    kernel = torch.compile(_masked_sdpa,backend=backend,dynamic=False)
    def run(q,k,v,mask,dropout):
        # Only the bucket population is dynamic; Q/K hardware widths stay fixed.
        for value in (q,k,v,mask): torch._dynamo.mark_dynamic(value,0)
        return kernel(q,k,v,mask,dropout)
    # Self and Audio together have more than eight width combinations. Scope
    # this allowance to the explicit diagnostic, never a global startup tweak.
    with torch._dynamo.config.patch(recompile_limit=64,accumulated_recompile_limit=256):
        token = _compiled_bucket.set(run)
        try:
            yield
        finally:
            _compiled_bucket.reset(token)


class _RowGather(torch.autograd.Function):
    """Compact gather with FP32 duplicate-row accumulation for low precision K/V.

    Overlapping windows (and padded Sink keys) cause heavily contended BF16
    atomics. Accumulate in FP32 once, then return the original activation dtype.
    """
    @staticmethod
    def forward(ctx, value, indices):
        ctx.save_for_backward(indices)
        ctx.shape = value.shape
        return value.index_select(0,indices)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        dtype = torch.float32 if gradient.dtype in (torch.float16,torch.bfloat16) else gradient.dtype
        result = torch.zeros(ctx.shape,device=gradient.device,dtype=dtype)
        return result.index_add_(0,indices,gradient.to(dtype)).to(gradient.dtype),None


def local_query_attention(module,query,bank,span,dropout=0.):
    """Fixed-width per-query gathers; absolute query RoPE is preserved."""
    ids,mask=span
    batch,heads,queries,dim=query.shape
    # Gather contiguous whole time rows: [B,Q,W,H,D] -> [BQ,H,W,D].
    # Flattening B/Q before the transpose keeps the last operation a pure view.
    k,v=(t[ids].reshape(batch*queries,ids.shape[-1],heads,dim).transpose(1,2)
         for t in bank.time_major())
    q=query.transpose(1,2).reshape(batch*queries,heads,1,dim)
    y=attend(q,k,v,mask.reshape(batch*queries,1,-1),dropout)
    return y.reshape(batch,queries,heads,dim).transpose(1,2)
