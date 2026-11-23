"""Full-depth v18 type readout with 2-gate RuleMMoE and local P0/G_local KV."""
from dataclasses import dataclass
from functools import cached_property
from bisect import bisect_left
from struct import pack, unpack
import math
import torch
from torch import nn
from maidata.decoder.attention import Attention
from maidata.decoder.blocks import ffn
from maidata.decoder.types import KV, DecoderOutput
from maidata.decoder.projections import linear_parameters
from maidata.nn.experts import dense_expert_values
from maidata.nn.position import apply_rope, rope_tables, dual_rope_tables
from .profiling import section, measured_call
from . import COMPACT_ARCHITECTURE, decoder_windows, integer_boundaries, self_integer_boundaries, moment_conditioning_enabled, stage1_pretrain
from .compact_schema import (ACTIVE_FAMILIES as NOTE_FAMILIES, INPUT_IDS as NOTE_INPUT_IDS,
                             WIDTHS, VOCAB_SIZE, FAMILIES, vocab_gate_map)
from .attention import layouts, rectangular, self_qkv, attend, RollingKV, local_query_attention


def scalar32(value):
    """Match token/grid float32 arithmetic using CPU scalars only."""
    return unpack('f', pack('f', value))[0]


def chart_steps(tokens, layout=None, position=0):
    """Per-chart monotonic integer token step index; packed chunks reset at each chart sink."""
    length = tokens.beats.shape[1]
    idx = torch.arange(position, position + length, device=tokens.beats.device, dtype=torch.long)
    if position == 0 and layout is not None and 'sinks' in layout and layout['sinks'].numel() > 1:
        sinks = layout['sinks'].to(device=idx.device, dtype=torch.long)
        idx = idx - sinks[torch.bucketize(idx, sinks, right=True) - 1]
    return idx


def token_steps(tokens, layout=None, position=0):
    """Per-chart monotonic token step index; packed chunks reset at each chart sink."""
    return chart_steps(tokens, layout, position).to(torch.float32)[None]


@dataclass(frozen=True)
class StreamTokens:
    types: torch.Tensor
    vocab_ids: torch.Tensor
    beats: torch.Tensor
    slide_beats: torch.Tensor
    causal_beats: torch.Tensor

    @property
    def valid(self):
        return torch.ones_like(self.types, dtype=torch.bool)

    def to(self, device):
        return StreamTokens(**{k: v.to(device) for k, v in vars(self).items()})


@dataclass(frozen=True)
class StreamMemory:
    audio: tuple
    beats: torch.Tensor
    valid: torch.Tensor
    beat_modulation: object = None
    condition: object = None
    chart_condition: object = None
    global_audio: object = None
    timing: object = None

    @cached_property
    def grid_cpu(self):
        # One transfer per inference bank, never a per-token search on CUDA.
        return self.beats.detach().cpu().tolist()


@dataclass(frozen=True)
class StreamCache:
    layers: tuple
    length: int
    causal: tuple
    indices: tuple
    integer_boundaries: bool = True
    self_before: float = 8.
    key_types: tuple = ()
    key_vocab_ids: tuple = ()
    moment_state: object = None

    def evict(self, beat, rollback_tokens):
        if self.integer_boundaries:
            beat = math.floor(beat)
        keep = tuple(i for i, (c, n) in enumerate(zip(self.causal, self.indices))
                     if round(c * 96) >= round(beat * 96) - round(self.self_before * 96)
                     or n == 0 or i >= len(self.indices) - rollback_tokens)
        if len(keep) == len(self.indices):
            return self
        ids = torch.tensor(keep, device=self.layers[0].key.device, dtype=torch.long)
        return StreamCache(
            tuple(RollingKV(kv.key.index_select(2, ids), kv.value.index_select(2, ids)) for kv in self.layers),
            self.length,
            tuple(self.causal[i] for i in keep),
            tuple(self.indices[i] for i in keep),
            self.integer_boundaries,
            self.self_before,
            tuple(self.key_types[i] for i in keep) if self.key_types else (),
            tuple(self.key_vocab_ids[i] for i in keep) if self.key_vocab_ids else (),
            self.moment_state,
        )


class RuleMMoE(nn.Module):
    """Input-token tasks share the configured dense experts in every M layer."""
    def __init__(self, dim, middle, experts, dropout, gates=2):
        super().__init__()
        self.experts = nn.ModuleList(ffn(dim, middle, dropout) for _ in range(experts))
        self.gates = nn.ModuleList(nn.Linear(dim, experts) for _ in range(gates))

    def forward(self, x, gate_ids):
        from torch.nn import functional as F
        logits = F.linear(x, *linear_parameters(self.gates))
        logits = logits.unflatten(-1, (len(self.gates), len(self.experts)))
        selected = logits.gather(-2, gate_ids[..., None, None].expand(*gate_ids.shape, 1, len(self.experts))).squeeze(-2)
        values = dense_expert_values(self.experts, x)
        return (selected.float().softmax(-1).to(values.dtype)[..., None] * values).sum(-2)


class TaskRuleMMoE(nn.Module):
    """Four shared experts, two private experts and two input roles per game."""

    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.experts = nn.ModuleList(ffn(dim, hidden, dropout) for _ in range(8))
        self.gates = nn.ModuleList(nn.Linear(dim, 6) for _ in range(4))

        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def weights(self, x, roles, task, dtype):
        start = 2 * task
        logits = torch.nn.functional.linear(x, *linear_parameters(self.gates[start:start + 2]))
        logits = logits.view(-1, 2, 6).gather(1, roles[:, None, None].expand(-1, 1, 6))
        return logits.float().softmax(-1).to(dtype)

    def forward(self, x, roles, mai_rows):
        if mai_rows == 0 or mai_rows == len(x):
            task = int(mai_rows == 0)
            private = 4 + 2 * task
            active = [*self.experts[:4], *self.experts[private:private + 2]]
            values = dense_expert_values(active, x)
            return torch.bmm(self.weights(x, roles, task, values.dtype), values).squeeze(1)
        sections = (slice(0, mai_rows), slice(mai_rows, None))
        shared = dense_expert_values(self.experts[:4], x)
        weights = torch.cat([
            self.weights(x[section], roles[section], task, shared.dtype)
            for task, section in enumerate(sections)
        ])
        shared_mix = torch.bmm(weights[:, :, :4], shared).squeeze(1)
        private_parts = []
        for task, section in enumerate(sections):
            start = 4 + 2 * task
            private = dense_expert_values(self.experts[start:start + 2], x[section])
            w = weights[section, 0]
            private_parts.append(torch.addcmul(private[:, 0] * w[:, 4:5], private[:, 1], w[:, 5:6]))
        return shared_mix + torch.cat(private_parts)


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        d = config['decoder_dim']
        args = (config['decoder_heads'], config['decoder_dropout'],
                config['decoder_rope_base'], config['decoder_rope_position_scale'])
        self.self_attn = Attention(d, d, *args)
        self.audio_attn = Attention(d, config['d_model'], *args)
        self.norms = nn.ModuleList(nn.LayerNorm(d, eps=config['decoder_norm_eps']) for _ in range(3))
        self.ffn = RuleMMoE(d, config['decoder_expert_ffn_dim'], config['decoder_experts'],
                            config['decoder_dropout'], gates=2)
        self.dropout = nn.Dropout(config['decoder_dropout'])
        from maidata.ymt3_muq_contract import enabled as is_ymt3_muq
        self.global_cross = bool(not is_ymt3_muq(config) and config.get('decoder_global_audio_mode') != 'none')
        if self.global_cross:
            self.global_attn = Attention(d, config['d_model'], *args)
            self.global_norm = nn.LayerNorm(d, eps=config['decoder_norm_eps'])
        from maidata.audio.beat_memory import zero_modulation
        self.chart_modulation = zero_modulation(d, d)

    @property
    def norm1(self):
        return self.norms[0]

    @property
    def norm2(self):
        return self.norms[1]

    @property
    def norm3(self):
        return self.norms[2]

    def forward(self, x, tokens, audio, layout, previous=None, position=0, audio_span=None,
                self_mask=None, self_tables=None, audio_tables=None, chart_modulation=None,
                gate_ids=None, global_audio=None, global_layout=None, global_span=None,
                global_visible=None, prepared_audio=None, moment_query=None):
        sa, aa = self.self_attn, self.audio_attn
        n = self.norms[0](x)
        q, k, v, raw = measured_call('self_projection', lambda n: self_qkv(sa, n, tokens.beats, tables=self_tables), n)
        if position == 0:
            # BOUNDARY's K has no rotary transform even at a negative start.
            if layout is not None and 'sinks' in layout:
                k = k.index_copy(2, layout['sinks'], raw.index_select(2, layout['sinks']))
            else:
                k = torch.cat((raw[:, :, :1], k[:, :, 1:]), 2)
        if layout is not None:
            y = rectangular(sa, q, KV(k, v, tokens.valid), layout['self_layout'], self_attention=True)
            cache = None
        else:
            if previous is not None:
                k, v = torch.cat((previous.key, k), 2), torch.cat((previous.value, v), 2)
            y = sa.output(attend(q, k, v, self_mask).transpose(1, 2).flatten(-2))
            cache = RollingKV(k, v)
        if moment_query is not None:
            y = y.masked_fill(moment_query[..., None], 0.0)
        x = x + self.dropout(y)
        with section('audio_query_rope'):
            q = apply_rope(aa.project_query(self.norms[1](x)), tokens.slide_beats, aa.base, aa.scale, tables=audio_tables)
        if layout is not None:
            y = rectangular(aa, q, audio, layout['audio_layout'],
                            prepared=None if prepared_audio is None else prepared_audio.setdefault('local', {}))
        else:
            if len(audio_span) == 2:
                y = aa.output(local_query_attention(aa, q, audio, audio_span).transpose(1, 2).flatten(-2))
            else:
                lo, hi, mask = audio_span
                k, v = (t.narrow(2, lo, hi - lo) for t in (audio.key, audio.value))
                y = aa.output(attend(q, k, v, mask).transpose(1, 2).flatten(-2))
        x = x + self.dropout(y)
        if global_audio is not None:
            ga = self.global_attn
            q = apply_rope(ga.project_query(self.global_norm(x)), tokens.slide_beats,
                           ga.base, ga.scale, tables=audio_tables)
            if global_span is None:
                y = rectangular(ga, q, global_audio, global_layout,
                                prepared=None if prepared_audio is None else prepared_audio.setdefault('global', {}))
            else:
                if len(global_span) == 2:
                    y = ga.output(local_query_attention(ga, q, global_audio, global_span,
                                                        ga.dropout if self.training else 0.).transpose(1, 2).flatten(-2))
                else:
                    lo, hi, mask = global_span
                    k, v = (t.narrow(2, lo, hi - lo) for t in (global_audio.key, global_audio.value))
                    y = ga.output(attend(q, k, v, mask, ga.dropout if self.training else 0.).transpose(1, 2).flatten(-2))
            # An empty window contributes no residual, including output bias.
            visible = global_visible[:, :, None] if global_visible.ndim == 2 else global_visible[None, :, None]
            x = x + self.dropout(y.masked_fill(~visible, 0))
        with section('ffn'):
            normalized = self.norms[-1](x)
            if chart_modulation is not None:
                chart_bank, chart_owners = chart_modulation
                chart_gamma, chart_beta = chart_bank[chart_owners].to(normalized.dtype).chunk(2, -1)
                normalized = normalized * (1 + chart_gamma) + chart_beta
            x = x + self.dropout(measured_call('ffn_kernel', self.ffn, normalized, gate_ids))
        return x, cache


class DenseEventDecoder(nn.Module):
    event_mode = True
    streaming_mode = True
    architecture = COMPACT_ARCHITECTURE

    def __init__(self, config):
        super().__init__()
        from maidata.ymt3_muq_contract import enabled as is_ymt3_muq
        self.dim = config['decoder_dim']
        self.pure_e_main = True
        self.integer_boundaries = integer_boundaries(config)
        self.self_integer_boundaries = self_integer_boundaries(config)
        self.moment_conditioning = moment_conditioning_enabled(config)
        self.stage1_pretrain = stage1_pretrain(config)
        self.history_dropout = bool(config.get('decoder_history_beat_dropout', not self.stage1_pretrain))
        self.step_rope_dims = int(config.get('decoder_step_rope_dims', 8))
        self.step_rope_base = float(config.get('decoder_step_rope_base', 256.))
        self.step_rope_scale = float(config.get('decoder_step_rope_scale', 1.))
        from .meta_conditioning import ChartMetaConditioner
        self.chart_meta = ChartMetaConditioner(config)
        self.windows = decoder_windows(config)
        self.markers = True
        self.notes_only = True
        self.conditioned = False
        self.physical_v6 = True
        self.no_placeholders = True
        self.architecture = config.get('decoder_architecture', COMPACT_ARCHITECTURE)
        self.global_audio = False
        self.global_cross = bool(not is_ymt3_muq(config) and config.get('decoder_global_audio_mode') != 'none')
        self.local_dual_stream = bool(config.get('encoder_local_dual_stream', not is_ymt3_muq(config)))
        self.global_full_song = False
        self.register_buffer('vocab_gate_map', vocab_gate_map(), persistent=False)
        self.widths, self.family_count = WIDTHS, len(FAMILIES)
        remap = torch.zeros(VOCAB_SIZE, dtype=torch.long)
        remap[list(NOTE_INPUT_IDS)] = torch.arange(len(NOTE_INPUT_IDS))
        self.register_buffer('input_remap', remap, persistent=False)
        self.register_buffer('active_families', torch.tensor(NOTE_FAMILIES), persistent=False)
        self.embedding = nn.Embedding(len(NOTE_INPUT_IDS), self.dim)
        if self.moment_conditioning:
            self.moment_pos_emb = nn.Embedding(97, self.dim)
            self.active_occupancy_emb = nn.Embedding(16, self.dim)
            nn.init.normal_(self.moment_pos_emb.weight, mean=0.0, std=0.5)
            nn.init.normal_(self.active_occupancy_emb.weight, mean=0.0, std=0.5)
        self.layers = nn.ModuleList(Block(config, i) for i in range(config['decoder_layers']))
        self.final_norm = nn.LayerNorm(self.dim, eps=config['decoder_norm_eps'])
        self.type_head = nn.Linear(self.dim, len(NOTE_FAMILIES))
        self.heads = nn.ModuleList(
            nn.Linear(self.dim, w) if w > 1 and f != 6 and f in NOTE_FAMILIES else nn.Identity()
            for f, w in enumerate(WIDTHS)
        )
        self.flag_heads = nn.ModuleList(nn.Linear(self.dim, 2) for _ in range(4))

    def reset_self_attention(self):
        """Re-initialize every Decoder layer's self-attention and its Pre-LN for Stage 2 transition."""
        for layer in self._all_blocks():
            nn.init.ones_(layer.norms[0].weight)
            nn.init.zeros_(layer.norms[0].bias)
            for proj in (layer.self_attn.query, layer.self_attn.key, layer.self_attn.value):
                nn.init.orthogonal_(proj.weight, gain=1.0)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            nn.init.orthogonal_(layer.self_attn.output.weight, gain=0.02)
            if layer.self_attn.output.bias is not None:
                nn.init.zeros_(layer.self_attn.output.bias)

    def _all_blocks(self):
        return tuple(self.layers)

    def field_logits(self, hidden, family, context='button'):
        if self.widths[family] == 1:
            return hidden.new_zeros((*hidden.shape[:-1], 1))
        if family != 6:
            return self.heads[family](hidden)
        if context != 'button':
            return self.flag_heads[2 if context == 'touch' else 3](hidden)
        lp = [head(hidden).float().log_softmax(-1) for head in self.flag_heads[:2]]
        return torch.stack([lp[0][..., v & 1] + lp[1][..., (v >> 1) & 1] for v in range(4)], -1)

    def _audio_rope_tables(self, memory, module, indices, prepared):
        # A caller-owned scope covers only this aligned P0/G_local pair.
        key = (module.dim // module.heads, module.base, module.scale, id(indices))
        shared = prepared if self.local_dual_stream else None
        if shared is not None and key in shared:
            return shared[key]
        if indices is None:
            beats = memory.beats.masked_fill(~memory.valid, 0)
        else:
            beats = memory.beats.flatten()
            if len(indices) != memory.valid.numel():
                beats = beats.index_select(0, indices)
            beats = beats[None]
        tables = rope_tables(beats, module.dim // module.heads, module.base, module.scale)
        if shared is not None:
            shared[key] = tables
        return tables

    def prepare_audio(self, memory, *, indices=None, prepared=None):
        tables = self._audio_rope_tables(memory, self.layers[0].audio_attn, indices, prepared)
        banks = []
        for layer in self._all_blocks():
            kv = layer.audio_attn.project(memory, indices=indices, tables=tables)
            # Arrange whole head rows once per shared projection, across M chunks.
            k, v = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (kv.key, kv.value))
            banks.append(KV(k, v, kv.valid))
        return tuple(banks)

    def set_global_audio_enabled(self, enabled):
        # Inference-only override; cached raw Encoder outputs remain reusable.
        self.global_audio_enabled = bool(enabled)

    def prepare_global_audio(self, memory, *, indices=None, prepared=None):
        if not getattr(self, 'global_audio_enabled', True):
            return None
        from .global_audio import GlobalAudio
        module = self.layers[0].global_attn
        if not self.local_dual_stream:
            indices = None
        tables = self._audio_rope_tables(memory, module, indices, prepared)
        banks = []
        for layer in self._all_blocks():
            kv = layer.global_attn.project(memory, indices=indices, tables=tables)
            k, v = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (kv.key, kv.value))
            banks.append(KV(k, v, kv.valid))
        return GlobalAudio(tuple(banks), memory.beats, memory.valid)

    def prepare_chart_meta(self, meta):
        value = self.chart_meta(meta)
        return torch.cat([value, *[layer.chart_modulation(value) for layer in self._all_blocks()]], -1)

    def memory(self, audio_kv, beats, valid, song=0, *, beat_modulation=None, condition=None, chart_condition=None, global_audio=None, timing=None):
        # Packed chunks address the shared [song,frame] bank directly.
        if song is None:
            return StreamMemory(audio_kv, beats, valid, beat_modulation, condition, chart_condition, global_audio, timing)
        if global_audio is not None:
            from dataclasses import replace
            global_audio = replace(global_audio, beats=global_audio.beats[song:song + 1], valid=global_audio.valid[song:song + 1],
                                   layers=tuple(KV(k.key[song:song + 1], k.value[song:song + 1], k.valid[song:song + 1]) for k in global_audio.layers))
        # Views, not repeated projection or copied full-song banks per chart.
        return StreamMemory(tuple(KV(k.key[song:song + 1], k.value[song:song + 1], k.valid[song:song + 1])
                                  for k in audio_kv), beats[song], valid[song], beat_modulation, condition, chart_condition, global_audio, timing)

    def forward(self, tokens, memory, *, layout=None, cache=None, need_cache=False, audio_span=None,
                positions=None, read_type=True, no_history=False, prepared=None):
        if layout is None and not need_cache:
            layout = layouts(tokens.causal_beats[0], tokens.slide_beats[0], memory.beats, memory.valid,
                             integer_boundaries=self.integer_boundaries, types=tokens.types[0],
                             vocab_ids=tokens.vocab_ids[0], stage1_pretrain=self.stage1_pretrain,
                             self_integer_boundaries=self.self_integer_boundaries,
                             **self.windows)
            layout = {k: {n: v.to(tokens.types.device) if isinstance(v, torch.Tensor) else v for n, v in b.items()}
                      for k, b in layout.items()}
        if self.history_dropout and self.training and not need_cache and not no_history and not self.stage1_pretrain:
            from .meta_conditioning import apply_self_history_dropout
            layout = apply_self_history_dropout(layout, tokens, len(memory.chart_condition))
        global_layout = global_span = global_visible = None
        if memory.global_audio is not None and layout is not None:
            global_layout = layout['audio_layout'] if self.local_dual_stream else layout.get('global_audio_layout')
            if global_layout is None:
                from .global_audio import layout as global_geometry
                from maidata.training_common.audio_data import map_tensors
                owners = layout.get('song_indices', torch.zeros_like(tokens.types))
                global_layout = global_geometry(tokens.slide_beats, owners, *memory.global_audio.cpu_geometry, full_song=self.global_full_song)
                global_layout = map_tensors(global_layout, lambda t: t.to(tokens.types.device))
        self_mask = None
        position = 0 if cache is None else cache.length
        key_types = key_vocab_ids = ()
        next_moment_state = None if cache is None else cache.moment_state
        if layout is None:
            current, cross = positions if positions is not None else (
                tokens.causal_beats[0].detach().cpu().tolist(), tokens.slide_beats[0].detach().cpu().tolist())
            cur_types = tuple(tokens.types[0].detach().cpu().tolist())
            cur_vocab = tuple(tokens.vocab_ids[0].detach().cpu().tolist())
            causal = (() if cache is None else cache.causal) + tuple(current)
            indices = (() if cache is None else cache.indices) + tuple(range(position, position + len(current)))
            key_types = (() if cache is None else cache.key_types) + cur_types
            key_vocab_ids = (() if cache is None else cache.key_vocab_ids) + cur_vocab
            key_ticks, query_ticks = tuple(round(c * 96) for c in causal), tuple(round(q * 96) for q in current)
            self_int_bounds = self.self_integer_boundaries
            self_mask = torch.tensor([[n == 0 or (
                (not self.stage1_pretrain or (kt != 12 and not (kt == 0 and kv == 96)))
                and ((q // 96 * 96 if self_int_bounds else q) - round(96 * self.windows.get('self_before', 8.)) <= c <= q and n <= position + i)
            ) for c, n, kt, kv in zip(key_ticks, indices, key_types, key_vocab_ids)] for i, q in enumerate(query_ticks)],
                device=tokens.types.device)
            if self.stage1_pretrain:
                for i, (qt, qv) in enumerate(zip(cur_types, cur_vocab)):
                    if qt in (16, 12) or (qt == 0 and qv == 96):
                        self_mask[i, 1:] = False
                        self_mask[i, 0] = True
        if layout is None and audio_span is None:
            q = tokens.slide_beats[0]
            if self.integer_boundaries:
                q = q.floor()
                cross = [math.floor(p) for p in cross]
            lo = bisect_left(memory.grid_cpu, scalar32(scalar32(min(cross)) - self.windows.get('audio_before', 8.)))
            hi = bisect_left(memory.grid_cpu, scalar32(scalar32(max(cross)) + self.windows.get('audio_after', 8.)))
            mask = ((memory.beats[None, lo:hi] >= q[:, None] - self.windows.get('audio_before', 8.)) &
                    (memory.beats[None, lo:hi] < q[:, None] + self.windows.get('audio_after', 8.)) & memory.valid[None, lo:hi])
            audio_span = lo, hi, mask
        if self.local_dual_stream and memory.global_audio is not None and layout is None:
            # Identical P0/G_local coordinates and masks: reuse the actual inference span.
            global_span = audio_span
            global_visible = audio_span[-1].any(-1)
        emb = self.embedding(self.input_remap[tokens.vocab_ids])
        x = emb
        chart_owners = torch.zeros_like(tokens.types) if layout is None else layout.get('chart_indices', torch.zeros_like(tokens.types))
        chart = memory.chart_condition
        if position == 0 and chart is not None:
            sinks = layout.get('sinks') if layout is not None else None
            if sinks is None:
                sinks = torch.zeros(1, device=x.device, dtype=torch.long)
            # Only gather the few boundary rows, not N_tokens x dim.
            chart_sink = (sinks, chart[:, :self.dim][chart_owners[:, sinks]])
            x = x.index_add(1, sinks, chart_sink[1].to(x.dtype))
        moment_query = None
        if self.moment_conditioning or self.stage1_pretrain:
            if layout is not None and 'moment_query' in layout and 'moment_pos_id' in layout and 'occupancy_id' in layout:
                mq = layout['moment_query']
                pos_id = layout['moment_pos_id']
                occ_id = layout['occupancy_id']
                if mq.ndim == 1:
                    mq, pos_id, occ_id = mq[None], pos_id[None], occ_id[None]
            elif layout is not None:
                from .moment_conditioning import prepare_moment_conditioning
                cond = prepare_moment_conditioning(
                    tokens,
                    getattr(memory, 'timing', None),
                    sinks=layout.get('sinks'),
                    physical_v6=self.physical_v6,
                    no_placeholders=self.no_placeholders,
                )
                mq, pos_id, occ_id = cond['moment_query'], cond['moment_pos_id'], cond['occupancy_id']
            else:
                from .moment_conditioning import step_moment_conditioning
                cond, next_moment_state = step_moment_conditioning(
                    None if cache is None else cache.moment_state,
                    tokens.types,
                    tokens.vocab_ids,
                    tokens.causal_beats,
                    getattr(memory, 'timing', None),
                    physical_v6=self.physical_v6,
                    no_placeholders=self.no_placeholders,
                )
                mq = cond['moment_query']
                pos_id = cond['moment_pos_id']
                occ_id = cond['occupancy_id']
            if self.moment_conditioning:
                pos_emb = self.moment_pos_emb(pos_id)
                occ_emb = self.active_occupancy_emb(occ_id) * mq.to(x.dtype)[..., None]
                x = x + pos_emb + occ_emb
            if self.stage1_pretrain:
                moment_query = mq
        reusable = prepared is not None
        if global_layout is not None:
            if reusable and 'global_visible' in prepared:
                global_visible = prepared['global_visible']
            else:
                from .attention import query_visibility
                global_visible = query_visibility(global_layout, x.device, memory.global_audio.valid)
                if reusable:
                    prepared['global_visible'] = global_visible
        if reusable and 'geometry' in prepared:
            self_tables, audio_tables = prepared['geometry']
        else:
            module = self.layers[0].self_attn
            head_dim = module.dim // module.heads
            self_tables = (dual_rope_tables(tokens.beats, token_steps(tokens, layout, position), head_dim,
                                            self.step_rope_dims, module.base, module.scale, self.step_rope_base, self.step_rope_scale)
                           if self.step_rope_dims else rope_tables(tokens.beats, head_dim, module.base, module.scale))
            audio_tables = rope_tables(tokens.slide_beats, head_dim, module.base, module.scale)
            if reusable:
                prepared['geometry'] = self_tables, audio_tables
        gate_ids = self.vocab_gate_map[tokens.vocab_ids]
        layers = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, tokens, memory.audio[i], layout,
                          None if cache is None else cache.layers[i], position, audio_span,
                          self_mask, self_tables, audio_tables,
                          None if chart is None else (chart[:, self.dim + i * 2 * self.dim:self.dim + (i + 1) * 2 * self.dim], chart_owners),
                          gate_ids=gate_ids,
                          global_audio=None if memory.global_audio is None else memory.global_audio.layers[i],
                          global_layout=global_layout, global_span=global_span, global_visible=global_visible,
                          prepared_audio=None if not reusable or getattr(self, 'recompute', False) else prepared.setdefault('audio_gathers', {}).setdefault(i, {}),
                          moment_query=moment_query)
            layers.append(kv)
        hidden = self.final_norm(x)
        result = StreamCache(
            tuple(layers), position + x.shape[1], causal, indices,
            self.self_integer_boundaries, self.windows.get('self_before', 8.),
            key_types, key_vocab_ids, next_moment_state,
        ) if need_cache else None
        return DecoderOutput(self.type_logits(hidden) if read_type else None, hidden, result, ())

    def type_logits(self, hidden):
        logits = self.type_head(hidden)
        return logits.new_full((*logits.shape[:-1], self.family_count), -torch.inf).index_copy(
            -1, self.active_families, logits)
