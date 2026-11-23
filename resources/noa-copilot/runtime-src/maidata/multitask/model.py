"""V14 tick96 E+D model, with task-separated vocabularies.

Only the supported maimai/MIDI tasks are instantiated by default. The audio
encoder is shared per song; chart metadata only enters the decoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import Attention, self_layout, window_layout
from .conditioning import ChartConditioner, sample_history_seconds, prepare_routing
from .audio_encoder import GatedAudioEncoder
from . import ARCHITECTURE


@dataclass
class ModelConfig:
    d_model: int = 512
    heads: int = 8
    decoder_dense_layers: int = 2
    decoder_expert_layers: int = 2
    decoder_ffn_dim: int = 2048
    shared_experts: int = 6
    shared_expert_dim: int = 1024
    step_rope_dims: int = 8
    tasks: tuple[str, ...] = ("maimai", "midi")
    difficulty_vocab_size: int = 16
    designer_vocab_size: int = 256
    level_ranges: dict = field(default_factory=lambda: {"maimai": (1., 15.), "midi": (0., 1.)})
    attention_query_block: int = 64
    stage0_no_history: bool = False
    stage0_history_seconds: float = 1.5
    stage1_history_seconds: float = 8.
    audio_window_seconds: float = 1.024
    adapter_layers: int = 2
    adapter_kernel: int = 5
    macro_enabled: bool = False
    yourmt3_unfreeze_layers: int = 2
    frontend_batch_windows: int = 128
    dropout: float = .1

    @classmethod
    def from_dict(cls, value):
        return value if isinstance(value, cls) else cls(**value)

@dataclass
class AudioStreams:
    p8: Tensor
    p_times: Tensor
    p_lengths: Tensor
    p_lengths_cpu: Tensor
    position_offset: float = 0.
    pointwise: Tensor | None = None

    def select(self, indices, indices_cpu=None):
        ids = indices.cpu() if indices_cpu is None else torch.as_tensor(indices_cpu)
        return AudioStreams(self.p8.index_select(0, indices), self.p_times.index_select(0, indices),
                            self.p_lengths.index_select(0, indices),
                            self.p_lengths_cpu.index_select(0, ids), self.position_offset)

class DenseFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class HybridMMoE(nn.Module):
    """Packed six shared GELU experts + one task-dedicated full-width expert.

    Both deterministic roles have task-specific gates over all seven experts.
    Packed tensors retain the expert dimension for batched GEMM and Muon.
    """
    def __init__(self, config):
        super().__init__()
        self.tasks = tuple(config.tasks)
        dim, hidden, count = config.d_model, config.shared_expert_dim, config.shared_experts
        self.shared_up = nn.Parameter(torch.empty(count, dim, hidden))
        self.shared_down = nn.Parameter(torch.empty(count, hidden, dim))
        self.shared_up_bias = nn.Parameter(torch.zeros(count, hidden))
        self.shared_down_bias = nn.Parameter(torch.zeros(count, dim))
        for n in range(count):
            nn.init.xavier_uniform_(self.shared_up[n])
            nn.init.xavier_uniform_(self.shared_down[n])
        self.dedicated = nn.ModuleDict({task: DenseFFN(dim, config.decoder_ffn_dim) for task in self.tasks})
        self.router = nn.ModuleDict({task: nn.ModuleList([nn.Linear(dim, count + 1) for _ in range(2)])
                                    for task in self.tasks})
        self.last_gate_weights = None
        self.collect_diagnostics = False
        self.gate_sums = {}
        self.gate_counts = {}

    def forward(self, x: Tensor, task_ids: Tensor, roles: Tensor, routing=None):
        flat = x.reshape(-1, x.shape[-1])
        if routing is None:
            routing = prepare_routing(task_ids.detach().cpu(), torch.full((x.shape[0],), x.shape[1]),
                                      roles.detach().cpu(), x.shape[1], len(self.tasks), x.device)
        valid_rows = torch.cat(routing.token_rows)
        compact = flat.index_select(0, valid_rows)
        # [experts,tokens,width]; one batched matrix multiplication per matrix.
        expanded = compact.unsqueeze(0).expand(self.shared_up.shape[0], -1, -1)
        hidden = F.gelu(torch.baddbmm(self.shared_up_bias[:, None], expanded, self.shared_up))
        shared = torch.baddbmm(self.shared_down_bias[:, None], hidden, self.shared_down)
        shared = shared.permute(1, 0, 2)
        output = shared.new_zeros(flat.shape)
        gate_report = (flat.new_zeros((len(flat), self.shared_up.shape[0] + 1))
                       if self.collect_diagnostics else None)
        role_flat = roles.reshape(-1)
        offset = 0
        for task_id, task in enumerate(self.tasks):
            indices = routing.token_rows[task_id]
            if not len(indices):
                continue
            selected = compact[offset:offset + indices.numel()]
            both_gates = torch.stack([router(selected) for router in self.router[task]], dim=1)
            logits = both_gates.gather(1, role_flat.index_select(0, indices)[:, None, None].expand(-1, 1, both_gates.shape[-1])).squeeze(1)
            weights = logits.softmax(-1)
            task_output = torch.bmm(weights[:, None, :-1].to(shared.dtype),
                                    shared[offset:offset + indices.numel()]).squeeze(1)
            task_output = task_output + self.dedicated[task](selected) * weights[:, -1:].to(task_output.dtype)
            output = output.index_copy_(0, indices, task_output)
            if gate_report is not None:
                gate_report = gate_report.index_copy_(0, indices, weights.detach().to(gate_report.dtype))
            offset += indices.numel()
        self.last_gate_weights = None if gate_report is None else gate_report.view(*x.shape[:2], -1)
        if self.training and self.collect_diagnostics:
            for task, role_groups in zip(self.tasks, routing.role_rows):
                for role, rows in zip(("moment", "note"), role_groups):
                    if rows.numel():
                        key = (task, role, routing.branch)
                        total = gate_report.index_select(0, rows).float().sum(0)
                        self.gate_sums[key] = self.gate_sums.get(key, 0) + total
                        self.gate_counts[key] = self.gate_counts.get(key, 0) + rows.numel()
        return output.view_as(x)


class DecoderLayer(nn.Module):
    def __init__(self, config, expert=False):
        super().__init__()
        dim = config.d_model
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, config.heads, config.step_rope_dims)
        self.p_norm = nn.LayerNorm(dim)
        self.p_attn = Attention(dim, config.heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.adaln = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = HybridMMoE(config) if expert else DenseFFN(dim, config.decoder_ffn_dim)

    def feed_forward(self, x, condition, tasks, roles, routing=None):
        gamma, beta = self.adaln(condition).chunk(2, dim=-1)
        h = self.ffn_norm(x) * (1 + gamma[:, None]) + beta[:, None]
        return x + self.dropout(self.ffn(h, tasks, roles, routing) if isinstance(self.ffn, HybridMMoE) else self.ffn(h))

    def forward(self, x, condition, tasks, roles, steps, causal_times, query_times,
                streams, layouts, audio_source=None, audio_indices=None, audio_kv=None,
                routing=None, auxiliary=None):
        self_attention, p_attention = layouts
        n = self.self_norm(x)
        # Only the clean/main token stream writes self K/V. The stage-0
        # history-aware readout queries never become keys of note-type queries.
        self_kv = self.self_attn.project_memory(n, causal_times, steps)
        x = x + self.self_attn(n, n, causal_times, causal_times, self_attention,
                               query_steps=steps, memory_steps=steps, memory_kv=self_kv)
        if auxiliary is not None:
            ax = auxiliary["x"]
            an = self.self_norm(ax)
            ax = ax + self.self_attn(an, n, auxiliary["causal"], causal_times,
                auxiliary["layouts"][0], query_steps=auxiliary["steps"], memory_kv=self_kv)
        source = streams if audio_source is None else audio_source
        if audio_kv is None:
            p_kv = self.p_attn.project_memory(source.p8, source.p_times)
        else:
            p_kv = audio_kv
        x = x + self.p_attn(self.p_norm(x), streams.p8, query_times, streams.p_times,
                             p_attention, memory_kv=p_kv, memory_indices=audio_indices)
        x = self.feed_forward(x, condition, tasks, roles, routing)
        if auxiliary is None:
            return x
        ax = ax + self.p_attn(self.p_norm(ax), streams.p8, auxiliary["times"], streams.p_times,
                              auxiliary["layouts"][1], memory_kv=p_kv, memory_indices=audio_indices)
        auxiliary["x"] = self.feed_forward(ax, condition, tasks, auxiliary["roles"], auxiliary["routing"])
        return x


@dataclass
class DecoderCache:
    task_id: int
    metadata: dict
    streams: AudioStreams
    audio_kv: list
    self_kv: list
    stage: int = 1
    steps_host: list = field(default_factory=list)
    times_host: list = field(default_factory=list)
    moments_host: list = field(default_factory=list)


class MultitaskModel(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config: ModelConfig | dict | None = None,
                 vocab_sizes: Mapping[str, int] | None = None):
        super().__init__()
        self.config = ModelConfig.from_dict(config or {})
        if vocab_sizes is None:
            from .codec import TaskVocabulary
            vocab_sizes = {task: TaskVocabulary(task).size for task in self.config.tasks}
        self.vocab_sizes = dict(vocab_sizes)
        self.encoder = GatedAudioEncoder(self.config)
        self.conditioner = ChartConditioner(self.config)
        self.embeddings = nn.ModuleDict({task: nn.Embedding(size, self.config.d_model)
                                        for task, size in vocab_sizes.items()})
        self.heads = nn.ModuleDict({task: nn.Linear(self.config.d_model, size)
                                   for task, size in vocab_sizes.items()})
        self.layers = nn.ModuleList([DecoderLayer(self.config, False)
                                    for _ in range(self.config.decoder_dense_layers)] +
                                   [DecoderLayer(self.config, True)
                                    for _ in range(self.config.decoder_expert_layers)])
        self.pointwise = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.output_norm = nn.LayerNorm(self.config.d_model)
        self.encoder_rhythm_head = nn.Linear(self.config.d_model, 24)
        self.decoder_rhythm_head = nn.Linear(self.config.d_model, 38)
        from .timbre_tags import MOMENT_TYPE_DIM
        self.moment_type_head = nn.Linear(self.config.d_model, MOMENT_TYPE_DIM)
        self.last_decoder_hidden: Tensor | None = None

    def encode(self, features, lengths=None, **inputs):
        return self.encoder(features, lengths, **inputs)

    def encode_waveforms(self, waveforms, sample_lengths, speeds=None, **inputs):
        features, lengths = self.encoder.frontend.batch(waveforms, sample_lengths, speeds)
        return self.encode(features, lengths, waveforms=waveforms, sample_lengths=sample_lengths,
                           speeds=speeds, **inputs)

    def encode_batch(self, batch):
        return self.encode(batch["features"], batch.get("audio_lengths"),
            **{key: batch[key] for key in ("waveforms", "sample_lengths", "speeds", "ymt3",
                                           "mert", "muq", "macro_lengths", "macro_origins") if key in batch})

    def pointwise_audio(self, streams, times, audio_indices):
        # Project once per song then gather only query frames, without making
        # chart-count copies of the full audio bank.
        if streams.pointwise is None:
            streams.pointwise = self.pointwise(streams.p8)
        memory = streams.pointwise
        positions = (times - streams.position_offset - .5).clamp_min(0)
        maximum = streams.p_lengths[audio_indices, None] - 1
        positions = torch.minimum(positions, maximum)
        left = positions.long()
        right = torch.minimum(left + 1, maximum)
        low, high = memory[audio_indices[:, None], left], memory[audio_indices[:, None], right]
        return low + (high - low) * (positions - left)[..., None]

    def _embed(self, tokens, task_ids, routing):
        output = self.encoder.input_projection.weight.new_zeros((*tokens.shape, self.config.d_model))
        for i, task in enumerate(self.config.tasks):
            indices = routing.chart_rows[i]
            if len(indices):
                output = output.index_copy(0, indices, self.embeddings[task](tokens[indices]))
        return output

    def _heads(self, x, task_ids, routing):
        x = self.output_norm(x)
        if routing.branch == "main":
            self.last_decoder_hidden = x
        width = max(self.vocab_sizes.values())
        result = None
        for i, task in enumerate(self.config.tasks):
            indices = routing.chart_rows[i]
            if len(indices):
                if len(indices) == len(x):
                    logits = self.heads[task](x)
                    return F.pad(logits, (0, width-logits.shape[-1]), value=-torch.inf) if width!=logits.shape[-1] else logits
                logits = self.heads[task](x.index_select(0,indices))
                if result is None:
                    result = logits.new_full((*x.shape[:2], width), -torch.inf)
                result = result.index_copy_(0, indices, F.pad(logits, (0,width-logits.shape[-1]), value=-torch.inf))
        return result

    def prepare_audio_kv(self, streams: AudioStreams):
        """One differentiable audio projection per layer/song, reusable by charts."""
        return [layer.p_attn.project_memory(streams.p8, streams.p_times) for layer in self.layers]

    def reset_gate_diagnostics(self, enabled=True):
        for layer in self.layers:
            if isinstance(layer.ffn, HybridMMoE):
                layer.ffn.collect_diagnostics = enabled
                layer.ffn.gate_sums.clear()
                layer.ffn.gate_counts.clear()

    def gate_diagnostics(self):
        result = {}
        for index, layer in enumerate(self.layers):
            if not isinstance(layer.ffn, HybridMMoE):
                continue
            result[str(index)] = {}
            totals = {}
            for (task, role, branch), total in layer.ffn.gate_sums.items():
                count = layer.ffn.gate_counts[task, role, branch]
                item = result[str(index)].setdefault(task, {}).setdefault(role, {"queries": 0, "branches": {}})
                item["queries"] += count
                item["branches"][branch] = {"queries": count, "mean": (total / count).detach().cpu().tolist()}
                totals[task, role] = totals.get((task, role), 0) + total
            for (task, role), total in totals.items():
                item = result[str(index)][task][role]
                item["mean"] = (total / item["queries"]).detach().cpu().tolist()
        return result

    def decode(self, batch: dict, streams: AudioStreams, audio_kv=None):
        tokens = batch["token_ids"]
        device, charts, length = tokens.device, *tokens.shape
        cpu = batch.get("_cpu", batch)
        def host(name, default=None):
            value = cpu.get(name, default)
            return torch.as_tensor(value).detach().cpu()
        from . import TASKS
        tasks_cpu = torch.tensor([self.config.tasks.index(TASKS[i]) for i in host("task_ids").long().tolist()])
        lengths_cpu = host("lengths", torch.full((charts,), length)).long()
        times_cpu = host("audio_positions").float()
        causal_cpu = host("onset_seconds", times_cpu/125).float()
        steps_cpu = host("token_steps", torch.arange(length).expand(charts, -1)).long()
        moments_cpu = host("moment_ids").long()
        roles_cpu = host("roles").long()
        audio_ids_cpu = host("audio_indices", torch.arange(charts)).long()
        visibility_cpu = host("history_visibility", (roles_cpu == 0).long()).long()
        routing = prepare_routing(tasks_cpu, lengths_cpu, roles_cpu, length, len(self.config.tasks), device)
        tasks = tasks_cpu.to(device)
        times = batch["audio_positions"].to(device)
        causal = batch["causal_beats"].to(device)
        steps = batch.get("token_steps", steps_cpu).to(device)
        roles = batch["roles"].to(device)
        audio_indices = batch.get("audio_indices", audio_ids_cpu).to(device)
        audio_source = streams
        stage = int(batch.get("stage", 1))
        default_history = (torch.full((charts,), self.config.stage0_history_seconds) if stage == 0
                           else sample_history_seconds(charts, "cpu", self.training).float() * self.config.stage1_history_seconds / 8)
        history = host("history_seconds") if "history_seconds" in cpu else default_history
        history = history.expand(charts)
        raw_meta = batch.get("metadata")
        if raw_meta is None:
            meta_list = [{} for _ in range(charts)]
        elif isinstance(raw_meta, dict):
            meta_list = [{k: (v[n] if isinstance(v, (Tensor, list, tuple)) else v)
                          for k, v in raw_meta.items()} for n in range(charts)]
        else:
            meta_list = list(raw_meta)
        no_history = torch.tensor([
            bool(meta_list[b].get("no_history", False)) or
            (stage == 0 and self.config.stage0_no_history and self.config.tasks[t] != "midi")
            for b, t in enumerate(routing.task_ids)
        ])
        protected = cpu.get("protected_spans", batch.get("protected_spans"))
        query_block = self.config.attention_query_block
        layout = self_layout(causal_cpu, moments_cpu, lengths_cpu, history, no_history,
            protected, query_block, protection_enabled=(~no_history)[:, None].expand(charts, length)).to(device)
        p_times_cpu = ((torch.arange(streams.p8.shape[1]).float() + .5) + streams.position_offset).expand(charts, -1)
        radius = self.config.audio_window_seconds * 125
        p_lengths_cpu = streams.p_lengths_cpu.index_select(0, audio_ids_cpu)
        p_layout = window_layout(times_cpu, p_times_cpu, lengths_cpu, p_lengths_cpu,
                                 radius, radius, query_block).to(device)
        if audio_kv is None:
            audio_kv = self.prepare_audio_kv(streams)
        condition, phase = self.conditioner(tasks, causal, batch.get("metadata"), stage,
            force_unconditional=bool(batch.get("force_unconditional", False)), task_ids_cpu=routing.task_ids)
        states = batch["state_features"].to(device=device)
        state_embedding = self.conditioner.state(states, routing)
        base = (self._embed(tokens, tasks, routing) + phase +
                (steps == 0)[..., None] * condition[:, None] +
                self.pointwise_audio(streams, times, audio_indices))
        x = base + state_embedding * (~no_history).to(device)[:, None, None]
        auxiliary = None
        rows = [torch.nonzero((visibility_cpu[b, :int(lengths_cpu[b])] > 0) & no_history[b],
                             as_tuple=False).flatten() for b in range(charts)]
        aux_lengths = torch.tensor([len(row) for row in rows])
        aux_length = max((len(row) for row in rows), default=0)
        if aux_length:
            columns = torch.zeros(charts, aux_length, dtype=torch.long)
            for b, row in enumerate(rows):
                columns[b, :len(row)] = row
            columns_device = columns.to(device, non_blocking=True)
            def gather(value):
                if value.ndim == 2:
                    return value.gather(1, columns_device)
                return value.gather(1, columns_device[..., None].expand(-1, -1, value.shape[-1]))
            aux_times = times_cpu.gather(1, columns)
            aux_roles = roles_cpu.gather(1, columns)
            aux_self = self_layout(causal_cpu, moments_cpu, lengths_cpu, history, no_history,
                protected, query_block, query_rows=rows).to(device)
            aux_p = window_layout(aux_times, p_times_cpu, aux_lengths, p_lengths_cpu,
                                   radius, radius, query_block).to(device)
            auxiliary = dict(x=gather(base + state_embedding), times=gather(times), causal=gather(causal),
                steps=gather(steps), roles=gather(roles), layouts=(aux_self, aux_p),
                routing=prepare_routing(tasks_cpu, aux_lengths, aux_roles, aux_length, len(self.config.tasks), device,
                                        branch="history_readout"))
        for layer_index, layer in enumerate(self.layers):
            x = layer(x, condition, tasks, roles, steps, causal, times, streams,
                (layout, p_layout), audio_source, audio_indices,
                None if audio_kv is None else audio_kv[layer_index], routing, auxiliary)
        logits = self._heads(x, tasks, routing)
        if auxiliary is not None:
            aux_logits = self._heads(auxiliary["x"], tasks, auxiliary["routing"]).flatten(0, 1)
            destinations = torch.cat([b * length + row for b, row in enumerate(rows)]).to(device)
            take = torch.cat([b * aux_length + torch.arange(len(row)) for b, row in enumerate(rows)]).to(device)
            logits = logits.flatten(0, 1).index_copy(0, destinations, aux_logits.index_select(0, take)).view_as(logits)
        return logits

    def forward(self, batch: dict, streams: AudioStreams | None = None):
        if streams is None:
            streams = self.encode_batch(batch)
        return self.decode(batch, streams)

    @torch.no_grad()
    def init_cache(self, streams: AudioStreams, task_id: int, metadata=None,
                   audio_index: int = 0, stage: int = 1, audio_cache=None):
        from . import TASKS
        from .grid import BeatGrid
        task_id=self.config.tasks.index(TASKS[task_id])
        device = streams.p8.device
        if audio_cache is None:
            streams = streams.select(torch.tensor([audio_index], device=device), [audio_index])
            audio_kv = self.prepare_audio_kv(streams)
        else:
            streams, audio_kv = audio_cache.streams, audio_cache.audio_kv
        meta = dict(metadata or {})
        meta.setdefault("_grid", BeatGrid(meta))
        return DecoderCache(task_id, meta, streams, audio_kv,
                            [None] * len(self.layers), stage)

    @torch.no_grad()
    def step(self, token_id, cache: DecoderCache, *, audio_positions,
             onset_seconds=None, causal_beats=None, query_beats=None, token_steps=None, moment_ids=0, roles=0,
             state_features=None, protected_token_indices=None, retain_active_token_indices=None,
             phase_features=None, history_visibility=None, legal_token_ids=None, **unused):
        """One clean token KV write plus optional history-aware readout query.

        All retention/visibility index decisions use prefix grammar CPU state;
        GPU integer gathers do not invoke nonzero or copy positions to CPU.
        legal_token_ids are the next-token candidates from this same prefix.
        """
        device = cache.streams.p8.device
        onset_seconds = audio_positions/125 if onset_seconds is None else onset_seconds
        token_steps = (cache.steps_host[-1] + 1 if cache.steps_host else 0) if token_steps is None else int(token_steps)
        history_visibility = int(roles == 0) if history_visibility is None else int(history_visibility)
        t = torch.tensor([[audio_positions]], device=device, dtype=torch.float32)
        ct = torch.tensor([[causal_beats]], device=device, dtype=torch.float32)
        st = torch.tensor([[token_steps]], device=device, dtype=torch.long)
        role = torch.tensor([[roles]], device=device, dtype=torch.long)
        task = torch.tensor([cache.task_id], device=device)
        routing = prepare_routing([cache.task_id], [1], [[roles]], 1, len(self.config.tasks), device)
        meta = dict(cache.metadata)
        if phase_features is not None:
            meta["phase_features"] = torch.as_tensor(phase_features, device=device).reshape(1, -1)
        elif meta.get("phase_features") is not None:
            meta["phase_features"] = torch.as_tensor(meta["phase_features"], device=device)[token_steps:token_steps + 1]
        condition, phase = self.conditioner(task, ct, [meta], cache.stage, task_ids_cpu=(cache.task_id,))
        state = torch.zeros((1, 1, 32), device=device) if state_features is None else torch.as_tensor(
            state_features, device=device).reshape(1, 1, 32)
        state_emb = self.conditioner.state(state, routing)
        base = (self.embeddings[self.config.tasks[cache.task_id]](torch.tensor([[token_id]], device=device)) +
                phase + self.pointwise_audio(cache.streams, t, torch.zeros(1, device=device, dtype=torch.long)))
        if token_steps == 0:
            base = base + condition[:, None]
        no_history = (bool(cache.metadata.get("no_history", False)) or
                      (cache.stage == 0 and self.config.stage0_no_history and self.config.tasks[cache.task_id] != "midi"))
        x = base if no_history else base + state_emb
        ax = base + state_emb if no_history and history_visibility else None
        steps = cache.steps_host + [token_steps]
        causal_times = cache.times_host + [float(onset_seconds)]
        moments = cache.moments_host + [int(moment_ids)]
        protected = set([] if protected_token_indices is None else protected_token_indices)
        retained = set([] if retain_active_token_indices is None else retain_active_token_indices)
        if no_history:
            visible_host = [m == int(moment_ids) or s == 0 for m, s in zip(moments, steps)]
        else:
            history = self.config.stage0_history_seconds if cache.stage == 0 else self.config.stage1_history_seconds
            cutoff = min(float(onset_seconds)-history,
                         math.floor(float(onset_seconds)))
            visible_host = [time >= cutoff or s == 0 or s in protected for time, s in zip(causal_times, steps)]
        aux_visible_host = [v or s in protected for v, s in zip(visible_host, steps)]
        keep_host = [i for i, (v, s) in enumerate(zip(aux_visible_host, steps)) if v or s in retained]
        visible = None if all(visible_host) else torch.tensor(visible_host, device=device)[None, None, None]
        aux_visible = (torch.tensor(aux_visible_host, device=device)[None, None, None]
                       if ax is not None and not all(aux_visible_host) else None)
        keep = torch.tensor(keep_host, device=device) if len(keep_host) < len(steps) else None
        offset = cache.streams.position_offset
        radius = self.config.audio_window_seconds * 125
        start = max(0, math.ceil(float(audio_positions) - radius - offset - .5))
        end = min(int(cache.streams.p_lengths_cpu[0]),
                  math.ceil(float(audio_positions) + radius - offset - .5))
        for i, layer in enumerate(self.layers):
            n = layer.self_norm(x)
            q = layer.self_attn.project_query(n, ct, st)
            new_k, new_v = layer.self_attn.project_memory(n, ct, st)
            old = cache.self_kv[i]
            k = torch.cat((old[0], new_k), dim=2) if old is not None else new_k
            v = torch.cat((old[1], new_v), dim=2) if old is not None else new_v
            x = x + layer.self_attn.finish(F.scaled_dot_product_attention(q, k, v, attn_mask=visible))
            if ax is not None:
                aq = layer.self_attn.project_query(layer.self_norm(ax), ct, st)
                ax = ax + layer.self_attn.finish(F.scaled_dot_product_attention(aq, k, v, attn_mask=aux_visible))
            attn, norm = layer.p_attn, layer.p_norm
            ka, va = cache.audio_kv[i]
            if end > start:
                q = attn.project_query(norm(x), t)
                x = x + attn.finish(F.scaled_dot_product_attention(q, ka[:, :, start:end], va[:, :, start:end]))
                if ax is not None:
                    aq = attn.project_query(norm(ax), t)
                    ax = ax + attn.finish(F.scaled_dot_product_attention(aq, ka[:, :, start:end], va[:, :, start:end]))
            x = layer.feed_forward(x, condition, task, role, routing)
            if ax is not None:
                ax = layer.feed_forward(ax, condition, task, role, routing)
            cache.self_kv[i] = (k,v) if keep is None else (k.index_select(2, keep), v.index_select(2, keep))
        cache.steps_host = [steps[i] for i in keep_host]
        cache.times_host = [causal_times[i] for i in keep_host]
        cache.moments_host = [moments[i] for i in keep_host]
        head = self.heads[self.config.tasks[cache.task_id]]
        logits = head(self.output_norm(x))[0, 0]
        if ax is not None:
            return head(self.output_norm(ax))[0, 0]
        return logits

    @torch.no_grad()
    def encode_long(self, features, window_seconds=600., overlap_seconds=60., **inputs):
        if features.ndim == 2:
            features = features[None]
        window, overlap = round(window_seconds*125), round(overlap_seconds*125)
        total=features.shape[1]
        if total<=window:
            local=dict(inputs)
            mert_origin=local.pop('mert_origin',None)
            muq_origin=local.pop('muq_origin',None)
            if 'mert' in local and 'macro_origins' not in local and (mert_origin is not None or muq_origin is not None):
                speed=1. if 'speeds' not in local or local['speeds'] is None else float(local['speeds'][0])
                local['macro_origins']=[(float(.5/75. if mert_origin is None else mert_origin)/speed,
                                         float(.5/25. if muq_origin is None else muq_origin)/speed)]
            return self.encode(features, **local)
        # Inference windows are a semantic long-recording policy, not training
        # recomputation. The farther-from-edge context owns each overlap frame.
        starts=list(range(0,total-overlap,window-overlap))
        pieces=[]
        for number,start in enumerate(starts):
            end=min(total,start+window)
            local={k:v for k,v in inputs.items() if k not in ('mert_origin','muq_origin')}
            if 'waveforms' in inputs:
                left,right=round(start/125*44100),round(end/125*44100)
                local['waveforms']=inputs['waveforms'][:,left:right]
                local['sample_lengths']=[local['waveforms'].shape[1]]
            if 'ymt3' in inputs:
                local['ymt3']=inputs['ymt3'][:,start:end]
            if 'mert' in inputs:
                origins=[]
                for name,rate in (('mert',75.),('muq',25.)):
                    base_origin=float(inputs.get(f'{name}_origin',.5/rate))
                    left=max(0,math.floor(start/125*rate-base_origin*rate))
                    right=min(inputs[name].shape[1],math.ceil(end/125*rate+base_origin*rate)+1)
                    local[name]=inputs[name][:,left:right]
                    origins.append((left+base_origin*rate)/rate-start/125)
                local['macro_origins']=[origins]
                local['macro_lengths']=[(local['mert'].shape[1],local['muq'].shape[1])]
            encoded=self.encoder(features[:,start:end],position_offset=start,**local)
            left=0 if number==0 else (overlap+1)//2
            right=end-start if number==len(starts)-1 else window-overlap//2
            pieces.append(encoded.p8[:,left:right])
        values=torch.cat(pieces,1)
        times=((torch.arange(total,device=values.device).float()+.5))[None]
        return AudioStreams(values,times,torch.tensor([total],device=values.device),torch.tensor([total]))
