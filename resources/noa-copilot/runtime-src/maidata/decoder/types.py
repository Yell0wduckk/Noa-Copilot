"""Tensor contracts, independent of Encoder, Draft and Physical implementations."""
from dataclasses import dataclass, fields
import torch
from torch import Tensor


@dataclass(frozen=True)
class Memory:
    values: Tensor
    beats: Tensor
    valid: Tensor


@dataclass(frozen=True)
class TokenBatch:
    types: Tensor
    beats: Tensor
    valid: Tensor
    kind: Tensor
    pad: Tensor
    flags: Tensor  # [B,L,5]; -1 = NA, never class zero
    route: Tensor
    track_break: Tensor
    time: Tensor  # v5: step beats, boundary zero; legacy inference/v6 storage retain M seconds

    slide_beats: Tensor | None = None
    causal_beats: Tensor | None = None  # duration CONT uses its note onset, never its endpoint

    def __post_init__(self):
        if self.slide_beats is None:
            object.__setattr__(self, "slide_beats", torch.where(self.types >= 7,self.beats,self.beats + 1))
        if self.causal_beats is None:
            object.__setattr__(self, "causal_beats", self.beats)

    def slice(self, start, stop):
        return TokenBatch(**{f.name: getattr(self, f.name)[:, start:stop] for f in fields(self)})

    def to(self, device):
        return TokenBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class KV:
    key: Tensor
    value: Tensor
    valid: Tensor
    query_layout: dict | None = None
    valid_context: tuple | None = None  # CPU layout, runtime valid tensor, mutation version
    compact_indices: Tensor | None = None  # [N_key] source positions for [N_key,H,D] condition K/V

    def time_major(self):
        """Shared-song [T,H,D] K/V, packed once for per-query row gathers.

        Prepared audio already has this layout, so contiguous() is a view-only
        no-op there. Legacy head-major banks pay the conversion once per bank.
        """
        tensors = (self.key, self.value)
        def pack():
            return tuple(t[0].transpose(0, 1).contiguous() for t in tensors)
        if torch.is_grad_enabled():
            # A legacy bank may outlive a backward. Scope its copy graph just
            # like parameter packs; outside a forward scope retain no graph.
            from .projections import scoped_parameter_pack
            return scoped_parameter_pack(tensors, 'audio_time_major', pack)
        try:
            versions = tuple(t._version for t in tensors)
        except RuntimeError:
            versions = None
        signature = (tuple(id(t) for t in tensors), versions, torch.is_grad_enabled())
        cached = getattr(self, '_time_major_cache', None)
        if versions is not None and cached is not None and cached[0] == signature:
            return cached[1]
        result = pack()
        if versions is not None:
            object.__setattr__(self, '_time_major_cache', (signature, result))
        return result

    def padded(self):
        """Restore the public bank only for incremental or dynamic-mask fallbacks."""
        if self.compact_indices is None:
            return self
        def restore(t):
            return t.new_zeros((self.valid.numel(), *t.shape[1:])).index_copy(
                0, self.compact_indices, t).reshape(*self.valid.shape, *t.shape[1:]).transpose(1, 2)
        return KV(restore(self.key), restore(self.value), self.valid, self.query_layout, self.valid_context)


@dataclass(frozen=True)
class DecoderMemory:
    audio: tuple[KV, ...]
    condition: tuple[KV, ...]
    plan_index: int | None
    row_indices: Tensor | None = None
    condition_beats: Tensor | None = None
    before_query: object = None  # inference-only transactional block refresher


@dataclass(frozen=True)
class DecoderCache:
    """Immutable self K/V; training probe-only caches leave front entries None."""
    layers: tuple[KV | None, ...]
    memory: DecoderMemory  # prevents mixing self caches from different windows


@dataclass(frozen=True)
class RouterOutput:
    hidden: Tensor
    logits: Tensor
    beats: Tensor
    valid: Tensor
    front_kv: tuple[KV, ...]
    memory: DecoderMemory
    previous: DecoderCache | None
    slide_beats: Tensor
    attention_cache: dict | None = None
    packed_layout: dict | None = None  # hidden stays compact; public logits stay [B,L,7]
    need_cache: bool = True
    causal_beats: Tensor | None = None  # generation onset; duration CONT keeps its origin
    kind: Tensor | None = None  # resolved input context for PAD/FLAGS dispatch
    route: Tensor | None = None


@dataclass(frozen=True)
class DecoderOutput:
    type_logits: Tensor | None  # streaming training reads only supervised rows
    hidden: Tensor
    cache: DecoderCache | None
    mtp_hidden: tuple[Tensor, ...] = ()
