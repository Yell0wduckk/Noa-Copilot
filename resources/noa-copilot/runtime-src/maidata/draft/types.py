"""Audio memory, chart metadata, and projected drafter keys/values."""
from dataclasses import dataclass
from torch import Tensor


@dataclass
class BeatMemory:
    values: Tensor
    beats: Tensor
    valid: Tensor


@dataclass
class DraftInputs:
    query_beats: Tensor
    draft_valid: Tensor
    audio_mapping: Tensor
    difficulty_id: Tensor
    level: Tensor
    span_beats: Tensor
    valid_indices: Tensor | None = None
    all_valid: bool = False
    charter_id: Tensor | None = None
    task_id: Tensor | None = None
    mai_anchors: int | None = None


@dataclass
class ProjectedAudio:
    key_values: tuple[tuple[Tensor, Tensor], ...]
    valid: Tensor
