"""Worker-prepared audio, Draft and Plan conditions for joint training."""

from dataclasses import dataclass
import torch


from maidata.decoder.types import Memory


@dataclass
class Conditions:
    audio: Memory
    decoder_z: Memory
    plan: torch.Tensor
    plan_valid: torch.Tensor
    occurrence_mapping: torch.Tensor
    audio_mapping: torch.Tensor
    audio_indices: torch.Tensor | None = None
    plan_beats: torch.Tensor | None = None
    pattern_embedding: torch.Tensor | None = None
    condition_kv: object = None
    chart_meta: object = None
    chart_condition: torch.Tensor | None = None
    global_audio: object = None
