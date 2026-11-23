"""Operational generation defaults shared by the CLI and Python API."""
from dataclasses import dataclass
import secrets


def resolve_seed(seed):
    """Resolve the random sentinel once, without changing global RNG state."""
    return secrets.randbelow(2**31) if seed == -1 else seed


def validate_slide_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError('max_slide_judge_areas must be a positive integer')


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = .4
    seed: int = 20261010
    max_tokens_per_window: int = 512
    max_tokens_per_moment: int = 20
    max_backtracks_per_window: int = 128
    max_backtracks_per_moment: int = 8
    rollback_tokens: int = 16
    physical_hard_mask: bool = True
    physical_enabled: bool = True
    trace: bool = False
    top_k: int = 0
    top_p: float = 1.
    max_slide_judge_areas: int = 40
    trajectory_history_beats: int = 4
    trajectory_power: float = 2.
    trajectory_moment: bool = False
    trajectory_nonmoment: bool = True
    trajectory_moment_penalty: float = 1.
    trajectory_nonmoment_penalty: float = 1.

    def __post_init__(self):
        validate_slide_limit(self.max_slide_judge_areas)
        object.__setattr__(self, 'seed', resolve_seed(self.seed))
