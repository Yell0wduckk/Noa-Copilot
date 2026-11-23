"""Natural integer-beat chart bounds, without 16-beat rounding or tail windows."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WholeCoverage:
    first_beat: int
    end_beat: int
    span_beats: float

    @property
    def target_beats(self): return 16

    @property
    def draft_beats(self): return self.end_beat-self.first_beat


def coverage(timing,duration,first_beat=0):
    end = timing.second_to_beat(duration)
    return WholeCoverage(first_beat,max(first_beat+1,math.ceil(end)),end-timing.second_to_beat(0))
