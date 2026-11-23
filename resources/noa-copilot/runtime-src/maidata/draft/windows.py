"""Song coverage metadata."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SongCoverage:
    first_beat: int
    audio_windows: int
    target_beats: int
    span_beats: float

    @property
    def draft_beats(self):
        return (self.audio_windows + 1) * self.target_beats

    def starts(self, *, include_tail):
        return tuple(self.first_beat + i * self.target_beats
                     for i in range(self.audio_windows + int(include_tail)))


def song_coverage(timing, audio_end_seconds, *, first_beat, target_beats):
    start, end = timing.second_to_beat(0), timing.second_to_beat(audio_end_seconds)
    return SongCoverage(first_beat, math.ceil((end - first_beat) / target_beats),
                        target_beats, end - start)
