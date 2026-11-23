"""Numeric metadata input, independent of the networks and chart event labels."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re


@dataclass(frozen=True)
class ChartCondition:
    audio_variant: str
    difficulty_id: int
    level: float
    level_text: str

    @property
    def key(self):
        return self.audio_variant, self.level, self.difficulty_id


def read_condition(source, *, numeric_level=None, validate_input=True):
    """Read only this chart's source level; explicit numeric metadata may replace it."""
    difficulty = str(source['difficulty'])
    if validate_input and (difficulty not in tuple(str(i) for i in range(1, 8))):
        raise ValueError('source difficulty must be in 1..7')
    raw = source.get('level_text', '')
    path = Path(source['source_path']) if source.get('source_path') else None
    if path is not None and path.is_file():
        text = path.read_text(encoding='utf-8-sig')
        match = re.search(rf'^&lv_{difficulty}=([^\r\n]*)', text, re.M)
        raw = match[1].strip() if match else raw
    elif validate_input and (numeric_level is None and not raw):
        raise ValueError('source text or bound exact numeric level metadata is required')
    value = raw if numeric_level is None else str(numeric_level)
    try:
        level = Decimal(value)
        if validate_input and (not level.is_finite() or not float('-inf') < float(level) < float('inf')):
            raise InvalidOperation()
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError(f'chart {source.get("chart_id", difficulty)} requires an exact numeric level; got {raw!r}') from exc
    return ChartCondition(str(source['audio_variant']), int(difficulty), float(level), raw)
