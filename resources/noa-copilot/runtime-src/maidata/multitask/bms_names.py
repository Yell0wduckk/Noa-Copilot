"""Inventory declared keysound names across all difficulties, including conditional branches."""
from collections import Counter
import json
from pathlib import Path
import re

from .bms import _read_text, is_appledouble

_WAV = re.compile(r'^[ \t]*#WAV([0-9a-z]{2})[ \t]+([^\r\n]+)', re.I | re.M)


def declarations(script):
    text = _read_text(script)
    if script.suffix.lower() == '.bmson':
        data = json.loads(text)
        channels = data.get('sound_channels', data.get('soundChannel', []))
        return [(str(index), channel['name']) for index, channel in enumerate(channels)]
    return [(match[1], match[2].split('//')[0].strip().strip('"')) for match in _WAV.finditer(text)]


def name_terms(name):
    stem = Path(name.replace('\\', '/')).stem
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    return set(re.findall(r'[^\W\d_]+', stem.casefold()))


def inventory_project(scripts, project):
    charts, terms, covered = [], {}, Counter()
    for script in sorted(scripts):
        if is_appledouble(script):
            continue
        row = {'chart': script.relative_to(project).as_posix()}
        try:
            entries = declarations(script)
        except (UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            row['error'] = repr(exc)
        else:
            row['declarations'] = entries
            counts = Counter(term for _, name in entries for term in name_terms(name))
            for term, count in counts.items():
                terms[term] = terms.get(term, 0) + count
            covered.update(counts.keys())
        charts.append(row)
    return charts, {t: [n, covered[t], 1] for t, n in terms.items()}
