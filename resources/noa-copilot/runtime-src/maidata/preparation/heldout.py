"""User-owned song exclusions and song-disjoint Master/Re:MASTER selection."""
import json
from pathlib import Path
from functools import lru_cache
from copy import deepcopy


@lru_cache(maxsize=1)
def exclusions():
    path = Path(__file__).resolve().parents[3] / 'configs/excluded-songs.json'
    return (json.loads(path.read_text(encoding='utf-8'))['songs'] if path.exists() else
            [dict(song_id='854',title='全世界共通リズム感テスト')])


def excluded_song(song_id, title=''):
    return any(str(song_id) == row['song_id'] or row['title'] in str(title or '') for row in exclusions())


def validation_inputs(manifest):
    """Apply chart-specific exclusions explicitly approved for this run."""
    result=deepcopy(manifest)
    path=Path(__file__).resolve().parents[3]/'configs/v13-validation-excluded-charts.json'
    excluded=json.loads(path.read_text(encoding='utf-8'))['charts']
    ids={r['chart_id'] for r in excluded}
    result['charts']=[r for r in result['charts'] if r['chart_id'] not in ids]
    variants={r['audio_variant'] for r in result['charts']}
    songs={r['song_id'] for r in result['charts']}
    assets={result['catalog']['variants'][v]['asset'] for v in variants}
    result['selection']['chart_exclusions']=excluded
    result['selection'].setdefault('candidate_songs',len(result['selection']['songs']))
    result['selection']['songs']=[s for s in result['selection']['songs'] if s['song_id'] in songs]
    result['catalog']['songs']=[s for s in result['catalog']['songs'] if s['song_id'] in songs]
    result['catalog']['assets']={k:v for k,v in result['catalog']['assets'].items() if k in assets}
    result['catalog']['variants']={k:v for k,v in result['catalog']['variants'].items() if k in variants}
    kept={r['chart_id'] for r in result['charts']}
    for song in result['catalog']['songs']:
        song['charts']={d:v for d,v in song.get('charts',{}).items() if f"{song['song_id']}:{d}" in kept}
    for variant in result['catalog']['variants'].values():
        if 'charts' in variant:variant['charts']=[k for k in variant['charts'] if k in kept]
    return result


def selection(catalog, train_plan):
    trained = {str(row['song_id']) for row in train_plan['charts']}
    chosen, excluded, unavailable = [], [], []
    for song in catalog['songs']:
        version = ''.join(song['version'].lower().split())
        if song['song_id'] in trained:
            continue
        if (version in ('maimai','maimaiplus') or '全世界共通' in song['title']
                or excluded_song(song['song_id'],song['title'])):
            excluded.append(song)
        elif set(song['charts']) & {'5','6'}:
            chosen.append(song)
        else:
            unavailable.append(song)
    return dict(songs=chosen, excluded=excluded, without_master_remaster=unavailable,
                training_song_count=len(trained))
