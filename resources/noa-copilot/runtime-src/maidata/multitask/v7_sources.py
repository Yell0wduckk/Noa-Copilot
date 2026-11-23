"""Metadata-only source index for an explicitly selected v7 chart cache.

Old tensors, audio grids, waveforms and chart bodies are never read here.
Source maidata headers supply chart-specific levels and designer names.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def _path(value, base):
    return (base / value).resolve()


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _optional_charts(value, base):
    if not value:
        return {}, None
    path = _path(value, base)
    if not path.is_file():
        return {}, path
    return {row['chart_id']: row for row in _json(path).get('charts', [])}, path


def _headers(path):
    if not path.is_file():
        return {}
    result = {}
    with path.open(encoding='utf-8-sig', errors='replace') as stream:
        for line in stream:
            if line.lstrip().startswith('&inote_'):
                break
            key, separator, value = line.strip().partition('=')
            if separator and key.startswith('&'):
                result[key[1:]] = value.strip()
    return result


def _original_audio(root, asset, record):
    relative = record.get('relative_source_path')
    if relative:
        normalized = relative.replace('\\', '/')
        for prefix in ('data/extracted/', 'dataset/sources/', 'sources/'):
            if normalized.startswith(prefix):
                return _path(normalized[len(prefix):], root)
        path = Path(relative)
        if path.is_absolute():
            return path
        return _path(path, root)
    # The accepted audio catalog preserves the source directory identity even
    # when the original normalization manifest was not copied with the cache.
    return root / str(asset['source_asset_id']) / 'track.mp3'


def build_source_index(source, plan, audio_catalog=None, source_root=None, designer_vocab=None):
    """Build rows in v7 manifest order, retaining the original plan's bindings.

    source/plan accept their directory or JSON path. The optional catalog and
    source-root arguments relocate metadata/audio without touching old assets.
    designer_vocab keeps caller-supplied IDs (index + 1); absent vocab uses
    chart-frequency Top-255 with lexical tie-breaking and unknown ID zero.
    """
    source, plan = Path(source).resolve(), Path(plan).resolve()
    source_file = source / 'manifest.json' if source.is_dir() else source
    plan_file = plan / 'plan.json' if plan.is_dir() else plan
    manifest, plan_data = _json(source_file), _json(plan_file)
    sources = plan_data.get('sources', {})
    catalog_file = (Path(audio_catalog).resolve() if audio_catalog is not None else
                    _path(sources['audio_catalog'], plan_file.parent))
    catalog = _json(catalog_file)
    root = (Path(source_root).resolve() if source_root is not None else
            _path(catalog['source_root'], catalog_file.parent))
    wav_file = _path(catalog['wav_manifest'], catalog_file.parent) if catalog.get('wav_manifest') else None
    wav_records = {}
    if wav_file is not None and wav_file.is_file():
        with wav_file.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    wav_records[item['output_sha256']] = item
    chart_catalog, chart_catalog_file = _optional_charts(sources.get('chart_catalog'), plan_file.parent)
    selection, selection_file = _optional_charts(sources.get('selection'), plan_file.parent)
    plan_charts = {row['chart_id']: row for row in plan_data['charts']}
    audio_groups = {row['audio_variant']: row for row in plan_data['audio_groups']}
    songs = {str(row['song_id']): row for row in catalog.get('songs', [])}
    header_cache, records = {}, []
    for chart_id in manifest['charts']:
        original = plan_charts[chart_id]
        details = {**selection.get(chart_id, {}), **chart_catalog.get(chart_id, {}), **original}
        song_id = str(original['song_id'])
        variant = original['audio_variant']
        group = audio_groups[variant]
        # Group binding is authoritative: do not substitute another variant
        # just because it belongs to the same song.
        asset_id = group['audio_asset']
        asset = catalog['assets'][asset_id]
        wav = wav_records.get(asset_id, {})
        audio = _original_audio(root, asset, wav)
        chart_root = _path(original.get('chart_cache_root') or sources['chart_cache'], plan_file.parent)
        chart_cache = _path(original['chart_cache_path'], chart_root)
        source_path = (root / song_id / Path(details.get('source_path') or 'maidata.txt').name
                       if source_root is not None else
                       _path(details['source_path'], plan_file.parent) if details.get('source_path') else
                       root / song_id / 'maidata.txt')
        if source_path not in header_cache:
            header_cache[source_path] = _headers(source_path)
        headers = header_cache[source_path]
        difficulty = int(original['difficulty'])
        metadata = deepcopy(details.get('metadata', {}))
        for field in ('level', 'level_value', 'designer_name', 'title', 'artist', 'version'):
            if details.get(field) is not None:
                metadata[field] = details[field]
        for field in ('title', 'artist', 'version'):
            if field in headers:
                metadata.setdefault(field, headers[field])
            elif songs.get(song_id, {}).get(field) is not None:
                metadata.setdefault(field, songs[song_id][field])
        raw_level = headers.get(f'lv_{difficulty}')
        if raw_level:
            metadata['level_raw'] = raw_level
            try:
                metadata['level'] = float(raw_level)
            except ValueError:
                metadata.setdefault('level', metadata.get('level_value'))
        else:
            metadata.setdefault('level', metadata.get('level_value'))
        name = headers.get(f'des_{difficulty}', metadata.get('designer_name'))
        metadata.update(difficulty=difficulty, song_id=song_id, source_path=str(source_path),
                        designer_name=name or None)
        timing = group.get('timing') or details.get('timing') or catalog['variants'][variant]['timing']
        from maidata.time import TimingMap
        clock = TimingMap.from_audio_timing(timing, validate_input=False)
        metadata.update(timing=deepcopy(timing), offset_seconds=clock.first_seconds,
                        bpm_segments=[[second, bpm] for second, (_, bpm) in zip(clock.seconds, clock.points)])
        row = dict(id=chart_id, task='maimai', dataset='maimai', split='train', song_id=song_id,
                   audio=str(audio), audio_id=asset.get('source_sha256') or wav.get('source_sha256') or str(audio),
                   chart_cache=str(chart_cache), audio_variant=variant, legacy_audio_asset=asset_id,
                   metadata=metadata)
        records.append(row)
    counts = Counter(row['metadata']['designer_name'] for row in records
                     if row['metadata']['designer_name'] not in (None, '', '-'))
    supplied = designer_vocab is not None
    vocabulary = list(designer_vocab) if supplied else sorted(counts, key=lambda name: (-counts[name], name))[:255]
    mapping = {}
    for i, name in enumerate(vocabulary):
        mapping.setdefault(name, i + 1)
    for row in records:
        row['metadata']['designer'] = mapping.get(row['metadata']['designer_name'], 0)
    provenance = dict(manifest=str(source_file), schema=manifest.get('schema'),
        architecture=manifest.get('architecture'), producer=deepcopy(manifest.get('producer')),
        plans=deepcopy(manifest.get('plans')), plan=str(plan_file), audio_catalog=str(catalog_file),
        source_root=str(root), wav_manifest=str(wav_file) if wav_file else None,
        chart_catalog=str(chart_catalog_file) if chart_catalog_file else None,
        selection=str(selection_file) if selection_file else None,
        metadata_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (('manifest', source_file), ('plan', plan_file), ('audio_catalog', catalog_file))})
    return dict(records=records, metadata=dict(v7_source=provenance,
        designer_vocab=vocabulary, designer_ids=mapping, designer_unknown_id=0,
        designer_vocab_rule='provided-order-index-plus-one' if supplied else 'selected-chart-frequency-top255-lexical-ties',
        source_chart_count=len(records), source_audio_count=len({row['audio_id'] for row in records})))
