"""Disk cache reader and Encoder input adapters; no hidden-state caching."""
from pathlib import Path
from functools import lru_cache
import json

import numpy as np

from .features import silence_mel_from_stats
from .contract import AUDIO_INPUT_CONTRACT


def grid_catalog_path(path):
    """Locate only the two supported adjacent catalogs, without scanning assets."""
    path = Path(path)
    for parent in (path.parent, path.parent.parent):
        catalog = parent / 'catalog.json'
        if catalog.is_file():
            return catalog.resolve()
    return None


@lru_cache(maxsize=32)
def _catalog_silence(path, mtime_ns, size):
    # The producer embeds the accepted fixed stats. Its fixed_stats_path can
    # point to another host, so never reopen that historical path.
    catalog = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if 'stats' not in catalog:
        raise ValueError(f'audio catalog lacks fixed Mel stats: {path}')
    silence = silence_mel_from_stats(catalog['stats'])
    silence.flags.writeable = False
    return silence


def _grid_silence(grid):
    """In-memory callers may explicitly use the mean=0/std=1 toy convention."""
    if 'silence_mel' in grid:
        return grid['silence_mel']
    return np.zeros(grid['mel_patches'].shape[-1], np.float32)


def validate_grid(a):
    if 'ymt3_frames' in a:
        from .ymt3_muq import validate_grid as validate_ymt3_muq
        return validate_ymt3_muq(a)
    if 'spec_frames' in a:
        from .compactspec import validate_grid as validate_compact
        return validate_compact(a)
    mel = a['mel_patches']
    if mel.ndim != 3:
        raise ValueError('mel_patches must be [J,R,M]')
    j,r,m = mel.shape
    shapes = dict(sample_valid=(j,r), beat_centers=(j,), aux32_grid=(j,r,32),
                  aux32_centers=(j,32), sample_times_sec=(j,r), sample_indices=(j,r),
                  grid_k=(j,r), first_beat=())
    if j<1 or r<1 or m<1:
        raise ValueError('empty grid')
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f'{key}: {a[key].shape} != {shape}')
    if a['sample_valid'].dtype != np.bool_:
        raise ValueError('sample_valid must be bool')
    for key in ('mel_patches','beat_centers','aux32_grid','aux32_centers'):
        if a[key].dtype != np.float32 or not np.isfinite(a[key]).all():
            raise ValueError(f'{key} must be finite float32')
    if not np.isfinite(a['sample_times_sec']).all():
        raise ValueError('nonfinite times')
    if not np.array_equal(a['beat_centers'], int(a['first_beat'])+np.arange(j,dtype=np.float32)+.5):
        raise ValueError('non-contiguous absolute beat centers')
    invalid = ~a['sample_valid']
    silence = np.asarray(_grid_silence(a))
    if silence.shape != (m,) or silence.dtype != np.float32 or not np.isfinite(silence).all():
        raise ValueError('silence_mel must be finite float32 [M]')
    outside = mel[invalid]
    # Existing disk payloads use zero padding; read adapters use physical
    # silence. Accept either complete row, never arbitrary invalid data.
    accepted = (outside == 0).all(-1) | (outside == silence).all(-1)
    if not accepted.all() or np.any(a['aux32_grid'][invalid]):
        raise ValueError('invalid positions must be zero or physical silence; aux must be zero')


@lru_cache(maxsize=8)
def _grid_references(catalog, mtime_ns, size):
    return json.loads(Path(catalog).read_text(encoding='utf-8-sig')).get('grid_references',{})


def load_grid(path, *, mel_root=None, stats=None, validate_input=True):
    path = Path(path)
    if not path.is_file():
        catalog = path.parent.parent/'catalog.json'
        if catalog.is_file():
            info = catalog.stat()
            reference = _grid_references(str(catalog),info.st_mtime_ns,info.st_size).get(path.stem)
            if reference is not None: path = Path(reference)
    with np.load(Path(path), allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if 'ymt3_frames' not in result and 'mel_patches' not in result and 'spec_patches' not in result and 'spec_frames' not in result:
        meta_file = path.with_suffix('.json')
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            if 'ymt3_muq_reference' in meta:
                ref = meta['ymt3_muq_reference']
                asset_path = path.parent.parent / ref['asset_path']
                with np.load(asset_path, allow_pickle=False) as asset_archive:
                    result['ymt3_frames'] = asset_archive['ymt3_frames']
                    result['muq_frames'] = asset_archive['muq_frames']
                    if 'silence_ymt3' not in result and 'silence_ymt3' in asset_archive.files:
                        result['silence_ymt3'] = asset_archive['silence_ymt3']
                    if 'silence_muq' not in result and 'silence_muq' in asset_archive.files:
                        result['silence_muq'] = asset_archive['silence_muq']
    if 'ymt3_frames' in result or 'spec_patches' in result or 'spec_frames' in result:
        result['timing'] = json.loads(str(result.pop('timing_json')))
        if validate_input:
            validate_grid(result)
        return result
    if 'mel_patches' not in result:
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        reference = meta['mel_reference']
        # Consumer relocation leaves the producer's metadata/identity untouched.
        from pathlib import PureWindowsPath
        feature = (Path(mel_root) / PureWindowsPath(reference['feature_path']).name
                   if mel_root is not None else Path(reference['feature_path']))
        old = np.load(feature,mmap_mode='r',allow_pickle=False)
        if validate_input and (old.dtype!=np.float32 or list(old.shape)!=reference['shape']):
            raise ValueError('referenced Mel shape/dtype mismatch')
        indices = result['mel_source_indices']
        reused = indices>=0
        if validate_input and (np.any(indices[reused]>=old.shape[1])):
            raise ValueError('referenced Mel index out of range')
        mel = np.zeros((*indices.shape,old.shape[0]),np.float32)
        mel[reused] = old[:,indices[reused]].T
        mel.reshape(-1,old.shape[0])[result['mel_new_flat_indices']] = result['mel_new_values']
        result['mel_patches'] = mel
    if stats is not None:
        silence = silence_mel_from_stats(stats)
    elif (catalog := grid_catalog_path(path)) is not None:
        metadata = catalog.stat()
        silence = _catalog_silence(str(catalog), metadata.st_mtime_ns, metadata.st_size)
    elif 'silence_mel' in result:
        # Standalone extracted grids can carry their already computed vector.
        silence = result['silence_mel']
    else:
        raise ValueError(f'fixed Mel stats required for physical silence: {path}; '
                         'provide stats, an adjacent catalog.json, or embedded silence_mel')
    if silence.shape != (result['mel_patches'].shape[-1],):
        raise ValueError('fixed Mel stats width differs from grid')
    result['silence_mel'] = silence
    if validate_input:
        validate_grid(result)
    # np.load arrays and referenced-Mel assembly are owned by this read. Only
    # this in-memory view changes; the original .npz/.npy remain untouched.
    result['mel_patches'][~result['sample_valid']] = silence
    return result


def song_inputs(grids, *, validate_input=True):
    import torch
    from maidata.audio.types import SongInputs
    if validate_input and (not grids):
        raise ValueError('empty song batch')
    r,m = grids[0]['mel_patches'].shape[1:]
    j = max(len(g['beat_centers']) for g in grids)
    mel = np.zeros((len(grids),j,r,m),np.float32)
    valid = np.zeros((len(grids),j,r),bool)
    centers = np.empty((len(grids),j),np.float32)
    aux = np.zeros((len(grids),j,32),np.float32)
    for i,g in enumerate(grids):
        if validate_input:
            validate_grid(g)
        if validate_input and (g['mel_patches'].shape[1:] != (r,m)):
            raise ValueError('incompatible grid representations in song batch')
        n = len(g['beat_centers'])
        mel[i,:n],aux[i,:n] = g['mel_patches'],g['aux32_centers']
        mel[i,:n][~g['sample_valid']] = _grid_silence(g)
        valid[i,:n] = True
        centers[i] = int(g['first_beat'])+np.arange(j,dtype=np.float32)+.5
    return SongInputs(*(torch.from_numpy(a) for a in (mel,valid,centers,aux)))


def window_inputs(grids, target_starts, window_to_song, *, target_beats, before_beats, after_beats, validate_input=True):
    import torch
    from maidata.audio.types import WindowInputs
    starts = np.asarray(target_starts)
    routes = np.asarray(window_to_song)
    if validate_input and (starts.ndim!=1 or routes.shape!=starts.shape or np.any(starts!=np.floor(starts))):
        raise ValueError('window starts must be integer beats and routes must align')
    if validate_input and (np.any(routes!=np.floor(routes)) or np.any(routes<0) or np.any(routes>=len(grids))):
        raise ValueError('invalid window_to_song')
    if validate_input and (any(not isinstance(v,int) or v<0 for v in (target_beats,before_beats,after_beats)) or target_beats==0):
        raise ValueError('invalid window extent')
    r,m = grids[0]['mel_patches'].shape[1:]
    c = target_beats+before_beats+after_beats
    mel = np.empty((len(starts),c,r,m),np.float32)
    aux = np.zeros((len(starts),c,r,32),np.float32)
    valid = np.ones((len(starts),c,r),bool)
    beats = np.empty((len(starts),c*r),np.float32)
    for w,(start,route) in enumerate(zip(starts,routes)):
        g = grids[int(route)]
        if validate_input and (g['mel_patches'].shape[1:]!=(r,m)):
            raise ValueError('incompatible grid representations')
        first = int(start)-before_beats
        indices = first+np.arange(c)-int(g['first_beat'])
        ok = (indices>=0)&(indices<len(g['beat_centers']))
        silence = _grid_silence(g)
        mel[w] = silence
        selected = g['mel_patches'][indices[ok]]
        selected[~g['sample_valid'][indices[ok]]] = silence
        mel[w,ok],aux[w,ok] = selected,g['aux32_grid'][indices[ok]]
        beats[w] = first+np.arange(c*r,dtype=np.float64)/r
    return WindowInputs(*(torch.from_numpy(a) for a in (
        mel.reshape(len(starts),c*r,m),aux.reshape(len(starts),c*r,32),beats,
        valid.reshape(len(starts),c*r),starts.astype(np.float32),routes.astype(np.int64))))
