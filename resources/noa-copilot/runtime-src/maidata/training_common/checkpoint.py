"""Complete-epoch state and recipe recovery; callers enforce their binding policy."""

import hashlib


from pathlib import Path
import random
import numpy as np
import torch


from maidata.training_common.audio_data import read_json


PATH_KEYS = ('pretrain_train_plan', 'pretrain_validation_plan', 'pretrain_chart_cache',
             'pretrain_audio_cache', 'pretrain_mel_root', 'pretrain_source_root', 'pretrain_output', 'pretrain_p0_cache')

def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_identity():
    """Read the saved-source marker or Git commit for caller-owned identity checks."""
    import subprocess
    root = Path(__file__).resolve().parents[3]
    marker = root / 'SOURCE_COMMIT'
    if marker.is_file():
        return {'commit': marker.read_text(encoding='utf-8').strip()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'commit': commit}

def data_binding(plans, config):
    """Bind metadata for an explicitly requested fresh supervision export."""
    result = {'plans': [], 'paths': {k: str(Path(config[k]).resolve()) for k in PATH_KEYS
                                    if k != 'pretrain_output' and config[k] is not None}}
    for plan in plans:
        names = ('plan.json', *plan.manifest['files'].values())
        result['plans'].append({name: digest_file(plan.root / name) for name in names})
    for key in ('pretrain_chart_cache', 'pretrain_audio_cache'):
        root = Path(config[key])
        result[key] = {name: digest_file(root / name) for name in ('catalog.json', 'production-summary.json')}
    charts = read_json(Path(config['pretrain_chart_cache']) / 'catalog.json')
    audio = read_json(Path(config['pretrain_audio_cache']) / 'catalog.json')
    from maidata.spec_contract import enabled as is_logspec
    if is_logspec(config):
        from maidata.spec_contract import extractor
        RECIPE = extractor(config).RECIPE
        if audio.get('spec_recipe') != RECIPE or audio.get('stats',{}).get('recipe') != RECIPE:
            raise ValueError('audio producer representation differs from LogSpec recipe')
    if charts['schema'] != 'maidata.full-chart-original.v2' or charts['view'] != 'original':
        raise ValueError('incorrect chart cache representation')
    chart_index = {c['chart_id']: c for c in charts['charts']}
    for plan in plans:
        if charts['producer_recipe'] != plan.manifest['sources']['chart_recipe']:
            raise ValueError('chart cache producer recipe differs from plan')
        for variant, group in plan.groups.items():
            actual = audio['variants'][variant]
            if (actual['asset'] != group['audio_asset'] or actual['timing'] != group['timing']
                    or actual['samples_per_beat'] != config['samples_per_beat']):
                raise ValueError('audio producer/plan identity differs')
        for key, chart in plan.charts.items():
            actual = chart_index[key]
            if actual['path'] != chart['chart_cache_path'] or actual['audio_variant'] != chart['audio_variant']:
                raise ValueError('chart catalog/plan identity differs')
    if set(plans[0].groups) & set(plans[1].groups):
        raise ValueError('train/validation audio groups overlap')
    if set(c['song_id'] for c in plans[0].charts.values()) & set(c['song_id'] for c in plans[1].charts.values()):
        raise ValueError('train/validation songs overlap')
    return result

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    if set(state) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('checkpoint is missing complete RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device inventory changed')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    elif state['cuda']:
        raise ValueError('cannot resume CUDA training on CPU')

def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)

def validate_cursor(payload, steps):
    epoch, step = payload['completed_epochs'], payload['step']
    if (type(epoch) is not int or not 1 <= epoch <= len(steps) or type(step) is not int
            or step != sum(steps[:epoch]) or payload['scheduler_state']['last_epoch'] != step):
        raise ValueError('checkpoint is not at a complete epoch boundary')
