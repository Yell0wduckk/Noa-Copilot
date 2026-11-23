"""Training controls; absolute stop and scheduler horizon are independent."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

DEFAULTS = dict(
    stage=0, model={}, manifest=None, output='artifacts/v14-tick96/stage0',
    optimizer='adamw', learning_rate=2e-4, muon_learning_rate=.02,
    betas=[.9, .95], weight_decay=.01, grad_clip=1.,
    warmup_steps=2000, scheduler_epochs=150, stop_after_epochs=75,
    workers=12, audio_groups_per_batch=4, token_budget=80000,
    seed=20261108, device='cuda', precision='bfloat16',
    task_proportions={'midi':1.}, dataset_temperature=.5, dataset_proportions={},
    speed_probability=.3, speed_range=[.8,1.2], max_audio_seconds=600.,
    stage0_checkpoint=None, yourmt3_checkpoint=None, yourmt3_lr_scale=.1,
    encoder_freeze_epochs=0, encoder_warmup_steps=0,
    validation_every=5, validation_subset_seconds=7200, validation_excerpt_seconds=60,
    generation_temperature=0., generation_max_tokens_per_second=1024,
    mirror_probability=.5, velocity_sigma=3., rhythm_loss_weight=.2, synthetic={},
    tag_moment=dict(
        enabled=False,
        include_unconditioned_chart=True,
        unconditioned_moment_only=False,
        max_queries_per_song=4,
        coarse_condition_prob=.5,
        max_coarse_subtypes=2,
        max_collision_rate=.05,
        min_events_per_query=1,
        moment_type_loss_weight=.5,
        residual_neg_weight=.25,
        ll_ignore_threshold=.5,
        ll_ignore_weight=.1,
        type_ll_ignore_threshold=.85,
    ),
)

@lru_cache(maxsize=1)
def _studio_defaults():
    path = Path(__file__).with_name('assets')/'studio-defaults.json'
    return json.loads(path.read_text(encoding='utf-8'))


def studio_options(overrides):
    defaults = _studio_defaults()
    values = {**defaults, **overrides}
    if values.get('reference_audio'):
        from .data import excluded_song
        values['reference_audio'] = [p for p in values['reference_audio']
            if not excluded_song(dict(song_id=Path(p).parent.name,title=str(p)))]
    for key in ('pan','width','role_balance_db','role_eq','preset_vet','drum_lowband_min','drum_lowband_max','drum_max_seconds','cooccurrence_pitch_ranges','preset_vet','drum_lowband_min','drum_lowband_max','drum_max_seconds','cooccurrence_pitch_ranges'):
        values[key] = {**defaults[key], **overrides.get(key,{})}
    return values


def read_studio_options(path, legacy_root=None):
    """Resolve a portable options file identically for offline and online rendering."""
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding='utf-8-sig'))
    def relative(value):
        return str((path.parent/value).resolve())
    for name in ('pattern_bank','sample_bank','studio_profile','preset_index','preset_vetted','preset_root','preset_vetted','preset_root','bms_archive','vital_vst3','voicevox_dir'):
        if not options.get(name):
            continue
        value = options[name]
        absolute = relative(value)
        if name in ('pattern_bank','sample_bank') and not Path(absolute).exists() and legacy_root:
            previous = Path(legacy_root)/value
            if previous.exists():
                absolute = str(previous.resolve())
        options[name] = absolute
    for name in ('reference_audio','room_irs','hall_irs'):
        if name in options:
            options[name] = [relative(p) for p in options[name]]
    def paths(bank):
        return {relative(p):w for p,w in bank.items()} if isinstance(bank,dict) else [relative(p) for p in bank]
    if 'presets' in options:
        bank = options['presets']
        options['presets'] = {k:paths(v) for k,v in bank.items()} if isinstance(bank,dict) else paths(bank)
    return options


def resolve(values=None):
    values = values or {}
    result = deepcopy(DEFAULTS)
    result.update(values)
    result['tag_moment'] = {**DEFAULTS['tag_moment'], **(values.get('tag_moment') or {})}
    if result['synthetic'].get('fraction',0) or result['synthetic'].get('samples_per_epoch'):
        result['synthetic'] = studio_options(result['synthetic'])
    if result['stage'] == 1:
        for key,value in dict(output='artifacts/v14-tick96/stage1', task_proportions={'maimai':.3,'midi':.7},
                              encoder_freeze_epochs=3, encoder_warmup_steps=200, warmup_steps=200).items():
            if key not in values:
                result[key] = value
    return result

def load(path=None):
    return resolve(json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else None)

def recipe(config):
    return {key:value for key,value in resolve(config).items() if key not in
            ('output','stop_after_epochs','device','manifest','yourmt3_checkpoint','stage0_checkpoint')}
