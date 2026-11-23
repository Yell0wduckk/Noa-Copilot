"""Whole-chart Simai v8 compact physical streaming contracts."""

PHYSICAL_ARCHITECTURE = 'maidata.decoder-physical-state-rule-mmoe-tick96.v17'
PHYSICAL_JOINT = 'maidata.joint-physical-state-rule-mmoe.v10'
PHYSICAL_CACHE_SCHEMA = 'maidata.whole-chart-physical-cache.v6'
PHYSICAL_V7_CACHE_SCHEMA = 'maidata.whole-chart-physical-cache.v7'

COMPACT_ARCHITECTURE = 'maidata.decoder-physical-no-placeholders-rule-mmoe-tick96.v18'
COMPACT_JOINT = 'maidata.joint-physical-no-placeholders-rule-mmoe.v11'
PHYSICAL_V8_CACHE_SCHEMA = 'maidata.whole-chart-physical-cache.v8'

ARCHITECTURE = COMPACT_ARCHITECTURE
JOINT_ARCHITECTURE = COMPACT_JOINT
CACHE_SCHEMA = PHYSICAL_V8_CACHE_SCHEMA


def compact(config):
    return True


def physical_v6(config):
    return True


def markers(config):
    return True


def cache_identity(config):
    return PHYSICAL_V8_CACHE_SCHEMA, COMPACT_ARCHITECTURE


def pure_e_main(config):
    return True


def integer_boundaries(config):
    if config.get('decoder_integer_boundaries') is not None:
        return bool(config['decoder_integer_boundaries'])
    return True


def self_integer_boundaries(config):
    if config.get('decoder_self_integer_boundaries') is not None:
        return bool(config['decoder_self_integer_boundaries'])
    return integer_boundaries(config) and not stage1_pretrain(config)


def moment_conditioning_enabled(config):
    if config.get('decoder_moment_conditioning') is not None:
        return bool(config['decoder_moment_conditioning'])
    from maidata.ymt3_muq_contract import enabled as is_ymt3_muq
    return bool(is_ymt3_muq(config) or config.get('decoder_stage1_pretrain', False))


def stage1_pretrain(config):
    return bool(config.get('decoder_stage1_pretrain', False))


def local_beat_ar(config):
    return False


def mixed_condition(config):
    return False


def decoder_windows(config):
    from maidata.ymt3_muq_contract import enabled as is_ymt3_muq
    default_self = 1.0 if config.get('decoder_stage1_pretrain', False) else 8.0
    default_audio = 4.0 if is_ymt3_muq(config) else 8.0
    return dict(
        self_before=float(config.get('decoder_self_before_beats', default_self)),
        audio_before=float(config.get('decoder_audio_before_beats', default_audio)),
        audio_after=float(config.get('decoder_audio_after_beats', default_audio)),
    )


def condition_before(config):
    return 0


def beat_ar(config):
    return False


def notes_only(config):
    return True


def conditioned(config):
    return False


def joint_architecture(config):
    return COMPACT_JOINT


def enabled(config):
    return config.get('decoder_architecture') == COMPACT_ARCHITECTURE


def mtp_depth(config):
    return 0


def inference_enabled(config):
    return enabled(config)
