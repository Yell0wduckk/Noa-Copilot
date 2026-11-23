"""Versioned, fixed CompactSpec T/O/S input and shallow frontend identities."""

ARCHITECTURE = 'maidata.audio-compactspec-e0-fold3.v1'
REPRESENTATION = 'compactspec-tos-249-bf16.v1'
DENSE = 'compactspec-dense.v1'
TCONV = 'compactspec-tconv.v1'
CONTEXT = 'bpm-progress.v1'
LEGACY_CONTEXT = 'bpm-phase24-progress.v1'
DEFAULTS = dict(audio_representation=REPRESENTATION, audio_frontend=DENSE,
                audio_context=CONTEXT, spec_bins=249)
KEYS = tuple(DEFAULTS)
# (name, FFT length, bins per octave, minimum Hz, count)
CHANNELS = (('T', 2048, 24, 30., 149), ('O', 512, 12, 30., 62), ('S', 1024, 6, 120., 38))
SLICES = ((0, 149), (149, 211), (211, 249))


def enabled(config):
    return config.get('encoder_architecture') == ARCHITECTURE


def require_input_config(config):
    for key, expected in DEFAULTS.items():
        if (config.get(key) not in (DENSE, TCONV) if key == 'audio_frontend'
                else config.get(key) not in (CONTEXT, LEGACY_CONTEXT) if key == 'audio_context'
                else config.get(key) != expected):
            raise ValueError(f'unsupported CompactSpec input identity: {key}')
