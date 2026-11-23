"""Versioned YMT3+ (8ms @ 512d) + MuQ Layer 10 (40ms @ 1024d) audio input and encoder contract."""

ARCHITECTURE = "maidata.audio-ymt3-muq-8ms.v1"
REPRESENTATION = "ymt3-8ms-512-muq-40ms-1024-bf16.v1"
FRONTEND = "ymt3-muq-linear-8ms.v1"
CONTEXT = "none"

YMT3_DIM = 512
MUQ_DIM = 1024
FRAME_SECONDS_8MS = 0.008
FRAME_SECONDS_40MS = 0.040
RATIO_40_TO_8 = 5

DEFAULTS = dict(
    encoder_architecture=ARCHITECTURE,
    audio_representation=REPRESENTATION,
    audio_frontend=FRONTEND,
    audio_context=CONTEXT,
    ymt3_dim=YMT3_DIM,
    muq_dim=MUQ_DIM,
    encoder_layers=2,
    encoder_window_frames=128,
)
KEYS = (
    "audio_representation",
    "audio_frontend",
    "audio_context",
    "ymt3_dim",
    "muq_dim",
    "encoder_layers",
    "encoder_window_frames",
)


def enabled(config):
    return (
        config.get("encoder_architecture") == ARCHITECTURE
        or config.get("audio_representation") == REPRESENTATION
    )


def require_input_config(config):
    for key in ("audio_representation", "audio_frontend", "audio_context", "ymt3_dim", "muq_dim"):
        expected = DEFAULTS[key]
        if config.get(key, expected) != expected:
            raise ValueError(f"unsupported YMT3+MuQ input identity: {key}={config.get(key)!r}")
    layers = config.get("encoder_layers", 2)
    if type(layers) is not int or layers <= 0:
        raise ValueError("encoder_layers must be a positive integer")
    window = config.get("encoder_window_frames", 128)
    if type(window) is not int or window <= 0:
        raise ValueError("encoder_window_frames must be a positive integer")
