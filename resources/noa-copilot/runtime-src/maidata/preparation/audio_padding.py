"""Physical silence required by active Decoder windows, without asset rebuilds."""
import math
import torch


def audio_padding(config):
    from maidata.streaming import decoder_windows
    windows = decoder_windows(config)
    return tuple(max(config[f'audio_{side}_beats'], math.ceil(windows.get(f'audio_{side}', 0)))
                 for side in ('before', 'after'))


def add_global_bounds(inputs, coverages, config):
    """P0 padding is valid silence; full-song cross attention uses chart beats."""
    if config.get('decoder_global_audio_full_song', False):
        inputs.layout['global_audio_end'] = torch.tensor(
            [c.first_beat+c.draft_beats for c in coverages], dtype=torch.float32)
