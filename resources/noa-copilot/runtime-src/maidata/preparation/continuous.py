"""BF16 full-song CompactSpec continuous domains, preserving absolute timing and silence."""
import numpy as np
import torch

from maidata.audio.types import ContinuousSongInputs, WindowRequests
from maidata.audio.continuous_layout import prepare_continuous_layout
from maidata.model import encoder_attention_radii, encoder_slot_radius
from .compactspec import from_bits, context_at
from .timing import TempoMap


def continuous_inputs(grids, coverages, config, *, validate_input=False):
    if not grids or len(grids) != len(coverages):
        raise ValueError('continuous audio requires one coverage per grid')
    from maidata.ymt3_muq_contract import enabled as is_ymt3_muq
    if 'ymt3_frames' in grids[0] or is_ymt3_muq(config):
        from .ymt3_muq import continuous_inputs as ymt3_continuous_inputs
        return ymt3_continuous_inputs(grids, coverages, config, validate_input=validate_input)
    rate = config['samples_per_beat']
    key = 'spec_frames'
    feature_shape = (249,)
    silence_array = grids[0]['silence_spec']
    from .audio_padding import audio_padding
    before,after = audio_padding(config)
    starts,ends = [],[]
    for grid,coverage in zip(grids,coverages,strict=True):
        if grid.get(key) is None or grid[key].shape[1:] != (rate,*feature_shape):
            raise ValueError('continuous CompactSpec grid representation differs from Encoder')
        if not np.array_equal(grid['silence_spec'],silence_array):
            raise ValueError('continuous CompactSpec grids have different silence_spec statistics')
        first = int(grid['first_beat'])
        starts.append(min(first,coverage.first_beat-before))
        ends.append(max(first+len(grid[key]),
                        coverage.first_beat+coverage.draft_beats+after))
    lengths = [(end-start)*rate for start,end in zip(starts,ends)]
    shape = (len(grids),max(lengths))
    spec = torch.zeros((*shape,*feature_shape),dtype=torch.bfloat16)
    context = torch.zeros((*shape,2),dtype=torch.float32)
    beats = torch.zeros(shape,dtype=torch.float32)
    valid = torch.zeros(shape,dtype=torch.bool)
    sample_valid = torch.zeros_like(valid)
    silence = torch.from_numpy(silence_array)
    for index,(grid,start,length) in enumerate(zip(grids,starts,lengths)):
        spec[index,:length] = silence.to(spec.dtype)
        offset = (int(grid['first_beat'])-start)*rate
        values = from_bits(grid[key]).reshape(-1,*feature_shape)
        selected = torch.from_numpy(grid['sample_valid'].reshape(-1))
        spec[index,offset:offset+len(values)] = torch.where(selected.reshape(-1,*([1]*len(feature_shape))),values,silence.to(spec.dtype))
        sample_valid[index,offset:offset+len(values)] = selected
        coordinates = start+np.arange(length,dtype=np.float64)/rate
        beats[index,:length] = torch.from_numpy(coordinates.astype(np.float32))
        valid[index,:length] = True
        stop = offset+len(values)
        tempo = TempoMap(grid['timing'])
        for left,right in ((0,offset),(stop,length)):
            if left < right:
                context[index,left:right] = torch.from_numpy(context_at(coordinates[left:right],
                    np.zeros(right-left,dtype=bool),tempo,float(grid['duration_seconds'])))
        context[index,offset:stop] = torch.from_numpy(grid['context_grid'].reshape(-1,2))

    result = ContinuousSongInputs(None,None,None,beats,valid,torch.tensor(starts,dtype=torch.float32),
        torch.tensor(ends,dtype=torch.float32),spec=spec,context=context,sample_valid=sample_valid,silence_spec=silence)
    result.layout = prepare_continuous_layout(result,radii=encoder_attention_radii(config),
        rates=config['fpn_rates'],slot_radius=encoder_slot_radius(config),
        e0_only=config.get('encoder_e0_fold3',True),samples_per_beat=rate,
        local_dual_stream=config.get('encoder_local_dual_stream',True))
    from .audio_padding import add_global_bounds
    add_global_bounds(result, coverages, config)
    return result


def continuous_requests(inputs, starts, mapping, config, *, scales=(0,)):
    """Prepare readout geometry only; no per-window acoustic tensors exist."""
    from maidata.audio.continuous_layout import prepare_continuous_readout_layout
    starts = torch.as_tensor(starts, dtype=torch.float32).detach().cpu()
    mapping = torch.as_tensor(mapping, dtype=torch.long).detach().cpu()
    if starts.ndim != 1 or mapping.shape != starts.shape:
        raise ValueError('continuous readout starts and song mapping must align')
    layout = prepare_continuous_readout_layout(inputs.layout, starts, mapping,
        rates=config['fpn_rates'], before=config['audio_before_beats'],
        context=config['audio_before_beats'] + config['target_beats'] + config['audio_after_beats'],
        scales=(0,))
    return WindowRequests(starts, mapping, layout)
