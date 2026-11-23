"""Per-beat audio banks with local windows or shared full-song attention."""
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from torch.nn import functional as F
from .bucket_layout import _pack_buckets


def enabled(config):
    return config.get('decoder_global_audio_mode', 'adaln') == 'cross_attention'


def full_song(config):
    return not config.get('encoder_local_dual_stream',False) and bool(config.get('decoder_global_audio_full_song',False))


@dataclass
class GlobalAudio:
    layers: tuple
    beats: torch.Tensor
    valid: torch.Tensor

    @cached_property
    def cpu_geometry(self):
        return self.beats.detach().cpu(),self.valid.detach().cpu()

    @cached_property
    def grid_cpu(self):
        return self.beats[0].detach().cpu().tolist()


def song_grid(songs, samples_per_beat):
    valid = F.pad(songs.valid, (0, (-songs.valid.shape[1]) % samples_per_beat))
    valid = valid.unflatten(1, (-1, samples_per_beat)).any(-1)
    beats = songs.domain_start[:, None] + torch.arange(valid.shape[1]) + .5
    end = (getattr(songs,'layout',None) or {}).get('global_audio_end')
    if end is not None:
        valid &= (beats>=0) & (beats<end[:,None])
    return beats.float(), valid


def layout(queries, owners, beats, valid, before=8., after=4., full_song=False):
    """CPU Q32 buckets, with keys owned by one audio variant only."""
    if full_song:
        return full_song_layout(owners,valid)
    q = queries.detach().cpu().reshape(-1).floor().numpy()
    owners = owners.detach().cpu().reshape(-1).numpy()
    beats, valid = beats.detach().cpu().numpy(), valid.detach().cpu().numpy()
    groups, visible = defaultdict(list), []
    width = beats.shape[1]
    # Pack nearby queries, even across adjacent beats, with exact per-row masks.
    # Sorting also handles Slide lookahead and multiple charts sharing a song.
    for song in np.unique(owners):
        qi = np.flatnonzero(owners == song)
        qi = qi[np.argsort(q[qi], kind='stable')]
        for start in range(0, len(qi), 32):
            rows = qi[start:start+32]
            anchors = q[rows]
            left, right = np.searchsorted(beats[song], [anchors.min()-before, anchors.max()+after])
            keys = np.arange(left, right)[valid[song, left:right]]
            mask = ((beats[song, keys][None] >= anchors[:,None]-before) &
                    (beats[song, keys][None] < anchors[:,None]+after))
            used = mask.any(0)
            keys, mask = keys[used], mask[:,used]
            live = mask.any(1)
            if not live.any(): continue
            rows, mask = rows[live], mask[live]
            kw = max(16, (len(keys)+7)//8*8)
            groups[32, kw].append((rows, keys + song*width, mask))
            visible.extend(rows)
    result = _pack_buckets(groups, len(q))
    result.update(verified_valid=False, max_k=0, visible_queries=torch.tensor(sorted(visible),dtype=torch.long))
    return result


def full_song_layout(owners, valid):
    """One ragged sequence per song: no Q32 duplication or dense True masks."""
    owners, valid = owners.detach().cpu().flatten(), valid.detach().cpu()
    queries, keys, q_lengths, k_lengths = [], [], [], []
    for song in owners.unique(sorted=True).tolist():
        qi = (owners==song).nonzero().flatten()
        ki = valid[song].nonzero().flatten()+song*valid.shape[1]
        if not len(ki): continue
        queries.append(qi); keys.append(ki)
        q_lengths.append(len(qi)); k_lengths.append(len(ki))
    empty = torch.empty(0,dtype=torch.long)
    qi = torch.cat(queries) if queries else empty
    ki = torch.cat(keys) if keys else empty
    return dict(full_song=True,length=len(owners),varlen_queries=qi,varlen_keys=ki,
                q_lengths=q_lengths,k_lengths=k_lengths,
                cu_q=torch.tensor([0,*np.cumsum(q_lengths)],dtype=torch.int32),
                cu_k=torch.tensor([0,*np.cumsum(k_lengths)],dtype=torch.int32),
                max_q=max(q_lengths,default=0),max_k=max(k_lengths,default=0),
                visible_queries=qi,verified_valid=True)



def add_layouts(chunks, songs, config):
    if config.get('encoder_local_dual_stream',False):
        return  # Both branches consume the existing audio_layout, including after OOM splitting.
    beats, valid = song_grid(songs, config['samples_per_beat'])
    for chunk in chunks:
        chunk.layout['global_audio_layout'] = layout(chunk.tokens.slide_beats,
            chunk.layout['song_indices'], beats, valid,full_song=full_song(config))
