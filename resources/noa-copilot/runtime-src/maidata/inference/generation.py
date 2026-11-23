"""Incremental grammar-masked generation on the tick96 score grid."""
from dataclasses import dataclass
import math
import torch
from maidata.multitask import TASKS
from maidata.multitask.codec import Grammar

class GenerationError(RuntimeError):
    pass

@dataclass
class GenerationResult:
    task: str
    tokens: list
    events: list
    truncated: bool
    duration_seconds: float

@torch.inference_mode()
def generate(model,features,task='maimai',*,metadata=None,temperature=0.,seed=20261104,
             max_tokens_per_second=1024,cfg_scale=1.,stage=1,streams=None,duration_seconds=None):
    model.eval()
    device=next(model.parameters()).device
    features=features.to(device)
    if streams is None:
        streams=model.encode_long(features)
    duration=features.shape[-2]/125 if duration_seconds is None else duration_seconds
    seconds=math.ceil(duration)
    grammar=Grammar(task,metadata=metadata)
    end_tick = grammar.grid.second_to_beat(duration)*96
    first_tick = grammar.grid.second_to_beat(0.)*96
    current=grammar.vocab.boundary_id
    grammar.advance(current)
    tokens=[current]
    cache=model.init_cache(streams,TASKS.index(task),metadata=metadata,stage=stage)
    unconditional=None
    if cfg_scale!=1.:
        blank=dict(metadata or {})
        blank.update(level=None,difficulty=0,designer=0,phase_available=False,style_tags=[0,0],
                     query_tag=None,query_coarse_id=0,query_fine_id=0)
        unconditional=model.init_cache(streams,TASKS.index(task),metadata=blank,stage=stage,audio_cache=cache)
    rng=torch.Generator(device=device).manual_seed(seed)
    maximum=max(1,seconds)*max_tokens_per_second
    truncated = True
    while len(tokens)<maximum:
        inputs=grammar.input_features()
        logits=model.step(current,cache,**inputs).float()
        if unconditional is not None:
            base=model.step(current,unconditional,**inputs).float()
            logits=base+cfg_scale*(logits-base)
        allowed = grammar.legal_tokens()
        if grammar.phase == 'moment':
            base = grammar.beat*96
            mo = grammar.vocab.offsets['MOMENT']
            allowed = tuple(i for i in allowed if i < mo or first_tick-1e-8 <= base+i-mo < end_tick-1e-8)
        if len(allowed)==1:
            current=allowed[0]
        else:
            ids=torch.tensor(allowed,device=device)
            candidates=logits[ids]
            chosen=candidates.argmax() if temperature==0 else torch.multinomial(
                (candidates/temperature).softmax(-1),1,generator=rng)[0]
            current=int(ids[chosen])
        if grammar.vocab.decode(current).family == 'BEAT' and (grammar.beat+1)*96 >= end_tick-1e-8:
            truncated = False
            break
        grammar.advance(current, validate=False)
        tokens.append(current)
    events=grammar.finish(end_time=duration)
    if task=='midi' and not grammar.moment_only:
        from maidata.multitask.midi import events_to_notes
        events=events_to_notes(events,end_time=duration)
    return GenerationResult(task,tokens,events,truncated,duration)
