"""Official MIDI validation only, every five epochs and at the absolute end."""
from collections import defaultdict
import torch
from maidata.joint_training.update import autocast
from .data import UpdateDataset
from .batching import chart_batch,cross_entropy,logical_batch
from .codec import crop_events,encode_chart
from .evaluation import midi_counts,scores


@torch.inference_mode()
def validate(model,rows,config,*,full=False):
    from maidata.inference.generation import generate
    selected=[i for i,r in enumerate(rows) if r['task']=='midi' and r.get('split')=='validation'
              and not r.get('pseudo',r.get('metadata',{}).get('pseudo',False))
              and r.get('source_family','real_midi') == 'real_midi']
    if not selected:
        return None
    # Manifest preserves official folds. No train/test random split is created.
    selected.sort(key=lambda i:str(rows[i]['id']))
    dataset=UpdateDataset(rows,[[[i]] for i in selected],
        {**config,'max_audio_seconds':float('inf')},0,training=False)
    was_training=model.training
    model.eval()
    device=next(model.parameters()).device
    ce=0.
    count=0
    budget=config['validation_subset_seconds']
    aggregate=defaultdict(lambda:[0,0,0])
    examples=0
    confusion=[[0,0],[0,0]]
    for n in range(len(dataset)):
        group=dataset[n][0]
        batch,targets,mask=logical_batch([group],0,device,model.encoder.frontend)
        features=batch['features'][0,:int(batch['audio_lengths'][0])]
        chart=group['charts'][0]
        with autocast(config,device):
            streams=model.encode_batch(batch)
            logits=model.decode(batch,streams)
        parts={}
        ce+=float(cross_entropy(logits,targets,mask,batch['legal_mask'],
                               stage=config['stage'],task_ids=batch['task_ids'],
                               current_slots=batch.get('current_slots'),pseudo=batch.get('pseudo'),
                               previous_pitches=batch['previous_pitches'],
                               target_pseudo=batch.get('target_pseudo'),target_groups=batch['target_groups'],
                               target_velocity_db=batch['target_velocity_db'],
                               velocity_sigma=config['velocity_sigma'],components=parts))
        count+=int(parts['supervision_count'])
        if full or budget>0:
            length=len(features)/125 if full else min(config['validation_excerpt_seconds'],budget,len(features)/125)
            excerpt=features[:max(1,round(length*125))]
            from .model import AudioStreams
            excerpt_frames=len(excerpt)
            excerpt_streams=AudioStreams(streams.p8[:,:excerpt_frames],streams.p_times[:,:excerpt_frames],
                torch.tensor([excerpt_frames],device=device),torch.tensor([excerpt_frames]))
            with autocast(config,device):
                result=generate(model,excerpt,'midi',metadata=chart['metadata'],temperature=0.,
                    seed=config['seed'],max_tokens_per_second=config['generation_max_tokens_per_second'],
                    stage=0,streams=streams if full else excerpt_streams)
            reference=crop_events(chart['events'],0,length,truncate_end=True)
            for name,values in midi_counts(reference,result.events).items():
                aggregate[name]=[a+b for a,b in zip(aggregate[name],values)]
            from .evaluation import envelope_confusion
            matrix=envelope_confusion(reference,result.events)
            confusion=[[a+b for a,b in zip(x,y)] for x,y in zip(confusion,matrix)]
            budget-=length
            examples+=1
    model.train(was_training)
    return dict(scores(aggregate),teacher_forcing_ce=ce/max(count,1),free_run_examples=examples,
                full_free_run=full,envelope_confusion=confusion)
