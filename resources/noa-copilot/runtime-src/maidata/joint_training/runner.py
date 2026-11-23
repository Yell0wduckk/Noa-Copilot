"""Current multi-task training, with checkpoint publication only at epoch end."""
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from maidata.model import build_model
from maidata.multitask.data import read_manifest,stage_sources,schedule_plans,UpdateDataset,loader
from maidata.multitask.optim import make_optimizer,make_scheduler
from .checkpoint import source_identity,initialize_encoder,restore,save
from .update import step_batch

def run(config,resume=None,*,cost_check=False):
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    rows,binding=read_manifest(config['manifest'])
    rows=stage_sources(rows,config['stage'])
    plans,planner_states=schedule_plans(rows,config)
    steps=[len(p) for p in plans]
    if not any(steps):
        # An empty selection is a no-op, not an epoch of trained weights.
        result=dict(status='no_training_records',completed_epochs=0,step=0,stage=config['stage'])
        print(json.dumps(result),flush=True)
        return result
    device=torch.device(config['device'])
    model=build_model(config).to(device)
    optimizer=make_optimizer(model,config)
    scheduler=make_scheduler(optimizer,config,steps)
    source=source_identity()
    loader_generator=torch.Generator().manual_seed(config['seed'])
    completed,step,best=0,0,-1.
    if resume:
        state=restore(resume,model,optimizer,scheduler,config=config,binding=binding,
                      steps_by_epoch=steps,source=source)
        completed,step,best=state['completed_epochs'],state['step'],state['best_f1']
        loader_generator.set_state(state['loader_rng'].cpu())
        if state['planner_state']!=planner_states[completed-1]:
            raise ValueError('checkpoint external dataset cursors differ from the saved data plan')
    else:
        if config.get('yourmt3_checkpoint'):
            from maidata.multitask.yourmt3_encoder import load_yourmt3_encoder_weights
            load_yourmt3_encoder_weights(model.encoder.yourmt3,config['yourmt3_checkpoint'])
        initialize_encoder(model,config['stage0_checkpoint'])
    output=Path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    for epoch in range(completed+1,config['stop_after_epochs']+1):
        model.train()
        final=epoch==config['stop_after_epochs']
        model.reset_gate_diagnostics(enabled=final)
        frozen=epoch<=config['encoder_freeze_epochs']
        model.encoder.set_trainable(frozen,top_enabled=step>=config['warmup_steps'])
        dataset=UpdateDataset(rows,plans[epoch-1],config,epoch)
        batches=loader(dataset,config['workers'],config['seed'],generator=loader_generator)
        started=time.perf_counter()
        for index,groups in enumerate(batches):
            if cost_check and device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            before=time.perf_counter()
            metrics=step_batch(model,groups,optimizer,scheduler,config)
            step+=1
            if cost_check and device.type=='cuda': torch.cuda.synchronize()
            metrics.update(epoch=epoch,update=index+1,step=step,seconds=time.perf_counter()-before,
                           encoder_frozen=frozen)
            if cost_check:
                metrics.update(parameters=sum(p.numel() for p in model.parameters()),
                               peak_allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                               device=str(device),original_batch_records=sum(len(g['charts']) for g in groups))
                print(json.dumps(metrics),flush=True)
                (output/'cost.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
                return metrics
            print(json.dumps(metrics),flush=True)
        validation=None
        if config['stage']==0 and (epoch%config['validation_every']==0 or final):
            from maidata.multitask.validation import validate
            validation=validate(model,rows,config,full=final)
        improved=validation is not None and validation.get('eight_class_onset',{}).get('f1',-1)>best
        if improved: best=validation['eight_class_onset']['f1']
        kwargs=dict(config=config,epoch=epoch,step=step,binding=binding,
                    planner_state=planner_states[epoch-1],steps_by_epoch=steps,
                    loader_rng=batches.generator.get_state(),best_f1=best,source=source)
        save(output/'last.ckpt',model,optimizer,scheduler,**kwargs)
        if epoch%5==0 or final:
            save(output/f'epoch-{epoch:03d}.ckpt',model,optimizer,scheduler,**kwargs)
        if improved: save(output/'best.ckpt',model,optimizer,scheduler,**kwargs)
        report=dict(epoch=epoch,seconds=time.perf_counter()-started,validation=validation)
        if final:
            report['gate_weights']=model.gate_diagnostics()
        with (output/'epochs.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
    return dict(completed_epochs=config['stop_after_epochs'],step=step)
