"""Complete-epoch, recipe/input-bound recovery with recorded source provenance."""
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from maidata.multitask import ARCHITECTURE,AUDIO_SCHEMA,CHECKPOINT_SCHEMA
from maidata.multitask.config import recipe

def source_identity():
    root=Path(__file__).resolve().parents[3]
    marker=root/'SOURCE_COMMIT'
    if marker.exists():
        return marker.read_text(encoding='utf-8').strip()
    try:
        cmd=['git','-c','safe.directory='+root.as_posix()]
        gitfile=root/'.git'
        if root.as_posix().startswith('/mnt/') and gitfile.is_file():
            from pathlib import PureWindowsPath
            ref=PureWindowsPath(gitfile.read_text(encoding='utf8').strip().removeprefix('gitdir: '))
            if ref.drive:
                cmd.extend(['--git-dir',str(Path('/mnt',ref.drive[0].lower(),*ref.parts[1:]))])
        return subprocess.check_output(cmd+['rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        torch.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])

def save(path,model,optimizer,scheduler,*,config,epoch,step,binding,planner_state,
         steps_by_epoch,loader_rng,best_f1,source):
    path=Path(path)
    payload=dict(schema=CHECKPOINT_SCHEMA,architecture=ARCHITECTURE,audio_schema=AUDIO_SCHEMA,config=config,
        recipe=recipe(config),model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),completed_epochs=epoch,step=step,
        binding=binding,planner_state=planner_state,steps_by_epoch=steps_by_epoch,
        rng=rng_state(),loader_rng=loader_rng,best_f1=best_f1,source=source)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)

def restore(path,model,optimizer,scheduler,*,config,binding,steps_by_epoch,source):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema']!=CHECKPOINT_SCHEMA or payload['architecture']!=ARCHITECTURE:
        raise ValueError('checkpoint architecture differs from the current tick96 model')
    for key,expected in dict(recipe=recipe(config),binding=binding,
                            steps_by_epoch=steps_by_epoch).items():
        if payload[key]!=expected:
            raise ValueError('complete-epoch resume '+key+' differs')
    epoch=payload['completed_epochs']
    if payload['step']!=sum(steps_by_epoch[:epoch]) or payload['scheduler_state']['last_epoch']!=payload['step']:
        raise ValueError('checkpoint does not describe a complete epoch')
    model.load_state_dict(payload['model_state'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    scheduler.load_state_dict(payload['scheduler_state'])
    restore_rng(payload['rng'])
    return payload

def initialize_encoder(model,path):
    """Transfer Stage 0's acoustic trunk and complete shared decoder."""
    if not path:
        return
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload['model_state']
    target=model.state_dict()
    selected={key:value for key,value in state.items()
              if key in target and key!='conditioner.task.weight' and '.maimai.' not in key and not key.startswith(('embeddings.maimai.','heads.maimai.'))}
    model.load_state_dict(selected,strict=False)
    saved=payload.get('config',{})
    saved_tasks=saved.get('model',{}).get('tasks',
        ('midi',) if len(state['conditioner.task.weight'])==1 else ('maimai','midi'))
    with torch.no_grad():
        for index,task in enumerate(saved_tasks):
            if task in model.config.tasks:
                model.conditioner.task.weight[model.config.tasks.index(task)].copy_(state['conditioner.task.weight'][index])
    # Independent timing vocabularies start from learned MIDI timing. They
    # remain separate parameters and diverge under their own task gradients.
    from maidata.multitask.grammar import SHARED_ROWS
    if 'midi' in model.embeddings and 'maimai' in model.embeddings:
        with torch.no_grad():
            model.embeddings['maimai'].weight[:SHARED_ROWS].copy_(model.embeddings['midi'].weight[:SHARED_ROWS])
            model.heads['maimai'].weight[:SHARED_ROWS].copy_(model.heads['midi'].weight[:SHARED_ROWS])
            model.heads['maimai'].bias[:SHARED_ROWS].copy_(model.heads['midi'].bias[:SHARED_ROWS])
