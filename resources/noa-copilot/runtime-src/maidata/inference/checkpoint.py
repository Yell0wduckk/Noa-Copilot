"""Strict current-model inference loading, without training asset traversal."""
from dataclasses import dataclass
import torch
from maidata.multitask import ARCHITECTURE,CHECKPOINT_SCHEMA
from maidata.model import build_model

@dataclass
class Models:
    model: object
    config: dict
    provenance: dict
    @property
    def encoder(self): return self.model.encoder
    @property
    def decoder(self): return self.model

class JointCheckpoint:
    def __init__(self,checkpoint):
        self.path=str(checkpoint)
        self.payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if self.payload['architecture']!=ARCHITECTURE or self.payload['schema']!=CHECKPOINT_SCHEMA:
            raise ValueError('inference requires the current tick96 multitask checkpoint')
        self.config=self.payload['config']
    def models(self,device='cpu'):
        model=build_model(self.config).to(device)
        model.load_state_dict(self.payload['model_state'],strict=True)
        model.eval()
        return Models(model,self.config,dict(checkpoint=self.path,source=self.payload.get('source'),
                                            completed_epochs=self.payload['completed_epochs']))
