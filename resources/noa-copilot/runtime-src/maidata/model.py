"""Construct the current model without launching asset production."""
from .flat import ARCHITECTURE as ARCHITECTURE, ARCHITECTURE_V187, ARCHITECTURE_V20

def build_model(config=None):
    if config is None or config.get("architecture") in (ARCHITECTURE, ARCHITECTURE_V187, ARCHITECTURE_V20) or "stage" not in config:
        from .flat.model import FlatModel, model_config
        return FlatModel(model_config(config or {}))
    from .multitask.model import ModelConfig, MultitaskModel
    values = dict(config.get('model', {}))
    values.setdefault('macro_enabled', config.get('stage', 0) == 1)
    values.setdefault('tasks', ('midi',) if config.get('stage',0)==0 else ('maimai','midi'))
    return MultitaskModel(ModelConfig(**values))

def build_decoder(config):
    return build_model(config)
