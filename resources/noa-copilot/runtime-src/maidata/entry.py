"""Current v14 model entrypoint."""
import argparse
import json
from .multitask.config import DEFAULTS,load
CONFIG=DEFAULTS

def main(argv=None):
    parser=argparse.ArgumentParser(description='Unified 10ms maimai/MIDI E+D')
    parser.add_argument('--config')
    parser.add_argument('--show-config',action='store_true')
    args=parser.parse_args(argv)
    config=load(args.config)
    if args.show_config:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        return 0
    from .model import build_model
    model=build_model(config)
    print(model)
    print('parameters:',sum(p.numel() for p in model.parameters()))
    return 0
