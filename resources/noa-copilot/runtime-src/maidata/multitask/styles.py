"""Discrete, editable song styles. MuQ-MuLan is an offline labeller only."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

STYLE_TAG_VOCAB = ('none','artcore','j_core','speedcore','tech','trance','denpa',
                  'vocaloid_rock','vocaloid_pop','metal','jazz_swing','funk_disco',
                  'symphonic','oriental','chiptune','acoustic_piano')
STYLE_PROMPTS = (
    'artcore drum and bass with flowing piano and strings',
    'J-core high energy four on the floor rave synthesizer music',
    'speedcore hardstyle gabber with fast distorted kicks',
    'tech bass house dubstep wobble bass glitch and syncopated breaks',
    'trance progressive music with sustained pads and arpeggios',
    'denpa kawaii future bass with cute vocals and playful bright synths',
    'Vocaloid J-rock with electric guitar riffs and live drums',
    'J-pop Vocaloid pop with clear catchy vocal melodies',
    'heavy metal djent with distorted guitars and double bass drums',
    'jazz swing big band with swung triplets and brass',
    'funk city pop disco with slap bass and tight snare backbeats',
    'symphonic classical gothic orchestral music and choir',
    'oriental Japanese Chinese traditional plucked pentatonic melodies',
    'chiptune eight bit square wave fast granular arpeggios',
    'acoustic piano solo gentle ballad music',
)

def parse_style_tags(value):
    if not value:
        return [0,0]
    if isinstance(value, dict):
        value = [value.get('primary_tag', 0), value.get('secondary_tag', 0)]
    elif isinstance(value, str):
        value = value.split(',')
    ids = [0 if x is None else STYLE_TAG_VOCAB.index(x.strip().lower().replace('-','_')) if isinstance(x,str) else int(x)
           for x in value]
    return (ids + [0,0])[:2]

def choose_tags(scores, *, secondary_gap=.05, threshold=.0):
    values, indices = torch.as_tensor(scores).topk(2)
    primary = int(indices[0]) + 1 if float(values[0]) >= threshold else 0
    secondary = int(indices[1]) + 1 if primary and float(values[0]-values[1]) < secondary_gap and float(values[1]) >= threshold else 0
    return dict(primary_tag=STYLE_TAG_VOCAB[primary],secondary_tag=STYLE_TAG_VOCAB[secondary],
                scores={tag:float(score) for tag,score in zip(STYLE_TAG_VOCAB[1:], scores)})

@torch.inference_mode()
def label_songs(records, model, *, device='cuda', secondary_gap=.05, threshold=.0):
    from .audio import read_waveform
    texts = F.normalize(model(texts=list(STYLE_PROMPTS)),dim=-1)
    result = {}
    for row in records:
        key=str(row.get('song_id',row['id']))
        if key in result:
            continue
        wave = read_waveform(row['audio'],24000).to(device)
        # One shared song tag, never a dense embedding passed to the decoder.
        chunks = [wave[start:start+30*24000] for start in range(0,len(wave),30*24000)]
        embeds = [F.normalize(model(wavs=F.pad(x,(0,30*24000-len(x)))[None]),dim=-1) for x in chunks]
        audio = F.normalize(torch.cat(embeds).mean(0),dim=-1)
        result[key] = choose_tags((audio @ texts.T).cpu(),
            secondary_gap=secondary_gap,threshold=threshold)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--secondary-gap',type=float,default=.05)
    parser.add_argument('--threshold',type=float,default=.0)
    args=parser.parse_args(argv)
    from muq import MuQMuLan
    model=MuQMuLan.from_pretrained(args.model).to(args.device).eval().requires_grad_(False)
    source=json.loads(args.index.read_text(encoding='utf-8-sig'))
    records=[dict(row,audio=str((args.index.parent/row['audio']).resolve())) for row in source['records']]
    result=label_songs(records,model,device=args.device,secondary_gap=args.secondary_gap,threshold=args.threshold)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
    main()
