"""Native joint intent model used inside generation, never as a judge."""
import threading
import math
import json
from pathlib import Path
import numpy as np
import torch
from .models.joint_plan import JointEventPlanModel
from ..io.timing import sample_positions
from ..io.audio import FRAME_SECONDS
from .intent import EventIntent,IntentChoices

_LOCK=threading.Lock()
_MODELS={}


def predict_heads(context):
    if context.slot not in (4,5,6):
        raise ValueError('Joint WHAT checkpoint supports EXPERT, MASTER and Re:MASTER only')
    path=Path(context.root)/'models/experimental/joint_plan.pt';key=(str(path),path.stat().st_mtime_ns)
    with _LOCK:
        if key not in _MODELS:
            model=JointEventPlanModel();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True);model.cuda().eval()
            _MODELS.clear();_MODELS[key]=model
        model=_MODELS[key]
    results={name:[] for name in ('arity','stars','holds','touches','tap_slide')}
    index=context.slot-4
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for begin in range(0,len(context.structure),32):
            end=min(begin+32,len(context.structure));bars=list(range(begin,end))
            audio=np.stack([sample_positions(context.mel,b*384+np.arange(384),context.bt,context.bv,FRAME_SECONDS) for b in bars])
            levels=torch.zeros((len(bars),3),dtype=torch.int64,device='cuda');levels[:,index]=round(context.level*10)
            _,out=model(torch.from_numpy(audio).cuda(),torch.from_numpy(context.structure[begin:end]).cuda(),torch.full((len(bars),),context.version,dtype=torch.int64,device='cuda'),levels)
            for name in results:results[name].append(out[index][name].float())
    return {name:torch.cat(values) for name,values in results.items()}


COMBINATIONS=tuple((a,s,h) for a in range(3) for s in range(a+1) for h in range(a-s+1))
def shifted_stars(heads,bias):
    result=dict(heads);count=torch.arange(3,device=heads['stars'].device)
    result['stars']=heads['stars']+float(bias)*count
    return result


def decode_intents(output,allowed,topk=8):
    """Joint likelihood over consistent WHAT counts, using every learned head.

    Heads were trained on occupied anchors. Rest is not a trained WHAT label;
    WHEN owns empty times. tap_slide is (arity=2, stars=1, holds=0), trained
    with positive-weight BCE=5, so undo that odds shift before joint scoring.
    """
    logs={name:output[name].float().log_softmax(-1) for name in ('arity','stars','holds','touches')}
    device=logs['arity'].device
    combinations=torch.tensor(COMBINATIONS,device=device)
    a,s,h=combinations.unbind(-1)
    score=logs['arity'][...,a]+logs['stars'][...,s]+logs['holds'][...,h]
    mixed=(a==2)&(s==1)&(h==0);aux=output['tap_slide'].float()-math.log(5.)
    score=score+.25*torch.where(mixed,torch.nn.functional.logsigmoid(aux)[...,None],torch.nn.functional.logsigmoid(-aux)[...,None])
    score=score[...,None]+logs['touches'][...,None,:]
    counts=torch.arange(logs['touches'].shape[-1],device=device)
    score=score.masked_fill((a[:,None]==0)&(counts[None,:]==0),-torch.inf)
    support=torch.zeros((len(COMBINATIONS),len(counts)),device=device,dtype=torch.bool)
    for arity,stars,holds,touches in allowed:
        if (arity,stars,holds) in COMBINATIONS and 0<=touches<len(counts):
            support[COMBINATIONS.index((arity,stars,holds)),touches]=True
    score=score.masked_fill(~support,-torch.inf)
    if not bool(torch.isfinite(score.flatten(-2)).any(-1).all()):raise RuntimeError('No slot-specific WHAT support')
    ids=score.flatten(-2).topk(min(topk,score.shape[-2]*score.shape[-1]),-1).indices.cpu().tolist()
    def row(values):
        result=[]
        for value in values:
            arity,stars,holds=COMBINATIONS[value//len(counts)]
            families=(0,)*(arity-stars-holds)+(1,)*holds+(2,)*stars
            result.append(EventIntent(families,value%len(counts)))
        return IntentChoices(tuple(result))
    return [row(values) for values in ids]


def intent_plan(context,target_stars):
    heads=predict_heads(context)
    ticks=np.asarray(context.ticks,dtype=np.int64)
    document=json.loads((Path(context.root)/'models/experimental/intent_support.json').read_text(encoding='utf8'))
    allowed={tuple(x) for x in document['slots'][str(context.slot)]['configurations']}
    def decode_with_bias(bias,topk):
        adjusted=shifted_stars(heads,bias)
        selected={name:value[ticks//384,ticks%384] for name,value in adjusted.items()}
        return decode_intents(selected,allowed,topk),adjusted
    target=max(0,min(int(target_stars),2*len(ticks)));best=None
    for bias in np.linspace(-5.,2.,57):
        choices,_=decode_with_bias(float(bias),1)
        count=sum(value.candidates[0].button_families.count(2) for value in choices)
        key=(abs(count-target),abs(float(bias)),count)
        if best is None or key<best[0]:best=(key,float(bias),count)
    choices,adjusted=decode_with_bias(best[1],8)
    probability=adjusted['stars'].softmax(-1)
    stars=(probability[...,1]+2*probability[...,2]).cpu().numpy()
    return dict(zip(map(int,ticks),choices)),stars,{'targetStars':target,'plannedStars':best[2],'starLogitBias':best[1]}


def star_scores(context):
    heads=predict_heads(context);p=heads['stars'].softmax(-1)
    return (p[...,1]+2*p[...,2]).cpu().numpy()
