"""Relational WHAT planning for delayed Slide launch on existing WHEN anchors."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import threading, math
from functools import lru_cache
import numpy as np
import torch
from .intent import EventIntent, IntentChoices
from .models.slide_timing import SlideTimingPlannerV1
from ..io.durations import slide_seconds
from ..io.timing import ticks_to_seconds

_LOCK=threading.Lock(); _CACHE={}

@dataclass(frozen=True)
class SlideLaunchLink:
    source_tick:int
    launch_tick:int
    cue_tick:int|None
    cue_intent:EventIntent|None
    min_move_ticks:int=48

@dataclass(frozen=True)
class RelationalWhatPlan:
    links:tuple[SlideLaunchLink,...]
    source_to_link:dict
    cue_to_source:dict
    duration_masks:dict
    decisions:tuple[dict,...]

    def __post_init__(self):
        from .contract_utils import freeze_plan
        freeze_plan(self)

def _load(context):
    path=Path(context.root)/'models/experimental/slide_timing_planner_v1.pt'
    device=torch.device(context.device)
    if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
    key=(str(path.resolve()).lower(),path.stat().st_mtime_ns,path.stat().st_size,str(device))
    with _LOCK:
        if key not in _CACHE:
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp.get('modelClass')!='SlideTimingPlannerV1':
                raise ValueError('Wrong Slide timing planner checkpoint')
            cfg=cp['config']; model=SlideTimingPlannerV1(**cfg)
            model.load_state_dict(cp['model'],strict=True); model.to(device).eval()
            move=np.asarray(cp['moveVocab'],dtype=np.int64)
            _CACHE.clear(); _CACHE[key]=(model,move)
        return _CACHE[key]

def _primary(value):
    return value.candidates[0] if isinstance(value,IntentChoices) else EventIntent.from_representation(value)

def _one_slide(intent):
    value=EventIntent.from_representation(intent)
    return value.button_families.count(2)==1

def _non_slide_choices(value):
    seq=value.candidates if isinstance(value,IntentChoices) else (EventIntent.from_representation(value),)
    kept=tuple(x for x in seq if 2 not in x.button_families)
    return IntentChoices(kept) if kept else None

def _relation_choices(value):
    seq=value.candidates if isinstance(value,IntentChoices) else (EventIntent.from_representation(value),)
    slides=tuple(x for x in seq if _one_slide(x))
    rest=tuple(x for x in seq if 2 not in x.button_families)
    return IntentChoices(slides+rest) if slides else None

def _cue_predictions(context, plan):
    model,_=_load(context); ticks=np.asarray(sorted(map(int,context.ticks)),np.int64)
    primary={t:_primary(plan[t]) for t in ticks if t in plan}
    rows=[]; row_ticks=[]
    for i,t in enumerate(ticks):
        intent=primary.get(int(t))
        if intent is None or not _one_slide(intent): continue
        delta=np.zeros(24,np.int16); arity=np.zeros(24,np.uint8)
        fam=np.zeros((24,3),np.uint8); touch=np.zeros(24,np.uint8)
        valid=np.zeros(24,np.bool_); accent=np.zeros(24,np.uint8)
        lo=max(0,i-8); hi=min(len(ticks),i+16)
        for j in range(lo,hi):
            d=j-i+8; other=primary.get(int(ticks[j]),EventIntent())
            delta[d]=np.clip(int(ticks[j])-int(t),-32768,32767)
            arity[d]=other.button_arity
            for f in other.button_families: fam[d,int(f)]+=1
            touch[d]=min(8,other.touch_count); valid[d]=True
        rows.append((delta,arity,fam,touch,accent,valid));row_ticks.append(int(t))
    if not rows:return {}
    b={
      'delta':torch.as_tensor(np.stack([r[0] for r in rows]),device=context.device),
      'arity':torch.as_tensor(np.stack([r[1] for r in rows]),device=context.device),
      'families':torch.as_tensor(np.stack([r[2] for r in rows]),device=context.device),
      'touch':torch.as_tensor(np.stack([r[3] for r in rows]),device=context.device),
      'accent':torch.as_tensor(np.stack([r[4] for r in rows]),device=context.device),
      'valid':torch.as_tensor(np.stack([r[5] for r in rows]),device=context.device),
      'slot':torch.full((len(rows),),int(context.slot),device=context.device),
      'ds':torch.full((len(rows),),float(context.level),device=context.device),
      'version':torch.full((len(rows),),int(context.version),device=context.device),
      'head_phase':torch.as_tensor(np.asarray(row_ticks)%384,device=context.device),
      'bpm':torch.as_tensor([float(context.bv[max(0,np.searchsorted(context.bt,t,side='right')-1)]) for t in row_ticks],device=context.device),
      'route':torch.zeros((len(rows),),dtype=torch.long,device=context.device)}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):out=model(b)['cue'].float()
    prob=out.softmax(-1).cpu().numpy(); pred=prob.argmax(-1)
    return {t:(int(c),tuple(map(float,p))) for t,c,p in zip(row_ticks,pred,prob)}

@lru_cache(maxsize=32)
def _parsed_slide_times(tokens, bpm):
    wait=np.full(len(tokens),np.nan,np.float64);move=wait.copy()
    for i,token in enumerate(tokens):
        if i<2:continue
        try:wait[i],move[i]=slide_seconds(token,bpm)
        except (ValueError,TypeError,ZeroDivisionError):continue
    wait.flags.writeable=False;move.flags.writeable=False
    return wait,move

def _duration_mask(context,source,launch,min_move_ticks=48,max_move_ticks=72):
    bpm=float(context.bv[max(0,np.searchsorted(context.bt,source,side='right')-1)])
    times=ticks_to_seconds(np.asarray([source,launch,launch+min_move_ticks,launch+max_move_ticks],np.int64),context.bt,context.bv)
    wait,move=_parsed_slide_times(tuple(map(str,context.factor_session[1]['durations'])),bpm)
    mask=(np.abs(wait-(times[1]-times[0]))<=1e-7)&(move+1e-7>=times[2]-times[1])&(move<=times[3]-times[1]+1e-7)
    return mask if mask.any() else None


def _single_tap_candidate(value):
    # Reuse a Tap in the complete event, including Tap+Slide chain backbones.
    seq=value.candidates if isinstance(value,IntentChoices) else (EventIntent.from_representation(value),)
    for rank,item in enumerate(seq):
        if 0 in item.button_families and item.touch_count==0:
            return rank,item
    return None

def _cue_context_safe(context,source,launch,plan,current):
    # This stage knows intent, not realized holds/routes. Do not count a
    # launch Tap as independent of its parent and do not veto intervening
    # half-beat Taps. Exact candidate and whole-chart rules remain authoritative.
    return _one_slide(current)

def _find_cue(context,source,plan,reserved):
    ticks=np.asarray(sorted(map(int,context.ticks)),np.int64); candidates=[]
    lo,hi=source+48,source+192
    for tick in ticks[(ticks>=lo)&(ticks<=hi)]:
        tick=int(tick)
        if tick in reserved or tick not in plan:continue
        found=_single_tap_candidate(plan[tick])
        if found is None:continue
        mask=_duration_mask(context,source,tick)
        if mask is None:continue
        rank,intent=found
        candidates.append(((abs((tick-source)-96),rank,tick),tick,intent,mask))
    return min(candidates,key=lambda x:x[0])[1:] if candidates else None

def _declare_idle_fallbacks(plan,protected=()):
    result=dict(plan);protected=set(protected)
    for tick,value in result.items():
        if tick in protected:continue
        seq=value.candidates if isinstance(value,IntentChoices) else (value,)
        if EventIntent() not in seq:seq=seq+(EventIntent(),)
        result[tick]=IntentChoices(seq)
    return result

def relationalize(context,plan):
    """Return WHAT choices plus conditional Slide->launch Tap links.

    The relation is chosen before WHERE.  If no launch/cue contract is available,
    Slide candidates are removed at that source anchor rather than compressed.
    """
    if int(context.slot)<5:
        return _declare_idle_fallbacks(plan),RelationalWhatPlan((),{}, {},{},()),{'enabled':False,'reason':'expert-or-lower','idleFallback':'explicit-last-choice'}
    predictions=_cue_predictions(context,plan); out=dict(plan); links=[]; decisions=[]
    reserved=set(); duration_masks={}; source_to_link={}; cue_to_source={}; reservations=[]
    for source in sorted(predictions):
        current=_primary(out[source]) if source in out else EventIntent()
        if not _one_slide(current):continue
        cue_class,prob=predictions[source]; relation_choices=_relation_choices(out[source])
        fallback=_non_slide_choices(out[source]) or IntentChoices((EventIntent((0,),0),))
        if relation_choices is None:
            out[source]=fallback;decisions.append({'sourceTick':source,'status':'FALLBACK_NO_SINGLE_SLIDE'});continue
        cue=None
        if cue_class:
            cue=_find_cue(context,source,out,reserved)
            if cue is None:
                out[source]=fallback
                decisions.append({'sourceTick':source,'status':'FALLBACK_NO_CUE_ANCHOR','cueClass':cue_class,'cueProb':prob})
                continue
            launch,cue_intent,mask=cue
            if not _cue_context_safe(context,source,launch,out,current):
                out[source]=fallback
                decisions.append({'sourceTick':source,'status':'FALLBACK_CUE_HAND_WINDOW','cueClass':cue_class,'cueProb':prob,'cueTick':launch})
                continue
            reserved.add(launch)
            prior=out[launch]
            seq=prior.candidates if isinstance(prior,IntentChoices) else (EventIntent.from_representation(prior),)
            matching=tuple(x for x in seq if 0 in x.button_families and x.touch_count==0)
            out[launch]=IntentChoices(matching) if matching else cue_intent
        else:
            launch=source+96; mask=_duration_mask(context,source,launch)
            if launch not in set(map(int,context.ticks)): mask=None
            if mask is None:
                out[source]=fallback;decisions.append({'sourceTick':source,'status':'FALLBACK_NO_TIMING_SUPPORT','cueProb':prob});continue
            cue_intent=None
        # Ordinary one-track chains reserve their full motion interval.
        # Adjacent half-beat sources cannot both reserve a 48--72 tick motion
        # plus release; select a different WHAT rather than shorten either.
        at,end=map(float,ticks_to_seconds(np.asarray([launch,launch+72]),context.bt,context.bv))
        conflict=any(at<old_end+1/60 and old_at<end+1/60 for old_at,old_end in reservations)
        if conflict:
            out[source]=fallback
            if cue_intent is not None: reserved.discard(launch);out[launch]=prior
            decisions.append({'sourceTick':source,'status':'FALLBACK_MOTION_RESERVATION','cueClass':cue_class,'cueProb':prob})
            continue
        out[source]=relation_choices; duration_masks[source]=mask
        link=SlideLaunchLink(source,launch,launch if cue_intent is not None else None,cue_intent,48)
        links.append(link);source_to_link[source]=link;reservations.append((at,end))
        if cue_intent is not None:cue_to_source[launch]=source
        decisions.append({'sourceTick':source,'launchTick':launch,'status':'RELATIONAL_SLIDE','cueClass':cue_class,'cueProb':prob,'cueTick':link.cue_tick,'durationSupport':int(mask.sum())})
    # The planner explicitly offers a one-Tap closure when its companion
    # cannot be realized. A decoder may not drop the required Tap or emit rest.
    for tick,value in tuple(out.items()):
        if tick in source_to_link:continue
        seq=value.candidates if isinstance(value,IntentChoices) else (value,)
        keep=tuple(x for x in seq if 2 not in x.button_families)
        out[tick]=IntentChoices(keep) if keep else EventIntent((0,),0)
    for tick in cue_to_source:
        value=out[tick];seq=value.candidates if isinstance(value,IntentChoices) else (value,)
        keep=tuple(x for x in seq if 0 in x.button_families)
        closure=EventIntent((0,),0)
        if closure not in keep:keep=keep+(closure,)
        out[tick]=IntentChoices(keep)
    out=_declare_idle_fallbacks(out,cue_to_source)
    relation=RelationalWhatPlan(tuple(links),source_to_link,cue_to_source,duration_masks,tuple(decisions))
    info={'enabled':True,'links':len(links),'cueLinks':len(cue_to_source),'fallbacks':sum(d['status'].startswith('FALLBACK') for d in decisions),'decisions':decisions}
    info['idleFallback']='explicit-last-choice-except-mandatory-cues'
    info['primaryStarsAfterRelations']=sum(_primary(x).button_families.count(2) for x in out.values())
    return out,relation,info
