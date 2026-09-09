"""Bounded WHERE launch alternatives over existing WHAT; exact runtime timing.

Runtime duration ids retain a documented legacy embedding projection. They are
never passed to the pretrained embedding and never change the timing checked
by the codec or Harness. No model or checkpoint vocabulary is mutated.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
import math
from hashlib import sha256
import numpy as np
from .intent import EventIntent, IntentChoices
from .relational_what import SlideLaunchLink, _cue_predictions, _primary, _declare_idle_fallbacks
from ..io.timing import ticks_to_seconds
from ..io.durations import slide_seconds

STANDARD_LAUNCH_TICKS = 96
TIMING_EPS = 1e-7

def canonicalize_emitted_text(text, rep, vocab, bpm):
    """Publish a standard beat-relative Slide token when exactly equivalent."""
    model_ids=vocab.get('_duration_model_ids')
    if not text or model_ids is None:return text
    model_count=int(vocab.get('_model_duration_count',len(vocab['durations'])))
    result=str(text); arity=min(2,int(rep['button_arity']))
    for i in range(arity):
        if int(rep['button_family'][i])!=2:continue
        duration=int(rep['button_duration'][i])
        if duration<model_count or duration>=len(vocab['durations']) or duration>=len(model_ids):continue
        model_id=int(model_ids[duration])
        if model_id<2 or model_id>=model_count:continue
        runtime=vocab['durations'][duration]; canonical=vocab['durations'][model_id]
        try:
            rw,rm=slide_seconds(runtime,bpm); cw,cm=slide_seconds(canonical,bpm)
        except (TypeError,ValueError,ZeroDivisionError):
            continue
        if abs(rw-cw)<=TIMING_EPS and abs(rm-cm)<=TIMING_EPS:
            result=result.replace(f'[{runtime}]',f'[{canonical}]',1)
    return result


@dataclass(frozen=True)
class WherePlan:
    links: tuple
    source_to_link: object
    cue_to_source: object
    duration_masks: object
    alternatives: object
    fingerprint: str
    runtime_where: bool = True


def prepare(context, plan):
    if context.slot < 5:
        from .relational_what import relationalize
        out, relation, info = relationalize(context, plan)
        return context, out, relation, info
    original = context.factor_session[1]
    vocab = dict(original)
    vocab['durations'] = list(original['durations'])
    model_ids = list(range(len(vocab['durations'])))
    ids = {s:i for i,s in enumerate(vocab['durations'])}
    from .sequence_runtime import maybe_profile
    profile=maybe_profile(context,plan,vocab)
    vocab['_sequence_profile']=profile
    predictions = {} if profile is not None and profile.cue_enabled else _cue_predictions(context, plan)
    options = {}; all_support = {}; sources = {}
    # Supported beat-relative movement lengths inherit their trained route-
    # conditioned score. Wait is independently planned over existing anchors.
    move_map={}
    for i,token in enumerate(original['durations']):
        if i<2 or '#' in token or ':' not in token: continue
        try:
            a,b=map(float,token.split(':')); span=384*b/a
        except (ValueError,ZeroDivisionError): continue
        if 24<=span<=192 and abs(span-round(span))<1e-7: move_map.setdefault(int(round(span)),i)
    moves=sorted(move_map.items())
    ticks=sorted(map(int,context.ticks))
    # Batch the shared tempo mapping once. Re-evaluating a NumPy BPM map for
    # every source/launch/movement alternative dominates CPU planning time.
    needed=np.asarray(sorted(set(ticks)|{t+96 for t in ticks}|{t+span+offset for t in ticks for span,_ in moves for offset in (0,96)}),np.int64)
    mapped=dict(zip(map(int,needed),map(float,ticks_to_seconds(needed,context.bt,context.bv))))
    times=lambda keys:np.asarray([mapped[int(t)] for t in keys],np.float64)
    for source,value in sorted(plan.items()):
        choices=value.candidates if isinstance(value,IntentChoices) else (value,)
        if not any(EventIntent.from_representation(x).button_families.count(2)==1 for x in choices): continue
        source=int(source); candidates=[]
        cue_prob=predictions.get(source,(0,(1.,0.,0.)))[1]
        cue_launches=[t for t in ticks if t-source==STANDARD_LAUNCH_TICKS and t in plan and 0 in _primary(plan[t]).button_families]
        # An explicit unbound branch competes with reuse. More available cue
        # anchors must not multiply the total probability of requesting cue.
        if source+STANDARD_LAUNCH_TICKS <= ticks[-1]+STANDARD_LAUNCH_TICKS:
            bpm=float(context.bv[max(0,np.searchsorted(context.bt,source,side='right')-1)])
            for span,model_id in moves:
                t=times([source,source+STANDARD_LAUNCH_TICKS,source+STANDARD_LAUNCH_TICKS+span]);w,m=slide_seconds(original['durations'][model_id],bpm)
                equivalent=abs(w-(t[1]-t[0]))<1e-7 and abs(m-(t[2]-t[1]))<1e-7
                if profile is not None and not profile.movement_fits(source,float(t[2])):continue
                if equivalent:
                    index=model_id
                elif profile is not None:
                    # A BPM change can cross the one-beat wait or movement.
                    # Preserve the exact anchors instead of deleting the Slide
                    # or silently falling back to the old categorical timing.
                    token=f'{float(t[1]-t[0]):.12f}##{float(t[2]-t[1]):.12f}'
                    if token not in ids:
                        ids[token]=len(vocab['durations']);vocab['durations'].append(token);model_ids.append(model_id)
                    index=ids[token]
                else:continue
                candidates.append((index,SlideLaunchLink(source,source+STANDARD_LAUNCH_TICKS,None,None,span),math.log(max(.1,cue_prob[0]))))
        for launch in ticks:
            delta=launch-source
            # Official charts overwhelmingly launch one beat after the Star
            # head. WHERE may reuse a Tap/Break/EX Tap only on that one-beat
            # anchor; do not stretch the Star head across nearby Tap anchors.
            if delta<STANDARD_LAUNCH_TICKS:continue
            if delta>STANDARD_LAUNCH_TICKS:break
            target=_primary(plan[launch]) if launch in plan else EventIntent()
            cue=target if 0 in target.button_families else None
            # Always retain the ordinary unbound option; reuse requires an
            # already planned Tap, not promotion of a different WHAT candidate.
            if cue is None or (profile is not None and profile.cue_enabled):continue
            for span,model_id in moves:
                if launch+span>ticks[-1]+96:continue
                t=times([source,launch,launch+span]);wait,move=float(t[1]-t[0]),float(t[2]-t[1])
                token=f'{wait:.12f}##{move:.12f}'
                if token not in ids:
                    ids[token]=len(vocab['durations']);vocab['durations'].append(token);model_ids.append(model_id)
                index=ids[token]
                link=SlideLaunchLink(source,launch,launch if cue is not None else None,cue,span)
                # Cue probability is a bounded prior, not a search veto.
                bias=math.log(max(.1,sum(cue_prob[1:]))) - math.log(max(1,len(cue_launches)))
                bias-=.25*abs(delta-STANDARD_LAUNCH_TICKS)/STANDARD_LAUNCH_TICKS
                candidates.append((index,link,bias))
        if candidates:options[source]=tuple(candidates);sources[source]=candidates[0][1]
    vocab['_sequence_blocked_sources']=frozenset(int(t) for t,v in plan.items() if profile is not None and any(EventIntent.from_representation(x).button_families.count(2)==1 for x in (v.candidates if isinstance(v,IntentChoices) else (v,))) and int(t) not in options)
    vocab['_duration_model_ids']=np.asarray(model_ids,np.int64)
    vocab['_duration_model_ids'].flags.writeable=False
    vocab['_model_duration_count']=len(original['durations'])
    for source,candidates in options.items():
        mask=np.zeros(len(model_ids),np.bool_);mask[[x[0] for x in candidates]]=True;mask.flags.writeable=False
        all_support[source]=mask
    digest=sha256(repr(tuple((t,_primary(v)) for t,v in sorted(plan.items()))).encode())
    for source,values in options.items():
        digest.update(str(source).encode())
        digest.update(np.asarray([(d,x.launch_tick,-1 if x.cue_tick is None else x.cue_tick,x.min_move_ticks,b) for d,x,b in values],np.float64).tobytes())
    if profile is not None:digest.update(repr((profile.digest,profile.weight,profile.start_weight,profile.cue_enabled)).encode())
    relation=WherePlan(tuple(sources.values()),MappingProxyType(sources),MappingProxyType({}),MappingProxyType(all_support),MappingProxyType(options),digest.hexdigest())
    ctx=replace(context,factor_session=(context.factor_session[0],vocab,*context.factor_session[2:]))
    out=_declare_idle_fallbacks(plan)
    stars={int(t) for t,v in plan.items() if 2 in _primary(v).button_families}
    edges={t:t+96 for t in stars if t+96 in stars and 0 in _primary(plan[t+96]).button_families}
    capacities=[]
    for origin in set(edges)-set(edges.values()):
        t=origin;length=1
        while t in edges:t=edges[t];length+=1
        capacities.append(length)
    return ctx,out,relation,{'enabled':True,'authority':'WHERE','sources':len(sources),'timingAlternatives':sum(map(len,options.values())),
        'runtimeDurationTokens':len(model_ids)-len(original['durations']),'modelDurationProjection':'legacy beat-relative movement embedding; exact timing retained by runtime',
        'whatPrimaryChanged':0,'cuePriorIsHardGate':False,'motionReservationHardGate':False,
        'primaryWhatMaxTapSlideChain':max(capacities,default=0),'primaryWhatChainLengths':sorted(capacities,reverse=True),
        'sequenceProfile':None if profile is None else profile.digest,'routeWeight':None if profile is None else profile.weight,
        'cueAuthority':'conditional WHERE geometry' if profile is not None and profile.cue_enabled else 'legacy linked timing'}


def selected_link(context, plan, tick, rep, events=None):
    if not getattr(plan,'runtime_where',False):return plan.source_to_link.get(tick)
    slide=[i for i in range(int(rep['button_arity'])) if int(rep['button_family'][i])==2]
    if len(slide)!=1:return None
    i=slide[0]; index=int(rep['button_duration'][i]); lane=int(rep['button_start'][i])
    alternatives=plan.alternatives.get(tick,())
    exact=[link for duration,link,_ in alternatives if duration==index]
    if events is None:return exact[0] if exact else None
    vocab=context.factor_session[1]
    bpm=float(context.bv[max(0,np.searchsorted(context.bt,int(tick),side='right')-1)])
    try: rw,rm=slide_seconds(vocab['durations'][index],bpm)
    except (IndexError,TypeError,ValueError,ZeroDivisionError): return exact[0] if exact else None
    equivalent=[]
    for duration,link,_ in alternatives:
        try: aw,am=slide_seconds(vocab['durations'][duration],bpm)
        except (IndexError,TypeError,ValueError,ZeroDivisionError):continue
        if abs(aw-rw)<=TIMING_EPS and abs(am-rm)<=TIMING_EPS:equivalent.append(link)
    from ..io.factors import factor_event
    for link in equivalent:
        if link.cue_tick is None:continue
        try: notes=factor_event(str(events.get(link.cue_tick,'')))['notes']
        except Exception:continue
        if any(n['family']=='tap' and int(n['start'])-1==lane for n in notes):return link
    unbound=next((link for link in exact if link.cue_tick is None),None)
    if unbound is not None:return unbound
    return next((link for link in equivalent if link.cue_tick is None),equivalent[0] if equivalent else None)


def candidate_snapshot(snapshot, plan, tick, active, references=None):
    if not getattr(plan,'runtime_where',False):return
    snapshot['_runtime_where']=True
    if tick not in plan.alternatives:return
    mask=np.array(plan.duration_masks[tick],copy=True);bias=np.zeros(len(mask),np.float32)
    fixed_lanes={}
    for duration,link,score in plan.alternatives[tick]:
        if link.cue_tick in active and active[link.cue_tick]['source']!=tick:mask[duration]=False
        bias[duration]=score
        if references is not None and link.cue_tick in references:
            from ..io.factors import factor_event
            fixed_lanes[duration]={int(n['start'])-1 for n in factor_event(references[link.cue_tick])['notes'] if n['family']=='tap'}
    prior=snapshot.get('allowedSlideDurationMask')
    snapshot['allowedSlideDurationMask']=mask if prior is None else mask&prior
    snapshot['_where_duration_bias']=bias
    snapshot['_where_fixed_cue_lanes']=fixed_lanes


def cue_binding_allowed(provider,context,link,lane):
    if link is None or link.cue_tick is None:return True
    from ..harness.source_head import TRACK_LIFECYCLE
    moment=float(ticks_to_seconds(np.asarray([link.cue_tick]),context.bt,context.bv)[0])
    return not TRACK_LIFECYCLE.lane_forbidden_at(provider,moment,lane)


def recent_head_lanes(provider, moment):
    """Compatibility view of the exact shared Tap-only open interval."""
    from ..harness.source_head import TRACK_LIFECYCLE
    lanes=TRACK_LIFECYCLE.blocked_lanes(provider,moment)
    provider._where_track_cache=provider._source_head_cache
    return lanes
