"""Bounded dependency rollback for relational generation; never drops a cue."""
from __future__ import annotations
from .renderer import render, RenderFailure
from .intent import EventIntent, IntentChoices
from .sampling import text_representation, intent_signature

def render_with_recovery(context, provider_factory, *, initial_provider=None,
                         max_attempts=4, **kwargs):
    if kwargs.get('references'):
        raise ValueError('recovery entry expects a fresh planned generation')
    plan=kwargs.get('intent_plan') or {};relation=kwargs.get('relational_what_plan')
    if relation is None or not plan:raise ValueError('relational WHAT is required')
    if not 1<=max_attempts<=8:raise ValueError('invalid recovery budget')
    all_ticks=sorted(map(int,context.ticks));targets=all_ticks;prefix={};failures=[];totals={}
    vocab=context.factor_session[1];provider=initial_provider
    for attempt in range(max_attempts):
        if attempt or provider is None:provider=provider_factory()
        call=dict(kwargs);call['targets']=targets
        call['intent_plan']={t:plan[t] for t in targets if t in plan}
        if attempt:call['references']={t:prefix.get(t,'') for t in all_ticks if t<targets[0]}
        call['seed_offset']=int(kwargs.get('seed_offset',0))+attempt*104729
        try:
            result=render(context,provider,**call);combined={**prefix,**result}
            for t,value in plan.items():
                seq=value.candidates if isinstance(value,IntentChoices) else (value,)
                actual=intent_signature(text_representation(combined.get(t,''),vocab))
                if actual not in {intent_signature(x) for x in seq}:
                    raise RenderFailure('WHAT drift after relational recovery',stage='provider_sampling',tick=t)
            audit=verify_complete_relations(context,relation,combined)
            context.cache['relational_what_audit']=audit
            context.cache['relation_recovery']={'attempts':attempt+1,'failures':failures}
            for k,v in provider.timings.items():totals[k]=totals.get(k,0)+v
            if initial_provider is not None:initial_provider.timings.update(totals)
            return combined
        except RenderFailure as exc:
            if exc.stage!='provider_sampling':raise
            for k,v in provider.timings.items():totals[k]=totals.get(k,0)+v
            partial=getattr(exc,'partial_events',None) or locals().get('combined',{})
            exc.partial_events={**prefix,**partial}
            exc.pending_ticks=tuple(t for t in all_ticks if exc.tick is None or t>=exc.tick)
            context.cache['relation_recovery']={'attempts':attempt+1,'failures':failures,'exhausted':True}
            if exc.tick is None or attempt+1==max_attempts:
                if initial_provider is not None:initial_provider.timings.update(totals)
                raise
            # Re-select the preceding source route and all dependent suffix
            # events, retaining verified earlier Tap/Star bindings and caches.
            parent=exc.details.get('relationalCueSourceTick',relation.cue_to_source.get(exc.tick,exc.tick-192))
            boundary=max(0,min(parent,exc.tick-96))
            from .contract_utils import dependency_window
            seed_window=dependency_window([boundary,exc.tick],relation,all_ticks,context=context,events=exc.partial_events)
            boundary=min(seed_window)
            targets=[t for t in all_ticks if t>=boundary]
            if not targets:raise
            partial=getattr(exc,'partial_events',None) or locals().get('combined',{})
            prefix={t:v for t,v in {**prefix,**partial}.items() if t<targets[0]}
            failures.append({'tick':exc.tick,'rollbackTick':targets[0],'reason':str(exc)[:300]})
    raise AssertionError('unreachable recovery exit')


def verify_complete_relations(context,relation,events):
    import numpy as np
    from ..io.durations import slide_seconds
    from ..io.timing import ticks_to_seconds
    vocab=context.factor_session[1]
    result={'activated':[],'fallback':[],'resolvedCues':[],'failedCues':[]}
    if int(context.slot)>=5 and not getattr(relation,'runtime_where',False):
        for tick,text in events.items():
            rep=text_representation(text,vocab)
            if any(int(x)==2 for x in rep['button_family'][:int(rep['button_arity'])]) and int(tick) not in relation.source_to_link:
                raise RenderFailure('Slide without declared launch plan',stage='provider_sampling',tick=int(tick))
    for declared in relation.links:
        link=declared
        t=link.source_tick;rep=text_representation(events.get(t,''),vocab)
        if getattr(relation,'runtime_where',False):
            from .relational_where import selected_link
            link=selected_link(context,relation,t,rep,events)
        ids=[i for i in range(int(rep['button_arity'])) if int(rep['button_family'][i])==2]
        if not ids:
            result['fallback'].append({'sourceTick':t,'reason':'NON_SLIDE_WHAT'});continue
        if len(ids)>1 and getattr(relation,'runtime_where',False):
            result.setdefault('outsideSingleTrackScope',[]).append(t);continue
        if link is None:raise RenderFailure('missing WHERE timing alternative',stage='provider_sampling',tick=t)
        if len(ids)!=1:raise RenderFailure('compound relation source',stage='provider_sampling',tick=t)
        i=ids[0];lane=int(rep['button_start'][i]);duration=int(rep['button_duration'][i])
        bpm=float(context.bv[max(0,np.searchsorted(context.bt,t,side='right')-1)])
        support=relation.duration_masks[t]
        if duration>=len(support) or not bool(support[duration]):
            raise RenderFailure('relation motion support drift',stage='provider_sampling',tick=t)
        wait,_=slide_seconds(vocab['durations'][duration],bpm)
        expected=float(np.diff(ticks_to_seconds(np.asarray([t,link.launch_tick]),context.bt,context.bv))[0])
        if abs(wait-expected)>1e-7:
            raise RenderFailure('relation launch timing drift',stage='provider_sampling',tick=t)
        result['activated'].append({'sourceTick':t,'launchTick':link.launch_tick,'lane':lane})
        observed=text_representation(events.get(link.launch_tick,''),vocab)
        if any(int(observed['button_family'][j])==0 and int(observed['button_start'][j])==lane for j in range(int(observed['button_arity']))):
            result.setdefault('observedSingleSourceTapCues',[]).append({'sourceTick':t,'cueTick':link.launch_tick,'lane':lane})
        if link.cue_tick is not None:
            cue=text_representation(events.get(link.cue_tick,''),vocab)
            ok=any(int(cue['button_family'][j])==0 and int(cue['button_start'][j])==lane for j in range(int(cue['button_arity'])))
            if not ok:raise RenderFailure('unresolved launch cue',stage='provider_sampling',tick=link.cue_tick)
            result['resolvedCues'].append({'sourceTick':t,'cueTick':link.cue_tick,'lane':lane})
    return result
