"""Optional causal WHAT recovery over the released R0 renderer and Harness."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .intent import EventIntent,IntentChoices
from .renderer import render,RenderFailure
from .sampling import RepresentationEncodingError,intent_signature,text_representation
from ..io.timing import ticks_to_seconds


EVENT_CANDIDATE_LIMIT=64
NODE_EVENT_LIMIT=32
RUN_CANDIDATE_LIMIT=4096
MAX_BACKTRACK_EVENTS=2
BEAM_WIDTH=4
MAX_NODES=32


def nonempty_plan(plan):
    result={}
    for tick,value in plan.items():
        seq=value.candidates if isinstance(value,IntentChoices) else (EventIntent.from_representation(value),)
        kept=tuple(candidate for candidate in seq if candidate.button_arity or candidate.touch_count)
        if not kept:raise RenderFailure('no non-empty declared WHAT candidate',stage='provider_sampling',tick=int(tick))
        result[int(tick)]=IntentChoices(kept)
    return result


class CausalBudget:
    def __init__(self):
        self.events=Counter();self.total=0
    def view(self):
        return CausalBudgetView(self)
    def metrics(self):
        return {'candidateCount':int(self.total),'maxEventCandidates':max(self.events.values() or [0]),
                'eventCandidates':{str(k):int(v) for k,v in sorted(self.events.items())},
                'eventLimit':EVENT_CANDIDATE_LIMIT,'runLimit':RUN_CANDIDATE_LIMIT,
                'scope':'one difficulty session including recovery and Harness-directed revisions'}


class CausalBudgetView:
    def __init__(self,shared):
        self.shared=shared;self.node=Counter()
    def _remaining(self,tick):
        return max(0,min(NODE_EVENT_LIMIT-self.node[int(tick)],
                         EVENT_CANDIDATE_LIMIT-self.shared.events[int(tick)],
                         RUN_CANDIDATE_LIMIT-self.shared.total))
    def limits(self,tick):
        remaining=self._remaining(tick)
        if remaining<=0:
            raise RenderFailure('causal candidate budget exhausted',stage='provider_sampling',tick=int(tick),
                                details={'causalBudget':self.shared.metrics()})
        same=min(8,remaining)
        choices=max(1,min(4,remaining//same))
        return {'_same_intent_candidate_limit':same,'_intent_choice_limit':choices,
                '_causal_candidate_remaining':remaining}
    def reserve(self,tick,count):
        tick=int(tick);count=int(count);remaining=self._remaining(tick)
        if count>remaining:
            raise RenderFailure('causal candidate budget exhausted',stage='provider_sampling',tick=tick,
                                details={'requestedCandidates':count,'remainingCandidates':remaining,
                                         'causalBudget':self.shared.metrics()})
        self.node[tick]+=count;self.shared.events[tick]+=count;self.shared.total+=count


def _plan_variant(plan,overrides):
    result=dict(plan)
    for tick,rank in overrides.items():
        seq=plan[int(tick)].candidates
        if not 0<=int(rank)<len(seq):raise ValueError('causal WHAT rank outside declared support')
        result[int(tick)]=IntentChoices((seq[int(rank)],))
    return result


def _prefix_key(prefix,overrides):
    return hashlib.sha256(json.dumps({'prefix':sorted(prefix.items()),'overrides':sorted(overrides.items())},
                                     ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def _causal_source(context,provider,events,failure_tick,all_ticks):
    if not events:return None
    payload=provider.codec.encode(events,context.bt,context.bv);c=payload.columns
    at=float(ticks_to_seconds(np.asarray([failure_tick]),context.bt,context.bv)[0])
    owners=[]
    if len(c['input_event']):
        live=(c['input_start']<=at+.2)&(c['input_end']+.2>=at)
        owners.extend(c['input_event'][live].detach().cpu().tolist())
    if len(c['track_event']):
        live=(c['track_start']<=at+.2)&(c['track_end']+.2>=at)
        owners.extend(c['track_event'][live].detach().cpu().tolist())
    event_ticks=c['event_tick'].detach().cpu().tolist()
    tick_index={int(t):i for i,t in enumerate(all_ticks)}
    failure_index=tick_index.get(int(failure_tick))
    sources=[]
    if failure_index is not None:
        for owner in owners:
            tick=int(event_ticks[int(owner)]);index=tick_index.get(tick)
            if index is not None and 0<failure_index-index<=MAX_BACKTRACK_EVENTS and tick in events:
                sources.append(tick)
    if sources:return max(sources)
    if failure_index is None or failure_index==0:return None
    return int(all_ticks[failure_index-1])


def _alternatives(plan,tick,actual_text,vocab):
    value=plan[int(tick)];seq=value.candidates
    try:actual=intent_signature(text_representation(actual_text,vocab))
    except RepresentationEncodingError as exc:
        raise RenderFailure(str(exc),stage='generated_encoding',tick=int(tick),cause=exc) from exc
    result=[]
    for rank,candidate in enumerate(seq):
        if intent_signature(candidate)==actual:continue
        result.append((rank,intent_signature(candidate)))
        if len(result)>=BEAM_WIDTH:break
    return result


def render_with_causal_recovery(context,provider_factory,*,initial_provider=None,budget=None,**kwargs):
    if kwargs.get('references'):raise ValueError('causal recovery expects a fresh generation')
    plan=kwargs.get('intent_plan') or {};relation=kwargs.get('relational_what_plan')
    if relation is None or not plan:raise ValueError('causal recovery requires relational WHAT')
    plan=nonempty_plan(plan);all_ticks=sorted(map(int,context.ticks));vocab=context.factor_session[1]
    budget=budget or CausalBudget();stack=[{'prefix':{},'targets':all_ticks,'overrides':{},'parent':None,'cause':None}]
    seen={_prefix_key({}, {})};failures=[];totals={};nodes=0
    best_partial={};best_failure_tick=None
    from .relation_recovery import verify_complete_relations
    while stack and nodes<MAX_NODES and budget.total<RUN_CANDIDATE_LIMIT:
        node=stack.pop();provider=initial_provider if nodes==0 and initial_provider is not None else provider_factory();combined=None
        provider.causal_budget=budget.view();variant=_plan_variant(plan,node['overrides'])
        call=dict(kwargs);call['targets']=node['targets'];call['intent_plan']={t:variant[t] for t in node['targets'] if t in variant}
        if node['targets'] and node['targets'][0]!=all_ticks[0]:
            call['references']={t:node['prefix'].get(t,'') for t in all_ticks if t<node['targets'][0]}
        call['seed_offset']=int(kwargs.get('seed_offset',0))+nodes*1000003
        if context.progress:context.progress(f"因果搜索：节点 {nodes+1}，从 {node['targets'][0] if node['targets'] else '结束'} 继续")
        try:
            generated=render(context,provider,**call);combined={**node['prefix'],**generated}
            for tick,value in plan.items():
                actual=intent_signature(text_representation(combined.get(tick,''),vocab))
                if actual not in {intent_signature(candidate) for candidate in value.candidates}:
                    raise RenderFailure('WHAT drift after causal recovery',stage='provider_sampling',tick=int(tick))
            audit=verify_complete_relations(context,relation,combined)
            for key,value in provider.timings.items():totals[key]=totals.get(key,0)+value
            receipt={'mode':'causal-v1','status':'PASS','nodes':nodes+1,'failures':failures,
                     'budget':budget.metrics(),'relationAudit':audit}
            context.cache['causal_recovery']=receipt;context.cache['relational_what_audit']=audit
            if initial_provider is not None:initial_provider.timings.update(totals)
            return combined
        except RenderFailure as exc:
            if exc.stage!='provider_sampling':raise
            for key,value in provider.timings.items():totals[key]=totals.get(key,0)+value
            partial=dict(combined) if combined is not None else {**node['prefix'],**dict(getattr(exc,'partial_events',{}) or {})}
            failure_tick=None if exc.tick is None else int(exc.tick)
            if failure_tick is None:raise
            if len(partial)>len(best_partial):best_partial=dict(partial);best_failure_tick=failure_tick
            source=_causal_source(context,provider,partial,failure_tick,all_ticks)
            record={'node':nodes,'tick':failure_tick,'reason':str(exc)[:300],'sourceTick':source,
                    'parent':node['parent'],'budget':budget.metrics()}
            failures.append(record);nodes+=1
            if source is None or source not in plan:continue
            alternatives=_alternatives(plan,source,partial.get(source,''),vocab)
            children=[]
            for rank,signature in alternatives:
                overrides=dict(node['overrides']);overrides[source]=rank
                prefix={t:text for t,text in partial.items() if int(t)<source}
                key=_prefix_key(prefix,overrides)
                if key in seen:continue
                seen.add(key)
                children.append({'prefix':prefix,'targets':[t for t in all_ticks if t>=source],
                                 'overrides':overrides,'parent':nodes-1,
                                 'cause':{'failureTick':failure_tick,'sourceTick':source,
                                          'rank':rank,'intent':repr(signature)}})
            for child in reversed(children):stack.append(child)
    receipt={'mode':'causal-v1','status':'BUDGET_EXHAUSTED' if budget.total>=RUN_CANDIDATE_LIMIT or
             any(v>=EVENT_CANDIDATE_LIMIT for v in budget.events.values()) else 'NO_VALIDATED_BRANCH',
             'nodes':nodes,'failures':failures,'budget':budget.metrics()}
    context.cache['causal_recovery']=receipt
    if initial_provider is not None:initial_provider.timings.update(totals)
    tick=failures[-1]['tick'] if failures else None
    failure=RenderFailure('causal recovery did not find a complete branch',stage='provider_sampling',tick=tick,
                          details={'causalRecovery':receipt})
    failure.partial_events=dict(best_partial)
    boundary=best_failure_tick if best_failure_tick is not None else tick
    failure.pending_ticks=tuple(t for t in all_ticks if boundary is None or t>=boundary)
    raise failure


__all__=['render_with_causal_recovery','nonempty_plan','CausalBudget']
