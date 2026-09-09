"""Generation authority: trained renderer plus bounded proposal/completion code."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import uuid
import torch

from ..domain import Chart,ChartRef,Proposal
from .renderer import render,RenderFailure
from ..runtime.payloads import Envelope
from .intent import EventIntent,IntentChoices


class GeneratorBackend:
    def __init__(self,codec,context,harness,anchor_scores=None,mert_structure=None):
        self.codec=codec;self.context=replace(context,cache={});self.harness=harness;self.anchor_scores=anchor_scores
        self.primary_events=None;self.timings=[]
        self.pending_intents=None
        self.what_plan=None
        self.what_info=None
        self.relational_plan=None
        self.relational_info=None
        self.causal_budget=None

    def _invoke(self,context,provider,**kwargs):
        self.pending_intents=kwargs.get('intent_plan')
        try:
            if kwargs.get('relational_what_plan') is not None and not kwargs.get('references'):
                if context.metadata.get('causalSearchMode')=='causal-v1':
                    from .causal_recovery import render_with_causal_recovery
                    events=render_with_causal_recovery(context,lambda:self.harness.sampling_provider(context),initial_provider=provider,budget=self.causal_budget,**kwargs)
                else:
                    from .relation_recovery import render_with_recovery
                    events=render_with_recovery(context,lambda:self.harness.sampling_provider(context),initial_provider=provider,**kwargs)
                return events,None
            if context.metadata.get('causalSearchMode')=='causal-v1' and self.causal_budget is not None:
                provider.causal_budget=self.causal_budget.view()
            events=render(context,provider,**kwargs)
            relation=kwargs.get('relational_what_plan')
            if relation is not None:
                from .relation_recovery import verify_complete_relations
                complete={**dict(kwargs.get('references') or {}),**events}
                try:
                    context.cache['relational_what_audit']=verify_complete_relations(context,relation,complete)
                except RenderFailure as exc:
                    exc.partial_events=complete
                    exc.pending_ticks=tuple(t for t in context.ticks if exc.tick is None or t>=exc.tick)
                    raise
            return events,None
        except RenderFailure as exc:
            if exc.stage!='provider_sampling':raise
            data={'tick':exc.tick,'pending_ticks':tuple(getattr(exc,'pending_ticks',context.ticks)),'reason':str(exc),
                  'decisions':tuple(exc.details.get('lastDecisions',()))}
            return dict(getattr(exc,'partial_events',{})),Envelope(data,torch.tensor([-1 if exc.tick is None else exc.tick],device=self.codec.device),'generation-interruption/1')

    def propose(self,request,feedback,budget):
        import time
        started=time.perf_counter();data=feedback.constraints.data;phase=data['phase'];ctx=self.context
        if ctx.progress:
            names={'initial':'主生成','repair':'按 Harness 约束重生成','resume':'恢复受阻的生成'}
            ctx.progress(f"难度 {ctx.slot}: {names[phase]}（轮次 {data['variant']+1}）")
        provider=self.harness.sampling_provider(ctx);interruption=None
        if phase=='initial':
            if ctx.slot>=4:
                from .planning import intent_plan
                target_stars=int(self.harness.conditions['star_plan_stars'])
                causal=ctx.metadata.get('causalSearchMode')=='causal-v1'
                if causal:
                    from .causal_recovery import CausalBudget
                    self.causal_budget=CausalBudget()
                self.what_plan,_,self.what_info=intent_plan(ctx,target_stars,topk=64 if causal else 8)
                if causal:
                    from .causal_recovery import nonempty_plan
                    self.what_plan=nonempty_plan(self.what_plan)
                from .relational_where import prepare
                ctx,self.what_plan,self.relational_plan,self.relational_info=prepare(ctx,self.what_plan)
                if causal:self.what_plan=nonempty_plan(self.what_plan)
                self.context=ctx
                provider=self.harness.sampling_provider(ctx)
                self.what_info=dict(self.what_info,relationalSlide=self.relational_info)
                events,interruption=self._invoke(ctx,provider,intent_plan=self.what_plan,allow_intent_revision=True,
                    duration_masks=self.relational_plan.duration_masks or None,relational_what_plan=self.relational_plan)
            else:
                events,interruption=self._invoke(ctx,provider)
            self.primary_events=dict(events)
        else:
            events=dict(data['base_events']);targets=list(data.get('targets',()))
            if self.relational_plan is not None and targets:
                from .contract_utils import dependency_window
                targets=dependency_window(targets,self.relational_plan,ctx.ticks,context=ctx,events=events)
            intents=None;duration_masks={};route_masks={};start_masks={}
            if phase=='repair':
                # Re-render the complete dependency window jointly, preserving
                # note families as proposals but allowing Harness rejection to
                # cause a different native intent where the model owns it.
                from .sampling import text_representation
                intents={tick:(self.what_plan[tick].rotated(data['variant']) if self.what_plan and tick in self.what_plan else EventIntent.from_representation(text_representation(events[tick],ctx.factor_session[1]))) for tick in targets if tick in events}
            elif phase=='resume':
                if data.get('fresh_intent') and ctx.slot<4:
                    intents=None
                else:
                    intents={tick:rep for tick,rep in (self.pending_intents or {}).items() if tick in targets}
                    if data.get('fresh_intent'):
                        intents={tick:(rep.rotated(data.get('escape_level',1)) if isinstance(rep,IntentChoices) else rep) for tick,rep in intents.items()}
                    if not intents:intents=None
            if not targets:return ()
            if self.relational_plan is not None and self.what_plan:
                # Relations are a contract across repairs, not an initial-only hint.
                intents={t:self.what_plan[t] for t in targets if t in self.what_plan}
                duration_masks={t:m for t,m in self.relational_plan.duration_masks.items() if t in targets}
            local_context=replace(ctx,progress=None)
            references=dict(events)
            for tick in ctx.ticks:references.setdefault(int(tick),'')
            generated,interruption=self._invoke(local_context,provider,targets=targets,references=references,intent_plan=intents,
                             seed_offset=int(data['variant'])*100003+(int(data.get('escape_level',0))*7919 if data.get('fresh_intent') else 0),allow_intent_revision=True,
                             duration_masks=duration_masks or None,route_masks=route_masks or None,start_masks=start_masks or None,relational_what_plan=self.relational_plan)
            for tick,text in generated.items():
                if text:events[tick]=text
                else:events.pop(tick,None)
        payload=self.codec.encode(events,ctx.bt,ctx.bv)
        if self.relational_plan is not None and interruption is None:
            audit=ctx.cache.get('relational_what_audit')
            if not isinstance(audit,dict) or audit.get('failedCues'):
                raise RuntimeError('complete relational proposal lacks a passing relation audit')
            audit_digest=hashlib.sha256(json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
            ctx.cache['relational_what_binding']={'chartDigest':payload.digest,'auditDigest':audit_digest,
                                                   'fallbackReasons':sorted({str(row.get('reason')) for row in audit.get('fallback',[])})}
        chart=Chart(ChartRef(request.request_id,str(uuid.uuid4()),payload.digest),request.definition,payload)
        self.timings.append({'phase':phase,'seconds':time.perf_counter()-started,'events':len(events),'harnessSampling':dict(provider.timings),
                             'reusedPrefixFrames':ctx.cache.get('last_reused_prefix',0),'forwardFrames':ctx.cache.get('last_forward_frames',0),
                             'whatAuthority':'joint_event_plan' if ctx.slot>=4 else 'v4_combined',
                             'causalSearchMode':ctx.metadata.get('causalSearchMode','stable'),
                             'causalRecovery':ctx.cache.get('causal_recovery'),
                             'causalBudget':self.causal_budget.metrics() if self.causal_budget is not None else None,
                             'relationAuditBinding':ctx.cache.get('relational_what_binding'),
                             'whatChoiceHistogram':dict(ctx.cache.get('what_choice_histogram',{})),
                             'whatPlan':self.what_info,'relationAudit':ctx.cache.get('relational_what_audit',{}),'declaredIdleAnchors':sum(not events.get(t) for t in self.what_plan) if self.what_plan and interruption is None else None})
        if ctx.progress:ctx.progress(f'难度 {ctx.slot}: 本轮模型已返回，Harness 正在检查完整草稿')
        return (Proposal(chart,feedback.base,feedback.feedback_id,complete=interruption is None,interruption=interruption),)
