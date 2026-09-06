"""Generation authority: trained renderer plus bounded proposal/completion code."""
from __future__ import annotations
from dataclasses import replace
import uuid
import numpy as np
import torch

from ..domain import Chart,ChartRef,Proposal
from .renderer import render,RenderFailure
from ..runtime.payloads import Envelope


class GeneratorBackend:
    def __init__(self,codec,context,harness,anchor_scores=None,mert_structure=None):
        self.codec=codec;self.context=replace(context,cache={});self.harness=harness;self.anchor_scores=anchor_scores
        self.primary_events=None;self.timings=[];self.attempted_stars=set()
        self.intent_scores=None
        self.pending_intents=None

    def _invoke(self,context,provider,**kwargs):
        self.pending_intents=kwargs.get('intent_plan')
        try:return render(context,provider,**kwargs),None
        except RenderFailure as exc:
            if exc.stage!='provider_sampling':raise
            data={'tick':exc.tick,'pending_ticks':tuple(exc.pending_ticks),'reason':str(exc),
                  'decisions':tuple(exc.details.get('lastDecisions',()))}
            return dict(exc.partial_events),Envelope(data,torch.tensor([exc.tick],device=self.codec.device),'generation-interruption/1')

    def _score_ticks(self,ticks):
        from ..io.timing import ticks_to_seconds
        from ..io.audio import FRAME_SECONDS
        sec=ticks_to_seconds(np.asarray(ticks),self.context.bt,self.context.bv)
        mel=self.context.mel;frames=np.clip(np.rint(sec/FRAME_SECONDS).astype(int),1,len(mel)-1)
        onset=np.maximum(mel[frames]-mel[frames-1],0).mean(1)
        logits=np.asarray([self.anchor_scores[int(t)//384,int(t)%384] if self.anchor_scores is not None else 0. for t in ticks])
        if self.intent_scores is None:
            from .planning import star_scores
            self.intent_scores=star_scores(self.context)
        joint=np.asarray([self.intent_scores[int(t)//384,int(t)%384] for t in ticks])
        return onset+logits*.05+joint*.5

    def propose(self,request,feedback,budget):
        import time
        started=time.perf_counter();data=feedback.constraints.data;phase=data['phase'];ctx=self.context
        if ctx.progress:
            names={'initial':'主生成','stars':'补全星星配额','reduce_stars':'调整星星配额','repair':'按 Harness 约束重生成','resume':'恢复受阻的生成'}
            ctx.progress(f"难度 {ctx.slot}: {names[phase]}（轮次 {data['variant']+1}）")
        provider=self.harness.sampling_provider(ctx);interruption=None
        if phase=='initial':
            events,interruption=self._invoke(ctx,provider)
            self.primary_events=dict(events)
        else:
            events=dict(data['base_events']);targets=list(data.get('targets',()))
            intents=None;duration_masks={};route_masks={};start_masks={}
            if phase=='stars':
                eligible=[]
                for tick,text in events.items():
                    notes=self.codec.parse_event(text)
                    if len(notes)==1 and notes[0]['family']=='tap' and tick not in self.attempted_stars:
                        eligible.append(tick)
                # A model-driven proposal ordering; legality comes only from
                # Harness. A failed optional Star never lowers the user floor.
                scores=self._score_ticks(eligible) if eligible else []
                count=min(len(eligible),max(4,int(data['star_deficit'])*2),int(data['star_room']))
                targets=[eligible[i] for i in np.argsort(scores)[::-1][:count]]
                targets.sort();self.attempted_stars.update(targets)
                if not targets:return ()
                from .sampling import empty_representation
                intents={}
                for tick in targets:
                    rep=empty_representation(len(ctx.factor_session[1]['touchPositions']));rep['button_arity']=1;rep['button_family'][0]=2;intents[tick]=rep
            elif phase=='reduce_stars':
                from .sampling import empty_representation
                candidates=[t for t,s in events.items() if len(self.codec.parse_event(s))==1 and self.codec.parse_event(s)[0]['family']=='slide']
                targets=sorted(candidates[-int(data['star_excess']):]);intents={}
                for tick in targets:
                    rep=empty_representation(len(ctx.factor_session[1]['touchPositions']));rep['button_arity']=1;intents[tick]=rep
            elif phase=='repair':
                # Re-render the complete dependency window jointly, preserving
                # note families as proposals but allowing Harness rejection to
                # cause a different native intent where the model owns it.
                from .sampling import text_representation
                intents={tick:text_representation(events[tick],ctx.factor_session[1]) for tick in targets if tick in events}
            elif phase=='resume':
                if data.get('fresh_intent'):
                    intents=None
                else:
                    intents={tick:rep for tick,rep in (self.pending_intents or {}).items() if tick in targets}
                    if not intents:intents=None
            if not targets:return ()
            local_context=replace(ctx,progress=None)
            references=dict(events)
            for tick in ctx.ticks:references.setdefault(int(tick),'')
            generated,interruption=self._invoke(local_context,provider,targets=targets,references=references,intent_plan=intents,
                             seed_offset=int(data['variant'])*100003+(int(data.get('escape_level',0))*7919 if data.get('fresh_intent') else 0),allow_intent_revision=True,
                             duration_masks=duration_masks or None,route_masks=route_masks or None,start_masks=start_masks or None)
            for tick,text in generated.items():
                if text:events[tick]=text
                else:events.pop(tick,None)
        payload=self.codec.encode(events,ctx.bt,ctx.bv)
        chart=Chart(ChartRef(request.request_id,str(uuid.uuid4()),payload.digest),request.definition,payload)
        self.timings.append({'phase':phase,'seconds':time.perf_counter()-started,'events':len(events),'harnessSampling':dict(provider.timings),
                             'reusedPrefixFrames':ctx.cache.get('last_reused_prefix',0),'forwardFrames':ctx.cache.get('last_forward_frames',0)})
        if ctx.progress:ctx.progress(f'难度 {ctx.slot}: 本轮模型已返回，Harness 正在检查完整草稿')
        return (Proposal(chart,feedback.base,feedback.feedback_id,complete=interruption is None,interruption=interruption),)
