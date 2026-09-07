"""Whole-chart authority and feedback compiler for the production session."""
from __future__ import annotations
from hashlib import sha256
import math
import uuid
import numpy as np
import torch

from ..domain import Evaluation,Feedback,FeedbackStop,PublishPermit,Scope,Verdict
from ..runtime.payloads import Envelope
from .kernel import Kernel,QUALITY_NAMES,RULES_ID
from .sampling import SamplingProvider


def interruption_attempt(history,tick):
    return sum(1 for observation in history
               if not observation.proposal.complete
               and observation.proposal.interruption is not None
               and int(observation.proposal.interruption.data.get('tick',-1))==int(tick))


class HarnessBackend:
    def __init__(self,codec,calibration,conditions):
        self.codec=codec;self.kernel=Kernel(codec);self.calibration=calibration;self.conditions=conditions
        self.minimum_stars=None;self.base_stars=None;self.known={};self.results={};self.observed=[]

    def sampling_provider(self,context):
        return SamplingProvider(self.kernel,context.factor_session[1],context.bt,context.bv,context.version,context.slot,self.conditions['end_seconds'],self.calibration)

    def feedback(self,request,history,budget):
        evidence=torch.empty((0,),device=self.codec.device)
        if not history:
            plan={'phase':'initial','variant':0,'base_events':(),'issues':(), 'minimum_stars':None}
            base=None;scope=()
        else:
            # Keep the most recent working draft and all prior observations.
            # Equal error counts do not cause stale feedback to be replayed.
            last=history[-1];payload=last.proposal.chart.payload;meta=last.evaluation.witnesses.data
            base=last.proposal.chart.ref;events=payload.events;issues=meta['issues'];ticks=np.asarray(sorted(events))
            phase='stars' if not issues and meta['stars']<(self.minimum_stars or 0) else 'repair'
            upper_stars=(int(self.conditions['star_target_stars'])
                         if self.conditions['star_control']
                         else max(self.base_stars or 0,int(self.conditions['official_stars'])))
            if not issues and self.conditions['star_control'] and meta['stars']>upper_stars:phase='reduce_stars'
            target=set()
            if not last.proposal.complete:
                phase='resume';interruption=last.proposal.interruption.data
                target.update(interruption['pending_ticks'])
                tick=interruption['tick'];columns=payload.columns
                escape=interruption_attempt(history,tick)
                from ..io.timing import ticks_to_seconds
                at=float(ticks_to_seconds(np.array([tick]),self.conditions['bt'],self.conditions['bv'])[0])
                # Causal owners become editable along with pending output;
                # renderer search failure is returned here, never promoted to
                # a claim that the requested chart is impossible.
                live_inputs=(columns['input_start']<=at+.2)&(columns['input_end']+.2>=at)
                live_tracks=(columns['track_start']<=at+.2)&(columns['track_end']+.2>=at)
                owners=torch.cat((columns['input_event'][live_inputs],columns['track_event'][live_tracks]))
                target.update(columns['event_tick'][owners].detach().cpu().tolist());target.add(tick)
                # A repeated stop at the same point means the narrow intent is
                # not making progress. Expand event context before asking the
                # renderer again; the prefix before this region remains cached.
                if escape>=2 and len(ticks):
                    index=int(np.searchsorted(ticks,tick));width=6 if escape==2 else 12
                    target.update(map(int,ticks[max(0,index-width):index+width+1]))
            for item in issues:
                tick=item['tick'];index=int(np.searchsorted(ticks,tick))
                # Features depend on predecessors and immediate successors.
                # HARD interactions can cross longer durations; include owner
                # witnesses and an expanded neighboring dependency window.
                width=2+min(4,len(history)//3)
                target.update(map(int,ticks[max(0,index-width):index+width+2]))
            scope=tuple(sorted(target))
            seen=tuple(x.proposal.chart.ref.content_digest for x in history)
            plan={'phase':phase,'variant':len(history),'base_events':tuple(events.items()),'issues':tuple(issues),
                  'targets':scope,'minimum_stars':self.minimum_stars,'seen_digests':seen,
                  'star_deficit':max(0,(self.minimum_stars or 0)-meta['stars']),
                  'star_excess':max(0,meta['stars']-upper_stars),'star_room':max(0,upper_stars-meta['stars']),
                  'quality_limits':None if self.calibration is None else tuple(self.calibration['thresholds'])}
            if phase=='resume':
                plan['escape_level']=escape
                plan['fresh_intent']=escape>=2
            evidence=last.evaluation.witnesses.evidence
        receipts=tuple(x.evaluation.receipt_id for x in history)
        return Feedback(str(uuid.uuid4()),request.request_id,request.definition,base,
                        Envelope(plan,evidence,'harness-feedback/1'),
                        Envelope({'ticks':scope},torch.tensor(scope,device=self.codec.device,dtype=torch.int64),'edit-scope/1'),receipts)

    def evaluate_batch(self,request,proposals,budget):
        B=len(proposals);c=self.conditions
        payloads=[p.chart.payload for p in proposals]
        result=self.kernel.evaluate(payloads,versions=[request.version_id]*B,end_seconds=[c['end_seconds']]*B,bpms=[float(c['bpm'])]*B,
                                    thresholds=[self.calibration['thresholds'] if self.calibration else None]*B,features=True)
        stars=result.star_counts.detach().cpu().tolist()
        if self.minimum_stars is None and proposals[0].complete:
            self.base_stars=stars[0]
            # The user parameter is now a direct target ratio against the
            # calibrated official reference.  Keep exact lower/upper bounds
            # for controlled difficulties so the slider has a deterministic
            # meaning independent of the model's first draft.
            self.minimum_stars=int(c['star_target_stars']) if c['star_control'] else 0
        verdicts=[]
        hc,qc,sc=result.summaries();hc=hc.detach().cpu().tolist();qc=qc.detach().cpu().tolist();sc=sc.detach().cpu().tolist()
        for b,proposal in enumerate(proposals):
            mask=result.event_batch==b;local_ticks=result.event_tick[mask];issues=[]
            for name,values in result.hard.items():
                for tick in result.event_tick[values&mask].detach().cpu().tolist():issues.append({'tick':tick,'reason':name,'severity':'HARD'})
            for j,name in enumerate(QUALITY_NAMES):
                for tick in result.event_tick[result.quality[:,j]&mask].detach().cpu().tolist():issues.append({'tick':tick,'reason':'quality:'+name,'severity':'QUALITY'})
            soft=dict(zip(result.soft,sc[b]));soft_budget=max(5,round(len(proposal.chart.payload.source)*.04))
            soft_ok=sum(soft[k] for k in ('TapOnSlide','SlideHeadTap','Overlap'))<=soft_budget
            if not soft_ok:
                # Full device masks are retained in this receipt; expand to
                # all Track owners only if a sound feel budget needs repair.
                columns=proposal.chart.payload.columns
                track_ticks=columns['event_tick'][columns['track_event']].unique().detach().cpu().tolist()
                issues.extend({'tick':tick,'reason':'quality:contact_budget','severity':'QUALITY'} for tick in track_ticks)
            upper_stars=(int(c['star_target_stars'])
                         if c['star_control']
                         else max(self.base_stars or 0,int(c['official_stars'])))
            accepted=proposal.complete and not issues and stars[b]>=(self.minimum_stars or 0) and (not c['star_control'] or stars[b]<=upper_stars)
            evidence=torch.cat((torch.stack([m[mask] for m in result.hard.values()],1).to(torch.float64),result.quality[mask].to(torch.float64),result.features[mask]),1)
            meta={'issues':issues,'hard_counts':dict(zip(result.hard,hc[b])),'quality_counts':dict(zip(QUALITY_NAMES,qc[b])),
                  'soft':soft,'soft_budget':soft_budget,'stars':stars[b],
                  'minimum_stars':self.minimum_stars,
                  'maximum_stars':upper_stars if c['star_control'] else None,
                  'target_stars':int(c['star_target_stars']) if c['star_control'] else None,
                  'target_ratio':float(c['star_target_ratio']) if c['star_control'] else None,
                  'official_stars':int(c['official_stars']),
                  'base_stars':self.base_stars,
                  'quality_calibration':'available' if self.calibration else 'not_available_for_this_slot',
                  'rules_id':RULES_ID,'coverage':'registered mechanisms and supported complete path tables',
                  'cpu_per_object_checks':False}
            if not proposal.complete:
                meta['interruption']=dict(proposal.interruption.data)
            receipt=Evaluation(proposal.chart.ref,request.definition,Scope.FULL_CHART if proposal.complete else Scope.CANDIDATE,Verdict.ACCEPT if accepted else Verdict.REVISE,proposal.complete,
                               Envelope(meta,evidence,'witnesses/1'),str(uuid.uuid4()))
            self.known[receipt.receipt_id]=receipt;self.results[proposal.chart.ref]=meta;verdicts.append(receipt)
        return tuple(verdicts)

    def permit(self,evaluation):
        if self.known.get(evaluation.receipt_id) is not evaluation or evaluation.verdict is not Verdict.ACCEPT or evaluation.scope is not Scope.FULL_CHART or not evaluation.coverage_complete:
            raise RuntimeError('Only this Harness full-chart acceptance can authorize publication')
        return PublishPermit(evaluation.chart,evaluation.definition,evaluation.receipt_id)
