"""Renderer sampling adapter to the same Kernel used by whole-chart review."""
from __future__ import annotations
import numpy as np
import torch
import time

from ..io.timing import ticks_to_seconds
from ..io.durations import hold_seconds, slide_seconds
from .durations import typed_duration_values
from .fused import TIME_EPSILON, INPUT_RELEASE_SECONDS


def representation_text(rep,vocab):
    notes=[]
    for j in range(int(rep['button_arity'])):
        family=int(rep['button_family'][j]);lane=int(rep['button_start'][j])+1;mods=int(rep['button_modifiers'][j])
        suffix=('b' if mods&1 else '')+('x' if mods&2 else '')+('f' if mods&8 else '')+('?' if mods&16 else '')
        duration=vocab['durations'][int(rep['button_duration'][j])]
        if family==0:notes.append(str(lane)+suffix)
        elif family==1:notes.append(f'{lane}h{suffix}[{duration}]')
        elif family==2:notes.append(f"{lane}{suffix}{vocab['routes'][int(rep['button_route'][j])]}[{duration}]")
    for j in np.flatnonzero(rep['touch_presence']):
        pos=vocab['touchPositions'][j];duration=int(rep['touch_duration'][j]);mods=int(rep['touch_modifiers'][j]);suffix=('x' if mods&2 else '')+('f' if mods&8 else '')
        notes.append(f"{pos}h{suffix}[{vocab['durations'][duration]}]" if duration else pos+suffix)
    return '/'.join(notes)


class SamplingProvider:
    def __init__(self,kernel,vocab,bt,bv,version,slot,end_seconds,calibration=None):
        self.kernel=kernel;self.codec=kernel.codec;self.tables=self.codec.tables;self.vocab=vocab
        self.bt=np.asarray(bt);self.bv=np.asarray(bv);self.version=version;self.slot=slot;self.end_seconds=end_seconds;self.calibration=calibration
        self.history={};self.references={};self.decisions=[];self.tick=0;self.max_cuda_matrix_elements=0
        self.device=self.codec.device;self._payload=None;self._durations={}
        self._reference_payload=None
        self.timings={'snapshotSeconds':0.,'packSeconds':0.,'kernelSeconds':0.,'decisionSeconds':0.,'candidateBatches':0,'candidates':0}
        names=list(vocab['touchPositions']);adj=self.tables['touchAdjacency']
        self.adjacency=torch.tensor([[a==b or b in adj.get(a,()) for b in names] for a in names],device=self.device,dtype=torch.bool)
        self.touch_pad_masks=torch.tensor([int(self.tables['simplePadMasks'][name]) for name in names],device=self.device,dtype=torch.int64)

    def bind_references(self,reference_events):
        self.references={int(t):str(x.get('text','')) if isinstance(x,dict) else str(x) for t,x in (reference_events or {}).items()}
        self._reference_payload=self.codec.encode(self.references,self.bt,self.bv)

    def seed_history(self,events):
        self.history=dict(events);self._payload=None

    def _context(self,moment,horizon):
        reference={}
        if self._reference_payload is not None:
            c=self._reference_payload.columns;times=c['event_time'];selected=(times>=moment-2)&(times<=horizon+2)
            live=(c['input_end']+.2>=moment)&(c['input_start']<=horizon+2)
            selected.scatter_(0,c['input_event'][live],True)
            tracks=(c['track_end']+.2>=moment)&(c['track_start']<=horizon+2)
            selected.scatter_(0,c['track_event'][tracks],True)
            ids=selected.nonzero(as_tuple=True)[0]
            if ids.numel():
                lower=(ids.min()-8).clamp_min(0);upper=(ids.max()+8).clamp_max(len(times)-1)
                kept=c['event_tick'][(torch.arange(len(times),device=self.device)>=lower)&(torch.arange(len(times),device=self.device)<=upper)].detach().cpu().tolist()
                reference={tick:self.references[tick] for tick in kept}
        return {**reference,**self.history}

    def snapshot(self,tick,enforce_recent=True):
        started=time.perf_counter()
        self.tick=int(tick);at=float(ticks_to_seconds(np.asarray([tick]),self.bt,self.bv)[0]);bpm=float(self.bv[max(0,np.searchsorted(self.bt,tick,side='right')-1)])
        if self._payload is None:self._payload=self.codec.encode(self.history,self.bt,self.bv)
        c=self._payload.columns;ie=c['input_event'];held=c['input_hold']&(c['input_start']<=at)&(c['input_end']>at+1e-7)
        outerheld=held&c['input_outer'];touchheld=held&~c['input_outer']
        lanes=c['input_sensor'][outerheld];touch=c['input_sensor'][touchheld]-8
        move=(c['track_shoot']<=at+TIME_EPSILON)&(c['track_end']>=at-TIME_EPSILON)
        active=(c['track_start']<=at+TIME_EPSILON)&(c['track_end']+.2>at-TIME_EPSILON)
        source_lanes=set((lanes+1).detach().cpu().tolist());touch_names=[self.codec.sensors[i] for i in touch.detach().cpu().tolist()]
        nh=int(outerheld.sum().item());nt=int(touchheld.sum().item());ns=int(move.sum().item())
        held_release=c['input_hold']&(c['input_start']<=at+1e-7)&(c['input_end']+1/180>at+1e-7)
        active_inputs=(c['input_start']<=at+1e-7)&(c['input_end']+1/180>at+1e-7)&c['input_outer']
        transient_outer=active_inputs&~c['input_hold']
        transient_touch=(c['input_start']<=at+1e-7)&(c['input_end']+1/180>at+1e-7)&~c['input_outer']&~c['input_hold']
        transient_touch_groups=int(torch.unique(c['input_event'][transient_touch]).numel())
        slide_hands=int((move.to(torch.int64)*(1+c['track_wifi'].to(torch.int64))).sum().item())
        available=max(0,2-int(held_release.sum().item())-int(transient_outer.sum().item())-transient_touch_groups-slide_hands)
        active_actions=(c['action_track']>=0)&(c['action_start']<=at+TIME_EPSILON)&(c['action_end']>at-TIME_EPSILON)
        covered_touch=((c['action_mask'][active_actions,None]&self.touch_pad_masks[None,:])!=0).any(0) if bool(active_actions.any()) else torch.zeros(len(self.touch_pad_masks),device=self.device,dtype=torch.bool)
        frame_tracks=(c['track_shoot']<=at+INPUT_RELEASE_SECONDS+TIME_EPSILON)&(c['track_end']>=at-TIME_EPSILON)
        moving_capacity=torch.where(frame_tracks.any(),torch.tensor(1,device=self.device),torch.tensor(2,device=self.device))
        outer_capacity=int((moving_capacity-active_inputs.sum()).clamp(0,2).item())
        if bpm not in self._durations:self._durations[bpm]=typed_duration_values(tuple(self.vocab['durations']),bpm)
        hold,wait,movement=self._durations[bpm];remain=self.end_seconds-at
        hm=torch.isfinite(hold)&(hold>0)&(hold<=remain+1e-7);sm=torch.isfinite(wait)&torch.isfinite(movement)&(wait>=0)&(movement>0)&(wait+movement<=remain+1e-7)
        # Keep the entire dependency suffix from the earliest live source,
        # plus its preceding boundary events. Old accepted intervals no
        # longer participating in any rule are not copied into every batch.
        if len(c['event_tick']):
            needed=c['event_time']>=at-2
            needed.scatter_(0,c['input_event'][c['input_end']+.2>=at],True)
            needed.scatter_(0,c['track_event'][c['track_end']+.2>=at],True)
            needed[-8:]=True
            lower=(needed.nonzero(as_tuple=True)[0].min()-8).clamp_min(0)
            keys=c['event_tick'][torch.arange(len(c['event_tick']),device=self.device)>=lower].detach().cpu().tolist()
            self.history={tick:self.history[tick] for tick in keys}
        # Tensor summaries condition the generator; musical rejection remains
        # evaluate_batch. No global tail cooldown is invented by the adapter.
        result=dict(blockedLanes=source_lanes,riskLanes=set(),holdLanes=source_lanes,slideObligations=[],
            slideStartActionCount=0,slideTailCount=0,slideEndpointActionCount=0,slideTailLanes=set(),slideTailCooldownCount=0,slideTailCooldownEndTicks=[],
            muriStateFeatures=np.zeros(32,np.float32),muriOracleSourceTick=int(tick),activeTouchHoldSensors=touch_names,lastSingleTapLane=None,motionDirection=0,motionRunLength=0,
            activeHands=min(2,nh+nt),activeHoldHands=min(2,nh),activeSlideHands=min(2,ns),activeSlideCount=int(active.sum().item()),activeTouchHoldHands=min(2,nt),availableHands=available,holdAvailableHands=available,
            maxOuterArity=outer_capacity,
            allowedTouchPresenceMask=covered_touch.detach().cpu().numpy(),
            allowedHoldDurationMask=hm.detach().cpu().numpy(),allowedSlideDurationMask=sm.detach().cpu().numpy())
        self.timings['snapshotSeconds']+=time.perf_counter()-started
        return result

    def update(self,tick,representation,bpm):
        text=representation_text(representation,self.vocab)
        if text:self.history[int(tick)]=text
        self._payload=None

    def commit(self,rep,moment,bpm):
        # update is the single commit path; the model's legacy callback invokes
        # commit too, but it must not duplicate notes or update another state.
        return None

    def touch_group_count(self,active):
        mask=torch.as_tensor(active,device=self.device,dtype=torch.bool)
        reach=self.adjacency&mask[:,None]&mask[None,:]
        for _ in range(6):reach=reach|((reach.float()@reach.float())>0)
        ids=torch.arange(len(mask),device=self.device);label=torch.where(reach,ids[None,:],len(mask)).amin(1)
        return int(torch.unique(label[mask]).numel())

    def touch_hold_hand_count(self,sensors):
        active=torch.zeros(len(self.vocab['touchPositions']),device=self.device,dtype=torch.bool)
        if sensors:active[torch.tensor([self.vocab['touchPositions'].index(s) for s in sensors],device=self.device)]=True
        return self.touch_group_count(active)

    def check_batch(self,reps,moment,bpm):
        if not reps:self.decisions=[];return []
        started=time.perf_counter();self.timings['candidateBatches']+=1;self.timings['candidates']+=len(reps)
        at_bpm=float(bpm)
        if at_bpm not in self._durations:self._durations[at_bpm]=typed_duration_values(tuple(self.vocab['durations']),at_bpm)
        hd,wait,move=self._durations[at_bpm]
        durations=torch.as_tensor([int(rep['button_duration'][j]) for rep in reps for j in range(int(rep['button_arity']))],device=self.device,dtype=torch.int64)
        extent=float(torch.maximum(hd[durations],wait[durations]+move[durations]).nan_to_num().max().item()) if durations.numel() else 0.
        context=self._context(moment,moment+extent);context.pop(self.tick,None)
        baseline=self.codec.encode(context,self.bt,self.bv)
        drafts=[baseline]+[self.codec.encode({**context,**({self.tick:representation_text(rep,self.vocab)} if representation_text(rep,self.vocab) else {})},self.bt,self.bv) for rep in reps]
        self.timings['packSeconds']+=time.perf_counter()-started
        count=len(drafts)
        thresholds=[self.calibration['thresholds']]*count if self.calibration else None
        started=time.perf_counter()
        result=self.kernel.evaluate(drafts,versions=[self.version]*count,end_seconds=[self.end_seconds]*count,bpms=[bpm]*count,thresholds=thresholds,features=bool(self.calibration))
        self.timings['kernelSeconds']+=time.perf_counter()-started;started=time.perf_counter()
        keys=result.event_tick[None,:]*len(result.hard)+torch.arange(len(result.hard),device=self.device)[:,None]
        flags=torch.stack(tuple(result.hard.values()));baseflags=flags&(result.event_batch[None]==0)
        existing=keys[baseflags]
        # Compare complete witness keys on device. Existing fixed-future
        # obligations remain a full-chart issue, but cannot veto an unrelated
        # new candidate whose addition does not change that obligation.
        new=flags&~torch.isin(keys,existing)
        reasons=list(result.hard);self.decisions=[];output=[]
        hc=torch.zeros((count,len(result.hard)),device=self.device,dtype=torch.int64)
        hc.index_add_(0,result.event_batch,new.T.to(torch.int64));hc=hc[1:]
        quality_keys=result.event_tick[None]*4+torch.arange(4,device=self.device)[:,None]
        existing_quality=quality_keys[result.quality.T&(result.event_batch[None]==0)]
        new_quality=result.quality.T&~torch.isin(quality_keys,existing_quality)
        qc=torch.zeros((count,4),device=self.device,dtype=torch.int64)
        qc.index_add_(0,result.event_batch,new_quality.T.to(torch.int64));qc=qc[1:]
        hc=torch.cat((hc,qc),1);reasons+=['quality:'+name for name in ('burst','motion_speed','track_speed','motion_change')]
        passed=(hc.sum(1)==0);soft=torch.stack(list(result.soft.values()),1);extra=(soft[1:,:3]-soft[:1,:3]).clamp_min(0).sum(1)
        records=hc.detach().cpu().tolist();passes=passed.detach().cpu().tolist();softs=extra.detach().cpu().tolist()
        for row,ok,soft_count in zip(records,passes,softs):
            reason='+'.join(name for name,n in zip(reasons,row) if n)
            decision={'severity':'HARD' if not ok else 'SOFT' if soft_count else 'CLEAN','evidence_status':'shared_cuda_kernel','replace_if_alternative':bool(soft_count),'reason':reason or ('ContactAdvisory' if soft_count else 'Clear'),'scope':'candidate_delta','soft_cost':int(soft_count)}
            self.decisions.append(decision);output.append((ok,decision['reason'],'same full-chart Kernel; bounded context'))
        self.timings['decisionSeconds']+=time.perf_counter()-started
        return output
