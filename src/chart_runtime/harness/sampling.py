"""Renderer sampling adapter to the same Kernel used by whole-chart review."""
from __future__ import annotations
import numpy as np
import torch
import time
import json

from ..io.timing import ticks_to_seconds
from ..io.durations import hold_seconds, slide_seconds
from .durations import typed_duration_values
from .fused import TIME_EPSILON, INPUT_RELEASE_SECONDS
from .snapshot_fast import SnapshotTracker
from .source_head import TRACK_LIFECYCLE


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
        self.device=self.codec.device;self._payload=None;self._durations={};self._duration_arrays_cpu={}
        self._reference_payload=None
        self._fast_snapshot=SnapshotTracker(vocab,self.tables,self.bt,self.bv)
        self._fast_snapshot_enabled=True
        pref_path=self.codec.root/'models/experimental/one_hand_motion_preference.json'
        if pref_path.is_file():
            document=json.loads(pref_path.read_text(encoding='utf8'))
            self.one_hand_preference=(document.get('slots') or {}).get(str(self.slot))
            self.one_hand_preference_quantile=document.get('preferenceQuantile')
        else:
            self.one_hand_preference=None;self.one_hand_preference_quantile=None
        slide_pref_path=self.codec.root/'models/experimental/slide_entry_motion_preference.json'
        if slide_pref_path.is_file():
            slide_document=json.loads(slide_pref_path.read_text(encoding='utf8'))
            self.slide_entry_preference=(slide_document.get('slots') or {}).get(str(self.slot))
            self.slide_entry_preference_quantile=slide_document.get('preferenceQuantile')
        else:
            self.slide_entry_preference=None;self.slide_entry_preference_quantile=None
        self._one_hand_state=None;self._tap_run_state=None
        self.timings={'snapshotSeconds':0.,'packSeconds':0.,'kernelSeconds':0.,'decisionSeconds':0.,'candidateBatches':0,'candidates':0}
        names=list(vocab['touchPositions']);adj=self.tables['touchAdjacency']
        self.adjacency=torch.tensor([[a==b or b in adj.get(a,()) for b in names] for a in names],device=self.device,dtype=torch.bool)
        self.touch_pad_masks=torch.tensor([int(self.tables['simplePadMasks'][name]) for name in names],device=self.device,dtype=torch.int64)

    def bind_references(self,reference_events):
        self.references={int(t):str(x.get('text','')) if isinstance(x,dict) else str(x) for t,x in (reference_events or {}).items()}
        self._reference_payload=self.codec.encode(self.references,self.bt,self.bv)
        c=self._reference_payload.columns
        if len(c['track_event']):
            self._source_head_reference_rows=np.column_stack(tuple(c[name].detach().cpu().numpy() for name in ('track_shoot','track_end','track_head'))+
                (c['event_tick'][c['track_event']].detach().cpu().numpy(),))
        else:self._source_head_reference_rows=np.empty((0,4),np.float64)
        if len(c['track_event']):
            self._track_tail_reference_rows=np.column_stack((c['track_shoot'].detach().cpu().numpy(),c['track_end'].detach().cpu().numpy(),c['track_tail'].detach().cpu().numpy(),c['event_tick'][c['track_event']].detach().cpu().numpy()))
        else:self._track_tail_reference_rows=np.empty((0,4),np.float64)
        if self.references:
            ticks=np.asarray(sorted(self.references),np.int64);times=ticks_to_seconds(ticks,self.bt,self.bv);tap_rows=[]
            for tick,at in zip(ticks,times):
                for note in self.codec.parse_event(self.references[int(tick)]):
                    if note['family']=='tap':tap_rows.append((float(at),int(note['start'])-1,int(tick)))
            self._fixed_tap_rows=np.asarray(tap_rows,np.float64).reshape(-1,3)
        else:self._fixed_tap_rows=np.empty((0,3),np.float64)
        if self.references:self._fast_snapshot_enabled=False

    def seed_history(self,events):
        self.history=dict(events);self._payload=None
        if self.history:self._fast_snapshot_enabled=False

    def _context(self,moment,horizon):
        reference={}
        if self._reference_payload is not None:
            c=self._reference_payload.columns;times=c['event_time'];selected=(times>=moment-2)&(times<=horizon+2)
            live=(c['input_end']+.2>=moment)&(c['input_start']<=horizon+2)
            selected.scatter_(0,c['input_event'][live],True)
            tracks=(c['track_end']+TRACK_LIFECYCLE.retention_seconds>=moment)&(c['track_start']<=horizon+2)
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
        if self._fast_snapshot_enabled:
            if bpm not in self._durations:self._durations[bpm]=typed_duration_values(tuple(self.vocab['durations']),bpm)
            if bpm not in self._duration_arrays_cpu:self._duration_arrays_cpu[bpm]=tuple(x.detach().cpu().numpy() for x in self._durations[bpm])
            result=self._fast_snapshot.snapshot(tick,bpm,self._duration_arrays_cpu[bpm],self.end_seconds)
            keep=self._fast_snapshot.prune(at);self.history={t:self.history[t] for t in keep if t in self.history}
            lifecycle=TRACK_LIFECYCLE.snapshot_masks(self,at)
            prior=result.get('allowedOuterStartMask');lifecycle['allowedOuterStartMask']=lifecycle['allowedOuterStartMask'] if prior is None else np.asarray(prior,bool)&lifecycle['allowedOuterStartMask']
            lifecycle['allowedTapStartMask']&=lifecycle['allowedOuterStartMask'];result.update(lifecycle)
            result['allowedSlideHeadDurationMask']=TRACK_LIFECYCLE.slide_mask(self,at,*self._duration_arrays_cpu[bpm][1:])
            self._one_hand_state=None
            if result.get('oneHandHoldConstraint') and 'freeHandLastLane' in result:
                self._one_hand_state={'last_lane':result['freeHandLastLane'],'last_time':result['freeHandLastTime'],
                    'previous_delta':result.get('freeHandPreviousDelta'),'constraint_start':result.get('oneHandConstraintStart')}
            self._tap_run_state=None
            if int(result.get('motionRunLength',0))>=3:
                self._tap_run_state={'last_lane':result['lastSingleTapLane'],'last_time':result['tapRunLastTime'],
                    'direction':result['motionDirection'],'run_length':result['motionRunLength']}
            self.timings['fastSnapshotCount']=int(self.timings.get('fastSnapshotCount',0))+1
            self.timings['snapshotSeconds']+=time.perf_counter()-started
            return result
        if self._payload is None:self._payload=self.codec.encode(self.history,self.bt,self.bv)
        c=self._payload.columns;ie=c['input_event'];held=c['input_hold']&(c['input_start']<=at)&(c['input_end']>at+1e-7)
        outerheld=held&c['input_outer'];touchheld=held&~c['input_outer']
        lanes=c['input_sensor'][outerheld];touch=c['input_sensor'][touchheld]-8
        move=(c['track_shoot']<=at+TIME_EPSILON)&(c['track_end']>=at-TIME_EPSILON)
        active=(c['track_start']<=at+TIME_EPSILON)&(c['track_end']+.2>at-TIME_EPSILON)
        source_lanes=set((lanes+1).detach().cpu().tolist());touch_names=[self.codec.sensors[i] for i in touch.detach().cpu().tolist()]
        nh=int(outerheld.sum().item());nt=int(touchheld.sum().item());ns=int(move.sum().item())
        held_release=c['input_hold']&(c['input_start']<=at+1e-7)&(c['input_end']+1/180>at+1e-7)
        outer_hold_hands=int((held_release&c['input_outer']).sum().item())
        touch_hold_inputs=held_release&~c['input_outer']
        if bool(touch_hold_inputs.any()):
            touch_hold_mask=torch.zeros(len(self.vocab['touchPositions']),device=self.device,dtype=torch.bool)
            touch_hold_mask[c['input_sensor'][touch_hold_inputs]-8]=True
            touch_hold_hands=self.touch_group_count(touch_hold_mask)
        else:touch_hold_hands=0
        self._one_hand_state=None
        one_hand_constraint=outer_hold_hands+touch_hold_hands==1
        if one_hand_constraint:
            owners=torch.unique(c['input_event'][held_release])
            start=float(c['input_start'][held_release].min().item())
            event_ids=torch.arange(len(c['event_tick']),device=self.device)
            free=(c['event_time']>=start-TIME_EPSILON)&(c['event_time']<at-TIME_EPSILON)&(c['event_lane_count']==1)&~torch.isin(event_ids,owners)
            recent=free.nonzero(as_tuple=True)[0][-2:]
            if recent.numel():
                last=int(recent[-1].item());last_lane=int(c['event_lanes'][last,0].item());last_time=float(c['event_time'][last].item())
                previous_delta=None
                if recent.numel()>=2:
                    previous=int(recent[-2].item());previous_lane=int(c['event_lanes'][previous,0].item())
                    previous_delta=(last_lane-previous_lane+4)%8-4
                self._one_hand_state={'last_lane':last_lane,'last_time':last_time,'previous_delta':previous_delta,'constraint_start':start,
                                      'outer_hold_hands':outer_hold_hands,'touch_hold_hands':touch_hold_hands}
        self._tap_run_state=None
        n_events=len(c['event_tick'])
        if n_events>=3:
            tap_event=torch.zeros(n_events,device=self.device,dtype=torch.bool)
            tap_notes=c['note_kind']==0
            if bool(tap_notes.any()):tap_event[c['note_event'][tap_notes]]=True
            single_tap=(c['event_notes']==1)&(c['event_lane_count']==1)&tap_event&(c['event_time']<at-TIME_EPSILON)
            past=(c['event_time']<at-TIME_EPSILON).nonzero(as_tuple=True)[0]
            recent3=past[-3:]
            if recent3.numel()==3 and bool(single_tap[recent3].all()):
                run_lanes=c['event_lanes'][recent3,0].to(torch.int64)
                d1=int(((run_lanes[1]-run_lanes[0]+4).remainder(8)-4).item());d2=int(((run_lanes[2]-run_lanes[1]+4).remainder(8)-4).item())
                if d1==d2 and abs(d2)==1:
                    self._tap_run_state={'last_lane':int(run_lanes[2].item()),'last_time':float(c['event_time'][recent3[2]].item()),'direction':d2,'run_length':3}
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
            needed.scatter_(0,c['track_event'][c['track_end']+TRACK_LIFECYCLE.retention_seconds>=at],True)
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
        if self._tap_run_state is not None:
            result.update(lastSingleTapLane=self._tap_run_state['last_lane'],motionDirection=self._tap_run_state['direction'],motionRunLength=self._tap_run_state['run_length'],tapRunLastTime=self._tap_run_state['last_time'])
        result['oneHandHoldConstraint']=bool(one_hand_constraint)
        if self._one_hand_state is not None:
            result.update(freeHandLastLane=self._one_hand_state['last_lane'],freeHandLastTime=self._one_hand_state['last_time'],
                          freeHandPreviousDelta=self._one_hand_state['previous_delta'],oneHandConstraintStart=self._one_hand_state['constraint_start'])
        self.timings['snapshotSeconds']+=time.perf_counter()-started
        lifecycle=TRACK_LIFECYCLE.snapshot_masks(self,at)
        prior=result.get('allowedOuterStartMask');lifecycle['allowedOuterStartMask']=lifecycle['allowedOuterStartMask'] if prior is None else np.asarray(prior,bool)&lifecycle['allowedOuterStartMask']
        lifecycle['allowedTapStartMask']&=lifecycle['allowedOuterStartMask'];result.update(lifecycle)
        if bpm not in self._duration_arrays_cpu:
            self._duration_arrays_cpu[bpm]=tuple(x.detach().cpu().numpy() for x in self._durations[bpm])
        result['allowedSlideHeadDurationMask']=TRACK_LIFECYCLE.slide_mask(self,at,*self._duration_arrays_cpu[bpm][1:])
        causal_budget=getattr(self,'causal_budget',None)
        if causal_budget is not None:result.update(causal_budget.limits(int(tick)))
        return result

    def update(self,tick,representation,bpm):
        text=representation_text(representation,self.vocab)
        if self._fast_snapshot_enabled:self._fast_snapshot.append(int(tick),representation,float(bpm))
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
        causal_budget=getattr(self,'causal_budget',None)
        if causal_budget is not None:causal_budget.reserve(int(self.tick),len(reps))
        started=time.perf_counter();self.timings['candidateBatches']+=1;self.timings['candidates']+=len(reps)
        at_bpm=float(bpm)
        if at_bpm not in self._durations:self._durations[at_bpm]=typed_duration_values(tuple(self.vocab['durations']),at_bpm)
        hd,wait,move=self._durations[at_bpm]
        if at_bpm not in self._duration_arrays_cpu:
            self._duration_arrays_cpu[at_bpm]=tuple(x.detach().cpu().numpy() for x in (hd,wait,move))
        h_cpu,w_cpu,m_cpu=self._duration_arrays_cpu[at_bpm]
        ids=[int(rep['button_duration'][j]) for rep in reps for j in range(int(rep['button_arity']))]
        extent=float(np.nan_to_num(np.maximum(h_cpu[ids],w_cpu[ids]+m_cpu[ids])).max()) if ids else 0.
        # Long Touch Holds also extend the future dependency window.
        touch_ids=[int(rep['touch_duration'][j]) for rep in reps for j in np.flatnonzero(rep['touch_presence']) if int(rep['touch_duration'][j])]
        if touch_ids:extent=max(extent,float(np.nan_to_num(h_cpu[touch_ids]).max()))
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
        passed=(hc.sum(1)==0);soft=torch.stack(list(result.soft.values()),1)
        contact_extra=(soft[1:,:min(3,soft.shape[1])]-soft[:1,:min(3,soft.shape[1])]).clamp_min(0).sum(1)
        slide_hand_extra=((soft[1:,3]-soft[:1,3]).clamp_min(0) if getattr(self,'slide_hand_order_enabled',False) and soft.shape[1]>3
                          else torch.zeros(len(reps),device=self.device,dtype=torch.int64))
        extra=contact_extra+slide_hand_extra
        # A Slide entry preference is deliberately narrow: it activates only
        # after three immediately preceding pure single Taps form an adjacent
        # directional run.  This catches abrupt Star pickup without flattening
        # unrelated high-motion Slide patterns elsewhere in a high-DS chart.
        slide_entry_preference=torch.zeros(len(reps),device=self.device,dtype=torch.float64)
        tap_state=self._tap_run_state;slide_pref=self.slide_entry_preference
        if tap_state is not None and slide_pref is not None:
            arity=torch.as_tensor([int(rep['button_arity']) for rep in reps],device=self.device,dtype=torch.int64)
            family=torch.as_tensor([int(rep['button_family'][0]) if int(rep['button_arity']) else -1 for rep in reps],device=self.device,dtype=torch.int64)
            starts=torch.as_tensor([int(rep['button_start'][0]) if int(rep['button_arity']) else 0 for rep in reps],device=self.device,dtype=torch.int64)
            delta=(starts-int(tap_state['last_lane'])+4).remainder(8)-4;gap=max(1e-6,float(moment)-float(tap_state['last_time']))
            step=delta.abs().to(torch.float64);jerk=(delta-int(tap_state['direction'])).abs().to(torch.float64)/gap
            step_limit=max(1.,float(slide_pref['step80']));jerk_limit=max(1e-6,float(slide_pref['jerk80']))
            slide_entry_preference=(step/step_limit-1.).clamp_min(0.)+(jerk/jerk_limit-1.).clamp_min(0.)
            slide_entry_preference=torch.where((arity==1)&(family==2),slide_entry_preference,torch.zeros_like(slide_entry_preference))
        slide_entry_milli=(slide_entry_preference*1000.).round().to(torch.int64)
        one_hand_preference=torch.zeros(len(reps),device=self.device,dtype=torch.float64)
        state=self._one_hand_state
        pref1=self.one_hand_preference
        if state is not None and pref1 is not None:
            arity=torch.as_tensor([int(rep['button_arity']) for rep in reps],device=self.device,dtype=torch.int64)
            starts=torch.as_tensor([int(rep['button_start'][0]) if int(rep['button_arity']) else 0 for rep in reps],device=self.device,dtype=torch.int64)
            delta=(starts-int(state['last_lane'])+4).remainder(8)-4
            step=delta.abs().to(torch.float64);gap=max(1e-6,float(moment)-float(state['last_time']))
            speed=step/gap
            previous_delta=state.get('previous_delta')
            if previous_delta is None:
                jerk=torch.zeros_like(speed);smooth=step<=1
            else:
                previous_delta=int(previous_delta);jerk=(delta-previous_delta).abs().to(torch.float64)/gap
                smooth=(step<=1)&((delta==0)|(previous_delta==0)|(delta==previous_delta))
            speed_limit=max(1e-6,float(pref1['speed90']));jerk_limit=max(1e-6,float(pref1['jerk90']));step_limit=max(1.,float(pref1['step90']))
            one_hand_preference=(speed/speed_limit-1.).clamp_min(0.)+(jerk/jerk_limit-1.).clamp_min(0.)+(step/step_limit-1.).clamp_min(0.)
            one_hand_preference=torch.where((arity==1)&~smooth,one_hand_preference,torch.zeros_like(one_hand_preference))
        one_hand_milli=(one_hand_preference*1000.).round().to(torch.int64)
        packed=torch.cat((hc,passed[:,None].to(torch.int64),extra[:,None],slide_hand_extra[:,None],slide_entry_milli[:,None],one_hand_milli[:,None]),dim=1).detach().cpu().tolist()
        records=[row[:-5] for row in packed];passes=[bool(row[-5]) for row in packed]
        softs=[row[-4] for row in packed];slide_hand_costs=[row[-3] for row in packed];slide_costs=[row[-2] for row in packed];one_hand_costs=[row[-1] for row in packed]
        for row,ok,soft_count,slide_hand_cost,slide_cost,one_hand_cost in zip(records,passes,softs,slide_hand_costs,slide_costs,one_hand_costs):
            hard_reason='+'.join(name for name,n in zip(reasons,row) if n)
            advisory=[]
            if soft_count-slide_hand_cost:advisory.append('ContactAdvisory')
            if slide_hand_cost:advisory.append('SlideHandOrder')
            if one_hand_cost:advisory.append('OneHandMotionPreference')
            if slide_cost:advisory.append('SlideEntryMotionPreference')
            decision={'severity':'HARD' if not ok else 'SOFT' if advisory else 'CLEAN','evidence_status':'shared_cuda_kernel','replace_if_alternative':bool(advisory),'reason':hard_reason or '+'.join(advisory) or 'Clear','scope':'candidate_delta','soft_cost':int(soft_count)*1000000+int(one_hand_cost)*100+int(slide_cost),'contact_soft_cost':int(soft_count-slide_hand_cost),'slide_hand_order_cost':int(slide_hand_cost),'slide_hand_order_clearance_seconds':float(self.kernel.fused.slide_hand_order_seconds),'slide_hand_order_max_attempts':int(self.kernel.fused.slide_hand_order_max_attempts),'slide_hand_order_try_next_what':bool(self.kernel.fused.slide_hand_order_try_next_what),'slide_hand_order_max_what_attempts':int(self.kernel.fused.slide_hand_order_max_what_attempts),'one_hand_motion_cost':int(one_hand_cost),'one_hand_preference_quantile':self.one_hand_preference_quantile,'slide_entry_motion_cost':int(slide_cost),'slide_entry_preference_quantile':self.slide_entry_preference_quantile}
            self.decisions.append(decision);output.append((ok,decision['reason'],'same full-chart Kernel; bounded context'))
        self.timings['decisionSeconds']+=time.perf_counter()-started
        return output
