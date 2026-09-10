"""Incremental mirror of SamplingProvider.snapshot state; never a legality judge."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from ..io.timing import ticks_to_seconds
from ..io.durations import hold_seconds,slide_seconds
from .fused import TIME_EPSILON,INPUT_RELEASE_SECONDS,INPUT_OVERLAY_SECONDS
from .source_head import TRACK_LIFECYCLE

@dataclass
class InputState:
    event:int; outer:bool; sensor:int; hold:bool; start:float; end:float; pad:int; overlay:bool
@dataclass
class TrackState:
    event:int; start:float; shoot:float; end:float; wifi:bool; actions:list=field(default_factory=list)
@dataclass
class EventState:
    tick:int; time:float; notes:int; lanes:tuple; pure_single_tap:bool; inputs:list=field(default_factory=list); tracks:list=field(default_factory=list)

class SnapshotTracker:
    def __init__(self,vocab,tables,bt,bv):
        self.vocab=vocab;self.tables=tables;self.bt=np.asarray(bt);self.bv=np.asarray(bv);self.events=[]
        self.touch_masks=np.asarray([int(tables['simplePadMasks'][x]) for x in vocab['touchPositions']],np.int64)
        self.outer_masks=np.asarray([int(tables['simplePadMasks']['A'+str(i+1)]) for i in range(8)],np.int64)
        names=list(vocab['touchPositions']);adj=tables['touchAdjacency']
        self.adj=[[a==b or b in adj.get(a,()) for b in names] for a in names]
    def time(self,tick):return float(ticks_to_seconds(np.asarray([tick]),self.bt,self.bv)[0])
    def groups(self,indices):
        active=set(map(int,indices));count=0
        while active:
            count+=1;stack=[active.pop()]
            while stack:
                a=stack.pop();linked=[b for b in tuple(active) if self.adj[a][b]]
                for b in linked:active.remove(b);stack.append(b)
        return count
    def append(self,tick,rep,bpm):
        tick=int(tick);at=round(self.time(tick)*1e9)/1e9;arity=int(rep['button_arity']);inputs=[];tracks=[]
        touch_ids=list(map(int,np.flatnonzero(rep['touch_presence'])));notes=arity+len(touch_ids);lanes=[]
        if notes==0:return
        for j in range(arity):
            family=int(rep['button_family'][j]);lane=int(rep['button_start'][j]);lanes.append(lane)
            mods=int(rep['button_modifiers'][j]);duration=int(rep['button_duration'][j])
            pad=int(self.outer_masks[lane])
            if family==1:
                length=hold_seconds(self.vocab['durations'][duration],bpm);inputs.append(InputState(tick,True,lane,True,at,at+length,pad,True))
            elif family in (0,2) and not (family==2 and mods&16):
                inputs.append(InputState(tick,True,lane,False,at,at,pad,family!=2))
            if family!=2:continue
            route=int(rep['button_route'][j]);entry=self.tables['slideConflicts'][f'{lane+1}:{route}']
            wait,move=slide_seconds(self.vocab['durations'][duration],bpm);shoot=at+wait;end=shoot+move;wifi=bool(entry.get('isWifi',False));actions=[]
            for c in entry['contactIntervals']:
                begin=shoot+float(c['startFraction'])*move;finish=shoot+float(c['endFraction'])*move
                if not wifi and float(c['endFraction'])>=1-1e-7:finish+=1/60
                actions.append((begin,max(begin+1e-9,finish),1<<int(c['pad'])))
            tracks.append(TrackState(tick,at,shoot,end,wifi,actions))
        for j in touch_ids:
            duration=int(rep['touch_duration'][j]);hold=duration!=0;length=hold_seconds(self.vocab['durations'][duration],bpm) if hold else 0.
            inputs.append(InputState(tick,False,j,hold,at,at+length,int(self.touch_masks[j]),True))
        pure=notes==1 and arity==1 and int(rep['button_family'][0])==0
        self.events.append(EventState(tick,at,notes,tuple(lanes),pure,inputs,tracks))
    def snapshot(self,tick,bpm,durations,end_seconds):
        at=self.time(int(tick));inputs=[x for e in self.events for x in e.inputs];tracks=[x for e in self.events for x in e.tracks]
        overlay_inputs=[x for x in inputs if x.overlay and x.start-INPUT_OVERLAY_SECONDS<=at+TIME_EPSILON and x.end+INPUT_OVERLAY_SECONDS>=at-TIME_EPSILON]
        blocked_pad=0
        for x in overlay_inputs: blocked_pad|=int(x.pad)
        allowed_non_slide=(self.outer_masks&blocked_pad)==0
        allowed_touch=(self.touch_masks&blocked_pad)==0
        held=[x for x in inputs if x.hold and x.start<=at and x.end>at+1e-7]
        outerheld=[x for x in held if x.outer];touchheld=[x for x in held if not x.outer]
        move=[x for x in tracks if x.shoot<=at+TIME_EPSILON and x.end>=at-TIME_EPSILON]
        active=[x for x in tracks if x.start<=at+TIME_EPSILON and x.end+.2>at-TIME_EPSILON]
        source_lanes={x.sensor+1 for x in outerheld};touch_names=[self.vocab['touchPositions'][x.sensor] for x in touchheld]
        nh,nt,ns=len(outerheld),len(touchheld),len(move)
        release=[x for x in inputs if x.hold and x.start<=at+1e-7 and x.end+1/180>at+1e-7]
        outer_hold_hands=sum(x.outer for x in release);touch_release=[x.sensor for x in release if not x.outer]
        touch_hold_hands=self.groups(touch_release) if touch_release else 0
        one_hand=outer_hold_hands+touch_hold_hands==1;one_state=None
        if one_hand:
            owners={x.event for x in release};start=min(x.start for x in release)
            free=[e for e in self.events if e.time>=start-TIME_EPSILON and e.time<at-TIME_EPSILON and len(e.lanes)==1 and e.tick not in owners][-2:]
            if free:
                last=free[-1];previous_delta=None
                if len(free)>=2:previous_delta=(last.lanes[0]-free[-2].lanes[0]+4)%8-4
                one_state={'last_lane':last.lanes[0],'last_time':last.time,'previous_delta':previous_delta,'constraint_start':start}
        tap_state=None;past=[e for e in self.events if e.time<at-TIME_EPSILON]
        if len(past)>=3:
            recent=past[-3:]
            if all(e.pure_single_tap and len(e.lanes)==1 for e in recent):
                d1=(recent[1].lanes[0]-recent[0].lanes[0]+4)%8-4;d2=(recent[2].lanes[0]-recent[1].lanes[0]+4)%8-4
                if d1==d2 and abs(d2)==1:tap_state={'last_lane':recent[2].lanes[0],'last_time':recent[2].time,'direction':d2,'run_length':3}
        active_inputs=[x for x in inputs if x.outer and x.start<=at+1e-7 and x.end+1/180>at+1e-7]
        transient_outer=sum(not x.hold for x in active_inputs)
        transient_touch_events={x.event for x in inputs if not x.outer and not x.hold and x.start<=at+1e-7 and x.end+1/180>at+1e-7}
        slide_hands=sum(1+int(x.wifi) for x in move)
        available=max(0,2-len(release)-transient_outer-len(transient_touch_events)-slide_hands)
        covered=np.zeros(len(self.touch_masks),np.bool_)
        for tr in tracks:
            for begin,finish,mask in tr.actions:
                if begin<=at+TIME_EPSILON and finish>at-TIME_EPSILON:
                    covered|=(mask&self.touch_masks)!=0
        frame=any(x.shoot<=at+INPUT_RELEASE_SECONDS+TIME_EPSILON and x.end>=at-TIME_EPSILON for x in tracks)
        moving_capacity=1 if frame else 2;outer_capacity=max(0,min(2,moving_capacity-len(active_inputs)))
        hold,wait,movement=durations;remain=end_seconds-at
        hm=np.isfinite(hold)&(hold>0)&(hold<=remain+1e-7);sm=np.isfinite(wait)&np.isfinite(movement)&(wait>=0)&(movement>0)&(wait+movement<=remain+1e-7)
        result=dict(blockedLanes=source_lanes,riskLanes=set(),holdLanes=source_lanes,slideObligations=[],slideStartActionCount=0,slideTailCount=0,
            slideEndpointActionCount=0,slideTailLanes=set(),slideTailCooldownCount=0,slideTailCooldownEndTicks=[],muriStateFeatures=np.zeros(32,np.float32),
            muriOracleSourceTick=int(tick),activeTouchHoldSensors=touch_names,lastSingleTapLane=None,motionDirection=0,motionRunLength=0,
            activeHands=min(2,nh+nt),activeHoldHands=min(2,nh),activeSlideHands=min(2,ns),activeSlideCount=len(active),activeTouchHoldHands=min(2,nt),
            availableHands=available,holdAvailableHands=available,maxOuterArity=outer_capacity,allowedNonSlideStartMask=allowed_non_slide,allowedTouchSensorMask=allowed_touch,allowedTouchPresenceMask=covered,
            allowedHoldDurationMask=np.asarray(hm),allowedSlideDurationMask=np.asarray(sm),oneHandHoldConstraint=bool(one_hand))
        if tap_state:result.update(lastSingleTapLane=tap_state['last_lane'],motionDirection=tap_state['direction'],motionRunLength=tap_state['run_length'],tapRunLastTime=tap_state['last_time'])
        if one_state:result.update(freeHandLastLane=one_state['last_lane'],freeHandLastTime=one_state['last_time'],freeHandPreviousDelta=one_state['previous_delta'],oneHandConstraintStart=one_state['constraint_start'])
        return result
    def prune(self,at):
        if not self.events:self.keep_ticks=[];return []
        needed=[]
        for i,e in enumerate(self.events):
            live=(e.time>=at-2 or any(x.end+.2>=at for x in e.inputs) or any(x.end+TRACK_LIFECYCLE.retention_seconds>=at for x in e.tracks))
            if live or i>=len(self.events)-8:needed.append(i)
        lower=max(0,min(needed)-8) if needed else max(0,len(self.events)-8)
        self.events=self.events[lower:];self.keep_ticks=[e.tick for e in self.events];return self.keep_ticks
