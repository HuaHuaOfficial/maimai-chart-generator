"""The single CUDA rule kernel for candidate batches and full charts.

All musical predicates execute on device. Host loops select named rules or
bounded tiles and serialize sparse witnesses, never test notes one by one.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .source_head import TRACK_LIFECYCLE

from ..io.codec import Codec, ChartPayload

RULES_ID = TRACK_LIFECYCLE.rules_id
QUALITY_NAMES = ('unpredictabilityWeightedInputs1s', 'outerKeyStepsPerSecond', 'slideSegmentsPerSecond', 'unexpectedMotionChangeRate')
HARD_NAMES = ('SameSensorStack','OuterMultiPress','HoldLaneConflict','DoubleHoldBlocksSlide',
              'TrackEndInput','UnprotectedSlideHeadInput','TapOnSlideCritical',
              'DoubleInputDuringSlide','WifiWithIndependentInput','DoubleWifi','WifiWithTwoTracks',
              'DoubleStartCompoundTrack','FullyOverlappingTrackWindow','SlideTooFast',
              'HoldBeyondChartEnd','TrackBeyondChartEnd','InvalidTiming','KnownHardFixture',
              'TouchUnavailableInVersion','TouchHoldSensorUnavailableInVersion','EXUnavailableInVersion','LongBreakUnavailableInVersion','MultiTouch')


@dataclass
class KernelResult:
    hard: dict[str, torch.Tensor]
    quality: torch.Tensor
    features: torch.Tensor
    soft: dict[str, torch.Tensor]
    event_batch: torch.Tensor
    event_tick: torch.Tensor
    track_counts: torch.Tensor
    star_counts: torch.Tensor
    note_counts: torch.Tensor
    feature_supported: torch.Tensor

    def summaries(self):
        size = self.track_counts.numel()
        hard_counts = torch.stack([torch.bincount(self.event_batch[mask], minlength=size) for mask in self.hard.values()], 1)
        quality_counts = torch.stack([torch.bincount(self.event_batch[self.quality[:,i]], minlength=size) for i in range(4)],1)
        soft_counts = torch.stack(list(self.soft.values()),1)
        return hard_counts, quality_counts, soft_counts


def _cat(payloads):
    columns={};offsets={'event':0,'track':0};batch_ids=[];cumulative=[0]
    for b,payload in enumerate(payloads):
        payload.assert_unmodified();c=payload.columns
        for key,value in c.items():
            if key=='event_time_ns':continue
            if key.endswith('_event') and offsets['event']:
                value=value+offsets['event']
            elif key in ('contact_track','queue_track') and offsets['track']:
                value=value+offsets['track']
            elif key=='action_track' and offsets['track']:
                value=torch.where(value>=0,value+offsets['track'],value)
            columns.setdefault(key,[]).append(value)
        count=c['event_tick'].numel()
        batch_ids.append(torch.full((count,),b,dtype=torch.int64,device=c['event_tick'].device))
        offsets['event']+=count;offsets['track']+=c['track_event'].numel();cumulative.append(offsets['event'])
    for key,values in columns.items():
        if values[0].ndim==2:
            width=max(x.shape[1] for x in values)
            fill=-1 if key=='event_lanes' else 0
            values=[F.pad(x,(0,width-x.shape[1]),value=fill) if x.shape[1]!=width else x for x in values]
        columns[key]=values[0] if len(values)==1 else torch.cat(values)
    columns['event_batch']=torch.cat(batch_ids)
    columns['action_batch']=torch.cat([torch.full((len(p.columns['action_start']),),b,device=p.columns['event_tick'].device,dtype=torch.int64) for b,p in enumerate(payloads)])
    return columns,cumulative


class Kernel:
    def __init__(self,codec:Codec):
        self.codec=codec;self.device=codec.device;self._scanner=None
        from .fused import FusedRules
        self.fused=FusedRules(codec)
        self._fixtures={}
        from ..io.fixtures import fixture_events as _fixture_events
        from ..io.symmetry import canonical_event_window
        for severity,name in (('clean','known_clean_muri_fixtures.json'),('hard','known_hard_muri_fixtures.json')):
            doc=json.loads((codec.root/'models/v2'/name).read_text(encoding='utf8'))
            self._fixtures[severity]=[(float(f['bpm']),canonical_event_window(_fixture_events(f)[:f.get('blockingEventCount')])) for f in doc['fixtures']]

    def _exemptions(self,payloads,starts,bpms):
        from ..io.symmetry import transform_event_group
        def digest(s):return int.from_bytes(sha256(s.encode()).digest()[:8],'little')&((1<<63)-1)
        clean=[];hard=[]
        # Parsing/hashing immutable strings only; all matching is done on CUDA.
        for b,payload in enumerate(payloads):
            eligible=[(kind,window) for kind,items in self._fixtures.items() for bpm,window in items if abs(bpm-bpms[b])<.01]
            if not eligible or not payload.source:continue
            hashes=torch.tensor([[digest(transform_event_group(s,sym)) for sym in range(16)] for _,s in payload.source],device=self.device)
            ticks=payload.columns['event_tick']
            for kind,window in eligible:
                width=len(window)
                if len(payload.source)<width:continue
                index=torch.arange(len(payload.source)-width+1,device=self.device)[:,None]+torch.arange(width,device=self.device)[None]
                expected=torch.tensor([digest(s) for _,s in window],device=self.device)
                delta=torch.tensor([t for t,_ in window],device=self.device)
                match=(hashes[index]==expected[None,:,None]).all(1).any(1)&((ticks[index]-ticks[index[:,0],None])==delta[None]).all(1)
                bounds=torch.stack((index[match,0],index[match,-1]),1)+starts[b]
                (clean if kind=='clean' else hard).append(bounds)
        empty=torch.empty((0,2),device=self.device,dtype=torch.int64)
        return torch.cat(clean) if clean else empty,torch.cat(hard) if hard else empty

    @torch.no_grad()
    def evaluate(self,payloads,*,versions,end_seconds,bpms,thresholds=None,tolerances=None,features=True):
        B=len(payloads)
        if not B or any(len(x)!=B for x in (versions,end_seconds,bpms)):
            raise ValueError('Batch metadata lengths must match the payload count')
        if any(not math.isfinite(float(x)) or float(x)<=0 for x in bpms):
            raise ValueError('BPM must be finite and positive')
        if any(not math.isfinite(float(x)) or float(x)<0 for x in end_seconds):
            raise ValueError('Chart ends must be finite and nonnegative')
        if thresholds is not None and (len(thresholds)!=B or any(x is not None and (len(x)!=4 or any(not math.isfinite(float(v)) or float(v)<=0 for v in x)) for x in thresholds)):
            raise ValueError('Malformed quality thresholds')
        for payload in payloads:
            if payload.definition_id!=self.codec.path_digest:
                raise ValueError('Payload path-table definition mismatch')
            if payload.columns['event_tick'].device!=self.device:
                raise ValueError('Payload belongs to another device')
        from .features import compute_features,violation_masks
        from ..runtime.resources import get_workspace
        workspace=get_workspace(str(self.device))
        with workspace.lease(64*1024*1024):
            c,starts=_cat(payloads);d=self.device;batch=c['event_batch'];n=len(batch);B=len(payloads)
            hard={key:torch.zeros(n,dtype=torch.bool,device=d) for key in HARD_NAMES}
            soft={key:torch.zeros(B,dtype=torch.int64,device=d) for key in ('TapOnSlide','SlideHeadTap','Overlap','SlideHandOrder')}
            clean,fixtures=self._exemptions(payloads,starts,bpms)
            def same(a,b):return batch[a[:,None]]==batch[b[None,:]]
            def exempt(a,b):
                if not clean.numel():return torch.zeros(torch.broadcast_shapes(a.shape,b.shape),device=d,dtype=torch.bool)
                lo=torch.minimum(a,b);hi=torch.maximum(a,b)
                return ((lo[...,None]>=clean[:,0])&(hi[...,None]<=clean[:,1])).any(-1)
            def add(reason,ea,eb,mask,allow_exempt=True):
                a=torch.broadcast_to(ea,mask.shape);b=torch.broadcast_to(eb,mask.shape)
                mask=mask&(~exempt(a,b) if allow_exempt else True)
                selected_a=a[mask];selected_b=b[mask]
                hard[reason].scatter_(0,selected_a,True);hard[reason].scatter_(0,selected_b,True)
            def unary(reason,event,mask,allow_exempt=False):add(reason,event,event,mask,allow_exempt)
            if fixtures.numel():hard['KnownHardFixture'].scatter_(0,fixtures[:,1],True)
            limits=torch.tensor(end_seconds,device=d,dtype=torch.float64);ver=torch.tensor(versions,device=d)
            flag_values,soft_values=self.fused.run(c,clean,ver,limits,8+self.codec.sensors.index('C'),B)
            masks=((flag_values[:,None]>>torch.arange(len(HARD_NAMES),device=d)[None,:])&1).bool()
            hard={name:masks[:,i] for i,name in enumerate(HARD_NAMES)}
            soft={name:soft_values[:,i] for i,name in enumerate(('TapOnSlide','SlideHeadTap','Overlap','SlideHandOrder'))}
            if fixtures.numel():hard['KnownHardFixture'].scatter_(0,fixtures[:,1],True)
            ie=c['input_event'];ib=batch[ie];I=len(ie)
            te=c['track_event'];tb=batch[te];T=len(te)
            ne=c['note_event'];nb=batch[ne];kind=c['note_kind'];mods=c['note_modifiers']
            if T:
                early=self._queues(c,B)
                if clean.numel():
                    # A registered clean window exempts only a queue whose
                    # entire interacting source set stays inside that window.
                    ae=c['action_event'];ab=batch[ae]
                    for lo in range(0,T,64):
                        sl=slice(lo,lo+64)
                        relevant=(tb[sl,None]==ab[None,:])&(c['action_start'][None,:]<c['track_early'][sl,None])&(c['action_end'][None,:]>c['track_start'][sl,None]-5/60)
                        lower=torch.where(relevant,ae[None,:],n).amin(1)
                        upper=torch.where(relevant,ae[None,:],-1).amax(1)
                        early[sl]&=~exempt(torch.minimum(lower,te[sl]),torch.maximum(upper,te[sl]))
                unary('SlideTooFast',te,early,False)
            values=compute_features(c['event_time'],c['event_notes'],c['event_lanes'],c['event_lane_count'],c['event_track_speed'],batch_ids=batch) if n and features else torch.zeros((n,4),dtype=torch.float64,device=d)
            supported=torch.tensor([x is not None for x in (thresholds or [None]*B)],dtype=torch.bool,device=d)
            quality=torch.zeros((n,4),dtype=torch.bool,device=d)
            if features and n and thresholds is not None:
                # Unsupported slots expose status; they never masquerade as a
                # verified calibration with huge fake numeric thresholds.
                ts=torch.tensor([x if x is not None else [1.]*4 for x in thresholds],dtype=torch.float64,device=d)
                quality=violation_masks(values,c['event_time'],ts[batch],tolerances or {'sustainedRatio':1.35,'extremeRatio':2.,'sustainedEvents':3,'windowSeconds':1.},batch_ids=batch)&supported[batch,None]
            return KernelResult(hard,quality,values,soft,batch,c['event_tick'],torch.bincount(tb,minlength=B),torch.bincount(nb[(kind==2)&((mods&16)==0)],minlength=B),torch.bincount(nb,minlength=B),supported)

    def _queues(self,c,B):
        """Device-side action timeline via scatter/cumsum, shared by all queues."""
        from .slide_queue import CudaSlideQueueScanner
        from .cuda_views import view
        if self._scanner is None:self._scanner=CudaSlideQueueScanner()
        te=c['track_event'];tb=c['event_batch'][te];T=len(te);d=self.device
        # Actions were concatenated per chart; encode their scenario by their
        # counts, which are metadata collected alongside payloads by evaluate.
        ab=c['action_batch']
        start=(c['action_start']*1e9).round().to(torch.int64)
        end=(c['action_end']*1e9).round().to(torch.int64)
        begin=((c['track_start']-5/60)*1e9).round().to(torch.int64)
        finish=((c['track_early']-1e-7)*1e9).round().to(torch.int64)
        stride=10**15;offset=10**12
        sb=start+ab*stride+offset;eb=end+ab*stride+offset;bb=begin+tb*stride+offset;fb=finish+tb*stride+offset
        grid=torch.unique(torch.cat((sb,eb,bb,fb)),sorted=True)
        si=torch.searchsorted(grid,sb);ei=torch.searchsorted(grid,eb)
        counts=torch.zeros((len(grid),33),dtype=torch.int32,device=d)
        pad_ids=torch.arange(33,device=d);on=((c['action_mask'][:,None]>>pad_ids[None,:])&1).to(torch.int32)
        counts.index_add_(0,si,on);counts.index_add_(0,ei,-on);held=counts.cumsum(0)>0
        states=(held.to(torch.int64)*(1<<pad_ids)[None]).sum(1);previous=torch.cat((states.new_zeros(1),states[:-1]));ups=previous&~states
        first=torch.searchsorted(grid,bb);last=torch.searchsorted(grid,fb,right=True)
        qt=c['queue_track'];cp=self._scanner.cp
        import cupy
        stream=torch.cuda.current_stream(d)
        with cupy.cuda.ExternalStream(stream.cuda_stream):
            completion,_=self._scanner.scan_global_device(view(cp,c['queue_masks']),view(cp,c['queue_skip']),view(cp,c['queue_count']),view(cp,states),view(cp,ups),view(cp,grid.to(torch.float64)),view(cp,first[qt].to(torch.int32)),view(cp,last[qt].to(torch.int32)))
            done=torch.isfinite(torch.from_dlpack(completion))
        total=torch.bincount(qt,minlength=T);complete=torch.zeros(T,dtype=torch.int32,device=d);complete.scatter_add_(0,qt,done.to(torch.int32))
        return torch.where(c['track_wifi'],(complete==total)&(total>0),complete>0)&(finish>begin)
