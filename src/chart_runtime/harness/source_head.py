"""Track head/tail lifecycle constraints shared by WHERE and CUDA.

Times are compared at the IR's nanosecond precision. The motion exclusion is
open; the post-end cooldown includes end and excludes end + 250 ms.
One nanosecond of equality tolerance covers independently rounded source and
Tap timestamps (including decimal duration tokens); it is not a gameplay window.
The head rule does not exempt EX. The tail rule preserves exact-end protection
for EX and exempts it only from the newly extended post-end cooldown.
"""
import numpy as np

WINDOW_NS = 350_000_000
POST_END_NS = 250_000_000
TAIL_WINDOW_NS = 300_000_000
TAIL_EX_WINDOW_NS = 200_000_000
TRACK_RETENTION_SECONDS = max(.2,POST_END_NS/1_000_000_000,TAIL_WINDOW_NS/1_000_000_000)
EQUALITY_NS = 1


def forbidden(tap, shoot, end):
    tap, shoot, end = (np.rint(np.asarray(x)*1_000_000_000).astype(np.int64)
                       for x in (tap, shoot, end))
    delta = tap-shoot
    motion=(delta > EQUALITY_NS) & (delta < WINDOW_NS-EQUALITY_NS) & (tap < end-EQUALITY_NS)
    cooldown=(tap >= end-EQUALITY_NS) & (tap < end+POST_END_NS-EQUALITY_NS)
    return motion|cooldown


def tail_forbidden(moment,shoot,end,is_ex=False):
    t,s,e=(np.rint(np.asarray(x)*1_000_000_000).astype(np.int64) for x in (moment,shoot,end));ex=np.asarray(is_ex,dtype=np.bool_)
    limit=np.where(ex,TAIL_EX_WINDOW_NS,TAIL_WINDOW_NS)
    return (t>=s-EQUALITY_NS)&(np.abs(t-e)<limit-EQUALITY_NS)


CUDA_PREDICATE = r'''
__device__ bool source_head_tap_forbidden(double tap, double shoot, double end) {
    long long t = llround(tap * 1000000000.0);
    long long s = llround(shoot * 1000000000.0);
    long long e = llround(end * 1000000000.0);
    bool motion=t-s>SOURCE_HEAD_EQUALITY_NS && t-s<SOURCE_HEAD_WINDOW_NS-SOURCE_HEAD_EQUALITY_NS && t<e-SOURCE_HEAD_EQUALITY_NS;
    bool cooldown=t>=e-SOURCE_HEAD_EQUALITY_NS && t<e+SOURCE_HEAD_POST_END_NS-SOURCE_HEAD_EQUALITY_NS;
    return motion||cooldown;
}
__device__ bool track_tail_input_forbidden(double input_time, double shoot, double end, bool is_ex) {
    long long t=llround(input_time*1000000000.0),s=llround(shoot*1000000000.0),e=llround(end*1000000000.0);
    long long limit=is_ex?TRACK_TAIL_EX_WINDOW_NS:TRACK_TAIL_WINDOW_NS;
    return t>=s-SOURCE_HEAD_EQUALITY_NS && llabs(t-e)<limit-SOURCE_HEAD_EQUALITY_NS;
}
'''.replace('SOURCE_HEAD_WINDOW_NS',str(WINDOW_NS)+'LL').replace('SOURCE_HEAD_EQUALITY_NS',str(EQUALITY_NS)+'LL').replace('SOURCE_HEAD_POST_END_NS',str(POST_END_NS)+'LL').replace('TRACK_TAIL_EX_WINDOW_NS',str(TAIL_EX_WINDOW_NS)+'LL').replace('TRACK_TAIL_WINDOW_NS',str(TAIL_WINDOW_NS)+'LL')


def blocked_lanes(provider, moment):
    """Zero-based Tap lanes from accepted history and fixed references."""
    from ..io.codec import Codec
    from ..io.timing import ticks_to_seconds
    from ..io.durations import slide_seconds
    from ..io.factors import parse_slide_tracks
    # Fixed references are packed once by bind_references. Rewalking and
    # reparsing a thousand-event reference map at every target dominated
    # recovery snapshots; only the newly generated history remains dynamic.
    history=dict(getattr(provider,'history',{}))
    tempo=(tuple(provider.bt),tuple(provider.bv))
    prior=getattr(provider,'_source_head_cache',{}) if getattr(provider,'_source_head_tempo',None)==tempo else {}
    fixed=np.asarray(getattr(provider,'_source_head_reference_rows',np.empty((0,4))),np.float64).reshape(-1,4)
    if history and len(fixed):fixed=fixed[~np.isin(fixed[:,3],np.fromiter(history,np.int64))]
    cache={};blocked=set();all_rows=list(fixed[:,:3])
    for tick,text in history.items():
        key=(int(tick),text);rows=prior.get(key)
        if rows is None:
            rows=[]
            for note in Codec.parse_event(text):
                if note['family']!='slide':continue
                at=float(ticks_to_seconds(np.asarray([tick]),provider.bt,provider.bv)[0])
                bpm=float(provider.bv[max(0,np.searchsorted(provider.bt,tick,side='right')-1)])
                for track in parse_slide_tracks(note['raw']):
                    wait,move=slide_seconds(track['duration'],bpm)
                    rows.append((at+wait,at+wait+move,int(note['start'])-1))
            rows=tuple(rows)
        cache[key]=rows
        all_rows.extend(rows)
    if all_rows:
        rows=np.asarray(all_rows)
        blocked.update(map(int,rows[forbidden(moment,rows[:,0],rows[:,1]),2]))
        provider._source_head_all_rows=rows
    else:provider._source_head_all_rows=np.empty((0,3),np.float64)
    provider._source_head_cache=cache;provider._source_head_tempo=tempo
    return blocked


def tap_mask(provider, moment):
    allowed=np.ones(8,dtype=np.bool_)
    for lane in blocked_lanes(provider,moment):allowed[lane]=False
    return allowed


def launch_heads(provider,moment):
    """Instant outer inputs at these heads can share a launch hand action."""
    rows=np.asarray(getattr(provider,'_source_head_all_rows',np.empty((0,3))),np.float64).reshape(-1,3)
    return set(map(int,rows[np.abs(rows[:,0]-moment)<1e-7,2])) if len(rows) else set()


def lane_forbidden_at(provider,moment,lane):
    """Whether a Tap cue is blocked by either head or tail lifecycle."""
    rows=np.asarray(getattr(provider,'_source_head_all_rows',np.empty((0,3))),np.float64).reshape(-1,3)
    rows=rows[rows[:,2]==int(lane)]
    head=bool(forbidden(moment,rows[:,0],rows[:,1]).any()) if len(rows) else False
    tails=np.asarray(getattr(provider,'_track_tail_all_rows',np.empty((0,3))),np.float64).reshape(-1,3)
    tails=tails[tails[:,2]==int(lane)]
    # A generated cue may become EX later, so only the non-exempt exact-end
    # collision cancels its binding at this stage.
    return head or (bool(tail_forbidden(moment,tails[:,0],tails[:,1],True).any()) if len(tails) else False)


def tail_lane_sets(provider,moment):
    """Exact-end hard lanes and post-end EX-exempt cooldown lanes."""
    from ..io.codec import Codec
    from ..io.timing import ticks_to_seconds
    from ..io.durations import slide_seconds
    from ..io.factors import parse_slide_tracks
    history=dict(getattr(provider,'history',{}));tempo=(tuple(provider.bt),tuple(provider.bv))
    prior=getattr(provider,'_track_tail_cache',{}) if getattr(provider,'_track_tail_tempo',None)==tempo else {}
    fixed=np.asarray(getattr(provider,'_track_tail_reference_rows',np.empty((0,4))),np.float64).reshape(-1,4)
    if history and len(fixed):fixed=fixed[~np.isin(fixed[:,3],np.fromiter(history,np.int64))]
    cache={};all_rows=list(fixed[:,:3])
    for tick,text in history.items():
        key=(int(tick),text);rows=prior.get(key)
        if rows is None:
            rows=[];at=float(ticks_to_seconds(np.asarray([tick]),provider.bt,provider.bv)[0]);bpm=float(provider.bv[max(0,np.searchsorted(provider.bt,tick,side='right')-1)])
            for note in Codec.parse_event(text):
                if note['family']!='slide':continue
                for track in parse_slide_tracks(note['raw']):
                    wait,move=slide_seconds(track['duration'],bpm);rows.append((at+wait,at+wait+move,int(track['route'][-1])-1))
            rows=tuple(rows)
        cache[key]=rows;all_rows.extend(rows)
    rows=np.asarray(all_rows,np.float64).reshape(-1,3);provider._track_tail_all_rows=rows;provider._track_tail_cache=cache;provider._track_tail_tempo=tempo
    if not len(rows):return set(),set()
    core=set(map(int,rows[tail_forbidden(moment,rows[:,0],rows[:,1],True),2]))
    all_nonex=set(map(int,rows[tail_forbidden(moment,rows[:,0],rows[:,1],False),2]))
    return core,all_nonex-core


def outer_start_mask(provider,moment):
    allowed=np.ones(8,dtype=np.bool_)
    core,_annulus=tail_lane_sets(provider,moment)
    for lane in core:allowed[lane]=False
    return allowed


def snapshot_masks(provider,moment):
    core,annulus=tail_lane_sets(provider,moment);outer=np.ones(8,dtype=np.bool_)
    for lane in core:outer[lane]=False
    tap=tap_mask(provider,moment)&outer
    return {'allowedOuterStartMask':outer,'allowedTapStartMask':tap,
            'launchShareLanes':launch_heads(provider,moment),'trackTailCoreLanes':core,'trackTailAnnulusLanes':annulus}


def slide_mask(provider, moment, wait, move):
    """Reverse mask for a new Star against already-fixed future Taps."""
    if not provider.references:return None
    rows=np.asarray(getattr(provider,'_fixed_tap_rows',np.empty((0,3))),np.float64).reshape(-1,3)
    history=getattr(provider,'history',{})
    if history and len(rows):rows=rows[~np.isin(rows[:,2],np.fromiter(history,np.int64))]
    rows=rows[rows[:,0]>moment+1e-9]
    if not len(rows):return None
    mask=np.ones((8,len(wait)),dtype=np.bool_)
    valid=np.flatnonzero(np.isfinite(wait)&np.isfinite(move))
    shoots=moment+wait[valid];ends=shoots+move[valid]
    shoot_ns=np.rint(shoots*1_000_000_000).astype(np.int64)
    for lane in range(8):
        tap_times=np.sort(rows[rows[:,1]==lane,0])
        if not len(tap_times):continue
        tap_ns=np.rint(tap_times*1_000_000_000).astype(np.int64)
        blocked=np.zeros(len(valid),np.bool_)
        end_ns=np.rint(ends*1_000_000_000).astype(np.int64)
        # Long Tracks may have an allowed gap between the first 350 ms and
        # end-cooldown. Query the first Tap in both forbidden intervals.
        for lower,side in ((shoot_ns+EQUALITY_NS,'right'),(end_ns-EQUALITY_NS,'left')):
            index=np.searchsorted(tap_ns,lower,side=side);present=index<len(tap_times)
            if present.any():
                selected=np.flatnonzero(present)
                blocked[selected]|=forbidden(tap_times[index[selected]],shoots[selected],ends[selected])
        mask[lane,valid]&=~blocked
    return mask


class TrackLifecycle:
    """The sole production interface for Track head/tail temporal constraints."""
    motion_window_ns=WINDOW_NS
    end_cooldown_ns=POST_END_NS
    tail_window_ns=TAIL_WINDOW_NS
    tail_ex_window_ns=TAIL_EX_WINDOW_NS
    equality_ns=EQUALITY_NS
    retention_seconds=TRACK_RETENTION_SECONDS
    rules_id='shared-cuda-rules/3.8-release180-sourcehold-slide-hand-order300ms'
    cuda_predicate=CUDA_PREDICATE

    tap_forbidden=staticmethod(forbidden)
    blocked_lanes=staticmethod(blocked_lanes)
    tap_mask=staticmethod(tap_mask)
    launch_heads=staticmethod(launch_heads)
    lane_forbidden_at=staticmethod(lane_forbidden_at)
    tail_input_forbidden=staticmethod(tail_forbidden)
    tail_lane_sets=staticmethod(tail_lane_sets)
    outer_start_mask=staticmethod(outer_start_mask)
    snapshot_masks=staticmethod(snapshot_masks)
    slide_mask=staticmethod(slide_mask)

    @staticmethod
    def phase(tap,shoot,end):
        t,s,e=(int(np.rint(float(x)*1_000_000_000)) for x in (tap,shoot,end))
        if abs(t-s)<=EQUALITY_NS:return 'shoot'
        if not bool(forbidden(t/1e9,s/1e9,e/1e9)):return 'open'
        return 'motion' if t<e-EQUALITY_NS else 'end-cooldown'


TRACK_LIFECYCLE=TrackLifecycle()
