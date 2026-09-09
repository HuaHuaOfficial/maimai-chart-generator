"""Pure contract utilities. These do not relax any Harness verdict."""
from __future__ import annotations
from hashlib import sha256
import numpy as np


def intersect_mask(snapshot, key, requested):
    mask = np.asarray(requested, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError(f'{key} must be a one-dimensional mask')
    prior = snapshot.get(key)
    if prior is not None:
        prior = np.asarray(prior, dtype=np.bool_)
        if prior.shape != mask.shape:
            raise ValueError(f'{key} shape mismatch: {prior.shape} != {mask.shape}')
        mask = prior & mask
    else:
        mask = mask.copy()
    snapshot[key] = mask


def dependency_window(targets, relation, all_ticks, *, context=None, events=None):
    if not targets or relation is None:
        return sorted(set(map(int, targets)))
    lo, hi = min(targets), max(targets)
    links=relation.links
    if getattr(relation,'runtime_where',False):
        if context is None or events is None:raise ValueError('WHERE recovery requires committed events')
        from .relational_where import selected_link
        from .sampling import text_representation
        links=tuple(link for tick,text in events.items() if tick in relation.source_to_link
                    if (link:=selected_link(context,relation,tick,text_representation(text,context.factor_session[1]),events)) is not None)
    while True:
        before = (lo, hi)
        for link in links:
            a, b = link.source_tick, link.cue_tick
            if b is not None and (lo <= a <= hi or lo <= b <= hi):
                lo, hi = min(lo,a,b), max(hi,a,b)
        if before == (lo, hi): break
    return sorted(set(targets) | {int(t) for t in all_ticks if lo <= int(t) <= hi})

def state_fingerprint(context, relation, *mask_maps):
    h = sha256()
    for value in (context.mel, context.structure, context.style):
        a = np.ascontiguousarray(value)
        h.update(repr((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    if relation is not None:
        if getattr(relation,'runtime_where',False):h.update(relation.fingerprint.encode())
        else:
            for link in relation.links: h.update(repr(link).encode())
        mask_maps = (relation.duration_masks,) + mask_maps
    for masks in mask_maps:
        for tick, mask in sorted((masks or {}).items()):
            a=np.asarray(mask,dtype=np.bool_)
            h.update(str(tick).encode());h.update(repr(a.shape).encode());h.update(a.tobytes())
    model=context.factor_session[0]
    h.update(repr(tuple((name,p._version) for name,p in model.named_parameters())).encode())
    return h.hexdigest()


def freeze_plan(plan):
    from types import MappingProxyType
    links=tuple(plan.links)
    expected={x.source_tick:x for x in links}
    cues={x.cue_tick:x.source_tick for x in links if x.cue_tick is not None}
    if len(expected)!=len(links) or expected!=dict(plan.source_to_link):
        raise ValueError('Duplicate or inconsistent relation sources')
    if len(cues)!=sum(x.cue_tick is not None for x in links) or cues!=dict(plan.cue_to_source):
        raise ValueError('Duplicate or inconsistent cue ownership')
    if set(plan.duration_masks)!=set(expected):
        raise ValueError('Every relation source needs exactly one duration mask')
    masks={}
    for x in links:
        if x.launch_tick<=x.source_tick or x.min_move_ticks<=0:
            raise ValueError('Delayed relation must have positive wait and motion')
        if x.cue_tick is not None:
            if x.cue_tick!=x.launch_tick or x.cue_intent is None or 0 not in x.cue_intent.button_families:
                raise ValueError('Cue must be a Tap at the declared launch')
        a=np.array(plan.duration_masks[x.source_tick],dtype=np.bool_,copy=True)
        if a.ndim!=1 or not a.any():raise ValueError('Empty or malformed relation duration support')
        a.flags.writeable=False;masks[x.source_tick]=a
    object.__setattr__(plan,'links',links)
    object.__setattr__(plan,'source_to_link',MappingProxyType(expected))
    object.__setattr__(plan,'cue_to_source',MappingProxyType(cues))
    object.__setattr__(plan,'duration_masks',MappingProxyType(masks))
