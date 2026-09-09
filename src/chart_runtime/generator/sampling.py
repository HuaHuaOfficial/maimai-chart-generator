"""Native V4 structured factor sampling primitives.

This module is deliberately independent of the legacy inference module.  It
contains only the representation, mask and structured-head sampling pieces
needed by the runtime renderer.  Musical judgement is not implemented here:
the caller supplies provider snapshots and consumes provider batch verdicts.
"""

from __future__ import annotations
from .sampler_host import geometry_tables, component_count

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from .intent import EventIntent
from ..version_semantics import (
    sanitize_slide_head_modifiers,
    slide_route_allowed,
    touch_enabled,
    touch_hold_enabled,
    touch_hold_sensor_allowed,
    touch_sensor_allowed,
)

from chart_runtime.io.audio import FRAME_SECONDS
from chart_runtime.io.factors import SHAPE_RE, factor_event, note_route
from chart_runtime.io.timing import sample_positions
from chart_runtime.generator.tokens import RUNTIME_METADATA_KEYS, metadata_tokens, rhythm_classes
from chart_runtime.generator.models.relational import (
    EMPTY_GEOMETRY_ID,
    geometry_candidate_index,
)


TPB = 384

# Slide syntax availability is version semantics, not an empirical dataset prior.
# Exact start/route geometry still comes from route_compatibility.json; its
# minVersion field is intentionally ignored here.


class RepresentationEncodingError(ValueError):
    """The newest V4 factor representation cannot encode an event losslessly."""

    def __init__(self, message: str, *, text: str = "", field: str = ""):
        self.text = text
        self.field = field
        super().__init__(message)


class WhereSamplingUnavailable(ValueError):
    """An explicit WHAT has no realization under the shared WHERE authority."""


def empty_representation(touch_count: int) -> dict[str, np.ndarray]:
    return {
        "button_arity": np.uint8(0),
        "button_family": np.zeros(2, np.uint8),
        "button_start": np.zeros(2, np.uint8),
        "button_route": np.zeros(2, np.uint16),
        "button_duration": np.zeros(2, np.uint16),
        "button_modifiers": np.zeros(2, np.uint8),
        "touch_presence": np.zeros(int(touch_count), np.bool_),
        "touch_duration": np.zeros(int(touch_count), np.uint16),
        "touch_modifiers": np.zeros(int(touch_count), np.uint8),
    }


def copy_representation(rep: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(rep,EventIntent):return rep
    return {
        key: value.copy() if isinstance(value, np.ndarray) else np.asarray(value).copy()
        for key, value in rep.items()
    }


def intent_signature(rep: Mapping[str, Any]) -> tuple:
    if isinstance(rep,EventIntent):return rep.button_arity,tuple(sorted(rep.button_families)),rep.touch_count
    arity = int(rep["button_arity"])
    families = tuple(sorted(int(value) for value in rep["button_family"][:arity]))
    return arity, families, int(np.count_nonzero(rep["touch_presence"]))


def representation_signature(rep: Mapping[str, Any]) -> tuple:
    arity = min(2, int(rep["button_arity"]))
    buttons = tuple(
        sorted(
            (
                int(rep["button_family"][index]),
                int(rep["button_start"][index]),
                int(rep["button_route"][index]),
                int(rep["button_duration"][index]),
                int(rep["button_modifiers"][index]),
            )
            for index in range(arity)
        )
    )
    touches = tuple(
        (
            int(index),
            int(rep["touch_duration"][index]),
            int(rep["touch_modifiers"][index]),
        )
        for index in np.flatnonzero(rep["touch_presence"])
    )
    return arity, buttons, touches


def text_representation(text: str, vocab: dict) -> dict[str, np.ndarray]:
    """Encode only events representable by the newest singular-track factor heads."""

    text = str(text)
    rep = empty_representation(len(vocab["touchPositions"]))
    family_to_id = {value: index for index, value in enumerate(vocab["families"])}
    position_to_id = {value: index for index, value in enumerate(vocab["positions"])}
    route_to_id = {value: index for index, value in enumerate(vocab["routes"])}
    duration_to_id = {value: index for index, value in enumerate(vocab["durations"])}
    touch_to_id = {value: index for index, value in enumerate(vocab["touchPositions"])}
    notes = factor_event(text)["notes"]
    buttons = [note for note in notes if note["family"] != "touch"]
    if len(buttons) > 2:
        raise RepresentationEncodingError(
            "event has more than two outer notes", text=text, field="button_arity"
        )
    rep["button_arity"] = np.uint8(len(buttons))
    for index, note in enumerate(buttons):
        try:
            rep["button_family"][index] = family_to_id[note["family"]]
            rep["button_start"][index] = position_to_id[note["start"]]
        except KeyError as exc:
            raise RepresentationEncodingError(
                f"unmapped factor {exc.args[0]!r}", text=text, field="button"
            ) from exc
        raw = str(note["raw"])
        if "*" in raw or raw.count("[") > 1 or raw.count("]") > 1:
            raise RepresentationEncodingError(
                "compound slide or per-segment timing is unsupported by the singular-track factor model",
                text=text,
                field="compound_slide",
            )
        route = note_route(raw)
        if note["family"] == "slide" and route is None:
            raise RepresentationEncodingError(
                "slide has no representable route", text=text, field="button_route"
            )
        if route is None:
            rep["button_route"][index] = 0
        else:
            if route not in route_to_id:
                raise RepresentationEncodingError(
                    f"unmapped route {route!r}", text=text, field="button_route"
                )
            rep["button_route"][index] = route_to_id[route]
        duration = note["duration"]
        if duration is None:
            rep["button_duration"][index] = 0
        elif duration not in duration_to_id:
            raise RepresentationEncodingError(
                f"unmapped duration {duration!r}",
                text=text,
                field="button_duration",
            )
        else:
            rep["button_duration"][index] = duration_to_id[duration]
        rep["button_modifiers"][index] = (
            int(note["is_break"])
            | (int(note["is_ex"]) << 1)
            | (int(note["is_star"]) << 2)
            | (int(note["is_firework"]) << 3)
            | (int(note["is_headless"]) << 4)
        )
    for note in (item for item in notes if item["family"] == "touch"):
        if note["start"] not in touch_to_id:
            raise RepresentationEncodingError(
                f"unmapped Touch sensor {note['start']!r}",
                text=text,
                field="touch_presence",
            )
        index = touch_to_id[note["start"]]
        rep["touch_presence"][index] = True
        duration = note["duration"]
        if duration is None:
            rep["touch_duration"][index] = 0
        elif duration not in duration_to_id:
            raise RepresentationEncodingError(
                f"unmapped Touch duration {duration!r}",
                text=text,
                field="touch_duration",
            )
        else:
            rep["touch_duration"][index] = duration_to_id[duration]
        rep["touch_modifiers"][index] = (
            int(note["is_break"])
            | (int(note["is_ex"]) << 1)
            | (int(note["is_star"]) << 2)
            | (int(note["is_firework"]) << 3)
            | (int(note["is_headless"]) << 4)
        )
    return rep


def load_route_support(asset_root: Path) -> dict:
    path = Path(asset_root) / "route_compatibility.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing V4 route support table: {path}")
    document = json.loads(path.read_text(encoding="utf8"))
    starts = document.get("starts")
    if not isinstance(starts, dict):
        raise ValueError("route_compatibility.json has no starts table")
    return starts


def allowed_route_ids(
    vocab: dict,
    version_id: int,
    start: int,
    route_support: dict,
) -> tuple[int, ...]:
    compatibility = route_support.get(str(int(start) + 1), {})
    result = []
    for index, route in enumerate(vocab["routes"]):
        if index < 2:
            continue
        # Presence in this table is used only for geometric start/route support.
        # Historical syntax availability comes from the explicit capability map.
        if compatibility.get(str(index)) is None:
            continue
        if slide_route_allowed(int(version_id), int(start), str(route)):
            result.append(index)
    return tuple(result)


def preload_route_ids(vocab: dict, version_id: int, route_support: dict) -> dict[tuple[int, int], tuple[int, ...]]:
    """Precompute the eight static start supports once per native render."""

    version = int(version_id)
    return {
        (version, start): allowed_route_ids(vocab, version, start, route_support)
        for start in range(8)
    }


def allowed_route_ids_for_snapshot(
    vocab: dict,
    version_id: int,
    start: int,
    snapshot: dict,
    route_support: dict,
) -> tuple[int, ...]:
    ids = allowed_route_ids(vocab, version_id, start, route_support)
    mask = snapshot.get("allowedSlideRouteMask")
    return ids if mask is None else tuple(index for index in ids if bool(mask[index]))


def start_allowed(snapshot: dict, index: int) -> bool:
    mask = snapshot.get("allowedOuterStartMask")
    return mask is None or bool(mask[int(index)])


def apply_geometry_start_mask(
    eligible: torch.Tensor,
    candidate_masks: torch.Tensor,
    snapshot: dict,
    device: torch.device,
) -> torch.Tensor:
    mask = snapshot.get("allowedOuterStartMask")
    if mask is None:
        return eligible
    allowed = torch.as_tensor(mask, device=device, dtype=torch.bool)
    return eligible & ~(candidate_masks.bool() & ~allowed[None]).any(1)


def choose_allowed(
    logits: torch.Tensor,
    allowed,
    temperature: float,
    rng: np.random.Generator,
    top_p: float = 1.0,
    host_bias=None,
) -> int:
    ids = np.asarray(list(allowed), np.int64)
    if len(ids) == 0:
        raise ValueError("no allowed factor ids")
    if len(ids) == 1:
        return int(ids[0])
    values = logits.float().detach().cpu().numpy()[ids]
    if host_bias is not None:values=values+np.asarray(host_bias)[ids]
    probability = np.exp((values - values.max()) / max(0.05, float(temperature)))
    probability /= probability.sum() + 1e-12
    if top_p < 1.0 and len(probability) > 1:
        order = np.argsort(probability)[::-1]
        keep = max(
            1,
            int(np.searchsorted(np.cumsum(probability[order]), max(0.05, top_p), side="left")) + 1,
        )
        ids = ids[order[:keep]]
        probability = probability[order[:keep]]
        probability /= probability.sum() + 1e-12
    return int(rng.choice(ids, p=probability))


def duration_ids_for_family(snapshot: dict, family: int, size: int) -> list[int]:
    key = "allowedSlideDurationMask" if int(family) == 2 else "allowedHoldDurationMask"
    runtime_timing=family==2 and '_where_duration_bias' in snapshot
    limit=int(size) if runtime_timing else min(int(size),int(snapshot.get('_model_duration_count',size)))
    return [index for index in range(2, limit) if key not in snapshot or snapshot[key][index]]


def slide_start_available(snapshot,start,duration_count):
    known=snapshot.get('_where_fixed_cue_lanes',{});future=snapshot.get('allowedSlideHeadDurationMask')
    return any((d not in known or int(start) in known[d]) and (future is None or future[int(start),d])
               for d in duration_ids_for_family(snapshot,2,duration_count))


def valid_outer_assignments(snapshot,starts,families,duration_count):
    """One WHERE feasibility authority used before and during sampling."""
    from itertools import permutations
    from .shared_launch import HAND_ACCOUNTING
    tap_allowed=np.asarray(snapshot.get('allowedTapStartMask',np.ones(8,np.bool_)),np.bool_)
    held={int(x)-1 for x in snapshot.get('holdLanes',())};cue=snapshot.get('_launch_cue_lane')
    return [order for order in set(permutations(families)) if HAND_ACCOUNTING.assignment_fits(snapshot,starts,order) and all(
        (f!=0 or tap_allowed[int(lane)]) and (f not in (0,1) or int(lane) not in held)
        and (f!=2 or slide_start_available(snapshot,lane,duration_count))
        and (cue is None or int(lane)!=int(cue) or f==0)
        for lane,f in zip(starts,order))]


@torch.no_grad()
def decode_structured_factor_event_fast(
    head,
    hidden: torch.Tensor,
    vocab: dict,
    version_id: int,
    sampling_temperature: float,
    rng: np.random.Generator,
    state,
    snapshot: dict,
    route_support: dict,
    route_cache: dict[tuple[int, int], tuple[int, ...]] | None = None,
    sampling_top_p: float = 1.0,
    previous_representation: dict | None = None,
    adapted_override: torch.Tensor | None = None,
    relational_logits_override: torch.Tensor | None = None,
    intent_override: dict | None = None,
    intent_hard_mask: bool = True,
    touch_hold_scale: float = 1.0,
) -> tuple[str, dict]:
    """Native newest-V4 structured sampler; no CPU legality predicates."""

    rep = empty_representation(len(vocab["touchPositions"]))
    device = hidden.device
    if not all(
        hasattr(head, name)
        for name in (
            "arity_head", "arity_embedding", "family_head", "family_embedding",
            "start_head", "start_embedding", "route_head", "route_embedding",
            "duration_head", "duration_embedding", "modifier_head", "touch_sensor",
            "touch_presence", "touch_hold_presence", "touch_duration", "touch_modifiers",
            "geometry_logits", "geometry_candidate_masks", "geometry_candidate_arities",
            "geometry_sequence_state",
        )
    ):
        raise TypeError("resident factor session is not the newest V4 structured head")
    blocked_for_decode: set[int] = set()
    adapted = head.adapter(hidden) if adapted_override is None else adapted_override.to(dtype=hidden.dtype)
    arity_logits = None if intent_override is not None else head.arity_head(adapted)[0, -1].float()
    relational_geometry = True
    geometry_logits = relational_logits_override
    if geometry_logits is None:
        geometry_target = {}
        if previous_representation is not None:
            for key in ("button_arity", "button_start"):
                value = torch.as_tensor(previous_representation[key], device=device)
                geometry_target[f"prev_{key}"] = value.long().reshape(1, 1, *value.shape)
        geometry_logits = head.geometry_logits(hidden, geometry_target)[0, -1].float()
    else:
        geometry_logits = geometry_logits[0, -1].float()

    profile=snapshot.get('_sequence_profile')
    calibrated=profile is not None and profile.cue_enabled
    model_score=0.0
    score_candidates=not calibrated and bool(snapshot.get('_launch_preferred_lanes')) and (snapshot.get('_launch_cue_lane') is None or snapshot.get('_cue_proposal_only',False))

    def where_slide_can_start(start):return slide_start_available(snapshot,start,len(vocab['durations']))

    tap_allowed=np.asarray(snapshot.get('allowedTapStartMask',np.ones(8,np.bool_)),np.bool_)
    from .shared_launch import HAND_ACCOUNTING
    launch_heads=HAND_ACCOUNTING.launch_heads(snapshot)

    def valid_assignments(starts,families):return valid_outer_assignments(snapshot,starts,families,len(vocab['durations']))

    def choose_relational_geometry(requested_arity: int):
        nonlocal model_score
        if requested_arity <= 0:
            return requested_arity, None
        candidate_arities, candidate_masks, candidate_starts = geometry_tables(head)
        allowed = np.asarray(snapshot.get("allowedOuterStartMask", np.ones(8, np.bool_)), dtype=np.bool_)
        eligible = (candidate_arities == requested_arity) & ~(candidate_masks & ~allowed[None]).any(1)
        candidate_ids = np.flatnonzero(eligible).tolist()
        if not candidate_ids and requested_arity == 2 and intent_override is None:
            eligible = (candidate_arities == 1) & ~(candidate_masks & ~allowed[None]).any(1)
            candidate_ids = np.flatnonzero(eligible).tolist()
            requested_arity = 1 if candidate_ids else 0
        if intent_hard_mask and intent_override is not None and candidate_ids:
            frozen_families = [int(value) for value in intent_override["button_family"][:requested_arity]]
            active_hold_lanes = {int(value)-1 for value in snapshot.get("holdLanes", ())}
            candidate_ids = [candidate for candidate in candidate_ids if not any(
                note_index < len(frozen_families) and frozen_families[note_index] in (0,1)
                and int(start) in active_hold_lanes
                for note_index, start in enumerate(candidate_starts[candidate]))]
        cue_lane = snapshot.get('_launch_cue_lane')
        if cue_lane is not None:
            candidate_ids = [i for i in candidate_ids if int(cue_lane) in candidate_starts[i]]
        if intent_override is not None and 2 in EventIntent.from_representation(intent_override).button_families:
            candidate_ids=[i for i in candidate_ids if any(where_slide_can_start(x) for x in candidate_starts[i])]
        if intent_override is not None and (launch_heads or not tap_allowed.all()):
            families=EventIntent.from_representation(intent_override).button_families
            candidate_ids=[i for i in candidate_ids if valid_assignments(candidate_starts[i],families)]
        if intent_override is None and not tap_allowed.all() and not any(
                duration_ids_for_family(snapshot,f,len(vocab['durations'])) for f in (1,2)):
            candidate_ids=[i for i in candidate_ids if all(tap_allowed[int(lane)] for lane in candidate_starts[i])]
        if intent_override is None and launch_heads:
            candidate_ids=[i for i in candidate_ids if HAND_ACCOUNTING.assignment_fits(snapshot,candidate_starts[i],(0,)*requested_arity)]
        if not candidate_ids:
            if intent_override is not None:raise WhereSamplingUnavailable('no geometry satisfies the explicit WHAT')
            if requested_arity==2 and intent_override is None and launch_heads:return choose_relational_geometry(1)
            return 0, None
        scores=geometry_logits
        preferred=snapshot.get('_launch_preferred_lanes',())
        if not calibrated and preferred and intent_override is not None and all(f==0 for f in EventIntent.from_representation(intent_override).button_families):
            bonus=np.asarray([bool(set(map(int,lanes)).intersection(preferred)) for lanes in candidate_starts],np.float32)
            scores=scores+torch.as_tensor(bonus,device=device)*3.0
        discouraged=snapshot.get('_recent_slide_heads',())
        # Pure Tap events have unambiguous geometry. Mixed families are
        # scored during assignment below, so a Star is never penalized as Tap.
        if discouraged and intent_override is not None and all(f==0 for f in EventIntent.from_representation(intent_override).button_families):
            cost=np.asarray([sum(int(x) in discouraged for x in lanes) for lanes in candidate_starts],np.float32)
            scores=scores-torch.as_tensor(cost,device=device)*3.0
        from .sequence_runtime import calibrated_cue,cue_calibration_eligible
        families=EventIntent.from_representation(intent_override).button_families if intent_override is not None else ()
        bias=None
        if profile is not None and 2 in families:
            lane_bias=profile.start_bias(snapshot['_sequence_tick'],snapshot['_sequence_history'])
            bias=np.array([np.mean(lane_bias[list(lanes)]) if lanes else 0. for lanes in candidate_starts])
        if calibrated and preferred and cue_calibration_eligible(families) and snapshot.get('_launch_cue_lane') is None:
            probability=profile.cue_probability(snapshot['_sequence_bpm'])
            selected=calibrated_cue(scores,candidate_ids,candidate_starts,preferred,probability,sampling_temperature,rng,sampling_top_p)
        else:
            selected=choose_allowed(scores,candidate_ids,sampling_temperature,rng,sampling_top_p,host_bias=bias)
        if score_candidates:model_score+=float(geometry_logits[selected].item())
        starts = list(candidate_starts[selected])
        return requested_arity, starts

    def one_value(embedding, value: int):
        return embedding(torch.full((1, 1), int(value), dtype=torch.long, device=device))

    tail_cooldown = int(snapshot.get("slideTailCooldownCount", 0)) > 0
    arity = (
        int(intent_override["button_arity"])
        if intent_override is not None
        else 0
        if tail_cooldown
        else choose_allowed(
            arity_logits,
            range(min(2,min(int(snapshot['holdAvailableHands']),int(snapshot.get('maxOuterArity',2)))+len(launch_heads)) + 1),
            sampling_temperature,
            rng,
            sampling_top_p,
        )
    )
    arity, starts = choose_relational_geometry(arity)
    rep["button_arity"] = arity
    if starts is not None:
        rep["button_start"][:arity] = np.asarray(starts, dtype=np.uint8)

    running = adapted + one_value(head.arity_embedding, int(rep["button_arity"]))
    summary = torch.zeros_like(running)
    duration_ids = duration_ids_for_family(snapshot, 1, len(vocab["durations"]))
    remaining_intent_families=list(EventIntent.from_representation(intent_override).button_families) if intent_override is not None else []
    for note_index in range(2):
        note_hidden = running + summary
        active_note = note_index < int(rep["button_arity"])
        if active_note:
            if intent_override is not None:
                # WHAT supplies an unordered family multiset. V4 assigns that
                # multiset to realized lanes using its learned family logits.
                family_logits=head.family_head[note_index](note_hidden)[0,-1].float()
                native_family_logits=family_logits
                allowed_families=sorted(set(remaining_intent_families))
                if 2 in allowed_families and (snapshot.get('_where_fixed_cue_lanes') or snapshot.get('allowedSlideHeadDurationMask') is not None):
                    if not where_slide_can_start(rep['button_start'][note_index]):allowed_families.remove(2)
                    elif not any(where_slide_can_start(x) for x in rep['button_start'][note_index+1:arity]):
                        allowed_families=[2]
                cue_lane = snapshot.get('_launch_cue_lane')
                if cue_lane is not None:
                    if int(rep['button_start'][note_index]) == int(cue_lane):
                        allowed_families = [0] if 0 in remaining_intent_families else []
                    elif int(cue_lane) in map(int,rep['button_start'][note_index+1:arity]):
                        remainder = list(remaining_intent_families)
                        if 0 in remainder: remainder.remove(0)
                        allowed_families = sorted(set(remainder))
                if int(rep['button_start'][note_index]) in snapshot.get('_recent_slide_heads',()) and len(allowed_families)>1 and 0 in allowed_families:
                    family_logits=family_logits.clone();family_logits[0]-=3.0
                if int(rep['button_start'][note_index]) in snapshot.get('_launch_preferred_lanes',()) and len(allowed_families)>1 and 0 in allowed_families:
                    family_logits=family_logits.clone();family_logits[0]+=3.0
                if launch_heads or not tap_allowed.all():
                    completions=valid_assignments(rep['button_start'][note_index:arity],remaining_intent_families)
                    allowed_families=[f for f in allowed_families if any(order[0]==f for order in completions)]
                if not allowed_families:raise WhereSamplingUnavailable('no family assignment satisfies the explicit WHAT')
                family=choose_allowed(family_logits,allowed_families,sampling_temperature,rng,sampling_top_p)
                if score_candidates:model_score+=float(native_family_logits.log_softmax(-1)[family].item())
                remaining_intent_families.remove(family)
            else:
                family_logits = head.family_head[note_index](note_hidden)[0, -1].float()
                family_logits[3] = -torch.inf
                allowed_families = [
                    family_id
                    for family_id in range(3)
                    if family_id == 0
                    or duration_ids_for_family(snapshot, family_id, len(vocab["durations"]))
                ]
                if snapshot.get('_forbid_unplanned_slide') or (snapshot.get('_single_slide_contract') and note_index>0 and 2 in rep['button_family'][:note_index]):
                    allowed_families=[f for f in allowed_families if f!=2]
                if not tap_allowed[int(rep['button_start'][note_index])]:
                    allowed_families=[f for f in allowed_families if f!=0]
                if not where_slide_can_start(rep['button_start'][note_index]):
                    allowed_families=[f for f in allowed_families if f!=2]
                if launch_heads:
                    from itertools import product
                    prefix=tuple(map(int,rep['button_family'][:note_index]))
                    future_lanes=rep['button_start'][note_index+1:arity]
                    def possible(lane,f):
                        if f==0:return bool(tap_allowed[int(lane)])
                        if f==1:return bool(duration_ids_for_family(snapshot,1,len(vocab['durations'])))
                        return not snapshot.get('_forbid_unplanned_slide') and where_slide_can_start(lane) and bool(duration_ids_for_family(snapshot,2,len(vocab['durations'])))
                    tails=list(product(*[tuple(f for f in (0,1,2) if possible(lane,f)) for lane in future_lanes]))
                    allowed_families=[f for f in allowed_families if any(
                        (not snapshot.get('_single_slide_contract') or (prefix+(f,)+tail).count(2)<=1)
                        and HAND_ACCOUNTING.assignment_fits(snapshot,rep['button_start'][:arity],prefix+(f,)+tail) for tail in tails)]
                family = choose_allowed(
                    family_logits, allowed_families, sampling_temperature, rng, sampling_top_p
                )
            rep["button_family"][note_index] = family
        else:
            family = int(rep["button_family"][note_index])
        family_condition = one_value(head.family_embedding, family)
        if active_note:
            start = int(rep["button_start"][note_index])
        else:
            start = int(rep["button_start"][note_index])
        start_condition = one_value(head.start_embedding, start)
        if active_note and family == 2:
            route_logits = head.route_head[note_index](
                note_hidden + family_condition + start_condition
            )[0, -1].float()
            cache_key = (int(version_id), int(start))
            if route_cache is not None and cache_key in route_cache:
                supported_routes = route_cache[cache_key]
                route_mask = snapshot.get("allowedSlideRouteMask")
                allowed_routes = (
                    supported_routes
                    if route_mask is None
                    else tuple(index for index in supported_routes if bool(route_mask[index]))
                )
            else:
                supported_routes = allowed_route_ids(
                    vocab, version_id, start, route_support
                )
                if route_cache is not None:
                    route_cache[cache_key] = supported_routes
                route_mask = snapshot.get("allowedSlideRouteMask")
                allowed_routes = (
                    supported_routes
                    if route_mask is None
                    else tuple(index for index in supported_routes if bool(route_mask[index]))
                )
            from .sequence_runtime import choose_route
            route = choose_route(route_logits,allowed_routes,start,snapshot,sampling_temperature,rng,sampling_top_p)
            rep["button_route"][note_index] = route
        else:
            route = int(rep["button_route"][note_index])
        route_condition = one_value(head.route_embedding, route)
        if active_note and family in (1, 2):
            duration_logits = head.duration_head[note_index](
                note_hidden + family_condition + start_condition + route_condition
            )[0, -1].float()
            if family==2 and '_duration_model_ids' in vocab:
                duration_logits=duration_logits[torch.tensor(vocab['_duration_model_ids'],device=device)]
                if '_where_duration_bias' in snapshot:
                    duration_logits=duration_logits+torch.as_tensor(snapshot['_where_duration_bias'],device=device)
            allowed_durations = duration_ids_for_family(
                snapshot, family, len(vocab["durations"])
            )
            if family==2:
                known=snapshot.get('_where_fixed_cue_lanes',{})
                allowed_durations=[d for d in allowed_durations if d not in known or start in known[d]]
                future=snapshot.get('allowedSlideHeadDurationMask')
                if future is not None:allowed_durations=[d for d in allowed_durations if future[start,d]]
            if not allowed_durations:
                raise ValueError(f"provider supplied no duration for family {family}")
            duration = choose_allowed(
                duration_logits,
                allowed_durations,
                sampling_temperature,
                rng,
                sampling_top_p,
            )
            rep["button_duration"][note_index] = duration
        else:
            duration = int(rep["button_duration"][note_index])
        model_duration=int(vocab['_duration_model_ids'][duration]) if '_duration_model_ids' in vocab else duration
        duration_condition = one_value(head.duration_embedding, model_duration)
        if active_note:
            modifier_logits = head.modifier_head[note_index](
                note_hidden
                + family_condition
                + start_condition
                + route_condition
                + duration_condition
            )[0, -1].float()
            probability = modifier_logits.sigmoid().detach().cpu().numpy()
            bits = sum(
                (1 << bit)
                for bit in range(5)
                if rng.random() < float(probability[bit])
            )
            bits &= 0b00011
            if family == 2:
                # For the singular-track V4 factor model these modifiers are
                # properties of the incoming Star head, not the Slide track.
                bits = sanitize_slide_head_modifiers(version_id, bits)
            else:
                if version_id < 13:
                    bits &= ~0b00010
                if family == 1 and version_id < 19:
                    bits &= ~0b00001
            rep["button_modifiers"][note_index] = bits
        summary = summary + family_condition + start_condition + route_condition + duration_condition

    event_context = running + summary
    touch_group_budget = max(
        0, HAND_ACCOUNTING.remaining_hands(snapshot,
            rep['button_start'][:int(rep['button_arity'])],tuple(map(int,rep['button_family'][:int(rep['button_arity'])])))
    )
    touch_active = np.zeros(head.c.touch_positions, dtype=np.bool_)
    names = vocab['touchPositions']
    sensor_allowed_np = np.asarray([touch_sensor_allowed(version_id,n) for n in names],np.bool_)
    covered_np = np.asarray(snapshot.get('allowedTouchPresenceMask',np.zeros(len(names),np.bool_)),np.bool_) & sensor_allowed_np
    adjacency = getattr(state,'tables',{}).get('touchAdjacency')
    def count_groups(mask):
        return component_count(mask,names,adjacency) if adjacency is not None else state.touch_group_count(mask)
    if intent_override is not None:
        intent=EventIntent.from_representation(intent_override);count=intent.touch_count
        if count and not touch_enabled(version_id):
            raise ValueError(f"Touch is unavailable before maimai DX (version {version_id})")
        if count:
            logits=head.touch_presence(event_context)[0,-1].float()
            if count>int(sensor_allowed_np.sum()):
                raise ValueError(f"Touch count {count} exceeds version-{version_id} sensor capacity")
            logits=logits.masked_fill(~torch.as_tensor(sensor_allowed_np,device=device),-torch.inf)
            probabilities=torch.softmax(logits/max(.1,sampling_temperature),-1).cpu().numpy().astype(np.float64)
            probabilities=np.maximum(probabilities,0.0);probabilities/=probabilities.sum()
            for _ in range(32):
                selected=rng.choice(len(probabilities),size=count,replace=False,p=probabilities)
                candidate=np.zeros(len(probabilities),np.bool_);candidate[selected]=True
                if count_groups(candidate&~covered_np)<=touch_group_budget:
                    touch_active=candidate;break
            else:touch_active=candidate
    elif version_id >= 13:
        if touch_group_budget or covered_np.any():
            touch_prob=head.touch_presence(event_context)[0,-1].float().sigmoid()
            touch_probability=(touch_prob*torch.as_tensor(sensor_allowed_np,device=device).float()).detach().cpu().numpy()
            for _ in range(32):
                candidate=rng.random(len(touch_probability))<touch_probability
                if count_groups(candidate&~covered_np)<=touch_group_budget:
                    touch_active=candidate;break
    touch_output=None
    if touch_active.any():
        sensors=torch.arange(head.c.touch_positions,device=device)[None,None]
        touch_hidden=event_context[...,None,:]+head.touch_sensor(sensors)
        hold_logits=head.touch_hold_presence(touch_hidden)[0,-1].squeeze(-1).float()
        duration_logits=head.touch_duration(touch_hidden)[0,-1].float()
        modifier_logits=head.touch_modifiers(touch_hidden)[0,-1].float()
        scale=max(0.,float(touch_hold_scale)) if touch_hold_enabled(version_id) else 0.
        hold_probs=(hold_logits+math.log(scale)).sigmoid() if scale>0 else torch.zeros_like(hold_logits)
        packed=torch.cat((hold_probs[:,None],duration_logits,modifier_logits.sigmoid()),dim=1).detach().cpu().numpy()
        nv=duration_logits.shape[-1]
        touch_output={'hold_probability':packed[:,0],'touch_duration':packed[:,1:1+nv],'modifier_probability':packed[:,1+nv:]}

    notes = []
    for note_index in range(int(rep["button_arity"])):
        family = int(rep["button_family"][note_index])
        start = int(rep["button_start"][note_index])
        route = int(rep["button_route"][note_index])
        duration = int(rep["button_duration"][note_index])
        bits = int(rep["button_modifiers"][note_index])
        sensor = str(start + 1)
        modifiers = ("b" if bits & 1 else "") + ("x" if bits & 2 else "")
        if family == 0:
            notes.append(sensor + modifiers)
        elif family == 1:
            notes.append(sensor + "h" + modifiers + f"[{vocab['durations'][duration]}]")
        elif family == 2:
            notes.append(sensor + modifiers + vocab["routes"][route] + f"[{vocab['durations'][duration]}]")

    if version_id >= 13 and (not tail_cooldown or intent_override is not None):
        touch_durations: dict[int, int] = {}
        touch_hold_scores: dict[int, float] = {}
        active_touch_hold_sensors = set(snapshot.get("activeTouchHoldSensors", ()))
        for sensor_index in np.flatnonzero(touch_active).tolist():
            if touch_output is None:
                raise AssertionError("Touch details missing for an active sensor")
            sensor = vocab["touchPositions"][sensor_index]
            if getattr(head, "touch_hold_enabled", False):
                scale=max(0.0,float(touch_hold_scale)) if touch_hold_enabled(version_id) else 0.0
                hold_probability=float(touch_output["hold_probability"][sensor_index])
                touch_hold_scores[sensor_index] = hold_probability
                duration = (
                    int(touch_output["touch_duration"][sensor_index, 2:].argmax()) + 2
                    if hold_probability >= float(getattr(head, "touch_hold_threshold", 0.5))
                    else 0
                )
            else:
                duration = int(touch_output["touch_duration"][sensor_index].argmax())
                duration = 0 if duration == 1 else duration
            if duration and sensor in active_touch_hold_sensors:
                duration = 0
            if duration and not touch_hold_sensor_allowed(version_id,sensor):
                duration = 0
            if duration and "allowedHoldDurationMask" in snapshot:
                allowed = duration_ids_for_family(snapshot, 1, len(vocab["durations"]))
                duration = (
                    allowed[int(touch_output["touch_duration"][sensor_index,allowed].argmax())]
                    if allowed
                    else 0
                )
            touch_durations[sensor_index] = duration
        hold_hand_budget = max(
            0, int(snapshot["holdAvailableHands"]) - int(rep["button_arity"])
        )
        while True:
            held = [
                vocab["touchPositions"][index]
                for index, duration in touch_durations.items()
                if duration
            ]
            if count_groups(np.asarray([name in held for name in names],np.bool_)) <= hold_hand_budget or not held:
                break
            drop = min(
                (index for index, duration in touch_durations.items() if duration),
                key=lambda index: touch_hold_scores.get(index, 0.0),
            )
            touch_durations[drop] = 0
        for sensor_index in np.flatnonzero(touch_active).tolist():
            sensor = vocab["touchPositions"][sensor_index]
            duration = touch_durations[sensor_index]
            modifier_probability = touch_output["modifier_probability"][sensor_index]
            bits = (
                (2 if rng.random() < float(modifier_probability[1]) else 0)
                | (8 if rng.random() < float(modifier_probability[3]) else 0)
            )
            rep["touch_presence"][sensor_index] = True
            rep["touch_duration"][sensor_index] = duration
            rep["touch_modifiers"][sensor_index] = bits
            modifier_text = ("x" if bits & 2 else "") + ("f" if bits & 8 else "")
            note = sensor + ("h" if duration else "") + modifier_text
            if duration:
                note += f"[{vocab['durations'][duration]}]"
            notes.append(note)
    text="/".join(notes)
    if score_candidates:snapshot.setdefault('_where_candidate_scores',{})[text]=model_score
    return text, rep


def model_metadata(metadata: dict) -> np.ndarray:
    metadata = {
        key: value for key, value in metadata.items()
        if key not in RUNTIME_METADATA_KEYS
    }
    return metadata_tokens({
        "maidataMetadata": metadata,
        "notesDesigner": {"id": 0, "name": "ChartTransformer AI"},
    })


def build_static_inputs(
    *,
    model,
    ticks: np.ndarray,
    mel: np.ndarray,
    structure: np.ndarray,
    bpm_ticks: np.ndarray,
    bpm_values: np.ndarray,
    version: int,
    slot: int,
    level: float,
    metadata: dict,
    style: np.ndarray,
    device: torch.device,
    group: int,
    prior_tick: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    audio = sample_positions(mel, ticks, bpm_ticks, bpm_values, FRAME_SECONDS)
    bars = np.clip(ticks // TPB, 0, len(structure) - 1)
    prior_tick = int(ticks[0]) if prior_tick is None else int(prior_tick)
    delta = np.clip(np.diff(np.r_[prior_tick, ticks]), 0, 1536)
    static = {
        "audio": torch.from_numpy(audio)[None].to(device),
        "structure": torch.from_numpy(structure[bars])[None].to(device),
        "tick": torch.from_numpy(ticks % TPB)[None].to(device),
        "delta": torch.from_numpy(delta.astype(np.int64))[None].to(device),
        "rhythm": torch.from_numpy(rhythm_classes(ticks))[None].to(device),
        "valid": torch.ones((1, len(ticks)), dtype=torch.bool, device=device),
        "version": torch.tensor([version], device=device),
        "difficulty": torch.tensor([slot - 2], device=device),
        "group": torch.tensor([group], device=device),
        "level": torch.tensor([round(level * 10)], device=device),
        "metadata": torch.from_numpy(model_metadata(metadata))[None].to(device),
        "style": torch.from_numpy(np.asarray(style, dtype=np.float32))[None].to(device),
    }
    condition = (
        model.version(static["version"].clamp(0, 31))
        + model.difficulty(static["difficulty"].clamp(0, 4))
        + model.level(static["level"].clamp(0, 200))
        + model._metadata(static["metadata"])
        + model.style(static["style"])
    )
    timing = model.tick(static["tick"]) + model.delta(static["delta"]) + model.rhythm(static["rhythm"])
    return static, condition, timing
