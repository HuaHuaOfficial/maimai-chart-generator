"""Native V4 structured factor sampling primitives.

This module is deliberately independent of the legacy inference module.  It
contains only the representation, mask and structured-head sampling pieces
needed by the runtime renderer.  Musical judgement is not implemented here:
the caller supplies provider snapshots and consumes provider batch verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from .intent import EventIntent

from chart_runtime.io.audio import FRAME_SECONDS
from chart_runtime.io.factors import SHAPE_RE, factor_event, note_route
from chart_runtime.io.timing import sample_positions
from chart_runtime.generator.tokens import metadata_tokens, rhythm_classes
from chart_runtime.generator.models.relational import (
    EMPTY_GEOMETRY_ID,
    geometry_candidate_index,
)


TPB = 384

# Native copy of the pinned static route-shape policy.  This is vocabulary
# support metadata, not a musical legality checker; candidate legality remains
# exclusively in the provider's CUDA batch verdict.
_SHAPE_POLICY = (
    (0, frozenset(("-", "<", ">", "^"))),
    (3, frozenset(("v", "p", "q", "s", "z", "V", "pp", "qq"))),
    (6, frozenset(("w",))),
)


def _allowed_shapes(version_id: int) -> frozenset[str]:
    result: set[str] = set()
    for minimum, shapes in _SHAPE_POLICY:
        if int(version_id) >= minimum:
            result.update(shapes)
    return frozenset(result)


class RepresentationEncodingError(ValueError):
    """The newest V4 factor representation cannot encode an event losslessly."""

    def __init__(self, message: str, *, text: str = "", field: str = ""):
        self.text = text
        self.field = field
        super().__init__(message)


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
    allowed_shapes = _allowed_shapes(int(version_id))
    compatibility = route_support.get(str(int(start) + 1), {})
    result = []
    for index, route in enumerate(vocab["routes"]):
        if index < 2:
            continue
        pair = compatibility.get(str(index))
        if pair is None or int(pair["minVersion"]) > int(version_id):
            continue
        shapes = SHAPE_RE.findall(str(route))
        if shapes and all(shape in allowed_shapes for shape in shapes):
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
) -> int:
    ids = np.asarray(list(allowed), np.int64)
    if len(ids) == 0:
        raise ValueError("no allowed factor ids")
    if len(ids) == 1:
        return int(ids[0])
    values = logits[torch.as_tensor(ids, device=logits.device)].float().detach().cpu().numpy()
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
    return [index for index in range(2, int(size)) if key not in snapshot or snapshot[key][index]]


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

    def choose_relational_geometry(requested_arity: int):
        if requested_arity <= 0:
            return requested_arity, None
        candidate_arities = head.geometry_candidate_arities
        candidate_masks = head.geometry_candidate_masks.bool()
        eligible = apply_geometry_start_mask(
            candidate_arities.eq(requested_arity), candidate_masks, snapshot, device
        )
        candidate_ids = torch.nonzero(eligible, as_tuple=True)[0].tolist()
        if not candidate_ids and requested_arity == 2 and intent_override is None:
            eligible = apply_geometry_start_mask(
                candidate_arities.eq(1), candidate_masks, snapshot, device
            )
            candidate_ids = torch.nonzero(eligible, as_tuple=True)[0].tolist()
            requested_arity = 1 if candidate_ids else 0
        if intent_hard_mask and intent_override is not None and candidate_ids:
            frozen_families = [
                int(value) for value in intent_override["button_family"][:requested_arity]
            ]
            active_hold_lanes = {int(value) - 1 for value in snapshot.get("holdLanes", ())}
            candidate_ids = [
                candidate
                for candidate in candidate_ids
                if not any(
                    note_index < len(frozen_families)
                    and frozen_families[note_index] in (0, 1)
                    and int(start) in active_hold_lanes
                    for note_index, start in enumerate(
                        torch.nonzero(candidate_masks[candidate], as_tuple=True)[0].tolist()
                    )
                )
            ]
        if not candidate_ids:
            return 0, None
        selected = choose_allowed(
            geometry_logits, candidate_ids, sampling_temperature, rng, sampling_top_p
        )
        starts = torch.nonzero(candidate_masks[selected], as_tuple=True)[0].tolist()
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
            range(min(int(snapshot['holdAvailableHands']),int(snapshot.get('maxOuterArity',2))) + 1),
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
                allowed_families=sorted(set(remaining_intent_families))
                family=choose_allowed(family_logits,allowed_families,sampling_temperature,rng,sampling_top_p)
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
            route = choose_allowed(
                route_logits,
                allowed_routes,
                sampling_temperature,
                rng,
                sampling_top_p,
            )
            rep["button_route"][note_index] = route
        else:
            route = int(rep["button_route"][note_index])
        route_condition = one_value(head.route_embedding, route)
        if active_note and family in (1, 2):
            duration_logits = head.duration_head[note_index](
                note_hidden + family_condition + start_condition + route_condition
            )[0, -1].float()
            allowed_durations = duration_ids_for_family(
                snapshot, family, len(vocab["durations"])
            )
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
        duration_condition = one_value(head.duration_embedding, duration)
        if active_note:
            modifier_logits = head.modifier_head[note_index](
                note_hidden
                + family_condition
                + start_condition
                + route_condition
                + duration_condition
            )[0, -1].float()
            probability = modifier_logits.sigmoid()
            bits = sum(
                (1 << bit)
                for bit in range(5)
                if rng.random() < float(probability[bit])
            )
            bits &= 0b00011
            if version_id < 13:
                bits &= ~0b00010
            if family in (1, 2) and version_id < 19:
                bits &= ~0b00001
            rep["button_modifiers"][note_index] = bits
        summary = summary + family_condition + start_condition + route_condition + duration_condition

    event_context = running + summary
    touch_group_budget = max(
        0, int(snapshot["holdAvailableHands"]) - int(rep["button_arity"])
    )
    touch_active = torch.zeros(head.c.touch_positions, dtype=torch.bool, device=device)
    if intent_override is not None:
        intent=EventIntent.from_representation(intent_override)
        count=intent.touch_count
        if count:
            # WHAT fixes only cardinality. WHERE resamples sensors from its
            # learned logits; no sensor bit mask crosses the intent boundary.
            logits=head.touch_presence(event_context)[0,-1].float()
            probabilities=torch.softmax(logits/max(.1,sampling_temperature),-1).cpu().numpy().astype(np.float64)
            probabilities=np.maximum(probabilities,1e-12);probabilities/=probabilities.sum()
            covered_touch=torch.as_tensor(snapshot.get('allowedTouchPresenceMask',np.zeros(head.c.touch_positions,np.bool_)),device=device,dtype=torch.bool)
            covered_np=covered_touch.cpu().numpy()
            for _ in range(32):
                selected=rng.choice(len(probabilities),size=count,replace=False,p=probabilities)
                candidate=np.zeros(len(probabilities),np.bool_);candidate[selected]=True
                if state.touch_group_count(candidate&~covered_np)<=touch_group_budget:
                    touch_active=torch.from_numpy(candidate).to(device);break
            else:
                # Preserve WHAT cardinality. Harness will reject this WHERE
                # realization and the next realization resamples sensors.
                touch_active=torch.from_numpy(candidate).to(device)
    elif version_id >= 13:
        covered_touch = torch.as_tensor(snapshot.get('allowedTouchPresenceMask',np.zeros(head.c.touch_positions,np.bool_)),device=device,dtype=torch.bool)
        if not touch_group_budget and not bool(covered_touch.any()):
            touch_probability = None
        else:
            touch_prob = head.touch_presence(event_context)[0, -1].float().sigmoid()
            touch_probability=touch_prob.detach().cpu().numpy()
        if touch_probability is not None:
            covered_np=covered_touch.detach().cpu().numpy()
            for _ in range(32):
                candidate = rng.random(len(touch_probability)) < touch_probability
                independent=candidate&~covered_np
                if state.touch_group_count(independent) <= touch_group_budget:
                    touch_active = torch.from_numpy(candidate).to(device)
                    break
    touch_output = None
    if bool(touch_active.any()):
        sensors = torch.arange(head.c.touch_positions, device=device)[None, None]
        touch_hidden = event_context[..., None, :] + head.touch_sensor(sensors)
        touch_output = {
            "touch_hold_presence": head.touch_hold_presence(touch_hidden)[0, -1].squeeze(-1).float(),
            "touch_duration": head.touch_duration(touch_hidden)[0, -1].float(),
            "touch_modifiers": head.touch_modifiers(touch_hidden)[0, -1].float(),
        }

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
        for sensor_index in touch_active.nonzero(as_tuple=True)[0].tolist():
            if touch_output is None:
                raise AssertionError("Touch details missing for an active sensor")
            sensor = vocab["touchPositions"][sensor_index]
            if getattr(head, "touch_hold_enabled", False):
                hold_probability = float(
                    touch_output["touch_hold_presence"][sensor_index].sigmoid()
                )
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
            if duration and sensor != "C" and version_id < 24:
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
            if state.touch_hold_hand_count(held) <= hold_hand_budget or not held:
                break
            drop = min(
                (index for index, duration in touch_durations.items() if duration),
                key=lambda index: touch_hold_scores.get(index, 0.0),
            )
            touch_durations[drop] = 0
        for sensor_index in touch_active.nonzero(as_tuple=True)[0].tolist():
            sensor = vocab["touchPositions"][sensor_index]
            duration = touch_durations[sensor_index]
            modifier_probability = touch_output["touch_modifiers"][sensor_index].sigmoid()
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
    return "/".join(notes), rep


def model_metadata(metadata: dict) -> np.ndarray:
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
