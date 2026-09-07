"""Native latest-V4 renderer backed by the caller-owned CUDA Harness.

The runtime deliberately does not import the legacy inference module.  Model
conditioning, causal chunking and structured sampling live in this native
path; the Harness remains the only source of candidate musical judgements.
"""

from __future__ import annotations
from .intent import EventIntent,IntentChoices

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from chart_runtime.io.timing import ticks_to_seconds
from chart_runtime.generator.models.motion import motion_state_from_representations
from chart_runtime.generator.models.relational import EMPTY_GEOMETRY_ID, geometry_candidate_index

from .sampling import (
    TPB,
    RepresentationEncodingError,
    build_static_inputs,
    copy_representation,
    decode_structured_factor_event_fast,
    empty_representation,
    intent_signature,
    load_route_support,
    preload_route_ids,
    text_representation,
)


class HarnessSamplingProvider(Protocol):
    tables: Any
    max_cuda_matrix_elements: int
    decisions: Sequence[Mapping[str, Any]]

    def snapshot(self, tick: int, enforce_recent: bool) -> dict: ...
    def update(self, tick: int, representation: dict, bpm: float) -> None: ...
    def commit(self, representation: dict, moment: float, bpm: float) -> None: ...
    def check_batch(self, representations: Sequence[dict], moment: float, bpm: float): ...
    def touch_group_count(self, active) -> int: ...
    def touch_hold_hand_count(self, sensors) -> int: ...
    def bind_references(self, reference_plan: Mapping[int, str]) -> None: ...


@dataclass(frozen=True)
class RenderContext:
    root: Path
    ticks: Sequence[int]
    mel: np.ndarray
    structure: np.ndarray
    bt: np.ndarray
    bv: np.ndarray
    version: int
    slot: int
    level: float
    metadata: dict
    device: torch.device
    checkpoint: Path
    style: np.ndarray
    temperature: float
    seed: int
    factor_session: Any = None
    progress: Callable[[str], None] | None = None
    cache: Any = None


class RenderFailure(RuntimeError):
    """Rich render failure preserving stage, tick, cause and provider context."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "render",
        tick: int | None = None,
        slot: int | None = None,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ):
        self.stage = str(stage)
        self.tick = None if tick is None else int(tick)
        self.slot = None if slot is None else int(slot)
        self.details = dict(details or {})
        self.cause = cause
        location = []
        if self.slot is not None:
            location.append(f"slot={self.slot}")
        if self.tick is not None:
            location.append(f"tick={self.tick}")
        suffix = f" ({', '.join(location)})" if location else ""
        detail_text = f" details={self.details!r}" if self.details else ""
        super().__init__(f"{self.stage}{suffix}: {message}{detail_text}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tick": self.tick,
            "slot": self.slot,
            "message": str(self),
            "details": dict(self.details),
            "causeType": type(self.cause).__name__ if self.cause else None,
            "causeMessage": str(self.cause) if self.cause else None,
        }


_CHECKPOINT_FLAGS_CACHE: dict[tuple[str, int, int], dict] = {}


def _target_ticks(
    value: Sequence[int] | Mapping[int, Any] | None,
    fallback: Sequence[int],
) -> list[int]:
    raw = fallback if value is None else value.keys() if isinstance(value, Mapping) else value
    try:
        result = [int(tick) for tick in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("targets must be an ordered sequence of integer ticks") from exc
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError("targets must be strictly increasing")
    return result


def _reference_texts(references: Mapping[int, Any] | None) -> dict[int, str]:
    if references is None:
        return {}
    if not isinstance(references, Mapping):
        raise TypeError("references must be a tick-keyed mapping")
    result: dict[int, str] = {}
    for tick, value in references.items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, Mapping) and isinstance(value.get("text"), str):
            text = value["text"]
        else:
            raise TypeError(f"references[{tick!r}] must contain chart text")
        result[int(tick)] = text
    return result


def _check_provider(provider: Any) -> None:
    required = (
        "snapshot", "update", "commit", "check_batch",
        "touch_group_count", "touch_hold_hand_count",
    )
    missing = [name for name in required if not callable(getattr(provider, name, None))]
    if missing:
        raise TypeError("Harness provider missing methods: " + ", ".join(missing))
    if not hasattr(provider, "tables"):
        raise TypeError("Harness provider must expose tables")
    if not hasattr(provider, "max_cuda_matrix_elements"):
        raise TypeError("Harness provider must expose max_cuda_matrix_elements")


def _provider_result(value) -> tuple[bool, str, str]:
    if isinstance(value, Mapping):
        severity = str(value.get("severity", "CLEAN")).upper()
        return (
            bool(value.get("ok", severity != "HARD")),
            str(value.get("reason", "")),
            str(value.get("detail", value.get("evidence_status", "harness"))),
        )
    try:
        ok, reason, detail = value
    except (TypeError, ValueError) as exc:
        raise TypeError("provider.check_batch must return triples or verdict mappings") from exc
    return bool(ok), str(reason), str(detail)


def _provider_decisions(provider: Any, count: int) -> list[dict]:
    values = getattr(provider, "decisions", None)
    if not isinstance(values, (list, tuple)) or len(values) != int(count):
        raise RuntimeError(
            f"provider decisions length {len(values) if isinstance(values, (list, tuple)) else 'missing'} "
            f"does not match check_batch size {count}"
        )
    result = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("provider.decisions must contain mappings")
        decision = dict(value)
        severity = str(decision.get("severity", "")).upper()
        if severity not in {"CLEAN", "SOFT", "HARD"}:
            raise ValueError(f"invalid provider severity {severity!r}")
        decision["severity"] = severity
        decision["evidence_status"] = str(decision.get("evidence_status", "harness"))
        decision["replace_if_alternative"] = bool(decision.get("replace_if_alternative", False))
        decision["reason"] = str(decision.get("reason", ""))
        result.append(decision)
    return result


def _locate_route_support(root: Path) -> dict:
    candidate = Path(root) / "models" / "v2"
    if not (candidate / "route_compatibility.json").is_file():
        raise FileNotFoundError(
            f"pinned native V4 route support table is missing: {candidate / 'route_compatibility.json'}"
        )
    return load_route_support(candidate)


def _validate_latest_session(context: RenderContext):
    if context.factor_session is None or not isinstance(context.factor_session, (tuple, list)):
        raise ValueError("native V4 renderer requires a resident factor_session")
    if len(context.factor_session) < 2:
        raise ValueError("factor_session must contain (model, vocab)")
    checkpoint_path = Path(context.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"native V4 checkpoint does not exist: {checkpoint_path}")
    stat = checkpoint_path.stat()
    cache_key = (str(checkpoint_path.resolve()).lower(), int(stat.st_size), int(stat.st_mtime_ns))
    checkpoint = _CHECKPOINT_FLAGS_CACHE.get(cache_key)
    if checkpoint is None:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint = {
            "modelClass": loaded.get("modelClass"),
            "geometrySequenceState": bool(loaded.get("geometrySequenceState", False)),
            "persistentMotionMemory": bool(loaded.get("persistentMotionMemory", False)),
            "contextualPipeline": bool(loaded.get("contextualPipeline", False)),
        }
        _CHECKPOINT_FLAGS_CACHE.clear()
        _CHECKPOINT_FLAGS_CACHE[cache_key] = checkpoint
    if checkpoint.get("modelClass") != "V4RelationalRenderer":
        raise ValueError(
            "native renderer accepts only V4RelationalRenderer; V3/Factor fallback is disabled"
        )
    if not bool(checkpoint.get("geometrySequenceState", False)):
        raise ValueError("checkpoint is not the newest geometry-sequence V4 renderer")
    if not bool(checkpoint.get("contextualPipeline", False)):
        raise ValueError("checkpoint is not the pinned native contextual workflow")
    model, vocab = context.factor_session[:2]
    if not hasattr(model, "audio_encoder") or not hasattr(model, "event_decoder"):
        raise TypeError("resident model lacks native Transformer encoder/decoder")
    if int(getattr(model.c, "sequence_length", -1)) != 96:
        raise ValueError("native V4 renderer requires sequence_length=96")
    if not bool(getattr(model, "geometry_sequence_state", False)):
        raise ValueError("resident model geometry_sequence_state disagrees with checkpoint")
    expected_motion_memory = bool(checkpoint.get("persistentMotionMemory", False))
    if bool(getattr(model, "persistent_motion_memory", False)) != expected_motion_memory:
        raise ValueError("resident model persistent_motion_memory disagrees with checkpoint")
    if not callable(getattr(model, "motion_step", None)) or not callable(
        getattr(model, "_state_embedding", None)
    ):
        raise TypeError("resident V4 model lacks native motion/state path")
    heads = getattr(model, "heads", None)
    if heads is None or not len(heads):
        raise TypeError("resident V4 model has no structured heads")
    for head in heads:
        if not bool(getattr(head, "geometry_sequence_state", False)):
            raise ValueError("all newest V4 structured heads must enable geometry_sequence_state")
        if not callable(getattr(head, "geometry_sequence_residual", None)):
            raise TypeError("newest V4 head lacks geometry_sequence_residual")
    return model, vocab, checkpoint


def _prefix(
    static: dict[str, torch.Tensor],
    previous_arrays: dict[str, np.ndarray],
    occupied: np.ndarray,
    blocked: np.ndarray,
    available: np.ndarray,
    conservative: np.ndarray,
    active_holds: np.ndarray,
    active_slides: np.ndarray,
    active_touch_holds: np.ndarray,
    local: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prefix = {
        key: value[:, : local + 1]
        for key, value in static.items()
        if value.ndim >= 2 and key != "metadata"
    }
    prefix.update(
        {key: value for key, value in static.items() if value.ndim < 2 or key == "metadata"}
    )
    for name, value in previous_arrays.items():
        prefix[name] = torch.from_numpy(value[:, : local + 1]).to(device)
    prefix["occupied_lanes"] = torch.from_numpy(occupied[:, : local + 1]).to(device)
    prefix["blocked_lanes"] = torch.from_numpy(blocked[:, : local + 1]).to(device)
    prefix["available_hands"] = torch.from_numpy(available[:, : local + 1]).to(device)
    prefix["conservative_available_hands"] = torch.from_numpy(
        conservative[:, : local + 1]
    ).to(device)
    prefix["active_holds"] = torch.from_numpy(active_holds[:, : local + 1]).to(device)
    prefix["active_slides"] = torch.from_numpy(active_slides[:, : local + 1]).to(device)
    prefix["active_touch_holds"] = torch.from_numpy(
        active_touch_holds[:, : local + 1]
    ).to(device)
    return prefix


def _shared_head_state(
    model,
    head,
    hidden: torch.Tensor,
    previous: dict,
    current_motion_state: np.ndarray,
    geometry_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    shared_adapted = head.adapter(hidden[:, -1:])
    target = {}
    for key in ("button_arity", "button_start", "button_route"):
        value = torch.as_tensor(previous[key], device=hidden.device)
        target[f"prev_{key}"] = value.long().reshape(1, 1, *value.shape)
    motion = torch.as_tensor(
        current_motion_state, device=hidden.device, dtype=hidden.dtype
    ).reshape(1, 1, -1)
    shared_geometry = head.geometry_logits(
        hidden[:, -1:], target, motion_state_override=motion
    )
    previous_arity = torch.as_tensor(
        previous["button_arity"], device=hidden.device, dtype=torch.long
    ).reshape(1, 1)
    previous_start = torch.as_tensor(
        previous["button_start"], device=hidden.device, dtype=torch.long
    ).reshape(1, 1, 2)
    previous_geometry = geometry_candidate_index(previous_start, previous_arity).masked_fill(
        previous_arity.eq(0), EMPTY_GEOMETRY_ID
    )
    residual, next_geometry_state = head.geometry_sequence_residual(
        hidden[:, -1:], previous_geometry, motion, geometry_state
    )
    return shared_adapted, shared_geometry + residual, next_geometry_state


def _candidate(
    *,
    head,
    hidden: torch.Tensor,
    vocab: dict,
    version: int,
    rng: np.random.Generator,
    provider,
    snapshot: dict,
    route_support: dict,
    route_cache: dict[tuple[int, int], tuple[int, ...]],
    temperature: float,
    shared_adapted: torch.Tensor,
    shared_geometry: torch.Tensor,
    intent: dict | None,
) -> tuple[str, dict]:
    text, rep = decode_structured_factor_event_fast(
        head,
        hidden[:, -1:],
        vocab,
        version,
        temperature,
        rng,
        provider,
        snapshot,
        route_support,
        route_cache=route_cache,
        adapted_override=shared_adapted,
        relational_logits_override=shared_geometry,
        intent_override=intent,
        intent_hard_mask=True,
    )
    if text:
        encoded = text_representation(text, vocab)
    else:
        encoded = empty_representation(len(vocab["touchPositions"]))
    if intent is not None and intent_signature(encoded) != intent_signature(intent):
        raise RepresentationEncodingError(
            "structured sampler changed an explicit event intent",
            text=text,
            field="intent_signature",
        )
    return text, encoded


def _sample_anchor(
    *,
    tick: int,
    moment: float,
    bpm: float,
    model,
    head,
    hidden: torch.Tensor,
    vocab: dict,
    version: int,
    provider,
    snapshot: dict,
    route_support: dict,
    temperature: float,
    rng: np.random.Generator,
    planned_intent: dict | None,
    allow_intent_revision: bool,
    shared_adapted: torch.Tensor,
    shared_geometry: torch.Tensor,
    route_cache: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[str, dict, dict]:
    options=(planned_intent.candidates if isinstance(planned_intent,IntentChoices)
             else (EventIntent.from_representation(planned_intent),) if planned_intent is not None else ())
    max_attempts=8 if isinstance(planned_intent,IntentChoices) else 32
    fixed_intent = options[0] if options else None
    revisions = 0
    batch_sizes: list[int] = []
    last_decisions: list[dict] = []
    last_candidates: list[dict] = []
    same_intent_attempts = 0
    best_soft: tuple[str, dict, dict] | None = None

    def partial_context() -> dict:
        def compact(value):
            if isinstance(value, np.ndarray):
                return {
                    "shape": list(value.shape),
                    "trueCount": int(value.astype(bool).sum()) if value.dtype == np.bool_ else None,
                }
            if isinstance(value, (set, tuple, list)):
                return list(value)[:16]
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return type(value).__name__

        return {
            "severity": "HARD",
            "intentRevision": bool(planned_intent is not None),
            "allowIntentRevision": bool(allow_intent_revision),
            "sameIntentAttempts": int(same_intent_attempts),
            "revisionRounds": int(revisions),
            "batchSizes": list(batch_sizes),
            "lastDecisions": [dict(item) for item in last_decisions],
            "lastCandidates": list(last_candidates),
            "providerTick": compact(getattr(provider, "tick", None)),
            "providerMaxCudaMatrixElements": compact(
                getattr(provider, "max_cuda_matrix_elements", None)
            ),
            "snapshot": {
                key: compact(snapshot.get(key))
                for key in (
                    "holdLanes",
                    "activeTouchHoldSensors",
                    "holdAvailableHands",
                    "availableHands",
                    "activeHoldHands",
                    "activeSlideHands",
                    "activeTouchHoldHands",
                    "allowedSlideDurationMask",
                    "allowedSlideRouteMask",
                    "allowedOuterStartMask",
                )
                if key in snapshot
            },
        }

    def record_batch(proposals, decisions, batch_size: int) -> None:
        nonlocal same_intent_attempts, last_decisions, last_candidates
        batch_sizes.append(int(batch_size))
        same_intent_attempts += int(batch_size)
        last_decisions = [dict(item) for item in decisions]
        last_candidates = [
            {"text": text[:160], "intent": repr(intent_signature(rep))}
            for text, rep in proposals
        ]

    while revisions < (len(options) if options and allow_intent_revision else 1 if options else 2):
        if options:fixed_intent=options[revisions]
        proposals: list[tuple[str, dict]] = []
        first_text, first_rep = _candidate(
            head=head,
            hidden=hidden,
            vocab=vocab,
            version=int(version),
            rng=rng,
            provider=provider,
            snapshot=snapshot,
            route_support=route_support,
            route_cache=route_cache,
            temperature=temperature,
            shared_adapted=shared_adapted,
            shared_geometry=shared_geometry,
            intent=fixed_intent,
        )
        # The provider owns version-independent judgement; the model version
        # is supplied by the caller through the closure below.
        proposals.append((first_text, first_rep))
        raw = provider.check_batch([first_rep], moment, bpm)
        _ = [_provider_result(value) for value in ([] if raw is None else list(raw))]
        decisions = _provider_decisions(provider, 1)
        record_batch(proposals, decisions, 1)
        if decisions[0]["severity"] == "CLEAN":
            return proposals[0][0], proposals[0][1], decisions[0]
        if decisions[0]["severity"] == "SOFT":
            best_soft = (proposals[0][0], proposals[0][1], decisions[0])
        if fixed_intent is None:
            fixed_intent = EventIntent.from_representation(first_rep)
        while len(proposals) < 4:
            proposals.append(
                _candidate(
                    head=head,
                    hidden=hidden,
                    vocab=vocab,
                    version=int(version),
                    rng=rng,
                    provider=provider,
                    snapshot=snapshot,
                    route_support=route_support,
                    route_cache=route_cache,
                    temperature=temperature,
                    shared_adapted=shared_adapted,
                    shared_geometry=shared_geometry,
                    intent=fixed_intent,
                )
            )
        raw = provider.check_batch([rep for _, rep in proposals], moment, bpm)
        _ = [_provider_result(value) for value in ([] if raw is None else list(raw))]
        decisions = _provider_decisions(provider, len(proposals))
        record_batch(proposals, decisions, len(proposals))
        for proposal, decision in zip(proposals, decisions):
            if decision["severity"] == "CLEAN":
                return proposal[0], proposal[1], decision
        for proposal, decision in zip(proposals, decisions):
            if decision["severity"] == "SOFT" and (best_soft is None or int(decision.get('soft_cost',0))<int(best_soft[2].get('soft_cost',0))):
                best_soft = (proposal[0],proposal[1],decision)
        while len(proposals) < max_attempts:
            batch = min(8, max_attempts - len(proposals))
            new = [
                _candidate(
                    head=head,
                    hidden=hidden,
                    vocab=vocab,
                    version=int(version),
                    rng=rng,
                    provider=provider,
                    snapshot=snapshot,
                    route_support=route_support,
                    route_cache=route_cache,
                    temperature=temperature,
                    shared_adapted=shared_adapted,
                    shared_geometry=shared_geometry,
                    intent=fixed_intent,
                )
                for _ in range(batch)
            ]
            raw = provider.check_batch([rep for _, rep in new], moment, bpm)
            _ = [_provider_result(value) for value in ([] if raw is None else list(raw))]
            new_decisions = _provider_decisions(provider, len(new))
            record_batch(new, new_decisions, len(new))
            for proposal, decision in zip(new, new_decisions):
                if decision["severity"] == "CLEAN":
                    return proposal[0], proposal[1], decision
            for proposal, decision in zip(new, new_decisions):
                if decision["severity"] == "SOFT" and (best_soft is None or int(decision.get('soft_cost',0))<int(best_soft[2].get('soft_cost',0))):
                    best_soft = (proposal[0],proposal[1],decision)
            proposals.extend(new)
        # SOFT means the requested WHAT is realizable.  It may trigger more
        # WHERE samples, but must never promote a lower-ranked WHAT.  Only an
        # all-HARD realization set proves this WHAT infeasible enough to move
        # to another IntentChoices candidate.
        if planned_intent is not None and best_soft is not None:
            return best_soft
        if planned_intent is not None and not allow_intent_revision:
            raise RenderFailure(
                "explicit intent exhausted 32 provider-judged realizations",
                stage="provider_sampling",
                tick=tick,
                details=partial_context(),
                )
        # Native infeasibility may redraw a fresh intent only after the full
        # same-intent budget.  It is never a silent change to an explicit plan.
        fixed_intent = None
        revisions += 1
    if best_soft is not None:
        return best_soft
    rest=empty_representation(len(vocab['touchPositions']))
    raw=provider.check_batch([rest],moment,bpm)
    _=[_provider_result(value) for value in ([] if raw is None else list(raw))]
    decision=_provider_decisions(provider,1)[0]
    if decision['severity']!='HARD':
        return '',rest,decision
    raise RenderFailure(
        "native intent search exhausted 32 fresh intents",
        stage="provider_sampling",
        tick=tick,
        details=partial_context(),
    )


def render(
    context: RenderContext,
    provider: HarnessSamplingProvider,
    *,
    targets: Sequence[int] | Mapping[int, Any] | None = None,
    references: Mapping[int, Any] | None = None,
    intent_plan: Mapping[int, dict] | None = None,
    route_masks: Mapping[int, Any] | None = None,
    duration_masks: Mapping[int, Any] | None = None,
    start_masks: Mapping[int, Any] | None = None,
    seed_offset: int = 0,
    allow_intent_revision: bool = False,
) -> dict[int, str]:
    """Run the native V4 causal renderer and return target tick -> Simai text."""

    try:
        _check_provider(provider)
        target_list = _target_ticks(targets, context.ticks)
        if not target_list:
            return {}
        if not isinstance(context.metadata, dict):
            raise TypeError("RenderContext.metadata must be a dict")
        model, vocab, _checkpoint = _validate_latest_session(context)
        route_support = _locate_route_support(Path(context.root))
        route_cache = preload_route_ids(vocab, int(context.version), route_support)
        reference_texts = _reference_texts(references)
        binder = getattr(provider, "bind_references", None)
        if callable(binder):
            binder(reference_texts)
        elif reference_texts:
            raise TypeError("provider must implement bind_references for references")
        if intent_plan is not None:
            outside = set(int(key) for key in intent_plan) - set(target_list)
            if outside:
                raise ValueError(f"intent_plan contains non-target ticks: {sorted(outside)[:8]}")

        target_set = set(target_list)
        reference_items: dict[int, tuple[str, dict]] = {}
        upper = max(target_list)
        for tick, text in reference_texts.items():
            if tick not in target_set and tick <= upper:
                try:
                    reference_items[tick] = (text, text_representation(text, vocab))
                except RepresentationEncodingError as exc:
                    raise RenderFailure(
                        str(exc),
                        stage="reference_encoding",
                        tick=tick,
                        slot=context.slot,
                        details={"unsupported": True, "field": exc.field, "text": text},
                        cause=exc,
                    ) from exc
        # Audio-encoder context is anchored to the full model schedule, not
        # truncated at the last edited point. This also makes prefix state
        # reuse valid when only a late dependency window changes.
        decode_ticks = sorted(target_set | set(reference_items) | set(map(int,context.ticks)))
        tick_seconds = ticks_to_seconds(
            np.asarray(decode_ticks, dtype=np.int64), context.bt, context.bv
        )
        rng = np.random.default_rng(int(context.seed) + int(seed_offset))
        group = {2: 0, 3: 1, 4: 2, 5: 3, 6: 3}.get(int(context.slot))
        if group is None:
            raise ValueError(f"unsupported V4 difficulty slot {context.slot}")
        representations: list[dict] = []
        emitted: dict[int, str] = {}
        motion_state = None
        geometry_state = None
        prototype = empty_representation(model.c.touch_positions)
        cache=context.cache if isinstance(context.cache,dict) else {}
        cache_key=(id(model),id(context.mel),id(context.structure),id(context.style),context.version,context.slot,context.level,
                   repr(sorted(context.metadata.items())),tuple(decode_ticks),tuple(map(int,context.bt)),tuple(map(float,context.bv)))
        if cache.get('key')!=cache_key:cache.clear();cache['key']=cache_key;cache['steps']=[];cache['memories']={}
        steps=cache['steps'];resume=0
        while resume<min(len(steps),len(decode_ticks)):
            tick=decode_ticks[resume]
            if tick in target_set or tick not in reference_items or steps[resume]['tick']!=tick or steps[resume]['text']!=reference_items[tick][0]:break
            resume+=1
        if resume:
            representations=[copy_representation(step['rep']) for step in steps[:resume]]
            geometry_state=steps[resume-1]['geometry'];motion_state=steps[resume-1]['motion']
            seed_history=getattr(provider,'seed_history',None)
            if not callable(seed_history):resume=0;representations=[];geometry_state=None;motion_state=None
            else:seed_history({step['tick']:step['text'] for step in steps[:resume] if step['text']})
        del steps[resume:]
        cache['last_reused_prefix']=resume;cache['last_forward_frames']=0
        work_done=0

        for chunk_start in range(0, len(decode_ticks), 96):
            if chunk_start+96<=resume:continue
            if decode_ticks[chunk_start]>upper:break
            chunk_ticks = np.asarray(decode_ticks[chunk_start : chunk_start + 96], dtype=np.int64)
            previous_tick = decode_ticks[chunk_start - 1] if chunk_start else int(chunk_ticks[0])
            static, condition, timing = build_static_inputs(
                model=model,
                ticks=chunk_ticks,
                mel=context.mel,
                structure=context.structure,
                bpm_ticks=context.bt,
                bpm_values=context.bv,
                version=int(context.version),
                slot=int(context.slot),
                level=float(context.level),
                metadata=context.metadata,
                style=context.style,
                device=context.device,
                group=group,
                prior_tick=previous_tick,
            )
            length = len(chunk_ticks)
            memory_input = model.audio(static["audio"]) + model.structure(static["structure"])
            memory_input = memory_input + timing + condition[:, None]
            if chunk_start in cache['memories']:
                memory=cache['memories'][chunk_start]
            else:
                with torch.inference_mode(), torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.float16,
                    enabled=context.device.type == "cuda",
                ):
                    memory = model.audio_encoder(memory_input)
                cache['memories'][chunk_start]=memory

            previous_arrays = {}
            occupied = np.zeros((1, length, 8), np.float32)
            blocked = np.zeros((1, length, 8), np.float32)
            available = np.full((1, length), 2, np.int64)
            conservative = np.full((1, length), 2, np.int64)
            active_holds = np.zeros((1, length), np.int64)
            active_slides = np.zeros((1, length), np.int64)
            active_touch_holds = np.zeros((1, length), np.int64)
            for name, value in prototype.items():
                previous_arrays[f"prev_{name}"] = np.zeros(
                    (1, length, *np.shape(value)), dtype=np.asarray(value).dtype
                )

            for local, absolute_tick in enumerate(chunk_ticks.tolist()):
                absolute_tick = int(absolute_tick)
                if absolute_tick>upper:break
                global_index=chunk_start+local
                if global_index<resume:
                    snapshot=steps[global_index]['snapshot']
                    for lane in snapshot.get('holdLanes',()):occupied[0,local,int(lane)-1]=1.
                    for lane in snapshot.get('riskLanes',()):blocked[0,local,int(lane)-1]=1.
                    available[0,local]=snapshot['holdAvailableHands'];conservative[0,local]=snapshot['availableHands']
                    active_holds[0,local]=snapshot['activeHoldHands'];active_slides[0,local]=snapshot['activeSlideHands'];active_touch_holds[0,local]=snapshot['activeTouchHoldHands']
                    previous=steps[global_index-1]['rep'] if global_index else prototype
                    for name,value in previous.items():previous_arrays[f'prev_{name}'][0,local]=value
                    continue
                absolute_time = float(tick_seconds[chunk_start + local])
                cache['last_forward_frames']+=1
                bpm_index = max(
                    0,
                    int(np.searchsorted(context.bt, absolute_tick, side="right") - 1),
                )
                current_bpm = float(context.bv[bpm_index])
                snapshot = provider.snapshot(absolute_tick, enforce_recent=True)
                if duration_masks is not None and absolute_tick in duration_masks:
                    snapshot["allowedSlideDurationMask"] = duration_masks[absolute_tick]
                if route_masks is not None and absolute_tick in route_masks:
                    snapshot["allowedSlideRouteMask"] = route_masks[absolute_tick]
                if start_masks is not None and absolute_tick in start_masks:
                    snapshot["allowedOuterStartMask"] = start_masks[absolute_tick]
                for lane in snapshot.get("holdLanes", ()):
                    occupied[0, local, int(lane) - 1] = 1.0
                for lane in snapshot.get("riskLanes", ()):
                    blocked[0, local, int(lane) - 1] = 1.0
                available[0, local] = int(snapshot["holdAvailableHands"])
                conservative[0, local] = int(snapshot["availableHands"])
                active_holds[0, local] = int(snapshot["activeHoldHands"])
                active_slides[0, local] = int(snapshot["activeSlideHands"])
                active_touch_holds[0, local] = int(snapshot["activeTouchHoldHands"])
                previous = representations[-1] if representations else prototype
                for name, value in previous.items():
                    previous_arrays[f"prev_{name}"][0, local] = value
                current_motion_state = motion_state_from_representations(representations)
                prefix = _prefix(
                    static,
                    previous_arrays,
                    occupied,
                    blocked,
                    available,
                    conservative,
                    active_holds,
                    active_slides,
                    active_touch_holds,
                    local,
                    context.device,
                )
                with torch.inference_mode(), torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.float16,
                    enabled=context.device.type == "cuda",
                ):
                    target_hidden = (
                        model._previous_event(prefix)
                        + timing[:, : local + 1]
                        + condition[:, None]
                        + model.occupied_lanes(prefix["occupied_lanes"].float())
                        + model.available_hands(prefix["available_hands"].long())
                        + model._state_embedding(prefix)
                    )
                    causal = torch.ones(
                        (local + 1, local + 1), device=context.device, dtype=torch.bool
                    ).triu(1)
                    hidden = model.event_decoder(
                        target_hidden,
                        memory,
                        tgt_mask=causal,
                        tgt_is_causal=True,
                    )[:, -1:]
                    if bool(getattr(model, "persistent_motion_memory", False)):
                        motion_hidden, motion_state = model.motion_step(
                            target_hidden[:, -1:], motion_state
                        )
                        hidden = hidden + motion_hidden
                    head = model.heads[group]
                    shared_adapted, shared_geometry, geometry_state = _shared_head_state(
                        model,
                        head,
                        hidden,
                        previous,
                        current_motion_state,
                        geometry_state,
                    )

                if absolute_tick in reference_items:
                    text, representation = reference_items[absolute_tick]
                    # Reference events are already provider-owned future
                    # context; they are teacher-forced, never re-judged.
                    chosen_text, chosen_representation = text, copy_representation(representation)
                else:
                    planned = (
                        intent_plan[absolute_tick]
                        if intent_plan is not None and absolute_tick in intent_plan
                        else None
                    )
                    chosen_text, chosen_representation, _decision = _sample_anchor(
                        tick=absolute_tick,
                        moment=absolute_time,
                        bpm=current_bpm,
                        model=model,
                        head=head,
                        hidden=hidden,
                        vocab=vocab,
                        version=int(context.version),
                        provider=provider,
                        snapshot=snapshot,
                        route_support=route_support,
                        route_cache=route_cache,
                        temperature=float(context.temperature),
                        rng=rng,
                        planned_intent=planned,
                        allow_intent_revision=bool(allow_intent_revision),
                        shared_adapted=shared_adapted,
                        shared_geometry=shared_geometry,
                    )
                    if planned is not None:
                        choices=planned.candidates if isinstance(planned,IntentChoices) else (EventIntent.from_representation(planned),)
                        realized=intent_signature(chosen_representation)
                        index=next((i for i,value in enumerate(choices) if intent_signature(value)==realized),-1)
                        histogram=cache.setdefault('what_choice_histogram',{})
                        histogram[str(index)]=int(histogram.get(str(index),0))+1
                    if chosen_text:
                        try:
                            chosen_representation = text_representation(chosen_text, vocab)
                        except RepresentationEncodingError as exc:
                            raise RenderFailure(
                                str(exc),
                                stage="generated_encoding",
                                tick=absolute_tick,
                                slot=context.slot,
                                details={"unsupported": True, "field": exc.field},
                                cause=exc,
                            ) from exc
                    emitted[absolute_tick] = chosen_text
                representations.append(chosen_representation)
                provider.update(absolute_tick, chosen_representation, current_bpm)
                provider.commit(chosen_representation, absolute_time, current_bpm)
                steps.append({'tick':absolute_tick,'text':chosen_text,'rep':copy_representation(chosen_representation),
                              'geometry':geometry_state,'motion':motion_state,'snapshot':snapshot})
                if absolute_tick in target_set:work_done+=1
            if context.progress is not None:
                context.progress(
                    f"难度 {context.slot}: 模型本轮 {work_done}/{len(target_list)}，复用上下文 {resume} 个位置"
                )
        return {tick: emitted.get(tick, "") for tick in target_list}
    except RenderFailure as exc:
        exc.partial_events=dict(reference_texts) if 'reference_texts' in locals() else {}
        if 'emitted' in locals():
            for tick,text in emitted.items():
                if text:exc.partial_events[int(tick)]=text
                else:exc.partial_events.pop(int(tick),None)
        exc.pending_ticks=tuple(t for t in target_list if exc.tick is None or t>=exc.tick) if 'target_list' in locals() else ()
        raise
    except Exception as exc:
        raise RenderFailure(
            str(exc),
            stage="native_v4_render",
            slot=getattr(context, "slot", None),
            details={
                "providerType": type(provider).__name__,
                "checkpoint": str(context.checkpoint),
                "seed": int(context.seed) + int(seed_offset),
                "hasIntentPlan": intent_plan is not None,
                "hasReferences": references is not None,
                "allowIntentRevision": bool(allow_intent_revision),
                "maxCudaMatrixElements": getattr(provider, "max_cuda_matrix_elements", None),
            },
            cause=exc,
        ) from exc
