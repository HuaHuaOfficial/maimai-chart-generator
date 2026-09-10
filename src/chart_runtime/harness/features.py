"""CUDA feature kernels shared by the runtime Harness.

The feature rows deliberately mirror ``sequence_harness.local_features``:

* ``lanes`` contains the caller-supplied outer lanes exactly as defined by
  the canonical IR; ``-1`` is padding.  This kernel does not reclassify or
  filter headless entries.  Touch-only events therefore have
  ``lane_counts == 0`` and do not replace the last non-empty outer-lane state.
* The first feature is the one-second, event-count-weighted burst.  A
  predictable single-lane move has weight ``0.35`` and an otherwise weighted
  event has weight ``1.0``.
* A pair of two-lane states uses the lower-cost complete matching.  All other
  arities use the mean nearest previous lane for each current lane.
* Delta history is reset by every non-single transition.  Surprise starts at
  one and jerk at zero until a prior delta exists; lag one or lag two makes a
  move predictable.

This module is intentionally CUDA-only.  Host-side parsing may transport the
small configuration vectors, but event features and violation masks are never
computed by a CPU reference path.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


# Bump whenever the feature ordering or any numerical boundary changes.  The
# string is part of receipts/calibration identity, not a user-facing label.
FEATURES_ID = "chart-runtime.local-features.cuda.v2"


def _require_cuda_tensors(*tensors: torch.Tensor | None) -> torch.device:
    """Require a common CUDA device without synchronizing event values."""

    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ValueError("at least one tensor is required")
    device = present[0].device
    if device.type != "cuda":
        raise RuntimeError("local difficulty features require CUDA")
    if any(tensor.device != device for tensor in present):
        raise ValueError("all feature tensors must be on one CUDA device")
    return device


def _validate_batch_ids(batch_ids: torch.Tensor | None, n: int, device: torch.device) -> None:
    if batch_ids is None:
        return
    if batch_ids.ndim != 1 or batch_ids.shape[0] != n:
        raise ValueError("batch_ids must have shape [N]")
    if batch_ids.dtype != torch.int64:
        raise TypeError("batch_ids must be torch.int64")
    if batch_ids.device != device:
        raise ValueError("batch_ids must share the feature CUDA device")


def _segment_layout(
    n: int, device: torch.device, batch_ids: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return segment starts, each row's start index, and a segment rank.

    ``batch_ids`` is expected to contain contiguous scenario blocks.  No host
    read or per-block Python loop is used.  The returned rank is only used to
    make reset-per-batch event times globally searchable.
    """

    if n == 0:
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_long = torch.empty(0, dtype=torch.int64, device=device)
        return empty_bool, empty_long, empty_long

    index = torch.arange(n, dtype=torch.int64, device=device)
    if batch_ids is None:
        starts = torch.zeros(n, dtype=torch.bool, device=device)
        starts[0] = True
    else:
        starts = torch.empty(n, dtype=torch.bool, device=device)
        starts[0] = True
        if n > 1:
            starts[1:] = batch_ids[1:] != batch_ids[:-1]

    # Since indices increase, cummax of the latest start marker gives the
    # current block's start for every row.
    start_markers = torch.where(starts, index, torch.zeros_like(index))
    start_index = torch.cummax(start_markers, dim=0).values
    segment_rank = starts.to(torch.int64).cumsum(0) - 1
    return starts, start_index, segment_rank


def _window_left_indices(
    times: torch.Tensor,
    starts: torch.Tensor,
    start_index: torch.Tensor,
    segment_rank: torch.Tensor,
    window_seconds: float,
    *,
    include_left_boundary: bool,
) -> torch.Tensor:
    """Find a window's first row in each block.

    ``include_left_boundary=False`` returns the first row strictly after
    ``time - window`` (the burst definition).  ``True`` returns the first row
    at or after that boundary (the historical full-quality sustained count).

    ``torch.searchsorted`` requires one sorted vector.  Adding a stride wider
    than the complete time range plus the window makes reset-per-batch blocks
    one globally sorted vector while preserving all within-block comparisons.
    The result is still an event-index tensor, so the caller can use one
    prefix sum for the window reduction.
    """

    n = times.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=times.device)

    # The zero-rank single-block case is unchanged by this encoding too, so
    # no host-side branch or scalar event reduction is needed here.
    span = (times.max() - times.min()).clamp_min(0.0)
    span = span + float(window_seconds) + 1.0
    encoded = times + segment_rank.to(times.dtype) * span
    query = encoded - float(window_seconds)
    return torch.searchsorted(encoded, query, right=not include_left_boundary)


def _validate_feature_inputs(
    times: torch.Tensor,
    note_counts: torch.Tensor,
    lanes: torch.Tensor,
    lane_counts: torch.Tensor,
    track_speed: torch.Tensor,
    batch_ids: torch.Tensor | None,
) -> tuple[torch.device, int, int]:
    device = _require_cuda_tensors(times, note_counts, lanes, lane_counts, track_speed, batch_ids)
    n = times.shape[0]
    if times.ndim != 1 or times.dtype != torch.float64:
        raise TypeError("times must be CUDA torch.float64 [N]")
    if note_counts.ndim != 1 or note_counts.shape[0] != n or note_counts.dtype != torch.int64:
        raise TypeError("note_counts must be CUDA torch.int64 [N]")
    if lanes.ndim != 2 or lanes.shape[0] != n or lanes.dtype != torch.int64:
        raise TypeError("lanes must be CUDA torch.int64 [N,K]")
    if lane_counts.ndim != 1 or lane_counts.shape[0] != n or lane_counts.dtype != torch.int64:
        raise TypeError("lane_counts must be CUDA torch.int64 [N]")
    if track_speed.ndim != 1 or track_speed.shape[0] != n or track_speed.dtype != torch.float64:
        raise TypeError("track_speed must be CUDA torch.float64 [N]")
    _validate_batch_ids(batch_ids, n, device)
    return device, n, lanes.shape[1]


def _circular_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distance = (left - right).abs()
    return torch.minimum(distance, 8.0 - distance)


def _previous_nonempty(
    lane_counts: torch.Tensor,
    starts: torch.Tensor,
    start_index: torch.Tensor,
) -> torch.Tensor:
    """Index the most recent non-empty lane row, never crossing a block."""

    n = lane_counts.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=lane_counts.device)
    index = torch.arange(n, dtype=torch.int64, device=lane_counts.device)
    block_floor = start_index - 1
    candidate = torch.where(lane_counts > 0, index, block_floor)
    latest = torch.cummax(candidate, dim=0).values
    previous = torch.cat((torch.full((1,), -1, dtype=torch.int64, device=lane_counts.device), latest[:-1]))
    previous = torch.where(previous < start_index, torch.full_like(previous, -1), previous)
    return torch.where(starts, torch.full_like(previous, -1), previous)


def compute_features(
    times: torch.Tensor,
    note_counts: torch.Tensor,
    lanes: torch.Tensor,
    lane_counts: torch.Tensor,
    track_speed: torch.Tensor,
    batch_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``[burst, movement, track_speed, jerk]`` on CUDA.

    ``batch_ids`` optionally identifies contiguous complete-chart blocks in a
    concatenated candidate batch.  Time windows, previous-lane state, and
    delta history reset at every block boundary.
    """

    device, n, k = _validate_feature_inputs(
        times, note_counts, lanes, lane_counts, track_speed, batch_ids
    )
    if n == 0:
        return torch.empty((0, 4), dtype=torch.float64, device=device)

    starts, start_index, segment_rank = _segment_layout(n, device, batch_ids)
    previous_index = _previous_nonempty(lane_counts, starts, start_index)
    previous_exists = previous_index >= 0
    safe_previous = previous_index.clamp_min(0)
    previous_counts = torch.where(
        previous_exists, lane_counts[safe_previous], torch.zeros_like(lane_counts)
    )
    gap = torch.where(previous_exists, times - times[safe_previous], torch.zeros_like(times))

    current_counts = lane_counts.clamp_min(0)
    if k:
        columns = torch.arange(k, dtype=torch.int64, device=device)
        current_valid = columns[None, :] < current_counts[:, None]
        previous_valid = columns[None, :] < previous_counts[:, None]
        previous_lanes = lanes[safe_previous]

        # The general arity path is also used when only one side has two
        # lanes.  Invalid padded entries are made unreachable by +inf.
        pairwise = _circular_distance(lanes[:, :, None].to(torch.float64), previous_lanes[:, None, :].to(torch.float64))
        pairwise = pairwise.masked_fill(~(current_valid[:, :, None] & previous_valid[:, None, :]), torch.inf)
        nearest = pairwise.min(dim=2).values
        nearest_mean = nearest.masked_fill(~current_valid, 0.0).sum(dim=1) / current_counts.clamp_min(1).to(torch.float64)

        if k >= 2:
            has_two = (current_counts == 2) & (previous_counts == 2) & previous_exists
            direct = _circular_distance(lanes[:, 0].to(torch.float64), previous_lanes[:, 0].to(torch.float64))
            direct = direct + _circular_distance(lanes[:, 1].to(torch.float64), previous_lanes[:, 1].to(torch.float64))
            swapped = _circular_distance(lanes[:, 0].to(torch.float64), previous_lanes[:, 1].to(torch.float64))
            swapped = swapped + _circular_distance(lanes[:, 1].to(torch.float64), previous_lanes[:, 0].to(torch.float64))
            paired_mean = torch.minimum(direct, swapped) / 2.0
            travel = torch.where(has_two, paired_mean, nearest_mean)
        else:
            travel = nearest_mean
    else:
        current_valid = torch.empty((n, 0), dtype=torch.bool, device=device)
        travel = torch.zeros(n, dtype=torch.float64, device=device)

    movement_valid = previous_exists & (current_counts > 0) & (gap > 0.0)
    speed = torch.where(movement_valid, travel / gap.clamp_min(1e-6), torch.zeros_like(gap))

    # A delta exists exactly when the current row and the latest prior
    # non-empty row are single-lane rows with positive elapsed time.
    single = current_counts == 1
    previous_single = previous_counts == 1
    if k:
        delta = (lanes[:, 0] - previous_lanes[:, 0] + 4) % 8 - 4
    else:
        delta = torch.zeros(n, dtype=torch.int64, device=device)
    delta_valid = single & previous_single & (gap > 0.0)

    prior_one = torch.zeros(n, dtype=torch.bool, device=device)
    prior_two = torch.zeros(n, dtype=torch.bool, device=device)
    delta_one = torch.zeros(n, dtype=torch.int64, device=device)
    delta_two = torch.zeros(n, dtype=torch.int64, device=device)
    if n > 1:
        prior_one[1:] = delta_valid[:-1]
        delta_one[1:] = delta[:-1]
    if n > 2:
        prior_two[2:] = delta_valid[:-2]
        delta_two[2:] = delta[:-2]
    predictable = delta_valid & prior_one & (
        (delta == delta_one) | (prior_two & (delta == delta_two))
    )
    surprise = torch.where(predictable, torch.zeros_like(times), torch.ones_like(times))
    jerk = torch.where(
        delta_valid & prior_one & ~predictable,
        (delta - delta_one).abs().to(torch.float64) / gap.clamp_min(1e-6),
        torch.zeros_like(times),
    )

    weights = note_counts.to(torch.float64) * (0.35 + 0.65 * surprise)
    left = _window_left_indices(
        times, starts, start_index, segment_rank, 1.0, include_left_boundary=False
    )
    prefix = torch.cat((torch.zeros(1, dtype=torch.float64, device=device), weights.cumsum(0)))
    burst = prefix[torch.arange(n, device=device, dtype=torch.int64) + 1] - prefix[left]
    return torch.stack((burst, speed, track_speed, jerk), dim=1)


def _tolerance_value(tolerance: Mapping[str, Any], key: str) -> float:
    try:
        value = tolerance[key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing local difficulty tolerance: {key}") from exc
    return float(value)


def violation_masks(
    features: torch.Tensor,
    times: torch.Tensor,
    thresholds: torch.Tensor | list[float] | tuple[float, ...],
    tolerance: Mapping[str, Any],
    batch_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact extreme/sustained masks for each local feature.

    The sustained count is a prefix-sum range reduction over the one-second
    ``(time - window, time]`` interval.  Batch blocks are independently
    searchable and never contribute to one another's sustained count.
    ``thresholds`` may be one broadcast row ``[4]`` or event rows ``[N,4]``;
    one row per batch ``[B,4]`` is also accepted when ``batch_ids`` is given.
    """

    device = _require_cuda_tensors(features, times, batch_ids)
    if features.ndim != 2 or features.shape[1] != 4 or features.dtype != torch.float64:
        raise TypeError("features must be CUDA torch.float64 [N,4]")
    n = features.shape[0]
    if times.ndim != 1 or times.shape[0] != n or times.dtype != torch.float64:
        raise TypeError("times must be CUDA torch.float64 [N]")
    _validate_batch_ids(batch_ids, n, device)
    threshold_tensor = torch.as_tensor(thresholds, dtype=torch.float64, device=device)
    if n == 0:
        if threshold_tensor.ndim == 1 and threshold_tensor.shape[0] == 4:
            return torch.empty((0, 4), dtype=torch.bool, device=device)
        if threshold_tensor.ndim == 2 and threshold_tensor.shape[1] == 4:
            return torch.empty((0, 4), dtype=torch.bool, device=device)
        raise ValueError("thresholds must have shape [4], [N,4], or [B,4]")
    if threshold_tensor.ndim == 1:
        if threshold_tensor.shape[0] != 4:
            raise ValueError("thresholds must have four feature values")
        threshold_rows = threshold_tensor
    elif threshold_tensor.ndim == 2 and threshold_tensor.shape[1] == 4:
        # A caller may pass one calibration row per event (already expanded)
        # or one row per contiguous chart block.  Both forms stay device-side.
        if threshold_tensor.shape[0] == n:
            threshold_rows = threshold_tensor
        elif batch_ids is not None:
            _, _, segment_rank = _segment_layout(n, device, batch_ids)
            threshold_rows = threshold_tensor[segment_rank]
        else:
            raise ValueError("two-dimensional thresholds require batch_ids unless expanded to [N,4]")
    else:
        raise ValueError("thresholds must have shape [4], [N,4], or [B,4]")

    ratios = features / threshold_rows
    sustained_ratio = _tolerance_value(tolerance, "sustainedRatio")
    sustained_events = int(_tolerance_value(tolerance, "sustainedEvents"))
    window_seconds = _tolerance_value(tolerance, "windowSeconds")
    extreme_ratio = _tolerance_value(tolerance, "extremeRatio")
    high = ratios > sustained_ratio

    starts, start_index, segment_rank = _segment_layout(n, device, batch_ids)
    left = _window_left_indices(
        times,
        starts,
        start_index,
        segment_rank,
        window_seconds,
        include_left_boundary=True,
    )
    prefix = torch.cat(
        (torch.zeros((1, 4), dtype=torch.int64, device=device), high.to(torch.int64).cumsum(0))
    )
    rows = torch.arange(n, dtype=torch.int64, device=device)
    counts = prefix[rows + 1] - prefix[left]
    sustained = high & (counts >= sustained_events)
    # This is the historical floor/tolerance rule: burst (feature 0) only
    # becomes sustained when the same event has nonzero unexpected jerk.
    sustained[:, 0] &= features[:, 3] != 0.0
    result=(ratios > extreme_ratio) | sustained
    for feature_index in tolerance.get('instantaneousFeatureIndexes', ()):
        index=int(feature_index)
        if not 0<=index<4:raise ValueError('instantaneous feature index out of range')
        result[:,index]|=ratios[:,index]>1.0
    return result
