"""Reference Star-count helper used to anchor the default WHAT distribution."""
from __future__ import annotations

import math


def target_stars(official_stars: int, target_ratio: float, event_count: int | None = None) -> int:
    """Scale a calibrated official-chart Star reference with a safe count cap.

    1.1.0 uses ratio=1 for the default planner anchor; user Star emphasis is
    applied later as a normalized WHAT relative-odds scale, not as a hard quota.
    """

    official = max(0, int(official_stars))
    ratio = float(target_ratio)
    if not math.isfinite(ratio):
        raise ValueError("Star 参考倍率必须是有限数值")
    ratio = min(3.0, max(0.0, ratio))
    value=max(0,int(round(official*ratio)))
    return min(2*int(event_count),value) if event_count is not None else value
