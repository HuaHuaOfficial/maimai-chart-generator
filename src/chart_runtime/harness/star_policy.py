"""Pure star-target policy shared by preparation, planning and the Harness."""
from __future__ import annotations

import math


def target_stars(official_stars: int, target_ratio: float) -> int:
    """Map the user-facing target ratio to an exact calibrated-star target.

    The ratio is relative to the calibrated official reference, not to the
    first model draft. Keeping this conversion pure makes the GUI contract
    testable without starting CUDA generation.
    """

    official = max(0, int(official_stars))
    ratio = float(target_ratio)
    if not math.isfinite(ratio):
        raise ValueError("星星目标比例必须是有限数值")
    ratio = min(1.0, max(0.0, ratio))
    return min(official, max(0, int(round(official * ratio))))
