from __future__ import annotations

import numpy as np


TICKS_PER_BAR = 384

def ticks_to_seconds(ticks: np.ndarray, bpm_ticks: np.ndarray, bpm_values: np.ndarray) -> np.ndarray:
    bpm_ticks = bpm_ticks.astype(np.int64, copy=False)
    bpm_values = bpm_values.astype(np.float64, copy=False)
    cumulative = np.zeros(len(bpm_ticks), dtype=np.float64)
    for index in range(1, len(bpm_ticks)):
        delta = bpm_ticks[index] - bpm_ticks[index - 1]
        cumulative[index] = cumulative[index - 1] + delta * 240.0 / (bpm_values[index - 1] * TICKS_PER_BAR)
    segment = np.searchsorted(bpm_ticks, ticks, side="right") - 1
    segment = np.clip(segment, 0, len(bpm_ticks) - 1)
    return cumulative[segment] + (ticks - bpm_ticks[segment]) * 240.0 / (bpm_values[segment] * TICKS_PER_BAR)

def sample_positions(mel: np.ndarray, ticks: np.ndarray, bpm_ticks: np.ndarray, bpm_values: np.ndarray, frame_seconds: float) -> np.ndarray:
    positions = ticks_to_seconds(ticks, bpm_ticks, bpm_values) / frame_seconds
    left = np.floor(positions).astype(np.int64); right = left + 1
    weight = (positions - left).astype(np.float32)
    output = np.zeros((len(ticks), mel.shape[1]), dtype=np.float32)
    valid = (left >= 0) & (right < mel.shape[0])
    if np.any(valid):
        lo = int(left[valid].min()); hi = int(right[valid].max()) + 1
        block = np.asarray(mel[lo:hi], dtype=np.float32)
        output[valid] = block[left[valid] - lo] * (1 - weight[valid, None]) + block[right[valid] - lo] * weight[valid, None]
    segment = np.searchsorted(bpm_ticks, ticks, side="right") - 1
    segment = np.clip(segment, 0, len(bpm_values) - 1)
    local_bpm = np.log1p(bpm_values[segment].astype(np.float32))[:, None] / 6.0
    bar_phase = ((ticks % TICKS_PER_BAR).astype(np.float32) / TICKS_PER_BAR)[:, None]
    return np.concatenate((output, local_bpm, bar_phase), axis=1)
