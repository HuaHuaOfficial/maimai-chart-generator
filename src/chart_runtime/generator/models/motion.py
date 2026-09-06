from __future__ import annotations

import numpy as np


MOTION_STATE_DIM = 28


def _features(outer_history: list[tuple[int, tuple[int, ...]]]) -> np.ndarray:
    result = np.zeros(MOTION_STATE_DIM, dtype=np.float32)
    if not outer_history:
        result[8 + 4] = 1.0
        result[8 + 9 + 1] = 1.0
        result[8 + 9 + 3] = 0.0
        result[8 + 9 + 3 + 1] = 1.0
        return result
    recent = outer_history[-4:]
    for lane, _ in recent:
        result[lane] += 0.25
    last_lane, last_families = outer_history[-1]
    delta = 0
    if len(outer_history) >= 2:
        previous_lane = outer_history[-2][0]
        raw = (last_lane - previous_lane) % 8
        delta = raw if raw <= 4 else raw - 8
    result[8 + delta + 4] = 1.0
    direction = int(np.sign(delta))
    result[8 + 9 + direction + 1] = 1.0
    run = 0
    if direction:
        run = 1
        lanes = [value[0] for value in outer_history[-9:]]
        deltas = []
        for previous, current in zip(lanes, lanes[1:]):
            raw = (current - previous) % 8
            signed = raw if raw <= 4 else raw - 8
            deltas.append(int(np.sign(signed)))
        for value in reversed(deltas[:-1]):
            if value != direction:
                break
            run += 1
    result[8 + 9 + 3] = min(run, 8) / 8.0
    arity_offset = 8 + 9 + 3 + 1
    result[arity_offset + min(2, len(last_families))] = 1.0
    family_offset = arity_offset + 3
    for family in last_families:
        if 0 <= family < 4:
            result[family_offset + family] += 0.5
    return result


def motion_state_from_representations(history: list[dict]) -> np.ndarray:
    outer = []
    for representation in history:
        arity = min(2, int(representation["button_arity"]))
        if arity <= 0:
            continue
        lane = int(representation["button_start"][0])
        families = tuple(
            int(value) for value in representation["button_family"][:arity]
        )
        outer.append((lane, families))
    return _features(outer)


def motion_state_sequence(
    arities: np.ndarray,
    families: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    arities = np.asarray(arities)
    families = np.asarray(families)
    starts = np.asarray(starts)
    result = np.zeros((len(arities), MOTION_STATE_DIM), dtype=np.float32)
    outer = []
    for index in range(len(arities)):
        result[index] = _features(outer)
        arity = min(2, int(arities[index]))
        if arity <= 0:
            continue
        outer.append(
            (
                int(starts[index, 0]),
                tuple(int(value) for value in families[index, :arity]),
            )
        )
    return result
