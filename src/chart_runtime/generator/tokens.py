from __future__ import annotations

import hashlib

import numpy as np


GROUP_BY_SLOT = {2: 0, 3: 1, 4: 2, 5: 3, 6: 3}
RHYTHM_DIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
# Orchestration switches are not musical conditioning.  Keeping them out of
# the frozen renderer token stream preserves the released stable path.
RUNTIME_METADATA_KEYS = frozenset({"causalSearchMode"})

def metadata_tokens(chart: dict, count: int = 32, vocab: int = 4096) -> np.ndarray:
    values = []
    for key, value in sorted((chart.get("maidataMetadata") or {}).items()):
        values.append(f"maidata:{key}={value}")
    designer = chart.get("notesDesigner") or {}
    values.extend((f"designer:id={designer.get('id', 0)}", f"designer:name={designer.get('name', '')}"))
    result = np.zeros(count, np.int64)
    for i, value in enumerate(values[:count]):
        digest = hashlib.blake2b(value.encode("utf8"), digest_size=8).digest()
        result[i] = int.from_bytes(digest, "little") % (vocab - 1) + 1
    return result

def rhythm_classes(ticks: np.ndarray) -> np.ndarray:
    pos = ticks % 384
    out = np.full(len(ticks), len(RHYTHM_DIVISIONS) - 1, np.int64)
    for i, division in enumerate(RHYTHM_DIVISIONS):
        step = 384 // division
        mask = (out == len(RHYTHM_DIVISIONS) - 1) & (pos % step == 0)
        out[mask] = i
    return out
