"""Host-side immutable geometry and exact touch-component helpers for decoding."""
from __future__ import annotations
import numpy as np

def geometry_tables(head):
    a, m = head.geometry_candidate_arities, head.geometry_candidate_masks
    key = (a.data_ptr(), a._version, m.data_ptr(), m._version, str(m.device))
    cached = getattr(head, '_sampler_host_geometry', None)
    if cached is None or cached[0] != key:
        arities = a.detach().cpu().numpy().copy()
        masks = m.detach().cpu().numpy().astype(np.bool_, copy=True)
        starts = tuple(tuple(map(int, np.flatnonzero(row))) for row in masks)
        cached = (key, arities, masks, starts)
        head._sampler_host_geometry = cached
    return cached[1:]

def component_count(active, names, adjacency):
    """Connected components of the induced Touch graph; no device round trip."""
    pending = {int(i) for i in np.flatnonzero(active)}
    by_name = {name: i for i, name in enumerate(names)}
    count = 0
    while pending:
        count += 1
        stack = [pending.pop()]
        while stack:
            i = stack.pop()
            neighbors = {by_name[x] for x in adjacency.get(names[i], ()) if x in by_name}
            reached = pending & neighbors
            pending.difference_update(reached)
            stack.extend(reached)
    return count
