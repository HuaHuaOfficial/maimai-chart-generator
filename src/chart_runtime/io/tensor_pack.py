"""Coalesce small host columns into dtype-wise transfers; preserve exact values."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch


def numpy_columns_to_device(columns, device):
    arrays = {key: np.ascontiguousarray(value) for key, value in columns.items()}
    groups = defaultdict(list)
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise TypeError(f'Object dtype is not a tensor column: {key}')
        groups[value.dtype.str].append(key)
    output = {}
    for keys in groups.values():
        sizes = [arrays[key].size for key in keys]
        flat = np.concatenate([arrays[key].reshape(-1) for key in keys])
        # Blocking transfer is intentional: the temporary host buffer may die
        # on return. No borrowed host memory or untracked asynchronous lifetime.
        storage = torch.from_numpy(flat).to(device)
        offset = 0
        for key, size in zip(keys, sizes):
            output[key] = storage.narrow(0, offset, size).reshape(arrays[key].shape)
            offset += size
    return {key: output[key] for key in columns}
