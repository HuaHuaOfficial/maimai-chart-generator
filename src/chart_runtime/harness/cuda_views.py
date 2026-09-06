"""Zero-copy views for an explicitly shared CUDA stream.

Callers must use the PyTorch launch stream for CuPy and keep it synchronized
before releasing the workspace. Tensor owners keep allocations alive.
"""
import numpy as np
import torch

DTYPES={torch.float64:np.float64,torch.float32:np.float32,torch.int64:np.int64,
        torch.int32:np.int32,torch.uint8:np.uint8,torch.bool:np.bool_}


def view(cp,tensor):
    if tensor.device.type!='cuda':raise TypeError('CUDA storage required')
    dtype=DTYPES[tensor.dtype]
    memory=cp.cuda.UnownedMemory(tensor.data_ptr(),tensor.numel()*tensor.element_size(),tensor,device_id=tensor.device.index)
    pointer=cp.cuda.MemoryPointer(memory,0)
    return cp.ndarray(tensor.shape,dtype=dtype,memptr=pointer,strides=tuple(s*tensor.element_size() for s in tensor.stride()))
