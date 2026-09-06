from dataclasses import dataclass
from types import MappingProxyType
import torch


@dataclass(frozen=True)
class Envelope:
    """Immutable transport metadata accompanying actual CUDA evidence buffers."""
    data: object
    evidence: torch.Tensor
    schema_id: str

    def __post_init__(self):
        if self.evidence.device.type != 'cuda':
            raise RuntimeError('Runtime evidence buffers must reside on CUDA')
        if isinstance(self.data,dict):
            object.__setattr__(self,'data',MappingProxyType(self.data))

    @property
    def device(self):return str(self.evidence.device)
