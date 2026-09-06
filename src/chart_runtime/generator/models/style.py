from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class StylePriorConfig:
    structure_dim: int = 256
    d_model: int = 256
    max_bars: int = 256
    layers: int = 2
    groups: int = 4
    codebooks: int = 4
    codebook_size: int = 128
    metadata_vocab: int = 4096

    def to_dict(self):
        return asdict(self)


class StylePrior(nn.Module):
    def __init__(self, c: StylePriorConfig):
        super().__init__()
        self.c = c
        d = c.d_model
        self.structure = nn.Linear(c.structure_dim, d)
        self.position = nn.Embedding(c.max_bars, d)
        self.version = nn.Embedding(32, d)
        self.difficulty = nn.Embedding(5, d)
        self.level = nn.Embedding(201, d)
        self.bpm = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.metadata = nn.Embedding(c.metadata_vocab, d, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d, 8, 1024, 0.1, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.summary = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.LayerNorm(d))
        self.adapters = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.codebooks * c.codebook_size)) for _ in range(c.groups))

    def metadata_pool(self, ids):
        mask = ids.ne(0)
        return (self.metadata(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def forward(self, batch):
        padding = batch["padding"].bool()
        pos = torch.arange(batch["structure"].shape[1], device=padding.device)
        condition = self.version(batch["version"].clamp(0, 31)) + self.difficulty(batch["difficulty"].clamp(0, 4)) + self.level(batch["level"].clamp(0, 200)) + self.bpm(torch.log1p(batch["bpm"]).unsqueeze(-1) / 6) + self.metadata_pool(batch["metadata"])
        hidden = self.encoder(self.structure(batch["structure"]) + self.position(pos)[None], src_key_padding_mask=padding)
        valid = ~padding
        mean = (hidden * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        attention = (hidden * condition[:, None]).sum(-1) / math.sqrt(self.c.d_model)
        attention = attention.masked_fill(padding, -torch.inf).softmax(-1)
        pooled = (hidden * attention[..., None]).sum(1)
        summary = self.summary(torch.cat((mean, pooled, condition), dim=-1))
        all_logits = torch.stack([adapter(summary).view(-1, self.c.codebooks, self.c.codebook_size) for adapter in self.adapters], dim=1)
        return all_logits[torch.arange(len(summary), device=summary.device), batch["group"].long()]
