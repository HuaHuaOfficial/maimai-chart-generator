from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class FactorConfig:
    audio_dim: int = 82
    structure_dim: int = 256
    d_model: int = 256
    sequence_length: int = 96
    encoder_layers: int = 3
    decoder_layers: int = 4
    groups: int = 4
    positions: int = 41
    touch_positions: int = 33
    routes: int = 1026
    durations: int = 258
    metadata_vocab: int = 4096
    metadata_count: int = 32
    max_button_notes: int = 2
    style_dim: int = 128

    def to_dict(self):
        return asdict(self)


class FactorHead(nn.Module):
    def __init__(self, c: FactorConfig):
        super().__init__()
        d, n = c.d_model, c.max_button_notes
        self.adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.button_arity = nn.Linear(d, n + 1)
        self.button_family = nn.Linear(d, n * 4)
        self.button_start = nn.Linear(d, n * 8)
        self.button_route = nn.Linear(d, n * c.routes)
        self.button_duration = nn.Linear(d, n * c.durations)
        self.button_modifiers = nn.Linear(d, n * 5)
        self.touch_presence = nn.Linear(d, c.touch_positions)
        self.touch_duration = nn.Linear(d, c.touch_positions * c.durations)
        self.touch_modifiers = nn.Linear(d, c.touch_positions * 5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.adapter(x)
        b, length, _ = x.shape
        return {
            "button_arity": self.button_arity(x),
            "button_family": self.button_family(x).view(b, length, 2, 4),
            "button_start": self.button_start(x).view(b, length, 2, 8),
            "button_route": self.button_route(x).view(b, length, 2, -1),
            "button_duration": self.button_duration(x).view(b, length, 2, -1),
            "button_modifiers": self.button_modifiers(x).view(b, length, 2, 5),
            "touch_presence": self.touch_presence(x),
            "touch_duration": self.touch_duration(x).view(b, length, 33, -1),
            "touch_modifiers": self.touch_modifiers(x).view(b, length, 33, 5),
        }


class FactorEventModel(nn.Module):
    def __init__(self, c: FactorConfig):
        super().__init__()
        self.c = c
        d = c.d_model
        self.audio = nn.Linear(c.audio_dim, d)
        self.structure = nn.Linear(c.structure_dim, d)
        self.tick = nn.Embedding(384, d)
        self.delta = nn.Embedding(1537, d)
        self.rhythm = nn.Embedding(16, d)
        self.version = nn.Embedding(32, d)
        self.difficulty = nn.Embedding(5, d)
        self.level = nn.Embedding(201, d)
        self.metadata = nn.Embedding(c.metadata_vocab, d, padding_idx=0)
        self.style = nn.Linear(c.style_dim, d, bias=False)
        nn.init.zeros_(self.style.weight)
        self.occupied_lanes = nn.Linear(8, d, bias=False)
        nn.init.zeros_(self.occupied_lanes.weight)
        self.available_hands = nn.Embedding(3, d)
        nn.init.zeros_(self.available_hands.weight)
        self.button_arity_emb = nn.Embedding(3, d)
        self.family_emb = nn.Embedding(4, d)
        self.button_start_emb = nn.Embedding(8, d)
        self.route_emb = nn.Embedding(c.routes, d)
        self.duration_emb = nn.Embedding(c.durations, d)
        self.modifier_emb = nn.Embedding(32, d)
        self.touch_start_emb = nn.Embedding(c.touch_positions, d)
        enc = nn.TransformerEncoderLayer(d, 8, 1024, 0.1, "gelu", batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d, 8, 1024, 0.1, "gelu", batch_first=True, norm_first=True)
        self.audio_encoder = nn.TransformerEncoder(enc, c.encoder_layers, enable_nested_tensor=False)
        self.event_decoder = nn.TransformerDecoder(dec, c.decoder_layers)
        self.heads = nn.ModuleList(FactorHead(c) for _ in range(c.groups))

    def _metadata(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0)
        return (self.metadata(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _previous_event(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        arity = batch["prev_button_arity"].long().clamp(0, 2)
        x = self.button_arity_emb(arity)
        note_mask = torch.arange(2, device=x.device)[None, None] < arity[..., None]
        button = self.family_emb(batch["prev_button_family"].long())
        button = button + self.button_start_emb(batch["prev_button_start"].long().clamp(0, 7))
        button = button + self.route_emb(batch["prev_button_route"].long())
        button = button + self.duration_emb(batch["prev_button_duration"].long())
        button = button + self.modifier_emb(batch["prev_button_modifiers"].long())
        x = x + (button * note_mask[..., None]).sum(2) / note_mask.sum(2, keepdim=True).clamp_min(1)
        touch_mask = batch["prev_touch_presence"].bool()
        sensor = torch.arange(self.c.touch_positions, device=x.device)[None, None]
        touch = self.touch_start_emb(sensor)
        touch = touch + self.duration_emb(batch["prev_touch_duration"].long())
        touch = touch + self.modifier_emb(batch["prev_touch_modifiers"].long())
        x = x + (touch * touch_mask[..., None]).sum(2) / touch_mask.sum(2, keepdim=True).clamp_min(1)
        return x

    def forward(self, batch: dict[str, torch.Tensor]):
        padding = ~batch["valid"].bool()
        condition = self.version(batch["version"].clamp(0, 31)) + self.difficulty(batch["difficulty"].clamp(0, 4)) + self.level(batch["level"].clamp(0, 200)) + self._metadata(batch["metadata"]) + self.style(batch["style"])
        timing = self.tick(batch["tick"]) + self.delta(batch["delta"].clamp(0, 1536)) + self.rhythm(batch["rhythm"])
        memory = self.audio(batch["audio"]) + self.structure(batch["structure"]) + timing + condition[:, None]
        memory = self.audio_encoder(memory, src_key_padding_mask=padding)
        target = self._previous_event(batch) + timing + condition[:, None] + self.occupied_lanes(batch["occupied_lanes"].float()) + self.available_hands(batch["available_hands"].long().clamp(0, 2))
        causal = torch.ones((target.shape[1], target.shape[1]), device=target.device, dtype=torch.bool).triu(1)
        hidden = self.event_decoder(target, memory, tgt_mask=causal, tgt_key_padding_mask=padding, memory_key_padding_mask=padding)
        result = []
        for group in batch["group"].unique(sorted=True):
            indices = batch["group"].eq(group).nonzero(as_tuple=True)[0]
            result.append((indices, self.heads[int(group)](hidden[indices])))
        return result
