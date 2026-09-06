from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from chart_runtime.generator.models.factor import FactorConfig
from chart_runtime.generator.models.motion import MOTION_STATE_DIM
from chart_runtime.generator.models.structured import (
    StructuredFactorHead,
    V3StructuredRenderer,
)


# The renderer scores complete outer-button sets instead of sampling the two
# button starts independently.  There are 8 single-button candidates and
# C(8, 2) unordered double-button candidates.
GEOMETRY_CANDIDATES: tuple[tuple[int, ...], ...] = tuple(
    (lane,) for lane in range(8)
) + tuple(
    (left, right)
    for left in range(8)
    for right in range(left + 1, 8)
)
GEOMETRY_CANDIDATE_COUNT = len(GEOMETRY_CANDIDATES)
EMPTY_GEOMETRY_ID = GEOMETRY_CANDIDATE_COUNT
CONFIGURATION_STATE_DIM = 42


def _candidate_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros((GEOMETRY_CANDIDATE_COUNT, 8), dtype=np.float32)
    arities = np.zeros(GEOMETRY_CANDIDATE_COUNT, dtype=np.int64)
    static = np.zeros((GEOMETRY_CANDIDATE_COUNT, 4), dtype=np.float32)
    for index, candidate in enumerate(GEOMETRY_CANDIDATES):
        masks[index, list(candidate)] = 1.0
        arities[index] = len(candidate)
        static[index, len(candidate)] = 1.0
        if len(candidate) == 2:
            distance = (candidate[1] - candidate[0]) % 8
            distance = min(distance, 8 - distance)
            static[index, 3] = distance / 4.0
    return masks, arities, static


def _relation_basis() -> np.ndarray:
    """Return candidate x previous-lane x signed-circular-delta counts."""

    basis = np.zeros((GEOMETRY_CANDIDATE_COUNT, 8, 8), dtype=np.float32)
    for candidate_index, candidate in enumerate(GEOMETRY_CANDIDATES):
        for previous_lane in range(8):
            for lane in candidate:
                basis[candidate_index, previous_lane, (lane - previous_lane) % 8] += 1.0
    return basis


def geometry_candidate_index(starts: torch.Tensor, arity: torch.Tensor) -> torch.Tensor:
    """Map target starts to the fixed single/pair candidate vocabulary."""

    first = starts[..., 0].long().clamp(0, 7)
    second = starts[..., 1].long().clamp(0, 7)
    low = torch.minimum(first, second)
    high = torch.maximum(first, second)
    pair_index = torch.full((8, 8), -1, dtype=torch.long, device=starts.device)
    cursor = 8
    for left in range(8):
        for right in range(left + 1, 8):
            pair_index[left, right] = cursor
            cursor += 1
    pair = pair_index[low, high]
    return torch.where(arity.long().eq(1), first, pair)


class RelationalStructuredFactorHead(StructuredFactorHead):
    """V3 factor head with a learned multi-event outer-key-set scorer.

    The old family/route/duration/Touch heads remain intact for a controlled
    ablation.  Only outer-key realization is changed: inference chooses one
    complete single/pair geometry candidate, then the existing factor heads
    realize its note families and attachments.
    """

    def __init__(
        self,
        config: FactorConfig,
        future_geometry_horizon: int = 0,
        explicit_motion_state: bool = False,
        geometry_sequence_state: bool = False,
        configuration_code_adapter: bool = False,
    ):
        super().__init__(config)
        d = config.d_model
        self.future_geometry_horizon = max(0, int(future_geometry_horizon))
        self.explicit_motion_state = bool(explicit_motion_state)
        self.geometry_sequence_state = bool(geometry_sequence_state)
        self.configuration_code_adapter = bool(configuration_code_adapter)
        masks, arities, static = _candidate_arrays()
        self.register_buffer(
            "geometry_candidate_masks", torch.from_numpy(masks), persistent=True
        )
        self.register_buffer(
            "geometry_candidate_arities", torch.from_numpy(arities), persistent=True
        )
        self.register_buffer(
            "geometry_candidate_static", torch.from_numpy(static), persistent=True
        )
        self.register_buffer(
            "geometry_relation_basis", torch.from_numpy(_relation_basis()), persistent=True
        )
        self.geometry_absolute = nn.Parameter(torch.empty(GEOMETRY_CANDIDATE_COUNT, d))
        nn.init.normal_(self.geometry_absolute, mean=0.0, std=0.02)
        self.geometry_query = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.geometry_feature = nn.Sequential(
            nn.Linear(8 + 8 + 4, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        self.geometry_bias = nn.Parameter(torch.zeros(GEOMETRY_CANDIDATE_COUNT))
        if self.explicit_motion_state:
            self.motion_route_embedding = nn.Embedding(config.routes, 32)
            self.motion_state_projection = nn.Sequential(
                nn.Linear(MOTION_STATE_DIM + 32, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Linear(d, d),
            )
            nn.init.zeros_(self.motion_state_projection[-1].weight)
            nn.init.zeros_(self.motion_state_projection[-1].bias)
        if self.geometry_sequence_state:
            state_size = 64
            self.geometry_state_hidden_projection = nn.Linear(d, state_size)
            self.geometry_state_motion_projection = nn.Linear(
                MOTION_STATE_DIM, state_size
            )
            self.geometry_state_previous = nn.Embedding(
                GEOMETRY_CANDIDATE_COUNT + 1, state_size
            )
            self.geometry_state_gru = nn.GRU(
                state_size, state_size, batch_first=True
            )
            self.geometry_state_output = nn.Linear(
                state_size, GEOMETRY_CANDIDATE_COUNT
            )
            nn.init.zeros_(self.geometry_state_output.weight)
            nn.init.zeros_(self.geometry_state_output.bias)
        if self.configuration_code_adapter:
            self.configuration_geometry_adapter = nn.Sequential(
                nn.Linear(CONFIGURATION_STATE_DIM, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Linear(d, GEOMETRY_CANDIDATE_COUNT),
            )
            nn.init.zeros_(self.configuration_geometry_adapter[-1].weight)
            nn.init.zeros_(self.configuration_geometry_adapter[-1].bias)
        # A small learned sequence scorer.  It is intentionally unnamed: it
        # ranks short geometry sequences, rather than classifying community
        # pattern names or enforcing a hand-written template.
        self.sequence_memory = nn.GRU(d, d, batch_first=True)
        self.sequence_score_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(32, d // 2)),
            nn.GELU(),
            nn.Linear(max(32, d // 2), 1),
        )
        if self.future_geometry_horizon:
            self.future_geometry_head = nn.Linear(
                d,
                self.future_geometry_horizon * GEOMETRY_CANDIDATE_COUNT,
            )

    def _previous_mask(self, target: dict[str, torch.Tensor] | None, hidden: torch.Tensor):
        batch, length = hidden.shape[:2]
        if target is None or "prev_button_arity" not in target:
            return hidden.new_zeros((batch, length, 8))
        arity = target["prev_button_arity"].long().clamp(0, 2)
        starts = target["prev_button_start"].long().clamp(0, 7)
        note_mask = (
            torch.arange(2, device=hidden.device)[None, None]
            < arity[..., None]
        )
        result = hidden.new_zeros((batch, length, 8))
        result.scatter_add_(
            -1,
            starts,
            note_mask.to(hidden.dtype).expand_as(starts),
        )
        return result

    def geometry_logits(
        self,
        hidden: torch.Tensor,
        target: dict[str, torch.Tensor] | None = None,
        previous_mask_override: torch.Tensor | None = None,
        motion_state_override: torch.Tensor | None = None,
        configuration_state_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        previous_mask = (
            self._previous_mask(target, hidden)
            if previous_mask_override is None
            else previous_mask_override.to(device=hidden.device, dtype=hidden.dtype)
        )
        basis = self.geometry_relation_basis.to(device=hidden.device, dtype=hidden.dtype)
        relative = torch.einsum("blp,cpd->blcd", previous_mask, basis)
        candidates = self.geometry_candidate_masks.to(
            device=hidden.device, dtype=hidden.dtype
        )
        static = self.geometry_candidate_static.to(
            device=hidden.device, dtype=hidden.dtype
        )
        candidate_features = torch.cat(
            (
                candidates[None, None].expand(hidden.shape[0], hidden.shape[1], -1, -1),
                relative,
                static[None, None].expand(hidden.shape[0], hidden.shape[1], -1, -1),
            ),
            dim=-1,
        )
        dynamic = self.geometry_feature(candidate_features)
        absolute = candidates @ self.start_embedding.weight.to(hidden.dtype)
        absolute = absolute / self.geometry_candidate_arities.clamp_min(1).to(
            hidden.dtype
        )[:, None]
        key = absolute + self.geometry_absolute.to(hidden.dtype)[None, :, :] + dynamic
        query_input = hidden
        if self.explicit_motion_state:
            motion_state = (
                target.get("motion_state")
                if motion_state_override is None and target is not None
                else motion_state_override
            )
            if motion_state is None:
                motion_state = hidden.new_zeros(
                    (*hidden.shape[:2], MOTION_STATE_DIM)
                )
            motion_state = motion_state.to(
                device=hidden.device, dtype=hidden.dtype
            )
            if target is not None and "prev_button_route" in target:
                route_ids = target["prev_button_route"].long()
                route_arity = target["prev_button_arity"].long().clamp(0, 2)
                route_mask = (
                    torch.arange(2, device=hidden.device)[None, None]
                    < route_arity[..., None]
                )
                route_state = (
                    self.motion_route_embedding(route_ids)
                    * route_mask[..., None]
                ).sum(dim=-2) / route_arity.clamp_min(1)[..., None]
            else:
                route_state = hidden.new_zeros((*hidden.shape[:2], 32))
            query_input = query_input + self.motion_state_projection(
                torch.cat((motion_state, route_state.to(hidden.dtype)), dim=-1)
            )
        query = self.geometry_query(query_input).unsqueeze(-2)
        logits = (
            (query * key).sum(dim=-1)
            / math.sqrt(self.c.d_model)
            + self.geometry_bias
        )
        if (
            self.configuration_code_adapter
            and configuration_state_override is not None
        ):
            logits = logits + self.configuration_geometry_adapter(
                configuration_state_override.to(
                    device=hidden.device, dtype=hidden.dtype
                )
            )
        return logits

    def sequence_score(
        self,
        hidden: torch.Tensor,
        geometry_ids: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a short geometry sequence under the learned renderer state.

        ``geometry_ids`` indexes the existing 36 single/pair candidate support.
        The score is used only for learned candidate ranking; it is not an
        exact legality gate and it does not name or require a configuration.
        """

        ids = geometry_ids.long().clamp(0, GEOMETRY_CANDIDATE_COUNT - 1)
        candidate_masks = self.geometry_candidate_masks.to(
            device=hidden.device, dtype=hidden.dtype
        )
        candidate_static = self.geometry_candidate_static.to(
            device=hidden.device, dtype=hidden.dtype
        )
        candidate_embed = (
            self.geometry_absolute.to(hidden.dtype)[ids]
            + candidate_masks[ids] @ self.start_embedding.weight.to(hidden.dtype)
            / self.geometry_candidate_arities[ids].clamp_min(1).to(hidden.dtype)[..., None]
        )
        # Include a compact pair-geometry descriptor without introducing a
        # second vocabulary.  Invalid/empty positions are zeroed before the
        # recurrent sequence scorer sees them.
        candidate_embed = candidate_embed + self.geometry_feature(
            torch.cat(
                (
                    candidate_masks[ids],
                    hidden.new_zeros((*ids.shape, 8)),
                    candidate_static[ids],
                ),
                dim=-1,
            )
        )
        if valid is not None:
            candidate_embed = candidate_embed * valid.bool()[..., None].to(
                candidate_embed.dtype
            )
        _, state = self.sequence_memory(hidden + candidate_embed)
        return self.sequence_score_head(state[-1]).squeeze(-1)

    def geometry_sequence_residual(
        self,
        hidden: torch.Tensor,
        previous_geometry: torch.Tensor,
        motion_state: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.geometry_sequence_state:
            raise RuntimeError("geometry sequence state is disabled")
        sequence_input = (
            self.geometry_state_hidden_projection(hidden)
            + self.geometry_state_motion_projection(
                motion_state.to(hidden.dtype)
            )
            + self.geometry_state_previous(
                previous_geometry.long().clamp(0, EMPTY_GEOMETRY_ID)
            )
        )
        encoded, next_state = self.geometry_state_gru(sequence_input, state)
        return self.geometry_state_output(encoded), next_state

    def forward(
        self,
        hidden: torch.Tensor,
        target: dict[str, torch.Tensor] | None = None,
        previous_mask_override: torch.Tensor | None = None,
        motion_state_override: torch.Tensor | None = None,
        configuration_state_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(hidden, target)
        output["geometry_logits"] = self.geometry_logits(
            hidden,
            target,
            previous_mask_override=previous_mask_override,
            motion_state_override=motion_state_override,
            configuration_state_override=configuration_state_override,
        )
        if self.geometry_sequence_state and target is not None:
            previous_id = geometry_candidate_index(
                target["prev_button_start"], target["prev_button_arity"]
            ).masked_fill(
                target["prev_button_arity"].long().eq(0),
                EMPTY_GEOMETRY_ID,
            )
            residual, _ = self.geometry_sequence_residual(
                hidden,
                previous_id,
                target["motion_state"],
            )
            output["geometry_logits"] = output["geometry_logits"] + residual
        output["geometry_sequence_hidden"] = hidden
        if self.future_geometry_horizon:
            output["future_geometry_logits"] = self.future_geometry_head(hidden).view(
                hidden.shape[0],
                hidden.shape[1],
                self.future_geometry_horizon,
                GEOMETRY_CANDIDATE_COUNT,
            )
        return output


class V4RelationalRenderer(V3StructuredRenderer):
    """Clean V4 renderer with relational event-group geometry generation.

    ``persistent_motion_memory`` is deliberately opt-in.  It is a compact,
    unnamed causal state over the renderer history; it is not a vocabulary of
    community pattern names and it is not a playability rule.  Keeping it
    opt-in preserves bit-for-bit behavior of older V4 checkpoints.
    """

    def __init__(
        self,
        config: FactorConfig,
        future_geometry_horizon: int = 0,
        persistent_motion_memory: bool = False,
        blockwise_context_size: int = 0,
        explicit_motion_state: bool = False,
        geometry_sequence_state: bool = False,
        configuration_code_adapter: bool = False,
    ):
        super().__init__(config)
        self.future_geometry_horizon = max(0, int(future_geometry_horizon))
        self.persistent_motion_memory = bool(persistent_motion_memory)
        self.blockwise_context_size = max(0, int(blockwise_context_size))
        self.explicit_motion_state = bool(explicit_motion_state)
        self.geometry_sequence_state = bool(geometry_sequence_state)
        self.configuration_code_adapter = bool(configuration_code_adapter)
        if self.persistent_motion_memory:
            self.motion_memory = nn.GRU(
                config.d_model,
                config.d_model,
                batch_first=True,
            )
            self.motion_memory_norm = nn.LayerNorm(config.d_model)
            # Start as a small residual so the clean relational renderer is
            # the initial behavior and the new state must earn its influence.
            self.motion_memory_gate = nn.Parameter(torch.tensor(-2.0))
        self.heads = nn.ModuleList(
            RelationalStructuredFactorHead(
                config,
                future_geometry_horizon=self.future_geometry_horizon,
                explicit_motion_state=self.explicit_motion_state,
                geometry_sequence_state=self.geometry_sequence_state,
                configuration_code_adapter=self.configuration_code_adapter,
            )
            for _ in range(config.groups)
        )

    def motion_step(
        self,
        target_hidden: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance the same motion state used by training-time sequences."""

        if not self.persistent_motion_memory:
            return target_hidden.new_zeros(target_hidden.shape), state
        state_out, next_state = self.motion_memory(target_hidden, state)
        scale = torch.sigmoid(self.motion_memory_gate).to(state_out.dtype)
        return self.motion_memory_norm(state_out) * scale, next_state

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev_event_embedding: torch.Tensor | None = None,
        soft_self_condition_rho: float = 0.0,
        previous_geometry_mask: torch.Tensor | None = None,
        motion_state_override: torch.Tensor | None = None,
        configuration_state_override: torch.Tensor | None = None,
    ):
        padding = ~batch["valid"].bool()
        condition = (
            self.version(batch["version"].clamp(0, 31))
            + self.difficulty(batch["difficulty"].clamp(0, 4))
            + self.level(batch["level"].clamp(0, 200))
            + self._metadata(batch["metadata"])
            + self.style(batch["style"])
        )
        timing = (
            self.tick(batch["tick"])
            + self.delta(batch["delta"].clamp(0, 1536))
            + self.rhythm(batch["rhythm"])
        )
        memory = self.audio(batch["audio"]) + self.structure(
            batch["structure"]
        ) + timing + condition[:, None]
        memory = self.audio_encoder(memory, src_key_padding_mask=padding)
        previous_event = self._previous_event(batch)
        block_size = self.blockwise_context_size
        if block_size:
            positions = torch.arange(
                previous_event.shape[1], device=previous_event.device
            )
            # The first anchor in a block retains the real preceding event;
            # later positions start without teacher-forced in-block history.
            previous_event = previous_event.masked_fill(
                (positions % block_size).ne(0)[None, :, None], 0.0
            )
        if prev_event_embedding is not None and soft_self_condition_rho > 0.0:
            rho = float(np.clip(soft_self_condition_rho, 0.0, 1.0))
            previous_event = (
                previous_event * (1.0 - rho)
                + prev_event_embedding.to(previous_event.dtype) * rho
            )
        target_hidden = (
            previous_event
            + timing
            + condition[:, None]
            + self.occupied_lanes(batch["occupied_lanes"].float())
            + self.available_hands(batch["available_hands"].long().clamp(0, 2))
            + self._state_embedding(batch)
        )
        causal = torch.ones(
            (target_hidden.shape[1], target_hidden.shape[1]),
            device=target_hidden.device,
            dtype=torch.bool,
        ).triu(1)
        if block_size:
            positions = torch.arange(
                target_hidden.shape[1], device=target_hidden.device
            )
            same_block = (positions[:, None] // block_size).eq(
                positions[None, :] // block_size
            )
            causal = causal & ~same_block
        hidden = self.event_decoder(
            target_hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        if self.persistent_motion_memory:
            motion_hidden, _ = self.motion_step(target_hidden)
            hidden = hidden + motion_hidden
        results = []
        target_names = (
            "button_arity",
            "button_family",
            "button_start",
            "button_route",
            "button_duration",
            "button_modifiers",
            "touch_presence",
            "touch_duration",
            "touch_modifiers",
            "blocked_lanes",
            "prev_button_arity",
            "prev_button_start",
            "prev_button_route",
            "motion_state",
        )
        for group in batch["group"].unique(sorted=True):
            indices = batch["group"].eq(group).nonzero(as_tuple=True)[0]
            target = {}
            for name in target_names:
                value = batch[name]
                if not value.dtype.is_floating_point and value.dtype != torch.bool:
                    value = value.long()
                if block_size and name.startswith("prev_"):
                    value = value.clone()
                    positions = torch.arange(
                        value.shape[1], device=value.device
                    )
                    value[:, positions % block_size != 0] = 0
                target[name] = value[indices]
            geometry_mask = None
            if previous_geometry_mask is not None:
                geometry_mask = previous_geometry_mask[indices]
            motion_state = None
            if motion_state_override is not None:
                motion_state = motion_state_override[indices]
            configuration_state = None
            if configuration_state_override is not None:
                configuration_state = configuration_state_override[indices]
            results.append(
                (
                    indices,
                    self.heads[int(group)](
                        hidden[indices],
                        target,
                        previous_mask_override=geometry_mask,
                        motion_state_override=motion_state,
                        configuration_state_override=configuration_state,
                    ),
                )
            )
        return results
