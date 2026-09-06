from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PlannerConfig:
    mert_dim: int = 768
    d_model: int = 256
    max_bars: int = 256
    segment_bars: int = 4
    director_layers: int = 4
    segment_layers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class FullSongBudgetPlanner(nn.Module):
    """Allocate one whole-chart event budget hierarchically over segments/bars."""

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.c = config
        d_model = config.d_model
        self.mert = nn.Linear(config.mert_dim, d_model)
        self.bpm = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.position = nn.Embedding(config.max_bars, d_model)
        self.segment_position = nn.Embedding(
            config.max_bars // config.segment_bars, d_model
        )
        self.version = nn.Embedding(32, d_model)
        self.slot = nn.Embedding(5, d_model)
        self.level = nn.Embedding(201, d_model)
        bar_layer = nn.TransformerEncoderLayer(
            d_model,
            8,
            1024,
            0.1,
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bar_director = nn.TransformerEncoder(
            bar_layer,
            config.director_layers,
            enable_nested_tensor=False,
        )
        segment_layer = nn.TransformerEncoderLayer(
            d_model,
            8,
            1024,
            0.1,
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.segment_director = nn.TransformerEncoder(
            segment_layer,
            config.segment_layers,
            enable_nested_tensor=False,
        )
        self.segment_allocation = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
        self.bar_allocation = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.total = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.version(batch["version"].long().clamp(0, 31))
            + self.slot(batch["slot"].long().clamp(0, 4))
            + self.level(batch["level"].long().clamp(0, 200))
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        segment_noise: torch.Tensor | None = None,
        section_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = batch["bar_mask"].bool()
        positions = torch.arange(self.c.max_bars, device=mask.device)
        bpm_features = torch.stack(
            (torch.log1p(batch["bpm"].float()) / 6.0, batch["bpm"].float() / 300.0),
            dim=-1,
        )
        condition = self.condition(batch)
        bars = self.bar_director(
            self.mert(batch["mert"].float())
            + self.bpm(bpm_features)
            + self.position(positions)[None]
            + condition[:, None],
            src_key_padding_mask=~mask,
        )
        segment_bars = self.c.segment_bars
        segment_count = self.c.max_bars // segment_bars
        grouped = bars.view(len(bars), segment_count, segment_bars, -1)
        grouped_mask = mask.view(len(mask), segment_count, segment_bars)
        segment_mask = grouped_mask.any(dim=-1)
        segments = (grouped * grouped_mask[..., None]).sum(dim=2) / grouped_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        segment_positions = torch.arange(segment_count, device=mask.device)
        segments = self.segment_director(
            segments + self.segment_position(segment_positions)[None],
            src_key_padding_mask=~segment_mask,
        )
        segment_logits = self.segment_allocation(segments).squeeze(-1)
        if segment_noise is not None:
            segment_logits = segment_logits + segment_noise
        segment_logits = segment_logits.masked_fill(~segment_mask, -torch.inf)
        segment_probability = segment_logits.softmax(dim=-1)
        expanded_segments = segments.repeat_interleave(segment_bars, dim=1)
        bar_logits = self.bar_allocation(
            torch.cat((bars, expanded_segments), dim=-1)
        ).squeeze(-1)
        bar_logits = bar_logits.masked_fill(~mask, -torch.inf)
        within_logits = bar_logits.view(
            len(bars), segment_count, segment_bars
        ).masked_fill(~grouped_mask, -torch.inf)
        within_probability = within_logits.softmax(dim=-1)
        within_probability = torch.nan_to_num(within_probability)
        allocation = (
            segment_probability[..., None] * within_probability
        ).view(len(bars), self.c.max_bars)
        bar_summary = (bars * mask[..., None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        segment_summary = (
            segments * segment_mask[..., None]
        ).sum(dim=1) / segment_mask.sum(dim=1, keepdim=True).clamp_min(1)
        total = torch.nn.functional.softplus(
            self.total(torch.cat((bar_summary, segment_summary), dim=-1)).squeeze(-1)
        )
        expected = allocation * total[:, None]
        return {
            "expected": expected,
            "allocation": allocation,
            "total": total,
            "segmentProbability": segment_probability,
            "withinProbability": within_probability,
            "segmentLogits": segment_logits,
            "barLogits": bar_logits,
            "barHidden": bars,
            "segmentHidden": segments,
        }


@dataclass
class PersistentPlannerConfig(PlannerConfig):
    """Configuration for the learned whole-song section-state planner.

    ``state_count`` is only the capacity of the latent state vocabulary.  The
    states have no hand-written names or meanings; their use and transitions
    are learned from the chart/audio objective.
    """

    state_count: int = 16


class PersistentSectionPlanner(FullSongBudgetPlanner):
    """Whole-song planner with a learned persistent section-state sequence.

    The original planner predicts every segment allocation directly.  This
    variant first infers a soft state sequence with a trainable transition
    matrix and explicit geometric durations (self-transition probabilities),
    then derives the coarse segment allocation from that sequence.  There is
    no minimum-duration rule, state naming, or manually assigned chart style.
    """

    supports_section_noise = True

    def __init__(self, config: PersistentPlannerConfig):
        # Recreate the V3.1 trunk with identical module names so a clean V3.1
        # checkpoint can be used as a warm start.  The new section modules are
        # initialized separately and are the only new parameters.
        super().__init__(
            PlannerConfig(
                mert_dim=config.mert_dim,
                d_model=config.d_model,
                max_bars=config.max_bars,
                segment_bars=config.segment_bars,
                director_layers=config.director_layers,
                segment_layers=config.segment_layers,
            )
        )
        self.c = config
        state_count = config.state_count
        d_model = config.d_model
        self.state_emission = nn.Linear(d_model, state_count)
        self.state_embedding = nn.Parameter(torch.randn(state_count, d_model) * 0.02)
        self.state_profile = nn.Parameter(torch.randn(state_count) * 0.08)
        # A soft starting preference for persistence, not a hard transition
        # mask.  Training remains free to learn any transition structure.
        self.transition_logits = nn.Parameter(torch.eye(state_count) * 1.5)
        self.section_residual = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
        self.residual_gate = nn.Parameter(torch.tensor(-1.0))
        self.bar_allocation_state = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _forward_backward(
        emission_logits: torch.Tensor,
        segment_mask: torch.Tensor,
        transition_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft state posteriors under a learned duration model."""

        # Do the small HMM calculation in fp32.  The surrounding model can
        # still use CUDA autocast; this prevents underflow for long songs.
        emission_logits = emission_logits.float()
        segment_mask = segment_mask.bool()
        transition_logp = F.log_softmax(transition_logits.float(), dim=-1)
        emission_logp = F.log_softmax(emission_logits, dim=-1)
        batch_size, segment_count, _ = emission_logp.shape
        forward = torch.zeros_like(emission_logp)
        if segment_count:
            forward[:, 0] = torch.where(
                segment_mask[:, 0, None], emission_logp[:, 0], torch.zeros_like(emission_logp[:, 0])
            )
        for index in range(1, segment_count):
            transitioned = torch.logsumexp(
                forward[:, index - 1, :, None] + transition_logp[None], dim=1
            ) + emission_logp[:, index]
            forward[:, index] = torch.where(
                segment_mask[:, index, None], transitioned, forward[:, index - 1]
            )

        backward = torch.zeros_like(emission_logp)
        for index in range(segment_count - 2, -1, -1):
            next_active = segment_mask[:, index + 1, None]
            next_value = torch.where(
                next_active,
                emission_logp[:, index + 1] + backward[:, index + 1],
                torch.zeros_like(emission_logp[:, index + 1]),
            )
            transitioned = torch.logsumexp(
                transition_logp[None] + next_value[:, None, :], dim=-1
            )
            backward[:, index] = torch.where(
                segment_mask[:, index, None], transitioned, torch.zeros_like(transitioned)
            )
        posterior = F.softmax(forward + backward, dim=-1)
        return posterior * segment_mask[..., None]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        segment_noise: torch.Tensor | None = None,
        section_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = batch["bar_mask"].bool()
        positions = torch.arange(self.c.max_bars, device=mask.device)
        bpm_features = torch.stack(
            (torch.log1p(batch["bpm"].float()) / 6.0, batch["bpm"].float() / 300.0),
            dim=-1,
        )
        condition = self.condition(batch)
        bars = self.bar_director(
            self.mert(batch["mert"].float())
            + self.bpm(bpm_features)
            + self.position(positions)[None]
            + condition[:, None],
            src_key_padding_mask=~mask,
        )
        segment_bars = self.c.segment_bars
        segment_count = self.c.max_bars // segment_bars
        grouped = bars.view(len(bars), segment_count, segment_bars, -1)
        grouped_mask = mask.view(len(mask), segment_count, segment_bars)
        segment_mask = grouped_mask.any(dim=-1)
        segments = (grouped * grouped_mask[..., None]).sum(dim=2) / grouped_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        segment_positions = torch.arange(segment_count, device=mask.device)
        segments = self.segment_director(
            segments + self.segment_position(segment_positions)[None],
            src_key_padding_mask=~segment_mask,
        )

        emission_logits = self.state_emission(segments)
        if section_noise is not None:
            emission_logits = emission_logits + section_noise
        state_probability = self._forward_backward(
            emission_logits, segment_mask, self.transition_logits
        )
        state_context = state_probability.to(segments.dtype) @ self.state_embedding
        state_profile = state_probability.to(segments.dtype) @ self.state_profile

        # The persistent state path is the coarse allocation backbone.  A
        # learned residual preserves local musical detail without making the
        # section path optional again.
        residual = self.section_residual(segments).squeeze(-1)
        residual = torch.sigmoid(self.residual_gate) * residual
        segment_logits = state_profile + residual
        if segment_noise is not None:
            segment_logits = segment_logits + segment_noise
        segment_logits = segment_logits.masked_fill(~segment_mask, -torch.inf)
        segment_probability = torch.nan_to_num(segment_logits.softmax(dim=-1))

        expanded_segments = segments.repeat_interleave(segment_bars, dim=1)
        expanded_state = state_context.repeat_interleave(segment_bars, dim=1)
        bar_logits = self.bar_allocation_state(
            torch.cat((bars, expanded_segments, expanded_state), dim=-1)
        ).squeeze(-1)
        bar_logits = bar_logits.masked_fill(~mask, -torch.inf)
        within_logits = bar_logits.view(
            len(bars), segment_count, segment_bars
        ).masked_fill(~grouped_mask, -torch.inf)
        within_probability = torch.nan_to_num(within_logits.softmax(dim=-1))
        allocation = (
            segment_probability[..., None] * within_probability
        ).view(len(bars), self.c.max_bars)

        bar_summary = (bars * mask[..., None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        segment_summary = (
            segments * segment_mask[..., None]
        ).sum(dim=1) / segment_mask.sum(dim=1, keepdim=True).clamp_min(1)
        total = torch.nn.functional.softplus(
            self.total(torch.cat((bar_summary, segment_summary), dim=-1)).squeeze(-1)
        )
        expected = allocation * total[:, None]
        transition_probability = self.transition_logits.float().softmax(dim=-1)
        if segment_count > 1:
            pair_mask = segment_mask[:, 1:] & segment_mask[:, :-1]
            switch_probability = 1.0 - (
                state_probability[:, 1:] * state_probability[:, :-1]
            ).sum(dim=-1)
            switch_probability = (
                switch_probability * pair_mask
            ).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1)
        else:
            switch_probability = state_probability.sum(dim=-1).sum(dim=-1) * 0
        return {
            "expected": expected,
            "allocation": allocation,
            "total": total,
            "segmentProbability": segment_probability,
            "withinProbability": within_probability,
            "segmentLogits": segment_logits,
            "barLogits": bar_logits,
            "barHidden": bars,
            "segmentHidden": segments,
            "stateProbability": state_probability,
            "stateLogits": emission_logits,
            "stateContext": state_context,
            "transitionProbability": transition_probability,
            "sectionSwitchProbability": switch_probability,
        }


class PersistentResidualSectionPlanner(PersistentSectionPlanner):
    """Persistent planner that preserves the clean V3.1 allocation heads.

    The first V3.3 experiment made a newly initialized section/bar head the
    entire budget path.  This control keeps the V3.1 segment and bar heads as
    the main path and lets the learned state sequence add a trainable,
    low-frequency residual.  It therefore tests section persistence without
    discarding the renderer-facing distribution already learned by V3.1.
    """

    supports_section_noise = True

    def __init__(self, config: PersistentPlannerConfig):
        super().__init__(config)
        self.residual_gate.data.fill_(-2.0)
        self.state_bar_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        segment_noise: torch.Tensor | None = None,
        section_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = batch["bar_mask"].bool()
        positions = torch.arange(self.c.max_bars, device=mask.device)
        bpm_features = torch.stack(
            (torch.log1p(batch["bpm"].float()) / 6.0, batch["bpm"].float() / 300.0),
            dim=-1,
        )
        condition = self.condition(batch)
        bars = self.bar_director(
            self.mert(batch["mert"].float())
            + self.bpm(bpm_features)
            + self.position(positions)[None]
            + condition[:, None],
            src_key_padding_mask=~mask,
        )
        segment_bars = self.c.segment_bars
        segment_count = self.c.max_bars // segment_bars
        grouped = bars.view(len(bars), segment_count, segment_bars, -1)
        grouped_mask = mask.view(len(mask), segment_count, segment_bars)
        segment_mask = grouped_mask.any(dim=-1)
        segments = (grouped * grouped_mask[..., None]).sum(dim=2) / grouped_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        segment_positions = torch.arange(segment_count, device=mask.device)
        segments = self.segment_director(
            segments + self.segment_position(segment_positions)[None],
            src_key_padding_mask=~segment_mask,
        )

        emission_logits = self.state_emission(segments)
        if section_noise is not None:
            emission_logits = emission_logits + section_noise
        state_probability = self._forward_backward(
            emission_logits, segment_mask, self.transition_logits
        )
        state_context = state_probability.to(segments.dtype) @ self.state_embedding
        state_profile = state_probability.to(segments.dtype) @ self.state_profile

        # V3.1 remains the renderer-facing coarse allocation.  The learned
        # state path can modulate it, but cannot replace it with an untrained
        # distribution.
        segment_logits = self.segment_allocation(segments).squeeze(-1)
        segment_logits = segment_logits + torch.sigmoid(self.residual_gate) * state_profile
        if segment_noise is not None:
            segment_logits = segment_logits + segment_noise
        segment_logits = segment_logits.masked_fill(~segment_mask, -torch.inf)
        segment_probability = torch.nan_to_num(segment_logits.softmax(dim=-1))

        expanded_segments = segments.repeat_interleave(segment_bars, dim=1)
        expanded_state = state_context.repeat_interleave(segment_bars, dim=1)
        base_bar_logits = self.bar_allocation(
            torch.cat((bars, expanded_segments), dim=-1)
        ).squeeze(-1)
        state_bar_logits = self.bar_allocation_state(
            torch.cat((bars, expanded_segments, expanded_state), dim=-1)
        ).squeeze(-1)
        bar_logits = base_bar_logits + torch.sigmoid(self.state_bar_gate) * state_bar_logits
        bar_logits = bar_logits.masked_fill(~mask, -torch.inf)
        within_logits = bar_logits.view(
            len(bars), segment_count, segment_bars
        ).masked_fill(~grouped_mask, -torch.inf)
        within_probability = torch.nan_to_num(within_logits.softmax(dim=-1))
        allocation = (
            segment_probability[..., None] * within_probability
        ).view(len(bars), self.c.max_bars)

        bar_summary = (bars * mask[..., None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        segment_summary = (
            segments * segment_mask[..., None]
        ).sum(dim=1) / segment_mask.sum(dim=1, keepdim=True).clamp_min(1)
        total = torch.nn.functional.softplus(
            self.total(torch.cat((bar_summary, segment_summary), dim=-1)).squeeze(-1)
        )
        expected = allocation * total[:, None]
        transition_probability = self.transition_logits.float().softmax(dim=-1)
        if segment_count > 1:
            pair_mask = segment_mask[:, 1:] & segment_mask[:, :-1]
            switch_probability = 1.0 - (
                state_probability[:, 1:] * state_probability[:, :-1]
            ).sum(dim=-1)
            switch_probability = (
                switch_probability * pair_mask
            ).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1)
        else:
            switch_probability = state_probability.sum(dim=-1).sum(dim=-1) * 0
        return {
            "expected": expected,
            "allocation": allocation,
            "total": total,
            "segmentProbability": segment_probability,
            "withinProbability": within_probability,
            "segmentLogits": segment_logits,
            "barLogits": bar_logits,
            "barHidden": bars,
            "segmentHidden": segments,
            "stateProbability": state_probability,
            "stateLogits": emission_logits,
            "stateContext": state_context,
            "transitionProbability": transition_probability,
            "sectionSwitchProbability": switch_probability,
        }
