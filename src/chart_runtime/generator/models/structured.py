from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chart_runtime.generator.models.factor import FactorConfig, FactorEventModel


class StructuredFactorHead(nn.Module):
    def __init__(self, config: FactorConfig):
        super().__init__()
        self.c = config
        d = config.d_model
        self.adapter = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.arity_head = nn.Linear(d, 3)
        self.arity_embedding = nn.Embedding(3, d)
        self.family_head = nn.ModuleList(nn.Linear(d, 4) for _ in range(2))
        self.family_embedding = nn.Embedding(4, d)
        self.start_head = nn.ModuleList(nn.Linear(d, 8) for _ in range(2))
        self.start_embedding = nn.Embedding(8, d)
        self.route_head = nn.ModuleList(nn.Linear(d, config.routes) for _ in range(2))
        self.route_embedding = nn.Embedding(config.routes, d)
        self.duration_head = nn.ModuleList(
            nn.Linear(d, config.durations) for _ in range(2)
        )
        self.duration_embedding = nn.Embedding(config.durations, d)
        self.modifier_head = nn.ModuleList(nn.Linear(d, 5) for _ in range(2))
        self.touch_sensor = nn.Embedding(config.touch_positions, d)
        self.touch_presence = nn.Linear(d, config.touch_positions)
        self.touch_hold_presence = nn.Linear(d, 1)
        self.touch_duration = nn.Linear(d, config.durations)
        self.touch_modifiers = nn.Linear(d, 5)
        self.event_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.context_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.muri_state_projection = nn.Sequential(
            nn.LayerNorm(32), nn.Linear(32, d), nn.GELU(), nn.Linear(d, d)
        )
        risk_hidden = max(32, d // 2)
        self.muri_risk_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, risk_hidden), nn.GELU(),
            nn.Linear(risk_hidden, 11),
        )
        nn.init.zeros_(self.muri_state_projection[-1].weight)
        nn.init.zeros_(self.muri_state_projection[-1].bias)
        nn.init.zeros_(self.muri_risk_head[-1].weight)
        nn.init.zeros_(self.muri_risk_head[-1].bias)

    @staticmethod
    def _condition(
        logits: torch.Tensor,
        embedding: nn.Embedding,
        target: torch.Tensor | None,
    ) -> torch.Tensor:
        if target is not None:
            return embedding(target.long())
        probability = logits.float().softmax(dim=-1).to(logits.dtype)
        return probability @ embedding.weight

    def event_embedding(self, target: dict[str, torch.Tensor]) -> torch.Tensor:
        arity = target["button_arity"].long().clamp(0, 2)
        result = self.arity_embedding(arity)
        note_mask = (
            torch.arange(2, device=arity.device)[None, None] < arity[..., None]
        )
        buttons = (
            self.family_embedding(target["button_family"].long())
            + self.start_embedding(target["button_start"].long().clamp(0, 7))
            + self.route_embedding(target["button_route"].long())
            + self.duration_embedding(target["button_duration"].long())
        )
        result = result + (buttons * note_mask[..., None]).sum(dim=2)
        touch_mask = target["touch_presence"].bool()
        sensors = torch.arange(self.c.touch_positions, device=arity.device)[None, None]
        touches = self.touch_sensor(sensors) + self.duration_embedding(
            target["touch_duration"].long()
        )
        result = result + (touches * touch_mask[..., None]).sum(dim=2)
        return result

    def soft_event_embedding(
        self,
        output: dict[str, torch.Tensor],
        hard_geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Expected event embedding under the model's factor distributions.

        This is used only for stop-gradient soft self-conditioning.  It keeps
        the event-factor decomposition intact instead of inventing a separate
        learned shortcut for the previous event.
        """

        arity_probability = F.softmax(output["button_arity"].float(), dim=-1)
        embedding_dtype = self.arity_embedding.weight.dtype
        if hard_geometry is None:
            result = arity_probability.to(embedding_dtype) @ self.arity_embedding.weight
        else:
            hard_arity = hard_geometry["button_arity"].long().clamp(0, 2)
            result = self.arity_embedding(hard_arity)
            hard_starts = hard_geometry["button_start"].long().clamp(0, 7)
        for note_index in range(2):
            if hard_geometry is None:
                gate = arity_probability[..., note_index + 1 :].sum(dim=-1, keepdim=True)
            else:
                gate = hard_arity.gt(note_index).to(embedding_dtype).unsqueeze(-1)
            family = F.softmax(output["button_family"][..., note_index, :].float(), dim=-1)
            route = F.softmax(output["button_route"][..., note_index, :].float(), dim=-1)
            duration = F.softmax(
                output["button_duration"][..., note_index, :].float(),
                dim=-1,
            )
            factor = (
                family.to(embedding_dtype) @ self.family_embedding.weight
                + (
                    self.start_embedding(hard_starts[..., note_index])
                    if hard_geometry is not None
                    else F.softmax(output["button_start"][..., note_index, :].float(), dim=-1).to(embedding_dtype)
                    @ self.start_embedding.weight
                )
                + route.to(embedding_dtype) @ self.route_embedding.weight
                + duration.to(embedding_dtype) @ self.duration_embedding.weight
            )
            result = result + gate.to(embedding_dtype) * factor
        touch_probability = output["touch_presence"].float().sigmoid()
        touch_duration = F.softmax(output["touch_duration"].float(), dim=-1)
        touch_embedding = self.touch_sensor.weight[None, None, :, :].to(embedding_dtype)
        touch_embedding = touch_embedding + (
            touch_duration.to(embedding_dtype) @ self.duration_embedding.weight
        )
        result = result + (
            touch_probability[..., None].to(embedding_dtype) * touch_embedding
        ).sum(dim=2)
        return result

    def geometry_soft_event_embedding(
        self,
        output: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Expected event embedding with only button starts softened."""

        arity = target["button_arity"].long().clamp(0, 2)
        embedding_dtype = self.arity_embedding.weight.dtype
        result = self.arity_embedding(arity)
        note_mask = (
            torch.arange(2, device=arity.device)[None, None] < arity[..., None]
        )
        for note_index in range(2):
            family = self.family_embedding(
                target["button_family"][..., note_index].long()
            )
            start_probability = F.softmax(
                output["button_start"][..., note_index, :].float(), dim=-1
            ).to(embedding_dtype)
            start = start_probability @ self.start_embedding.weight
            route = self.route_embedding(
                target["button_route"][..., note_index].long()
            )
            duration = self.duration_embedding(
                target["button_duration"][..., note_index].long()
            )
            result = result + note_mask[..., note_index, None].to(embedding_dtype) * (
                family + start + route + duration
            )
        touch_mask = target["touch_presence"].bool()
        sensors = torch.arange(self.c.touch_positions, device=arity.device)[None, None]
        touches = self.touch_sensor(sensors) + self.duration_embedding(
            target["touch_duration"].long()
        )
        result = result + (
            touches * touch_mask[..., None].to(embedding_dtype)
        ).sum(dim=2)
        return result

    def forward(
        self,
        hidden: torch.Tensor,
        target: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.adapter(hidden)
        arity_logits = self.arity_head(hidden)
        arity_target = None if target is None else target["button_arity"]
        arity_condition = self._condition(
            arity_logits, self.arity_embedding, arity_target
        )
        running = hidden + arity_condition
        family_logits = []
        start_logits = []
        route_logits = []
        duration_logits = []
        modifier_logits = []
        summaries = []
        for note_index in range(2):
            note_hidden = running + sum(summaries, torch.zeros_like(running))
            family = self.family_head[note_index](note_hidden)
            family_condition = self._condition(
                family,
                self.family_embedding,
                None if target is None else target["button_family"][..., note_index],
            )
            start = self.start_head[note_index](note_hidden + family_condition)
            start_condition = self._condition(
                start,
                self.start_embedding,
                None if target is None else target["button_start"][..., note_index],
            )
            route = self.route_head[note_index](
                note_hidden + family_condition + start_condition
            )
            route_condition = self._condition(
                route,
                self.route_embedding,
                None if target is None else target["button_route"][..., note_index],
            )
            duration = self.duration_head[note_index](
                note_hidden + family_condition + start_condition + route_condition
            )
            duration_condition = self._condition(
                duration,
                self.duration_embedding,
                None if target is None else target["button_duration"][..., note_index],
            )
            modifier = self.modifier_head[note_index](
                note_hidden
                + family_condition
                + start_condition
                + route_condition
                + duration_condition
            )
            summaries.append(
                family_condition
                + start_condition
                + route_condition
                + duration_condition
            )
            family_logits.append(family)
            start_logits.append(start)
            route_logits.append(route)
            duration_logits.append(duration)
            modifier_logits.append(modifier)
        event_context = running + sum(summaries, torch.zeros_like(running))
        touch_presence = self.touch_presence(event_context)
        sensors = torch.arange(
            self.c.touch_positions, device=hidden.device
        )[None, None]
        touch_hidden = event_context[..., None, :] + self.touch_sensor(sensors)
        touch_hold_presence = self.touch_hold_presence(touch_hidden).squeeze(-1)
        touch_duration = self.touch_duration(touch_hidden)
        touch_modifiers = self.touch_modifiers(touch_hidden)
        output = {
            "button_arity": arity_logits,
            "button_family": torch.stack(family_logits, dim=2),
            "button_start": torch.stack(start_logits, dim=2),
            "button_route": torch.stack(route_logits, dim=2),
            "button_duration": torch.stack(duration_logits, dim=2),
            "button_modifiers": torch.stack(modifier_logits, dim=2),
            "touch_presence": touch_presence,
            "touch_hold_presence": touch_hold_presence,
            "touch_duration": touch_duration,
            "touch_modifiers": touch_modifiers,
        }
        if target is not None:
            positive_event = self.event_embedding(target)
            negative_target = {
                key: value.clone()
                for key, value in target.items()
            }
            negative_target["button_start"] = (
                negative_target["button_start"].long() + 3
            ) % 8
            if "blocked_lanes" in target:
                blocked = target["blocked_lanes"].bool()
                blocked_start = blocked.float().argmax(dim=-1)
                has_blocked = blocked.any(dim=-1)
                negative_target["button_start"][..., 0] = torch.where(
                    has_blocked,
                    blocked_start,
                    negative_target["button_start"][..., 0],
                )
            negative_target["button_duration"] = torch.roll(
                negative_target["button_duration"], shifts=1, dims=1
            )
            negative_target["touch_presence"] = torch.roll(
                negative_target["touch_presence"], shifts=5, dims=-1
            )
            negative_event = self.event_embedding(negative_target)
            context = self.context_projection(hidden)
            output["qualityPositive"] = (
                context * self.event_projection(positive_event)
            ).sum(dim=-1) / np.sqrt(self.c.d_model)
            output["qualityNegative"] = (
                context * self.event_projection(negative_event)
            ).sum(dim=-1) / np.sqrt(self.c.d_model)
        return output

    def muri_legality_tensors(
        self,
        hidden: torch.Tensor,
        target: dict[str, torch.Tensor],
        state_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        event = self.event_projection(self.event_embedding(target))
        context = self.context_projection(hidden)
        # Preserve the accepted V1 preference ordering exactly.  The explicit
        # Slide/judge state is consumed only by the new risk critic, so a weak
        # or poorly calibrated V2 critic cannot silently destroy V1 quality.
        preference = (context * event).sum(dim=-1) / np.sqrt(self.c.d_model)
        state = state_features.float()
        if state.ndim == 2:
            state = state[:, None, :]
        state_context = self.muri_state_projection(state).to(context.dtype)
        risk_logits = self.muri_risk_head(
            torch.tanh(context + state_context + event)
        )
        return preference, risk_logits

    def _single_representation_target(
        self, representation: dict, device: torch.device
    ) -> dict[str, torch.Tensor]:
        target = {}
        for key in (
            "button_arity", "button_family", "button_start", "button_route",
            "button_duration", "touch_presence", "touch_duration",
        ):
            value = torch.as_tensor(representation[key], device=device)
            target[key] = value.reshape(1, 1, *value.shape)
        return target

    @torch.no_grad()
    def score_representation(
        self, hidden: torch.Tensor, representation: dict,
        state_features=None,
    ) -> float:
        target = self._single_representation_target(representation, hidden.device)
        if state_features is None:
            event = self.event_projection(self.event_embedding(target))
            context = self.context_projection(hidden)
            return float((context * event).sum() / np.sqrt(self.c.d_model))

        state = torch.as_tensor(state_features, device=hidden.device).reshape(1, 1, 32)
        preference, _ = self.muri_legality_tensors(hidden, target, state)
        return float(preference[0, 0])

    @torch.no_grad()
    def score_legality(
        self, hidden: torch.Tensor, representation: dict, state_features
    ) -> dict[str, float | list[float]]:
        target = self._single_representation_target(representation, hidden.device)
        state = torch.as_tensor(state_features, device=hidden.device).reshape(1, 1, 32)
        preference, logits = self.muri_legality_tensors(hidden, target, state)
        probability = logits[0, 0].float().softmax(dim=-1)
        interaction = probability[1] + probability[4] + probability[5]
        risk = (1.0 - probability[0]) + 2.0 * probability[2] + 0.5 * interaction
        return {
            "preferenceScore": float(preference[0, 0]),
            "legalProbability": float(probability[0]),
            "slideTooFastProbability": float(probability[2]),
            "predictedRisk": float(risk),
            "classProbabilities": probability.cpu().tolist(),
        }

    @torch.no_grad()
    def score_legality_batch(
        self, hidden: torch.Tensor, representations: list[dict], state_features
    ) -> list[dict[str, float]]:
        if not representations:
            return []
        target = {}
        for key in (
            "button_arity", "button_family", "button_start", "button_route",
            "button_duration", "touch_presence", "touch_duration",
        ):
            values = np.stack([np.asarray(rep[key]) for rep in representations], axis=0)
            tensor = torch.as_tensor(values, device=hidden.device)
            target[key] = tensor[:, None, ...]
        count = len(representations)
        hidden_batch = hidden.expand(count, -1, -1)
        state = torch.as_tensor(state_features, device=hidden.device).reshape(1, 1, 32)
        state = state.expand(count, -1, -1)
        preference, logits = self.muri_legality_tensors(hidden_batch, target, state)
        probability = logits[:, 0].float().softmax(dim=-1)
        interaction = probability[:, 1] + probability[:, 4] + probability[:, 5]
        risk = (1.0 - probability[:, 0]) + 2.0 * probability[:, 2] + 0.5 * interaction
        packed = torch.stack(
            (preference[:, 0], probability[:, 0], probability[:, 2], risk), dim=-1
        ).float().cpu().numpy()
        return [
            {
                "preferenceScore": float(row[0]),
                "legalProbability": float(row[1]),
                "slideTooFastProbability": float(row[2]),
                "predictedRisk": float(row[3]),
            }
            for row in packed
        ]



class V3StructuredRenderer(FactorEventModel):
    def __init__(self, config: FactorConfig):
        super().__init__(config)
        d = config.d_model
        self.heads = nn.ModuleList(
            StructuredFactorHead(config) for _ in range(config.groups)
        )
        self.active_holds = nn.Embedding(3, d)
        self.active_slides = nn.Embedding(3, d)
        self.active_touch_holds = nn.Embedding(3, d)
        self.conservative_available_hands = nn.Embedding(3, d)
        self.blocked_lanes = nn.Linear(8, d, bias=False)

    def _state_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.active_holds(batch["active_holds"].long().clamp(0, 2))
            + self.active_slides(batch["active_slides"].long().clamp(0, 2))
            + self.active_touch_holds(
                batch["active_touch_holds"].long().clamp(0, 2)
            )
            + self.conservative_available_hands(
                batch["conservative_available_hands"].long().clamp(0, 2)
            )
            + self.blocked_lanes(batch["blocked_lanes"].float())
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev_event_embedding: torch.Tensor | None = None,
        soft_self_condition_rho: float = 0.0,
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
        hidden = self.event_decoder(
            target_hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        results = []
        target_names = (
            "button_arity",
            "button_family",
            "button_start",
            "button_route",
            "button_duration",
            "touch_presence",
            "touch_duration",
            "blocked_lanes",
        )
        for group in batch["group"].unique(sorted=True):
            indices = batch["group"].eq(group).nonzero(as_tuple=True)[0]
            target = {}
            for name in target_names:
                value = batch[name]
                if not value.dtype.is_floating_point and value.dtype != torch.bool:
                    value = value.long()
                target[name] = value[indices]
            results.append(
                (indices, self.heads[int(group)](hidden[indices], target))
            )
        return results
