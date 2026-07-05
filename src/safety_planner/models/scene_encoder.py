from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..datasets.tensorizer import tensorize_scene
from ..interfaces import SceneSample, TensorizedScene


class VectorSceneEncoder(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config = config or {}
        self.d_scene = int(self._config.get("d_scene", 64))
        d_entity = int(self._config.get("d_entity", max(16, self.d_scene // 4)))
        self.ego_encoder = _entity_mlp(4, d_entity)
        self.agent_encoder = _entity_mlp(7, d_entity)
        self.map_encoder = _entity_mlp(5, d_entity)
        self.route_encoder = _entity_mlp(4, d_entity)
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_entity, self.d_scene),
            nn.ReLU(),
            nn.Linear(self.d_scene, self.d_scene),
        )

    def forward(
        self,
        sample: SceneSample | TensorizedScene | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        batch = self._as_batch(sample)
        ego = self.ego_encoder(batch["ego_history"])
        agents = self.agent_encoder(batch["agent_history"])
        map_features = self.map_encoder(batch["map_polylines"])
        route = self.route_encoder(batch["route_polyline"])
        ego_pool = _masked_mean(ego, batch["ego_history_mask"], (1,))
        agent_pool = _masked_mean(agents, batch["agent_history_mask"], (1, 2))
        map_pool = _masked_mean(map_features, batch["map_mask"], (1, 2))
        route_pool = _masked_mean(route, batch["route_mask"], (1,))
        return self.fusion(torch.cat((ego_pool, agent_pool, map_pool, route_pool), dim=-1))

    def _as_batch(
        self,
        sample: SceneSample | TensorizedScene | Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(sample, SceneSample):
            sample = tensorize_scene(sample)
        names = (
            "ego_history",
            "ego_history_mask",
            "agent_history",
            "agent_history_mask",
            "map_polylines",
            "map_mask",
            "route_polyline",
            "route_mask",
        )
        if isinstance(sample, TensorizedScene):
            return {
                name: torch.as_tensor(getattr(sample, name), device=self._device).unsqueeze(0)
                for name in names
            }
        missing = [name for name in names if name not in sample]
        if missing:
            raise ValueError(f"Tensorized scene batch is missing: {missing}.")
        return {name: sample[name].to(self._device) for name in names}

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


def _entity_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
        nn.ReLU(),
    )


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dimensions: tuple[int, ...],
) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = values * weights
    denominator = weights
    for dimension in sorted(dimensions, reverse=True):
        numerator = numerator.sum(dim=dimension)
        denominator = denominator.sum(dim=dimension)
    return numerator / denominator.clamp_min(1.0)
