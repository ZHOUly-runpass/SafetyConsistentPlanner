from __future__ import annotations

import torch
from torch import nn

from ..interfaces.scene import SceneSample


class VectorSceneEncoder(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config = config or {}
        self.d_scene = int(self._config.get("d_scene", 64))
        self.encoder = nn.Sequential(
            nn.Linear(16, self.d_scene),
            nn.ReLU(),
            nn.Linear(self.d_scene, self.d_scene),
        )

    def forward(self, sample: SceneSample) -> torch.Tensor:
        sample.validate()
        features = torch.as_tensor(self._extract_stats(sample), dtype=torch.float32).unsqueeze(0)
        return self.encoder(features)

    def _extract_stats(self, sample: SceneSample) -> list[float]:
        ego_last = [0.0, 0.0, 0.0, 0.0]
        ego_mean = [0.0, 0.0, 0.0, 0.0]
        if sample.ego_history is not None and sample.ego_history.shape[0] > 0:
            valid = sample.ego_history_mask if sample.ego_history_mask is not None else None
            ego = sample.ego_history[valid] if valid is not None and valid.any() else sample.ego_history
            ego_last = ego[-1, :4].astype(float).tolist()
            ego_mean = ego[:, :4].mean(axis=0).astype(float).tolist()

        agent_count = 0.0
        valid_agent_steps = 0.0
        if sample.agent_history is not None:
            agent_count = float(sample.agent_history.shape[0])
            if sample.agent_history_mask is not None:
                valid_agent_steps = float(sample.agent_history_mask.sum())

        map_count = float(len(sample.map_polylines or []))
        route_points = float(sample.route_polyline.shape[0]) if sample.route_polyline is not None else 0.0
        expert_points = float(sample.expert_future.shape[0]) if sample.expert_future is not None else 0.0
        timestamp_sec = float(sample.timestamp_us) / 1_000_000.0

        return (
            ego_last
            + ego_mean
            + [
                agent_count,
                valid_agent_steps,
                map_count,
                route_points,
                expert_points,
                timestamp_sec,
                1.0 if sample.scenario_id else 0.0,
                1.0,
            ]
        )
