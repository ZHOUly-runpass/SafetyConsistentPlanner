from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .bspline_candidate_head import BSplineCandidateHead
from .il_head import ILHead
from .safety_heads import SafetyPredictionHeads
from .scene_encoder import VectorSceneEncoder


class SafetyConsistentPlannerModel(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        d_scene = int(cfg.get("d_scene", 256))
        num_candidates = int(cfg.get("num_candidates", 4))
        num_future = int(cfg.get("num_future", 8))
        self.scene_encoder = VectorSceneEncoder({**cfg.get("scene_encoder", {}), "d_scene": d_scene})
        self.il_head = ILHead(d_scene, num_future, int(cfg.get("hidden_dim", 256)))
        self.candidate_head = BSplineCandidateHead(
            d_scene,
            num_candidates=num_candidates,
            num_future=num_future,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
        )
        self.safety_heads = SafetyPredictionHeads(
            d_scene,
            num_candidates=num_candidates,
            hidden_dim=int(cfg.get("safety_hidden_dim", 128)),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.scene_encoder(batch)
        base = self.il_head(embedding)
        candidates = self.candidate_head(embedding, base)
        h_min, feasibility, correction, risk = self.safety_heads(embedding, candidates)
        return {
            "scene_embedding": embedding,
            "base_trajectory": base,
            "candidate_trajectories": candidates,
            "predicted_h_min": h_min,
            "feasibility_logits": feasibility,
            "predicted_correction": correction,
            "predicted_risk": risk,
        }

