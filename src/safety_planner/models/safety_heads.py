from __future__ import annotations

import torch
from torch import nn


class SafetyPredictionHeads(nn.Module):
    def __init__(self, d_scene: int, num_candidates: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_candidates = num_candidates

        self.shared_mlp = nn.Sequential(
            nn.Linear(d_scene, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.h_min_head = nn.Linear(hidden_dim, num_candidates)
        self.feasibility_head = nn.Linear(hidden_dim, num_candidates)
        self.correction_head = nn.Linear(hidden_dim, num_candidates)
        self.risk_head = nn.Linear(hidden_dim, num_candidates)

    def forward(
        self, scene_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input: [B, D_scene]
        # Returns:
        #   predicted_h_min:      [B, G]  continuous
        #   feasibility_logits:   [B, G]  binary classification
        #   predicted_correction: [B, G]  continuous
        #   predicted_risk:       [B, G]  continuous
        feat = self.shared_mlp(scene_embedding)
        return (
            self.h_min_head(feat),
            self.feasibility_head(feat),
            self.correction_head(feat),
            self.risk_head(feat),
        )


class LightweightRanker(nn.Module):
    def __init__(self, d_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        # Input:  [B, G, d_features]  (stacked analytical cost + safety prediction features)
        # Output: [B, G]  (ranking scores)
        return self.mlp(candidate_features).squeeze(-1)
