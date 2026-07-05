from __future__ import annotations

import torch
from torch import nn


class SafetyPredictionHeads(nn.Module):
    def __init__(
        self,
        d_scene: int,
        num_candidates: int = 4,
        hidden_dim: int = 128,
        candidate_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_candidates = num_candidates

        self.candidate_encoder = nn.Sequential(
            nn.Linear(4, candidate_dim),
            nn.ReLU(),
            nn.Linear(candidate_dim, candidate_dim),
            nn.ReLU(),
        )
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_scene + candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.h_min_head = nn.Linear(hidden_dim, 1)
        self.feasibility_head = nn.Linear(hidden_dim, 1)
        self.correction_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        scene_embedding: torch.Tensor,
        candidate_trajectories: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input: [B, D_scene]
        # Returns:
        #   predicted_h_min:      [B, G]  continuous
        #   feasibility_logits:   [B, G]  binary classification
        #   predicted_correction: [B, G]  continuous
        #   predicted_risk:       [B, G]  continuous
        batch = scene_embedding.shape[0]
        if candidate_trajectories is None:
            candidate_trajectories = torch.zeros(
                (batch, self.num_candidates, 1, 4),
                dtype=scene_embedding.dtype,
                device=scene_embedding.device,
            )
        if candidate_trajectories.shape[:2] != (batch, self.num_candidates):
            raise ValueError("candidate_trajectories batch/candidate shape mismatch.")
        candidate_features = self.candidate_encoder(candidate_trajectories).mean(dim=2)
        scene_features = scene_embedding[:, None, :].expand(-1, self.num_candidates, -1)
        feat = self.shared_mlp(torch.cat((scene_features, candidate_features), dim=-1))
        return (
            self.h_min_head(feat).squeeze(-1),
            self.feasibility_head(feat).squeeze(-1),
            self.correction_head(feat).squeeze(-1),
            self.risk_head(feat).squeeze(-1),
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
