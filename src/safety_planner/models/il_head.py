from __future__ import annotations

import torch
from torch import nn


class ILHead(nn.Module):
    def __init__(self, d_scene: int, num_future: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_future = num_future
        self.mlp = nn.Sequential(
            nn.Linear(d_scene, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_future * 4),
        )

    def forward(self, scene_embedding: torch.Tensor) -> torch.Tensor:
        # Input:  [B, D_scene]
        # Output: [B, Np, 4]  (x, y, yaw, velocity per future point)
        B = scene_embedding.shape[0]
        out = self.mlp(scene_embedding)
        return out.view(B, self.num_future, 4)
