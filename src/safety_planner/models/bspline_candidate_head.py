from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BSplineCandidateHead(nn.Module):
    def __init__(
        self,
        d_scene: int,
        num_candidates: int = 4,
        num_future: int = 8,
        num_control_points: int = 6,
        degree: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.num_future = num_future
        self.num_control_points = num_control_points
        self.degree = degree

        self.control_point_net = nn.Sequential(
            nn.Linear(d_scene, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_candidates * num_control_points * 2),
        )

    def forward(
        self,
        scene_embedding: torch.Tensor,
        base_trajectory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = scene_embedding.shape[0]
        control_points = self.control_point_net(scene_embedding)
        control_points = control_points.view(
            batch * self.num_candidates,
            self.num_control_points,
            2,
        )
        residual_xy = F.interpolate(
            control_points.transpose(1, 2),
            size=self.num_future,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        residual_xy = residual_xy.view(batch, self.num_candidates, self.num_future, 2)
        if base_trajectory is None:
            base_xy = torch.zeros(
                (batch, 1, self.num_future, 2),
                dtype=residual_xy.dtype,
                device=residual_xy.device,
            )
        else:
            if base_trajectory.shape != (batch, self.num_future, 4):
                raise ValueError(
                    f"base_trajectory must have shape [{batch}, {self.num_future}, 4]."
                )
            base_xy = base_trajectory[:, None, :, :2]
        xy = base_xy + residual_xy

        deltas = torch.zeros_like(xy)
        deltas[:, :, 1:, :] = xy[:, :, 1:, :] - xy[:, :, :-1, :]
        deltas[:, :, 0, :] = deltas[:, :, 1, :] if self.num_future > 1 else 0.0
        yaw = torch.atan2(deltas[..., 1], deltas[..., 0]).unsqueeze(-1)
        velocity = torch.linalg.norm(deltas, dim=-1, keepdim=True) / 0.5
        return torch.cat((xy, yaw, velocity), dim=-1)
