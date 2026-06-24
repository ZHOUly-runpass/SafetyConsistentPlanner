from __future__ import annotations

import torch
from torch import nn


def expert_imitation_loss(
    pred_states: torch.Tensor,
    expert_states: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        diff = (pred_states - expert_states) * mask.unsqueeze(-1)
    else:
        diff = pred_states - expert_states
    return torch.mean(diff**2)


def pseudo_label_loss(
    pred_states: torch.Tensor,
    pseudo_states: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        diff = (pred_states - pseudo_states) * mask.unsqueeze(-1)
    else:
        diff = pred_states - pseudo_states
    return torch.mean(diff**2)


def cbf_feasibility_loss(
    predicted_h_min: torch.Tensor,
    true_h_min: torch.Tensor,
    feasible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = predicted_h_min - true_h_min
    if feasible_mask is not None:
        diff = diff * feasible_mask
    return torch.mean(diff**2)


def feasibility_classification_loss(
    predicted_logits: torch.Tensor,
    true_feasible: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        predicted_logits, true_feasible.float()
    )


def correction_prediction_loss(
    predicted_correction: torch.Tensor,
    true_correction: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = predicted_correction - true_correction
    if mask is not None:
        diff = diff * mask
    return torch.mean(diff**2)


def smoothness_loss(
    pred_states: torch.Tensor,
) -> torch.Tensor:
    accel = pred_states[:, 1:, :2] - 2.0 * pred_states[:, 1:-1, :2] + pred_states[:, :-2, :2]
    return torch.mean(accel**2)


def diversity_loss(
    candidate_trajectories: torch.Tensor,
) -> torch.Tensor:
    B, G, T, D = candidate_trajectories.shape
    if G <= 1:
        return torch.tensor(0.0, device=candidate_trajectories.device)
    flat = candidate_trajectories.view(B, G, -1)
    pairwise_dist = torch.cdist(flat, flat, p=2.0)
    mask = 1.0 - torch.eye(G, device=flat.device).unsqueeze(0)
    return -torch.mean(pairwise_dist * mask)


def compute_total_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for name, loss_value in losses.items():
        total = total + weights.get(name, 1.0) * loss_value
    return total
