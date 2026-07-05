from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models import SafetyConsistentPlannerModel


class PlannerWorker:
    def __init__(self, checkpoint_path: str | Path, device: str = "cuda") -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SafetyConsistentPlannerModel(checkpoint.get("model_config", {})).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def handle(self, request: dict) -> dict:
        scene = request.get("scene")
        if not isinstance(scene, dict):
            raise ValueError("Planner request requires a scene mapping.")
        batch = {
            name: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for name, value in scene.items()
            if isinstance(value, np.ndarray)
        }
        output = self.model(batch)
        candidates = output["candidate_trajectories"][0].detach().cpu().numpy()
        scores = (
            output["predicted_h_min"][0]
            + torch.sigmoid(output["feasibility_logits"][0])
            - output["predicted_correction"][0]
            - output["predicted_risk"][0]
        )
        selected = int(torch.argmax(scores).item())
        return {
            "ok": True,
            "selected_candidate_id": selected,
            "candidate_trajectories": candidates,
            "selected_trajectory": candidates[selected],
            "scores": scores.detach().cpu().numpy(),
        }

