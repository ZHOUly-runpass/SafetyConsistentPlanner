from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..dcbf_mpc import CasadiVehicleDcbfMpcSolver, DcbfMpcSolver
from ..interfaces import PlannerOutput, PredictedObstacle
from ..models import SafetyConsistentPlannerModel
from .executor import execute_top_candidate


class PlannerWorker:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        solver: DcbfMpcSolver | None = None,
        solver_config: dict | None = None,
        mpc_num_intervals: int = 15,
        mpc_dt: float = 0.2,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SafetyConsistentPlannerModel(checkpoint.get("model_config", {})).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.solver = solver or CasadiVehicleDcbfMpcSolver(solver_config)
        self.mpc_num_intervals = mpc_num_intervals
        self.mpc_dt = mpc_dt

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
        planner_output = PlannerOutput(
            base_trajectory=_numpy(output.get("base_trajectory")),
            candidate_trajectories=_numpy(output["candidate_trajectories"]),
            candidate_logits=_numpy(output.get("candidate_logits")),
            predicted_h_min=_numpy(output.get("predicted_h_min")),
            feasibility_logits=_numpy(output.get("feasibility_logits")),
            predicted_correction=_numpy(output.get("predicted_correction")),
            predicted_risk=_numpy(output.get("predicted_risk")),
        )
        initial_state = np.asarray(request.get("initial_state"), dtype=np.float64)
        planner_timestamps = np.asarray(request.get("planner_timestamps"), dtype=np.float64)
        if initial_state.shape != (4,):
            raise ValueError("Planner request initial_state must have shape [4].")
        candidates = planner_output.candidate_trajectories
        if planner_timestamps.shape != (candidates.shape[2],):
            raise ValueError("planner_timestamps must match candidate horizon.")
        obstacles = [_predicted_obstacle(item) for item in request.get("predicted_obstacles", [])]
        execution = execute_top_candidate(
            planner_output,
            planner_timestamps,
            initial_state,
            obstacles,
            self.solver,
            self.mpc_num_intervals,
            self.mpc_dt,
            scenario_id=str(request.get("scenario_id", "")),
        )
        return {
            "ok": True,
            "selected_candidate_id": execution.selected_candidate_id,
            "candidate_trajectories": candidates[0],
            "selected_trajectory": execution.safe_trajectory,
            "first_control": execution.first_control,
            "mpc": {
                "feasible": execution.feasible,
                "h_min": execution.h_min,
                "slack_max": execution.slack_max,
                "solve_time_ms": execution.solve_time_ms,
                "diagnostics": execution.diagnostics or {},
            },
            "fallback": execution.fallback_mode.value,
        }


def _numpy(value):
    return None if value is None else value.detach().cpu().numpy()


def _predicted_obstacle(payload: dict) -> PredictedObstacle:
    obstacle = PredictedObstacle(
        track_id=str(payload["track_id"]),
        obstacle_type=int(payload.get("obstacle_type", 0)),
        timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
        centers=np.asarray(payload["centers"], dtype=np.float64),
        yaws=np.asarray(payload["yaws"], dtype=np.float64),
        lengths=np.asarray(payload["lengths"], dtype=np.float64),
        widths=np.asarray(payload["widths"], dtype=np.float64),
        velocities=np.asarray(payload["velocities"], dtype=np.float64),
        valid_mask=np.asarray(payload["valid_mask"], dtype=np.bool_),
    )
    obstacle.validate(num_intervals=len(obstacle.timestamps) - 1)
    return obstacle
