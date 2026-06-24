from __future__ import annotations

import numpy as np

from src.safety_planner.dcbf_mpc import NumpyVehicleDcbfMpcSolver
from src.safety_planner.interfaces import FallbackMode, PlannerOutput
from src.safety_planner.planning import execute_top_candidate, hard_filter_candidates


def test_hard_filter_candidates_uses_safety_predictions() -> None:
    planner_output = PlannerOutput(
        candidate_trajectories=np.zeros((1, 3, 8, 4), dtype=np.float64),
        predicted_h_min=np.array([[0.2, -0.1, 0.5]], dtype=np.float64),
        feasibility_logits=np.array([[2.0, 2.0, -2.0]], dtype=np.float64),
    )

    candidate_ids = hard_filter_candidates(
        planner_output,
        batch_index=0,
        min_predicted_h=0.0,
        min_feasibility_probability=0.5,
    )

    assert candidate_ids.tolist() == [0]


def test_execute_top_candidate_returns_execution_output() -> None:
    candidate = np.zeros((8, 4), dtype=np.float64)
    candidate[:, 0] = np.linspace(0.0, 4.0, 8)
    candidate[:, 3] = 2.0
    planner_output = PlannerOutput(
        candidate_trajectories=candidate.reshape(1, 1, 8, 4),
        predicted_h_min=np.array([[1.0]], dtype=np.float64),
        feasibility_logits=np.array([[4.0]], dtype=np.float64),
    )

    out = execute_top_candidate(
        planner_output=planner_output,
        planner_timestamps=np.linspace(0.0, 3.5, 8),
        initial_state=np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float64),
        predicted_obstacles=[],
        solver=NumpyVehicleDcbfMpcSolver(),
        mpc_num_intervals=15,
        mpc_dt=0.2,
    )

    out.validate()
    assert out.feasible
    assert out.fallback_mode == FallbackMode.NONE
    assert out.first_control.shape == (2,)
