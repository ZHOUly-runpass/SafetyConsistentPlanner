from __future__ import annotations

import numpy as np

from src.safety_planner.dcbf_mpc import NumpyVehicleDcbfMpcSolver
from src.safety_planner.interfaces import MpcRequest, PredictedObstacle


def test_numpy_solver_returns_complete_result() -> None:
    nm = 15
    dt = 0.2
    timestamps = np.arange(nm + 1, dtype=np.float64) * dt
    reference = np.zeros((nm + 1, 4), dtype=np.float64)
    reference[:, 0] = np.linspace(0.0, 6.0, nm + 1)
    reference[:, 3] = 2.0
    obstacle = PredictedObstacle(
        track_id="front_vehicle",
        obstacle_type=1,
        timestamps=timestamps,
        centers=np.column_stack((np.full(nm + 1, 20.0), np.zeros(nm + 1))),
        yaws=np.zeros(nm + 1),
        lengths=np.full(nm + 1, 4.5),
        widths=np.full(nm + 1, 2.0),
        velocities=np.zeros((nm + 1, 2)),
        valid_mask=np.ones(nm + 1, dtype=np.bool_),
    )
    request = MpcRequest(
        schema_version="1.0",
        scenario_id="synthetic",
        candidate_id=0,
        initial_state=np.array([0.0, 0.0, 0.0, 2.0]),
        reference_states=reference,
        reference_timestamps=timestamps,
        predicted_obstacles=[obstacle],
    )

    solver = NumpyVehicleDcbfMpcSolver(
        {
            "vehicle": {"length": 4.8, "width": 2.0, "wheelbase": 3.0},
            "dcbf": {"gamma": 0.2, "inflation": 0.1},
        }
    )
    result = solver.solve(request)

    result.validate(num_intervals=nm, num_obstacles=1)
    assert result.safe_states.shape == (nm + 1, 4)
    assert result.controls.shape == (nm, 2)
    assert result.h_values.shape == (1, nm + 1)
    assert result.cbf_residuals.shape == (1, nm)
    assert result.obstacle_slack.shape == (1, nm)
    assert result.first_control.shape == (2,)
    assert result.feasible
    assert result.h_min > 0.0
