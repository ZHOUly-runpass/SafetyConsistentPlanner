from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("casadi")

from src.safety_planner.dcbf_mpc import CasadiVehicleDcbfMpcSolver
from src.safety_planner.interfaces import MpcRequest, PredictedObstacle, RoadCorridor


def _request(with_obstacle: bool = False) -> MpcRequest:
    nm, dt = 10, 0.2
    timestamps = np.arange(nm + 1, dtype=np.float64) * dt
    reference = np.zeros((nm + 1, 4), dtype=np.float64)
    reference[:, 0] = np.linspace(0.0, 4.0, nm + 1)
    reference[:, 3] = 2.0
    obstacles = []
    if with_obstacle:
        obstacles.append(
            PredictedObstacle(
                track_id="blocked_lane",
                obstacle_type=1,
                timestamps=timestamps,
                centers=np.column_stack((np.full(nm + 1, 2.5), np.zeros(nm + 1))),
                yaws=np.zeros(nm + 1),
                lengths=np.full(nm + 1, 1.0),
                widths=np.full(nm + 1, 1.0),
                velocities=np.zeros((nm + 1, 2)),
                valid_mask=np.ones(nm + 1, dtype=np.bool_),
            )
        )
    corridor = RoadCorridor(
        normals=np.tile(np.array([0.0, 1.0]), (nm + 1, 1)),
        lower_bounds=np.full(nm + 1, -3.0),
        upper_bounds=np.full(nm + 1, 3.0),
        valid_mask=np.ones(nm + 1, dtype=np.bool_),
    )
    return MpcRequest(
        schema_version="1.0",
        scenario_id="casadi_synthetic",
        candidate_id=0,
        initial_state=np.array([0.0, 0.0, 0.0, 2.0]),
        reference_states=reference,
        reference_timestamps=timestamps,
        predicted_obstacles=obstacles,
        road_corridor=corridor,
    )


def _config() -> dict:
    return {
        "allow_numpy_fallback": False,
        "vehicle": {"wheelbase": 3.0, "length": 4.8, "width": 2.0},
        "limits": {
            "velocity_min": 0.0,
            "velocity_max": 10.0,
            "acceleration_min": -5.0,
            "acceleration_max": 3.0,
            "steering_min": -0.5,
            "steering_max": 0.5,
            "steering_rate_min": -1.0,
            "steering_rate_max": 1.0,
        },
        "dcbf": {"gamma": 0.2, "ellipse_inflation": 0.1, "max_slack": 100.0},
        "timeouts": {"solve_ms": 3000},
        "ipopt": {"max_iter": 300, "print_level": 0},
    }


def test_casadi_solver_uses_real_backend_and_dynamics() -> None:
    request = _request()
    result = CasadiVehicleDcbfMpcSolver(_config()).solve(request)
    assert result.metadata["backend"] == "casadi_ipopt"
    assert result.feasible
    assert np.max(np.abs(result.safe_states[0] - request.initial_state)) < 1e-8
    assert result.road_slack.shape == (2, 11)
    result.validate(num_intervals=10, num_obstacles=0)


def test_casadi_solver_changes_colliding_reference() -> None:
    request = _request(with_obstacle=True)
    result = CasadiVehicleDcbfMpcSolver(_config()).solve(request)
    assert result.metadata["backend"] == "casadi_ipopt"
    assert result.safe_states is not None
    assert not np.allclose(result.safe_states, request.reference_states)
    assert result.cbf_residuals.shape == (1, 10)
    assert np.min(result.cbf_residuals) >= -1e-4


def test_casadi_solver_accepts_warm_start() -> None:
    request = _request()
    request.initial_guess_states = request.reference_states.copy()
    request.initial_guess_controls = np.zeros((10, 2), dtype=np.float64)
    result = CasadiVehicleDcbfMpcSolver(_config()).solve(request)
    assert result.metadata["backend"] == "casadi_ipopt"
    assert result.iterations >= 0
    assert result.first_control.shape == (2,)
