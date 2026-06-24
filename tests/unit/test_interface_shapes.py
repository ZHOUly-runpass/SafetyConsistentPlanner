from __future__ import annotations

import numpy as np

from src.safety_planner.interfaces import (
    VehicleState, VehicleControl, Trajectory,
    MpcRequest, MpcResult, SceneSample,
    PlannerOutput, PredictedObstacle,
    PseudoLabelRecord, ExecutionOutput, FallbackMode,
    RankingFeatures, RankingResult,
    MpcStateIndex, MpcControlIndex, TrajectoryIndex,
)


def test_vehicle_state_shape() -> None:
    s = VehicleState(0.0, 0.0, 0.0, 5.0)
    arr = s.as_array()
    assert arr.shape == (4,)


def test_vehicle_control_shape() -> None:
    c = VehicleControl(0.5, 0.1)
    arr = c.as_array()
    assert arr.shape == (2,)


def test_trajectory_validate() -> None:
    t = Trajectory(
        timestamps=np.array([0.0, 0.5, 1.0]),
        states=np.array([[0.0, 0.0, 0.0, 10.0]] * 3),
        valid_mask=np.array([True, True, True]),
    )
    t.validate()


def test_mpc_request_validate() -> None:
    obstacle = PredictedObstacle(
        track_id="agent_1",
        obstacle_type=1,
        timestamps=np.linspace(0.0, 3.0, 16),
        centers=np.zeros((16, 2)),
        yaws=np.zeros(16),
        lengths=np.full(16, 4.5),
        widths=np.full(16, 2.0),
        velocities=np.zeros((16, 2)),
        valid_mask=np.ones(16, dtype=np.bool_),
    )
    req = MpcRequest(
        schema_version="1.0",
        scenario_id="test",
        candidate_id=0,
        initial_state=np.array([0.0, 0.0, 0.0, 5.0]),
        reference_states=np.zeros((16, 4)),
        reference_timestamps=np.linspace(0.0, 3.0, 16),
        predicted_obstacles=[obstacle],
    )
    req.validate(num_intervals=15)


def test_mpc_result_validate() -> None:
    r = MpcResult(
        safe_states=np.zeros((16, 4)),
        controls=np.zeros((15, 2)),
        feasible=True,
        first_control=np.array([0.0, 0.0]),
    )
    r.validate(num_intervals=15, num_obstacles=2)


def test_scene_sample_validate() -> None:
    s = SceneSample(
        scenario_id="test",
        ego_history=np.zeros((5, 8)),
        ego_history_mask=np.ones(5, dtype=np.bool_),
        expert_future=np.zeros((8, 4)),
        expert_future_mask=np.ones(8, dtype=np.bool_),
    )
    s.validate()


def test_planner_output_validate() -> None:
    p = PlannerOutput(
        base_trajectory=np.zeros((2, 8, 4)),
        candidate_trajectories=np.zeros((2, 4, 8, 4)),
        candidate_logits=np.zeros((2, 4)),
    )
    p.validate(batch=2, num_candidates=4, num_future=8)


def test_predicted_obstacle_validate() -> None:
    obstacle = PredictedObstacle(
        track_id="agent_1",
        obstacle_type=1,
        timestamps=np.linspace(0.0, 3.0, 16),
        centers=np.zeros((16, 2)),
        yaws=np.zeros(16),
        lengths=np.full(16, 4.5),
        widths=np.full(16, 2.0),
        velocities=np.zeros((16, 2)),
        valid_mask=np.ones(16, dtype=np.bool_),
        position_covariance=np.zeros((16, 2, 2)),
    )
    obstacle.validate(num_intervals=15)
    assert obstacle.schema_version == "1.0"


def test_ranking_interfaces_validate() -> None:
    features = RankingFeatures(
        candidate_ids=np.array([0, 1, 2], dtype=np.int64),
        analytical_cost=np.array([3.0, 2.0, 1.0]),
        predicted_h_min=np.array([0.2, 0.5, 1.0]),
        feasibility_probability=np.array([0.7, 0.8, 0.9]),
        predicted_correction=np.array([0.3, 0.2, 0.1]),
        predicted_risk=np.array([0.4, 0.2, 0.1]),
    )
    features.validate(num_candidates=3)

    result = RankingResult(
        candidate_ids=np.array([2, 1, 0], dtype=np.int64),
        scores=np.array([0.9, 0.5, 0.1]),
        selected_candidate_id=2,
    )
    result.validate(num_candidates=3)


def test_pseudo_label_record_validate() -> None:
    r = PseudoLabelRecord(
        scenario_id="test",
        candidate_id=0,
        controls=np.zeros((15, 2)),
        h_values=np.zeros((3, 16)),
        cbf_residuals=np.zeros((3, 15)),
        slack=np.zeros((3, 15)),
        feasible=True,
        quality_grade="A",
    )
    r.validate()


def test_execution_output_validate() -> None:
    out = ExecutionOutput(
        selected_candidate_id=2,
        first_control=np.array([0.5, 0.1]),
        feasible=True,
        fallback_mode=FallbackMode.NONE,
    )
    out.validate()


def test_all_interfaces_have_version() -> None:
    classes = [
        MpcRequest,
        MpcResult,
        SceneSample,
        PlannerOutput,
        PredictedObstacle,
        PseudoLabelRecord,
        ExecutionOutput,
        Trajectory,
        RankingFeatures,
        RankingResult,
    ]
    for cls in classes:
        assert "schema_version" in cls.__dataclass_fields__
