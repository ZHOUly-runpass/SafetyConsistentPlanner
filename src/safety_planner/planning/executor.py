from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dcbf_mpc import DcbfMpcSolver, resample_trajectory_to_mpc_grid
from ..interfaces import ExecutionOutput, FallbackMode, MpcRequest, PlannerOutput, PredictedObstacle
from .candidate_filter import hard_filter_candidates, score_candidates


def execute_top_candidate(
    planner_output: PlannerOutput,
    planner_timestamps: NDArray[np.float64],
    initial_state: NDArray[np.float64],
    predicted_obstacles: list[PredictedObstacle],
    solver: DcbfMpcSolver,
    mpc_num_intervals: int,
    mpc_dt: float,
    batch_index: int = 0,
    scenario_id: str = "",
) -> ExecutionOutput:
    planner_output.validate(
        batch=planner_output.candidate_trajectories.shape[0],
        num_candidates=planner_output.candidate_trajectories.shape[1],
        num_future=planner_output.candidate_trajectories.shape[2],
    )
    candidate_ids = hard_filter_candidates(planner_output, batch_index=batch_index)
    if candidate_ids.size == 0:
        return _emergency_stop(initial_state, reason="all_candidates_filtered")

    scores = score_candidates(planner_output, batch_index=batch_index, candidate_ids=candidate_ids)
    sorted_ids = candidate_ids[np.argsort(scores)[::-1]]

    for order, candidate_id in enumerate(sorted_ids):
        candidate = planner_output.candidate_trajectories[batch_index, candidate_id]
        valid_mask = np.ones(candidate.shape[0], dtype=np.bool_)
        mpc_timestamps, reference_states, _ = resample_trajectory_to_mpc_grid(
            planner_timestamps,
            candidate,
            valid_mask,
            mpc_num_intervals,
            mpc_dt,
            current_time=float(planner_timestamps[0]),
        )
        request = MpcRequest(
            schema_version="1.0",
            scenario_id=scenario_id,
            candidate_id=int(candidate_id),
            initial_state=initial_state,
            reference_states=reference_states,
            reference_timestamps=mpc_timestamps,
            predicted_obstacles=predicted_obstacles,
            prediction_source="planner_candidate",
        )
        result = solver.solve(request)
        fallback_mode = FallbackMode.NONE if order == 0 else FallbackMode.NEXT_CANDIDATE
        if result.feasible:
            return _result_to_output(result, int(candidate_id), fallback_mode)

    best_id = int(sorted_ids[0])
    reduced_candidate = planner_output.candidate_trajectories[batch_index, best_id].copy()
    reduced_candidate[:, 3] *= 0.5
    reduced_request = _request_from_candidate(
        reduced_candidate,
        planner_timestamps,
        initial_state,
        predicted_obstacles,
        mpc_num_intervals,
        mpc_dt,
        scenario_id,
        best_id,
        "reduced_speed",
    )
    reduced_result = solver.solve(reduced_request)
    if reduced_result.feasible:
        return _result_to_output(reduced_result, best_id, FallbackMode.REDUCED_SPEED)

    conservative_timestamps = np.arange(mpc_num_intervals + 1, dtype=np.float64) * mpc_dt
    conservative_states = np.zeros((mpc_num_intervals + 1, 4), dtype=np.float64)
    conservative_states[0] = initial_state
    for k in range(mpc_num_intervals):
        speed = max(0.0, conservative_states[k, 3] - 2.0 * mpc_dt)
        conservative_states[k + 1] = conservative_states[k]
        conservative_states[k + 1, 0] += mpc_dt * speed * np.cos(initial_state[2])
        conservative_states[k + 1, 1] += mpc_dt * speed * np.sin(initial_state[2])
        conservative_states[k + 1, 3] = speed
    conservative_request = MpcRequest(
        schema_version="1.0",
        scenario_id=scenario_id,
        candidate_id=best_id,
        initial_state=initial_state,
        reference_states=conservative_states,
        reference_timestamps=conservative_timestamps,
        predicted_obstacles=predicted_obstacles,
        prediction_source="conservative_reference",
    )
    conservative_result = solver.solve(conservative_request)
    if conservative_result.feasible:
        return _result_to_output(
            conservative_result,
            best_id,
            FallbackMode.CONSERVATIVE_REFERENCE,
        )

    return _emergency_stop(initial_state, reason="solver_infeasible")


def _request_from_candidate(
    candidate,
    planner_timestamps,
    initial_state,
    predicted_obstacles,
    mpc_num_intervals,
    mpc_dt,
    scenario_id,
    candidate_id,
    prediction_source,
) -> MpcRequest:
    valid_mask = np.ones(candidate.shape[0], dtype=np.bool_)
    mpc_timestamps, reference_states, _ = resample_trajectory_to_mpc_grid(
        planner_timestamps,
        candidate,
        valid_mask,
        mpc_num_intervals,
        mpc_dt,
        current_time=float(planner_timestamps[0]),
    )
    return MpcRequest(
        schema_version="1.0",
        scenario_id=scenario_id,
        candidate_id=int(candidate_id),
        initial_state=initial_state,
        reference_states=reference_states,
        reference_timestamps=mpc_timestamps,
        predicted_obstacles=predicted_obstacles,
        prediction_source=prediction_source,
    )


def _result_to_output(result, candidate_id: int, fallback_mode: FallbackMode) -> ExecutionOutput:
    output = ExecutionOutput(
        selected_candidate_id=candidate_id,
        safe_trajectory=result.safe_states,
        first_control=result.first_control,
        feasible=True,
        fallback_mode=fallback_mode,
        h_min=result.h_min,
        slack_max=result.slack_max,
        solve_time_ms=result.solve_time_ms,
        diagnostics=result.metadata,
    )
    output.validate()
    return output


def _emergency_stop(initial_state: NDArray[np.float64], reason: str) -> ExecutionOutput:
    safe_trajectory = np.tile(initial_state.astype(np.float64), (2, 1))
    output = ExecutionOutput(
        selected_candidate_id=-1,
        safe_trajectory=safe_trajectory,
        first_control=np.array([-5.0, 0.0], dtype=np.float64),
        feasible=False,
        fallback_mode=FallbackMode.EMERGENCY_STOP,
        h_min=0.0,
        slack_max=0.0,
        diagnostics={"reason": reason},
    )
    output.validate()
    return output
