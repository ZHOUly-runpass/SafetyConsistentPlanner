from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resample_trajectory_to_mpc_grid(
    planner_timestamps: NDArray[np.float64],
    planner_states: NDArray[np.float64],
    planner_valid_mask: NDArray[np.bool_],
    mpc_num_intervals: int,
    mpc_dt: float,
    current_time: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    nm = mpc_num_intervals
    mpc_timestamps = current_time + np.arange(nm + 1, dtype=np.float64) * mpc_dt
    mpc_states = np.empty((nm + 1, 4), dtype=np.float64)
    mpc_valid = np.ones(nm + 1, dtype=np.bool_)

    valid_planner = planner_timestamps[planner_valid_mask]
    if len(valid_planner) < 2:
        mpc_states[:, :2] = planner_states[0, :2]
        mpc_states[:, 2] = 0.0
        mpc_states[:, 3] = 0.0
        mpc_valid[:] = False
        return mpc_timestamps, mpc_states, mpc_valid

    planner_positions = planner_states[planner_valid_mask, :2]
    planner_yaws = planner_states[planner_valid_mask, 2]
    planner_velocities = planner_states[planner_valid_mask, 3]

    planner_yaws_unwrapped = np.unwrap(planner_yaws)

    t_min = valid_planner[0]
    t_max = valid_planner[-1]

    for i, t in enumerate(mpc_timestamps):
        if t < t_min:
            mpc_states[i, :2] = planner_positions[0]
            mpc_states[i, 2] = planner_yaws[0]
            mpc_states[i, 3] = planner_velocities[0]
            if t + 1e-9 < t_min:
                mpc_valid[i] = False
        elif t > t_max:
            mpc_states[i, :2] = planner_positions[-1]
            mpc_states[i, 2] = planner_yaws[-1]
            mpc_states[i, 3] = planner_velocities[-1]
            if t - 1e-9 > t_max:
                mpc_valid[i] = False
        else:
            px = np.interp(t, valid_planner, planner_positions[:, 0])
            py = np.interp(t, valid_planner, planner_positions[:, 1])
            pyaw_unwrapped = np.interp(t, valid_planner, planner_yaws_unwrapped)
            pv = np.interp(t, valid_planner, planner_velocities)
            mpc_states[i, 0] = px
            mpc_states[i, 1] = py
            mpc_states[i, 2] = (pyaw_unwrapped + np.pi) % (2.0 * np.pi) - np.pi
            mpc_states[i, 3] = pv

    return mpc_timestamps, mpc_states, mpc_valid


def resample_obstacle_to_mpc_grid(
    obstacle_timestamps: NDArray[np.float64],
    obstacle_states: NDArray[np.float64],
    obstacle_valid_mask: NDArray[np.bool_],
    mpc_timestamps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    nm_plus_1 = len(mpc_timestamps)
    state_dim = obstacle_states.shape[1]
    resampled = np.empty((nm_plus_1, state_dim), dtype=np.float64)
    resampled_valid = np.ones(nm_plus_1, dtype=np.bool_)

    valid_obs_t = obstacle_timestamps[obstacle_valid_mask]
    if len(valid_obs_t) < 2:
        resampled[:] = obstacle_states[0]
        resampled_valid[:] = False
        return resampled, resampled_valid

    valid_obs_states = obstacle_states[obstacle_valid_mask]
    t_min = valid_obs_t[0]
    t_max = valid_obs_t[-1]

    for i, t in enumerate(mpc_timestamps):
        if t < t_min:
            resampled[i] = valid_obs_states[0]
            if t + 1e-9 < t_min:
                resampled_valid[i] = False
        elif t > t_max:
            resampled[i] = valid_obs_states[-1]
            if t - 1e-9 > t_max:
                resampled_valid[i] = False
        else:
            for d in range(state_dim):
                resampled[i, d] = np.interp(t, valid_obs_t, valid_obs_states[:, d])

    return resampled, resampled_valid
