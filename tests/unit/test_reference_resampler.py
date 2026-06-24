from __future__ import annotations

import numpy as np

from src.safety_planner.dcbf_mpc import resample_trajectory_to_mpc_grid


def test_resample_downsample() -> None:
    planner_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float64)
    planner_s = np.zeros((9, 4), dtype=np.float64)
    planner_s[:, 0] = np.linspace(0.0, 40.0, 9)
    planner_s[:, 3] = 10.0
    planner_mask = np.ones(9, dtype=np.bool_)

    mpc_t, mpc_s, mpc_mask = resample_trajectory_to_mpc_grid(
        planner_t, planner_s, planner_mask,
        mpc_num_intervals=15, mpc_dt=0.2,
    )

    assert mpc_t.shape == (16,)
    assert mpc_s.shape == (16, 4)
    assert mpc_mask.shape == (16,)
    assert np.all(np.diff(mpc_t) > 0)
    assert np.all(np.isfinite(mpc_s[mpc_mask]))


def test_resample_partial_coverage() -> None:
    planner_t = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    planner_s = np.array([
        [5.0, 0.0, 0.0, 10.0],
        [10.0, 0.0, 0.0, 10.0],
        [15.0, 0.0, 0.0, 10.0],
    ], dtype=np.float64)
    planner_mask = np.ones(3, dtype=np.bool_)

    mpc_t, mpc_s, mpc_mask = resample_trajectory_to_mpc_grid(
        planner_t, planner_s, planner_mask,
        mpc_num_intervals=15, mpc_dt=0.2,
    )

    assert mpc_s.shape == (16, 4)
    assert not mpc_mask[0]
    assert mpc_s[0, 0] == 5.0


def test_resample_yaw_unwrap() -> None:
    planner_t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    planner_s = np.array([
        [0.0, 0.0, 3.0, 5.0],
        [2.5, 0.0, -3.0, 5.0],
        [5.0, 0.0, 3.0, 5.0],
    ], dtype=np.float64)
    planner_mask = np.ones(3, dtype=np.bool_)

    mpc_t, mpc_s, mpc_mask = resample_trajectory_to_mpc_grid(
        planner_t, planner_s, planner_mask,
        mpc_num_intervals=5, mpc_dt=0.2,
    )

    for i in range(6):
        yaw = mpc_s[i, 2]
        assert -np.pi <= yaw < np.pi
