from __future__ import annotations

import numpy as np

from src.safety_planner.dcbf_mpc import compute_ellipse_barrier, compute_dcbf_residual, compute_slack_penalty


def test_ellipse_barrier_zero_distance() -> None:
    h = compute_ellipse_barrier(
        0.0, 0.0, 0.0, 4.8, 2.0,
        0.0, 0.0, 0.0, 4.8, 2.0,
    )
    assert h <= 0.0


def test_ellipse_barrier_positive_distance() -> None:
    h = compute_ellipse_barrier(
        0.0, 0.0, 0.0, 4.8, 2.0,
        10.0, 0.0, 0.0, 4.8, 2.0,
    )
    assert h > 0.0


def test_ellipse_barrier_side_by_side() -> None:
    h = compute_ellipse_barrier(
        0.0, 0.0, 0.0, 4.8, 2.0,
        0.0, 3.0, 0.0, 4.8, 2.0,
    )
    assert h > 0.0


def test_ellipse_barrier_array() -> None:
    h = compute_ellipse_barrier(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        0.0,
        4.8, 2.0,
        np.array([10.0, 0.0]),
        np.array([0.0, 10.0]),
        0.0,
        np.array([4.8, 4.8]),
        np.array([2.0, 2.0]),
    )
    assert h.shape == (2,)
    assert np.all(np.isfinite(h))


def test_dcbf_residual_without_slack() -> None:
    h_k = np.array([2.0])
    h_kp1 = np.array([1.9])
    gamma = 0.1
    r = compute_dcbf_residual(h_kp1, h_k, gamma)
    assert np.isclose(r[0], 1.9 - 0.9 * 2.0)


def test_dcbf_residual_with_slack() -> None:
    h_k = np.array([2.0])
    h_kp1 = np.array([1.5])
    gamma = 0.1
    slack = np.array([1.0])
    r = compute_dcbf_residual(h_kp1, h_k, gamma, slack)
    expected = 1.5 - 0.9 * 2.0 + 1.0
    assert np.isclose(r[0], expected)


def test_slack_penalty() -> None:
    slack = np.array([0.1, 0.2, 0.1])
    p = compute_slack_penalty(slack)
    assert np.isclose(p, 0.06)
