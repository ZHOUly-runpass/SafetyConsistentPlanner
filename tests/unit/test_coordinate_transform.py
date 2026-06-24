from __future__ import annotations

import numpy as np

from src.safety_planner.vehicle import wrap_yaw, global_to_ego, ego_to_global, transform_yaw_to_ego, transform_yaw_to_global


def test_wrap_yaw_identity() -> None:
    assert np.isclose(wrap_yaw(0.0), 0.0)
    assert np.isclose(wrap_yaw(np.pi), -np.pi)
    assert np.isclose(wrap_yaw(-np.pi), -np.pi)


def test_wrap_yaw_wrap() -> None:
    assert np.isclose(wrap_yaw(3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_yaw(-3.0 * np.pi), -np.pi)


def test_wrap_yaw_array() -> None:
    yaws = np.array([0.0, 3.0 * np.pi, -3.0 * np.pi])
    result = wrap_yaw(yaws)
    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], -np.pi)
    assert np.isclose(result[2], -np.pi)


def test_global_to_ego_identity() -> None:
    ex, ey = global_to_ego(5.0, 3.0, 5.0, 3.0, 0.0)
    assert np.isclose(ex, 0.0)
    assert np.isclose(ey, 0.0)


def test_global_to_ego_with_yaw() -> None:
    ex, ey = global_to_ego(2.0, 0.0, 0.0, 0.0, np.pi / 2.0)
    assert np.isclose(ex, 0.0, atol=1e-9)
    assert np.isclose(ey, -2.0, atol=1e-9)


def test_global_ego_roundtrip() -> None:
    gx, gy = 10.0, 5.0
    ox, oy = 3.0, 2.0
    oyaw = 0.5
    ex, ey = global_to_ego(gx, gy, ox, oy, oyaw)
    gx2, gy2 = ego_to_global(ex, ey, ox, oy, oyaw)
    assert np.isclose(gx2, gx, atol=1e-9)
    assert np.isclose(gy2, gy, atol=1e-9)


def test_transform_yaw_roundtrip() -> None:
    for g_yaw in [0.0, 0.5, -0.5, 2.0, -2.0]:
        e_yaw = transform_yaw_to_ego(g_yaw, 0.3)
        g_yaw2 = transform_yaw_to_global(e_yaw, 0.3)
        assert np.isclose(np.cos(g_yaw2), np.cos(g_yaw), atol=1e-9)
        assert np.isclose(np.sin(g_yaw2), np.sin(g_yaw), atol=1e-9)
