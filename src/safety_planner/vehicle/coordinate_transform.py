from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wrap_yaw(yaw: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return (np.asarray(yaw, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_yaw(yaw: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.unwrap(yaw.astype(np.float64))


def global_to_ego(
    global_x: NDArray[np.float64] | float,
    global_y: NDArray[np.float64] | float,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dx = np.asarray(global_x, dtype=np.float64) - ego_x
    dy = np.asarray(global_y, dtype=np.float64) - ego_y
    c = np.cos(-ego_yaw)
    s = np.sin(-ego_yaw)
    ego_x_out = c * dx - s * dy
    ego_y_out = s * dx + c * dy
    return ego_x_out, ego_y_out


def ego_to_global(
    ego_x: NDArray[np.float64] | float,
    ego_y: NDArray[np.float64] | float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ex = np.asarray(ego_x, dtype=np.float64)
    ey = np.asarray(ego_y, dtype=np.float64)
    c = np.cos(origin_yaw)
    s = np.sin(origin_yaw)
    global_x_out = c * ex - s * ey + origin_x
    global_y_out = s * ex + c * ey + origin_y
    return global_x_out, global_y_out


def transform_yaw_to_ego(global_yaw: NDArray[np.float64] | float, ego_yaw: float) -> NDArray[np.float64] | float:
    return wrap_yaw(np.asarray(global_yaw, dtype=np.float64) - ego_yaw)


def transform_yaw_to_global(ego_yaw: NDArray[np.float64] | float, origin_yaw: float) -> NDArray[np.float64] | float:
    return wrap_yaw(np.asarray(ego_yaw, dtype=np.float64) + origin_yaw)
