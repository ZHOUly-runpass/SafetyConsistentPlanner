from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vehicle_footprint(
    length: float,
    width: float,
    position: NDArray[np.float64],
    yaw: float,
) -> NDArray[np.float64]:
    l2 = length / 2.0
    w2 = width / 2.0
    corners_local = np.array([
        [l2, w2],
        [l2, -w2],
        [-l2, -w2],
        [-l2, w2],
        [l2, w2],
    ], dtype=np.float64)

    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    corners = corners_local @ rot.T + position[:2]
    return corners


def ellipse_barrier_distance(
    ego_pos: NDArray[np.float64],
    ego_yaw: float,
    ego_length: float,
    ego_width: float,
    obs_pos: NDArray[np.float64],
    obs_yaw: float,
    obs_length: float,
    obs_width: float,
    inflation: float = 0.0,
) -> float:
    a_ex = (ego_length + inflation) / 2.0
    b_ex = (ego_width + inflation) / 2.0
    a_ox = (obs_length + inflation) / 2.0
    b_ox = (obs_width + inflation) / 2.0

    dx = obs_pos[0] - ego_pos[0]
    dy = obs_pos[1] - ego_pos[1]
    d = np.sqrt(dx**2 + dy**2)
    if d < 1e-9:
        return 0.0

    cos_theta = dx / d
    sin_theta = dy / d

    ego_angle = np.arctan2(sin_theta * np.cos(-ego_yaw) - cos_theta * np.sin(-ego_yaw),
                           cos_theta * np.cos(-ego_yaw) + sin_theta * np.sin(-ego_yaw))
    obs_angle = np.arctan2(-sin_theta * np.cos(-obs_yaw) + cos_theta * np.sin(-obs_yaw),
                           -cos_theta * np.cos(-obs_yaw) - sin_theta * np.sin(-obs_yaw))

    ego_r = a_ex * b_ex / np.sqrt((b_ex * np.cos(ego_angle))**2 + (a_ex * np.sin(ego_angle))**2)
    obs_r = a_ox * b_ox / np.sqrt((b_ox * np.cos(obs_angle))**2 + (a_ox * np.sin(obs_angle))**2)

    return d - ego_r - obs_r


def compute_obstacle_ellipse_params(
    length: float,
    width: float,
    inflation: float = 0.0,
) -> tuple[float, float]:
    a = (length + inflation) / 2.0
    b = (width + inflation) / 2.0
    return a, b
