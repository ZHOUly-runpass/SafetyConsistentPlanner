from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_ellipse_barrier(
    ego_x: NDArray[np.float64] | float,
    ego_y: NDArray[np.float64] | float,
    ego_yaw: NDArray[np.float64] | float,
    ego_length: float,
    ego_width: float,
    obs_x: NDArray[np.float64] | float,
    obs_y: NDArray[np.float64] | float,
    obs_yaw: NDArray[np.float64] | float,
    obs_length: NDArray[np.float64] | float,
    obs_width: NDArray[np.float64] | float,
    inflation: float = 0.0,
) -> NDArray[np.float64]:
    ex = np.asarray(ego_x, dtype=np.float64)
    ey = np.asarray(ego_y, dtype=np.float64)
    e_yaw = np.asarray(ego_yaw, dtype=np.float64)
    ox = np.asarray(obs_x, dtype=np.float64)
    oy = np.asarray(obs_y, dtype=np.float64)
    o_yaw = np.asarray(obs_yaw, dtype=np.float64)
    ol = np.asarray(obs_length, dtype=np.float64)
    ow = np.asarray(obs_width, dtype=np.float64)

    a_ex = (ego_length + inflation) / 2.0
    b_ex = (ego_width + inflation) / 2.0
    a_ox = (ol + inflation) / 2.0
    b_ox = (ow + inflation) / 2.0

    dx = ox - ex
    dy = oy - ey
    d = np.sqrt(dx**2 + dy**2)

    result = np.zeros_like(d, dtype=np.float64)

    mask = d > 1e-9
    if not np.any(mask):
        return result

    dm = d[mask]
    dxm = dx[mask]
    dym = dy[mask]
    e_ym = e_yaw if e_yaw.ndim == 0 else e_yaw[mask]
    o_ym = o_yaw if o_yaw.ndim == 0 else o_yaw[mask]
    a_om = a_ox if a_ox.ndim == 0 else a_ox[mask]
    b_om = b_ox if b_ox.ndim == 0 else b_ox[mask]

    cos_theta = dxm / dm
    sin_theta = dym / dm

    cos_neg_ey = np.cos(-e_ym)
    sin_neg_ey = np.sin(-e_ym)
    ego_angle = np.arctan2(
        sin_theta * cos_neg_ey - cos_theta * sin_neg_ey,
        cos_theta * cos_neg_ey + sin_theta * sin_neg_ey,
    )

    cos_neg_oy = np.cos(-o_ym)
    sin_neg_oy = np.sin(-o_ym)
    obs_angle = np.arctan2(
        -sin_theta * cos_neg_oy + cos_theta * sin_neg_oy,
        -cos_theta * cos_neg_oy - sin_theta * sin_neg_oy,
    )

    ego_r = a_ex * b_ex / np.sqrt(
        (b_ex * np.cos(ego_angle)) ** 2 + (a_ex * np.sin(ego_angle)) ** 2
    )
    obs_r = a_om * b_om / np.sqrt(
        (b_om * np.cos(obs_angle)) ** 2 + (a_om * np.sin(obs_angle)) ** 2
    )

    result[mask] = dm - ego_r - obs_r
    return result


def compute_dcbf_residual(
    h_kp1: NDArray[np.float64],
    h_k: NDArray[np.float64],
    gamma: float,
    slack: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    if slack is None:
        return h_kp1 - (1.0 - gamma) * h_k
    return h_kp1 - (1.0 - gamma) * h_k + slack


def compute_slack_penalty(slack: NDArray[np.float64]) -> float:
    return float(np.sum(slack**2))


def inflate_ego_footprint(
    length: float,
    width: float,
    inflation: float,
) -> tuple[float, float]:
    return length + inflation, width + inflation
