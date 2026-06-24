from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bicycle_step(
    state: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
    wheelbase: float,
) -> NDArray[np.float64]:
    x, y, yaw, v = state
    a, delta = control

    x_next = x + dt * v * np.cos(yaw)
    y_next = y + dt * v * np.sin(yaw)
    yaw_next = yaw + dt * v / wheelbase * np.tan(delta)
    v_next = v + dt * a

    return np.array([x_next, y_next, yaw_next, v_next], dtype=np.float64)


def bicycle_rollout(
    initial_state: NDArray[np.float64],
    controls: NDArray[np.float64],
    dt: float,
    wheelbase: float,
) -> NDArray[np.float64]:
    nm = controls.shape[0]
    states = np.empty((nm + 1, 4), dtype=np.float64)
    states[0] = initial_state

    for k in range(nm):
        states[k + 1] = bicycle_step(states[k], controls[k], dt, wheelbase)

    return states


def bicycle_rollout_batch(
    initial_states: NDArray[np.float64],
    controls: NDArray[np.float64],
    dt: float,
    wheelbase: float,
) -> NDArray[np.float64]:
    B, nm = controls.shape[0], controls.shape[1]
    states = np.empty((B, nm + 1, 4), dtype=np.float64)
    states[:, 0, :] = initial_states

    for k in range(nm):
        x = states[:, k, 0]
        y = states[:, k, 1]
        yaw = states[:, k, 2]
        v = states[:, k, 3]
        a = controls[:, k, 0]
        delta = controls[:, k, 1]

        states[:, k + 1, 0] = x + dt * v * np.cos(yaw)
        states[:, k + 1, 1] = y + dt * v * np.sin(yaw)
        states[:, k + 1, 2] = yaw + dt * v / wheelbase * np.tan(delta)
        states[:, k + 1, 3] = v + dt * a

    return states
