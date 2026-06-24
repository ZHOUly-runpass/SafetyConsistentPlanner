from __future__ import annotations

import numpy as np

from src.safety_planner.vehicle import bicycle_step, bicycle_rollout


def test_bicycle_step_straight() -> None:
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float64)
    control = np.array([0.0, 0.0], dtype=np.float64)
    dt = 0.2
    wheelbase = 3.0

    next_state = bicycle_step(state, control, dt, wheelbase)
    assert next_state.shape == (4,)
    assert np.isclose(next_state[0], 1.0)
    assert np.isclose(next_state[1], 0.0)
    assert np.isclose(next_state[2], 0.0)
    assert np.isclose(next_state[3], 5.0)


def test_bicycle_step_turn() -> None:
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float64)
    control = np.array([0.0, 0.3], dtype=np.float64)
    dt = 0.2
    wheelbase = 3.0

    next_state = bicycle_step(state, control, dt, wheelbase)
    assert next_state[2] > 0.0


def test_bicycle_step_accelerate() -> None:
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float64)
    control = np.array([2.0, 0.0], dtype=np.float64)
    dt = 0.2
    wheelbase = 3.0

    next_state = bicycle_step(state, control, dt, wheelbase)
    assert np.isclose(next_state[3], 5.4)


def test_bicycle_rollout() -> None:
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float64)
    nm = 15
    controls = np.zeros((nm, 2), dtype=np.float64)
    controls[:, 0] = 0.5
    dt = 0.2
    wheelbase = 3.0

    states = bicycle_rollout(state, controls, dt, wheelbase)
    assert states.shape == (nm + 1, 4)
    assert np.all(np.isfinite(states))
    assert states[-1, 3] > states[0, 3]
