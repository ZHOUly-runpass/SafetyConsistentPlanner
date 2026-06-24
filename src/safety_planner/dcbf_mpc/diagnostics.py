from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..interfaces.mpc import MpcResult


def compute_h_min(h_values: NDArray[np.float64]) -> float:
    if h_values.size == 0:
        return float("inf")
    return float(np.min(h_values))


def compute_slack_statistics(slack: NDArray[np.float64]) -> tuple[float, float]:
    if slack.size == 0:
        return 0.0, 0.0
    return float(np.max(slack)), float(np.sum(slack))


def compute_correction_statistics(
    reference_states: NDArray[np.float64],
    safe_states: NDArray[np.float64],
) -> tuple[float, float]:
    diff = safe_states[:, :2] - reference_states[:, :2]
    l2_per_step = np.sqrt(np.sum(diff**2, axis=1))
    return float(np.sqrt(np.sum(l2_per_step**2))), float(np.max(l2_per_step))


def compute_safety_margin(h_values: NDArray[np.float64], threshold: float = 0.0) -> float:
    if h_values.size == 0:
        return 0.0
    return float(np.min(h_values)) - threshold


def populate_diagnostics(
    result: MpcResult,
    h_values: NDArray[np.float64] | None,
    slack: NDArray[np.float64] | None,
    reference_states: NDArray[np.float64] | None,
    safe_states: NDArray[np.float64] | None,
) -> None:
    if h_values is not None:
        result.h_min = compute_h_min(h_values)
    if slack is not None:
        result.slack_max, result.slack_sum = compute_slack_statistics(slack)
    if reference_states is not None and safe_states is not None:
        result.correction_l2, result.correction_max = compute_correction_statistics(
            reference_states, safe_states
        )
