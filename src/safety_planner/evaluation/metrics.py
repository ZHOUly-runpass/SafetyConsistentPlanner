from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..vehicle.coordinate_transform import wrap_yaw


def _masked_diff(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes.")
    if mask is None:
        return prediction - target
    if mask.shape != prediction.shape[:1]:
        raise ValueError("mask must have shape [T].")
    return prediction[mask] - target[mask]


def average_displacement_error(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    diff = _masked_diff(prediction[:, :2], target[:, :2], mask)
    if diff.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def final_displacement_error(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    if mask is not None:
        valid_indices = np.nonzero(mask)[0]
        if valid_indices.size == 0:
            return 0.0
        idx = int(valid_indices[-1])
    else:
        idx = prediction.shape[0] - 1
    return float(np.linalg.norm(prediction[idx, :2] - target[idx, :2]))


def yaw_error(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    diff = wrap_yaw(prediction[:, 2] - target[:, 2])
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return 0.0
    return float(np.mean(np.abs(diff)))


def velocity_error(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    diff = prediction[:, 3] - target[:, 3]
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return 0.0
    return float(np.mean(np.abs(diff)))
