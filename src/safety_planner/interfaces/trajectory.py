from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .enums import MpcStateIndex, MpcControlIndex, TrajectoryIndex

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    velocity: float

    def as_array(self) -> FloatArray:
        value = np.asarray(
            [self.x, self.y, self.yaw, self.velocity],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(value)):
            raise ValueError("VehicleState contains non-finite values.")
        return value


@dataclass(frozen=True)
class VehicleControl:
    acceleration: float
    steering: float

    def as_array(self) -> FloatArray:
        value = np.asarray(
            [self.acceleration, self.steering],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(value)):
            raise ValueError("VehicleControl contains non-finite values.")
        return value


@dataclass
class Trajectory:
    timestamps: FloatArray
    states: FloatArray
    valid_mask: BoolArray
    frame: str = "ego"
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must have shape [T].")

        if self.states.ndim != 2:
            raise ValueError("states must have shape [T, D].")

        if self.states.shape[0] != self.timestamps.shape[0]:
            raise ValueError("states and timestamps length mismatch.")

        if self.valid_mask.shape != self.timestamps.shape:
            raise ValueError("valid_mask must have shape [T].")

        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing.")

        if not np.all(np.isfinite(self.states[self.valid_mask])):
            raise ValueError("Valid trajectory states contain NaN or Inf.")
