from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PredictedObstacle:
    track_id: str
    obstacle_type: int

    # Relative to the current planning time, [Nm+1], seconds.
    timestamps: NDArray[np.float64]

    # Obstacle centers in ego frame, [Nm+1, 2], meters.
    centers: NDArray[np.float64]

    # Yaw [rad], length [m], width [m], all [Nm+1].
    yaws: NDArray[np.float64]
    lengths: NDArray[np.float64]
    widths: NDArray[np.float64]

    # Velocity in ego frame, [Nm+1, 2], m/s.
    velocities: NDArray[np.float64]

    # Valid prediction mask, [Nm+1].
    valid_mask: NDArray[np.bool_]

    # Optional position covariance, [Nm+1, 2, 2], m^2.
    position_covariance: NDArray[np.float64] | None = None
    schema_version: str = "1.0"

    def validate(self, num_intervals: int) -> None:
        expected_points = num_intervals + 1
        if self.timestamps.shape != (expected_points,):
            raise ValueError(f"timestamps must have shape [{expected_points}].")
        if self.centers.shape != (expected_points, 2):
            raise ValueError(f"centers must have shape [{expected_points}, 2].")
        for name in ("yaws", "lengths", "widths"):
            value = getattr(self, name)
            if value.shape != (expected_points,):
                raise ValueError(f"{name} must have shape [{expected_points}].")
        if self.velocities.shape != (expected_points, 2):
            raise ValueError(f"velocities must have shape [{expected_points}, 2].")
        if self.valid_mask.shape != (expected_points,):
            raise ValueError(f"valid_mask must have shape [{expected_points}].")
        if self.position_covariance is not None:
            if self.position_covariance.shape != (expected_points, 2, 2):
                raise ValueError(
                    f"position_covariance must have shape [{expected_points}, 2, 2]."
                )

        numeric_arrays = (
            self.timestamps,
            self.centers,
            self.yaws,
            self.lengths,
            self.widths,
            self.velocities,
        )
        if not all(np.all(np.isfinite(value)) for value in numeric_arrays):
            raise ValueError("PredictedObstacle contains NaN or Inf.")
        if np.any(self.lengths <= 0.0) or np.any(self.widths <= 0.0):
            raise ValueError("Obstacle lengths and widths must be positive.")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing.")
