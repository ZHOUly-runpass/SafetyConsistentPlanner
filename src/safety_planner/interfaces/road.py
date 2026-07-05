from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RoadCorridor:
    """Linearized drivable corridor on the MPC time grid.

    A position ``p_k`` is inside the corridor when
    ``lower_bounds[k] <= normals[k] @ p_k <= upper_bounds[k]``.
    """

    normals: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    schema_version: str = "1.0"

    def validate(self, num_intervals: int) -> None:
        num_points = num_intervals + 1
        if self.normals.shape != (num_points, 2):
            raise ValueError(f"normals must have shape [{num_points}, 2].")
        for name in ("lower_bounds", "upper_bounds", "valid_mask"):
            value = getattr(self, name)
            if value.shape != (num_points,):
                raise ValueError(f"{name} must have shape [{num_points}].")
        if not np.all(np.isfinite(self.normals)):
            raise ValueError("Road corridor normals contain NaN or Inf.")
        if not np.all(np.isfinite(self.lower_bounds)) or not np.all(
            np.isfinite(self.upper_bounds)
        ):
            raise ValueError("Road corridor bounds contain NaN or Inf.")
        if np.any(self.lower_bounds[self.valid_mask] > self.upper_bounds[self.valid_mask]):
            raise ValueError("Road corridor lower bound exceeds upper bound.")
        norms = np.linalg.norm(self.normals[self.valid_mask], axis=1)
        if norms.size and np.any(norms <= 1e-9):
            raise ValueError("Valid road corridor normals must be non-zero.")

