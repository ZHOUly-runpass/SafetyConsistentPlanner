from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class PlannerOutput:
    schema_version: str = "1.0"

    base_trajectory: FloatArray | None = None
    candidate_trajectories: FloatArray | None = None
    candidate_logits: FloatArray | None = None

    predicted_h_min: FloatArray | None = None
    feasibility_logits: FloatArray | None = None
    predicted_correction: FloatArray | None = None
    predicted_risk: FloatArray | None = None

    def validate(self, batch: int = 1, num_candidates: int = 4, num_future: int = 8) -> None:
        if self.base_trajectory is not None:
            if self.base_trajectory.shape != (batch, num_future, 4):
                raise ValueError(
                    f"base_trajectory must have shape [{batch}, {num_future}, 4]."
                )

        if self.candidate_trajectories is not None:
            if self.candidate_trajectories.shape != (batch, num_candidates, num_future, 4):
                raise ValueError(
                    f"candidate_trajectories must have shape [{batch}, {num_candidates}, {num_future}, 4]."
                )

        if self.candidate_logits is not None:
            if self.candidate_logits.shape != (batch, num_candidates):
                raise ValueError(
                    f"candidate_logits must have shape [{batch}, {num_candidates}]."
                )

        if self.predicted_h_min is not None:
            if self.predicted_h_min.shape != (batch, num_candidates):
                raise ValueError(
                    f"predicted_h_min must have shape [{batch}, {num_candidates}]."
                )

        if self.feasibility_logits is not None:
            if self.feasibility_logits.shape != (batch, num_candidates):
                raise ValueError(
                    f"feasibility_logits must have shape [{batch}, {num_candidates}]."
                )

        if self.predicted_correction is not None:
            if self.predicted_correction.shape != (batch, num_candidates):
                raise ValueError(
                    f"predicted_correction must have shape [{batch}, {num_candidates}]."
                )

        if self.predicted_risk is not None:
            if self.predicted_risk.shape != (batch, num_candidates):
                raise ValueError(
                    f"predicted_risk must have shape [{batch}, {num_candidates}]."
                )
