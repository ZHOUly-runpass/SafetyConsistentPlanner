from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class RankingFeatures:
    """Per-candidate ranking features in SI units unless otherwise noted."""

    schema_version: str = "1.0"

    candidate_ids: IntArray | None = None
    analytical_cost: FloatArray | None = None
    predicted_h_min: FloatArray | None = None
    feasibility_probability: FloatArray | None = None
    predicted_correction: FloatArray | None = None
    predicted_risk: FloatArray | None = None
    candidate_confidence: FloatArray | None = None

    def validate(self, num_candidates: int) -> None:
        expected = (num_candidates,)
        for name in (
            "candidate_ids",
            "analytical_cost",
            "predicted_h_min",
            "feasibility_probability",
            "predicted_correction",
            "predicted_risk",
            "candidate_confidence",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != expected:
                raise ValueError(f"{name} must have shape [{num_candidates}].")

        for name in (
            "analytical_cost",
            "predicted_h_min",
            "feasibility_probability",
            "predicted_correction",
            "predicted_risk",
            "candidate_confidence",
        ):
            value = getattr(self, name)
            if value is not None and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf.")

        if self.feasibility_probability is not None:
            if np.any((self.feasibility_probability < 0.0) | (self.feasibility_probability > 1.0)):
                raise ValueError("feasibility_probability must be in [0, 1].")


@dataclass
class RankingResult:
    """Sorted candidate ids and ranking scores. Higher score is better."""

    schema_version: str = "1.0"

    candidate_ids: IntArray | None = None
    scores: FloatArray | None = None
    selected_candidate_id: int = -1
    metadata: dict | None = None

    def validate(self, num_candidates: int) -> None:
        expected = (num_candidates,)
        if self.candidate_ids is not None and self.candidate_ids.shape != expected:
            raise ValueError(f"candidate_ids must have shape [{num_candidates}].")
        if self.scores is not None:
            if self.scores.shape != expected:
                raise ValueError(f"scores must have shape [{num_candidates}].")
            if not np.all(np.isfinite(self.scores)):
                raise ValueError("scores contains NaN or Inf.")
        if self.candidate_ids is not None and self.selected_candidate_id >= 0:
            if self.selected_candidate_id not in set(self.candidate_ids.tolist()):
                raise ValueError("selected_candidate_id is not present in candidate_ids.")
