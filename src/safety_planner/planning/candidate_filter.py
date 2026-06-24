from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..interfaces.planner import PlannerOutput


def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def hard_filter_candidates(
    planner_output: PlannerOutput,
    batch_index: int = 0,
    min_predicted_h: float = 0.0,
    min_feasibility_probability: float = 0.5,
) -> NDArray[np.int64]:
    if planner_output.candidate_trajectories is None:
        raise ValueError("candidate_trajectories is required.")
    num_candidates = planner_output.candidate_trajectories.shape[1]
    keep = np.ones(num_candidates, dtype=np.bool_)

    if planner_output.predicted_h_min is not None:
        keep &= planner_output.predicted_h_min[batch_index] >= min_predicted_h
    if planner_output.feasibility_logits is not None:
        keep &= _sigmoid(planner_output.feasibility_logits[batch_index]) >= min_feasibility_probability

    return np.nonzero(keep)[0].astype(np.int64)


def score_candidates(
    planner_output: PlannerOutput,
    batch_index: int = 0,
    candidate_ids: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    if planner_output.candidate_trajectories is None:
        raise ValueError("candidate_trajectories is required.")
    num_candidates = planner_output.candidate_trajectories.shape[1]
    ids = candidate_ids if candidate_ids is not None else np.arange(num_candidates, dtype=np.int64)
    scores = np.zeros(ids.shape[0], dtype=np.float64)

    if planner_output.candidate_logits is not None:
        scores += planner_output.candidate_logits[batch_index, ids]
    if planner_output.predicted_h_min is not None:
        scores += planner_output.predicted_h_min[batch_index, ids]
    if planner_output.feasibility_logits is not None:
        scores += _sigmoid(planner_output.feasibility_logits[batch_index, ids])
    if planner_output.predicted_correction is not None:
        scores -= planner_output.predicted_correction[batch_index, ids]
    if planner_output.predicted_risk is not None:
        scores -= planner_output.predicted_risk[batch_index, ids]

    return scores
