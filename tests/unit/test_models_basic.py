from __future__ import annotations

import numpy as np
import pytest

from src.safety_planner.interfaces import SceneSample
from src.safety_planner.models.lightweight_ranker import GBDTRanker

torch = pytest.importorskip("torch")

from src.safety_planner.models.bspline_candidate_head import BSplineCandidateHead
from src.safety_planner.models.scene_encoder import VectorSceneEncoder


def test_vector_scene_encoder_returns_embedding() -> None:
    sample = SceneSample(
        scenario_id="synthetic",
        ego_history=np.zeros((5, 4), dtype=np.float64),
        ego_history_mask=np.ones(5, dtype=np.bool_),
        agent_history=np.zeros((2, 5, 4), dtype=np.float64),
        agent_history_mask=np.ones((2, 5), dtype=np.bool_),
        expert_future=np.zeros((8, 4), dtype=np.float64),
        expert_future_mask=np.ones(8, dtype=np.bool_),
    )
    encoder = VectorSceneEncoder({"d_scene": 16})
    embedding = encoder(sample)
    assert embedding.shape == (1, 16)
    assert torch.isfinite(embedding).all()


def test_bspline_candidate_head_returns_candidate_states() -> None:
    head = BSplineCandidateHead(d_scene=16, num_candidates=4, num_future=8)
    out = head(torch.zeros((2, 16), dtype=torch.float32))
    assert out.shape == (2, 4, 8, 4)
    assert torch.isfinite(out).all()


def test_gbdt_ranker_falls_back_to_linear_model() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ranker = GBDTRanker()
    ranker.fit(features, labels)
    pred = ranker.predict(features)
    assert pred.shape == (3,)
    assert pred[2] > pred[0]
