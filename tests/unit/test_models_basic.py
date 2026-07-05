from __future__ import annotations

import numpy as np
import pytest

from src.safety_planner.interfaces import SceneSample
from src.safety_planner.models.lightweight_ranker import GBDTRanker

torch = pytest.importorskip("torch")

from src.safety_planner.models.bspline_candidate_head import BSplineCandidateHead
from src.safety_planner.models.scene_encoder import VectorSceneEncoder
from src.safety_planner.datasets import tensorize_scene
from src.safety_planner.models.planner_model import SafetyConsistentPlannerModel


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


def test_bspline_candidates_are_residuals_around_base() -> None:
    head = BSplineCandidateHead(d_scene=8, num_candidates=2, num_future=8)
    for parameter in head.parameters():
        torch.nn.init.zeros_(parameter)
    base = torch.zeros((1, 8, 4), dtype=torch.float32)
    base[..., 0] = torch.arange(8, dtype=torch.float32)
    candidates = head(torch.zeros((1, 8)), base)
    torch.testing.assert_close(
        candidates[:, :, :, :2], base[:, None, :, :2].expand(-1, 2, -1, -1)
    )


def test_vector_model_accepts_tensorized_batch_and_backpropagates() -> None:
    sample = SceneSample(
        scenario_id="batch",
        ego_history=np.zeros((5, 4), dtype=np.float64),
        ego_history_mask=np.ones(5, dtype=np.bool_),
        agent_history=np.zeros((2, 5, 7), dtype=np.float64),
        agent_history_mask=np.ones((2, 5), dtype=np.bool_),
        expert_future=np.zeros((8, 4), dtype=np.float64),
        expert_future_mask=np.ones(8, dtype=np.bool_),
    )
    tensorized = tensorize_scene(sample)
    names = (
        "ego_history",
        "ego_history_mask",
        "agent_history",
        "agent_history_mask",
        "map_polylines",
        "map_mask",
        "route_polyline",
        "route_mask",
    )
    batch = {
        name: torch.as_tensor(getattr(tensorized, name)).unsqueeze(0)
        for name in names
    }
    model = SafetyConsistentPlannerModel(
        {"d_scene": 32, "hidden_dim": 32, "num_candidates": 4, "num_future": 8}
    )
    output = model(batch)
    assert output["base_trajectory"].shape == (1, 8, 4)
    assert output["candidate_trajectories"].shape == (1, 4, 8, 4)
    assert output["predicted_h_min"].shape == (1, 4)
    loss = sum(value.float().mean() for value in output.values())
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


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
