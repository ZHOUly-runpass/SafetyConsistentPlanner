from __future__ import annotations

import json

import numpy as np
import pytest

from src.safety_planner.datasets import (
    deterministic_subsets,
    tensorize_scene,
    write_npz_shards,
    write_supervised_npz_shards,
)
from src.safety_planner.interfaces import SceneSample


def _sample(scenario_id: str = "scene") -> SceneSample:
    return SceneSample(
        scenario_id=scenario_id,
        timestamp_us=123,
        ego_history=np.ones((3, 4), dtype=np.float64),
        ego_history_mask=np.ones(3, dtype=np.bool_),
        agent_history=np.ones((2, 3, 7), dtype=np.float64),
        agent_history_mask=np.ones((2, 3), dtype=np.bool_),
        agent_type=np.array([1, 2], dtype=np.int32),
        map_polylines=[np.ones((4, 5), dtype=np.float64)],
        route_polyline=np.ones((6, 4), dtype=np.float64),
        expert_future=np.ones((8, 4), dtype=np.float64),
    )


def test_tensorizer_pads_and_masks_without_future_leakage() -> None:
    tensorized = tensorize_scene(_sample())
    tensorized.validate()
    assert tensorized.ego_history.shape == (5, 4)
    assert tensorized.agent_history.shape == (64, 5, 7)
    assert tensorized.map_polylines.shape == (128, 20, 5)
    assert tensorized.route_polyline.shape == (64, 4)
    assert tensorized.expert_future.shape == (8, 4)
    assert tensorized.ego_history_mask.sum() == 3
    assert tensorized.agent_history_mask.sum() == 6


def test_deterministic_subsets_are_disjoint_and_repeatable() -> None:
    ids = [f"scene-{index:04d}" for index in range(30)]
    first = deterministic_subsets(ids, 10, 5, 3, seed=7)
    second = deterministic_subsets(list(reversed(ids)), 10, 5, 3, seed=7)
    assert first == second
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["closed_loop"]))


def test_deterministic_subsets_reject_insufficient_scenarios() -> None:
    with pytest.raises(ValueError, match="Need 6"):
        deterministic_subsets(["a", "b"], 2, 2, 2)


def test_npz_shards_and_manifest_round_trip(tmp_path) -> None:
    scenes = [tensorize_scene(_sample(f"scene-{index}")) for index in range(3)]
    rows = write_npz_shards(scenes, tmp_path, "train", shard_size=2, config_hash="abc")
    assert [row["shard"] for row in rows] == [
        "train-00000.npz",
        "train-00000.npz",
        "train-00001.npz",
    ]
    with np.load(tmp_path / "train-00000.npz") as shard:
        assert shard["ego_history"].shape == (2, 5, 4)
        assert shard["scenario_id"].tolist() == ["scene-0", "scene-1"]
    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "train-manifest.jsonl").read_text().splitlines()
    ]
    assert manifest_rows == rows


def test_supervised_shards_align_targets_with_scenes(tmp_path) -> None:
    scenes = [tensorize_scene(_sample(f"scene-{index}")) for index in range(2)]
    targets = {"target_h_min": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    write_supervised_npz_shards(scenes, targets, tmp_path, "train", shard_size=2)
    with np.load(tmp_path / "train-00000.npz") as shard:
        np.testing.assert_array_equal(shard["target_h_min"], targets["target_h_min"])
