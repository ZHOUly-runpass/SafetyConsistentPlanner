from __future__ import annotations

import json

import numpy as np
import pytest

from src.safety_planner.training.train_il import train_il_baseline
from src.safety_planner.training.train_ranker import train_ranker
from src.safety_planner.training.train_safety_alignment import train_safety_alignment
from src.safety_planner.datasets import tensorize_scene, write_npz_shards
from src.safety_planner.interfaces import SceneSample

torch = pytest.importorskip("torch")


def test_training_entrypoints_support_local_dry_run(tmp_path) -> None:
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps({"dry_run": True}), encoding="utf-8")

    il = train_il_baseline(str(config_path))
    safety = train_safety_alignment(str(config_path))
    ranker = train_ranker(str(config_path))

    assert il["status"] == "validated_entrypoint"
    assert safety["status"] == "validated_entrypoint"
    assert ranker["status"] == "validated_entrypoint"


def test_il_training_writes_recoverable_checkpoint(tmp_path) -> None:
    scenes = []
    for index in range(4):
        sample = SceneSample(
            scenario_id=f"train-{index}",
            ego_history=np.zeros((5, 4), dtype=np.float64),
            ego_history_mask=np.ones(5, dtype=np.bool_),
            agent_history=np.zeros((1, 5, 7), dtype=np.float64),
            agent_history_mask=np.ones((1, 5), dtype=np.bool_),
            expert_future=np.zeros((8, 4), dtype=np.float64),
            expert_future_mask=np.ones(8, dtype=np.bool_),
        )
        scenes.append(tensorize_scene(sample))
    data_dir = tmp_path / "data"
    write_npz_shards(scenes[:3], data_dir, "train", shard_size=2)
    write_npz_shards(scenes[3:], data_dir, "val", shard_size=2)
    config = {
        "dry_run": False,
        "data": {
            "train_manifest": str(data_dir / "train-manifest.jsonl"),
            "val_manifest": str(data_dir / "val-manifest.jsonl"),
        },
        "output_dir": str(tmp_path / "output"),
        "model": {"d_scene": 16, "d_entity": 8, "hidden_dim": 16},
        "training": {"epochs": 1, "batch_size": 2, "device": "cpu"},
    }
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = train_il_baseline(str(config_path), dry_run=False)
    assert result["status"] == "completed"
    checkpoint = torch.load(tmp_path / "output" / "best.pt", map_location="cpu")
    assert checkpoint["schema_version"] == "1.0"
    assert checkpoint["history"][0]["val_loss"] >= 0.0


def test_ranker_training_writes_gbdt_and_small_mlp(tmp_path) -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    features = rng.normal(size=(24, 5))
    features[:, 2] = 1.0
    labels = -features[:, 0] + 0.25 * features[:, 1]
    feature_path = tmp_path / "ranker.npz"
    output_path = tmp_path / "ranker.joblib"
    np.savez(feature_path, features=features, labels=labels)
    config_path = tmp_path / "ranker.json"
    config_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "feature_npz": str(feature_path),
                "output_path": str(output_path),
                "ranker": {"backend": "sklearn", "random_state": 4, "n_estimators": 10},
                "small_mlp": {"random_state": 4, "max_iter": 500},
            }
        ),
        encoding="utf-8",
    )

    result = train_ranker(str(config_path), dry_run=False)

    assert result["status"] == "completed"
    assert output_path.exists()
    assert (tmp_path / "ranker-small-mlp.joblib").exists()
    assert result["metrics"]["num_rows"] == 24
    assert np.isfinite(result["metrics"]["small_mlp_mse"])
