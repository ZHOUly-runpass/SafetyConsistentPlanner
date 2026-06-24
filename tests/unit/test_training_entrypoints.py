from __future__ import annotations

import json

from src.safety_planner.training.train_il import train_il_baseline
from src.safety_planner.training.train_ranker import train_ranker
from src.safety_planner.training.train_safety_alignment import train_safety_alignment


def test_training_entrypoints_support_local_dry_run(tmp_path) -> None:
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps({"dry_run": True}), encoding="utf-8")

    il = train_il_baseline(str(config_path))
    safety = train_safety_alignment(str(config_path))
    ranker = train_ranker(str(config_path))

    assert il["status"] == "validated_entrypoint"
    assert safety["status"] == "validated_entrypoint"
    assert ranker["status"] == "validated_entrypoint"
