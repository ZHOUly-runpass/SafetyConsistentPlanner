from __future__ import annotations

from typing import Any

from ..utils import load_config
from ..models.lightweight_ranker import GBDTRanker


def train_ranker(config_path: str, dry_run: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = {
        "stage": "ranker",
        "config_path": config_path,
        "dry_run": dry_run,
        "inputs": ["analytical_cost", "safety_predictions", "pseudo_label_quality"],
        "baselines": [
            "random",
            "candidate_confidence",
            "analytical_cost",
            "analytical_plus_safety",
            "gbdt_or_linear_fallback",
            "small_mlp",
            "mpc_oracle",
        ],
    }
    if dry_run or config.get("dry_run", True):
        manifest["status"] = "validated_entrypoint"
        return manifest

    import json
    from pathlib import Path

    import joblib
    import numpy as np

    feature_path = config.get("feature_npz")
    output_path = config.get("output_path")
    if not feature_path or not output_path:
        raise ValueError("Ranker training requires feature_npz and output_path.")
    with np.load(feature_path) as data:
        features = data["features"]
        labels = data["labels"]
    ranker = GBDTRanker(config.get("ranker", {}))
    ranker.fit(features, labels)
    predictions = ranker.predict(features)
    mse = float(np.mean((predictions - labels) ** 2))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ranker, target)
    metrics = {"training_mse": mse, "num_rows": int(features.shape[0])}
    target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {**manifest, "status": "completed", "metrics": metrics}
