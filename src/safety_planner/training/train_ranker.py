from __future__ import annotations

from typing import Any

from ..utils import load_config
from ..models.lightweight_ranker import GBDTRanker, SmallMLPRanker


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
    ranker_config = config.get("ranker", {})
    gbdt = GBDTRanker(ranker_config)
    gbdt.fit(features, labels)
    mlp = SmallMLPRanker(config.get("small_mlp", ranker_config))
    mlp.fit(features, labels)
    analytical = -features[:, 0] - np.where(features[:, 2] >= 0.5, 0.0, 1e6)
    gbdt_predictions = gbdt.predict(features)
    mlp_predictions = mlp.predict(features)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mlp_target = target.with_name(target.stem + "-small-mlp" + target.suffix)
    joblib.dump(gbdt, target)
    joblib.dump(mlp, mlp_target)
    metrics = {
        "training_mse": float(np.mean((gbdt_predictions - labels) ** 2)),
        "analytical_mse": float(np.mean((analytical - labels) ** 2)),
        "gbdt_mse": float(np.mean((gbdt_predictions - labels) ** 2)),
        "small_mlp_mse": float(np.mean((mlp_predictions - labels) ** 2)),
        "num_rows": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "artifacts": {"gbdt": str(target), "small_mlp": str(mlp_target)},
    }
    target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {**manifest, "status": "completed", "metrics": metrics}
