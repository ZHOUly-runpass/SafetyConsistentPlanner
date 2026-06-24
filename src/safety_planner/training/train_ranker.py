from __future__ import annotations

from typing import Any

from ..utils import load_config


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

    raise RuntimeError(
        "Ranker training requires pseudo-label feature tables generated on the "
        "development machine. Set dry_run=true for local validation."
    )
