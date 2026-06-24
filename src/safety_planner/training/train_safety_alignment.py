from __future__ import annotations

from typing import Any

from ..utils import load_config


def train_safety_alignment(config_path: str, dry_run: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = {
        "stage": "safety_alignment",
        "config_path": config_path,
        "dry_run": dry_run,
        "model": "encoder + BSplineCandidateHead + SafetyPredictionHeads",
        "losses": [
            "pseudo_label_loss",
            "cbf_feasibility_loss",
            "feasibility_classification_loss",
            "correction_prediction_loss",
            "smoothness_loss",
            "diversity_loss",
        ],
    }
    if dry_run or config.get("dry_run", True):
        manifest["status"] = "validated_entrypoint"
        return manifest

    _require_training_dependency("torch", "PyTorch")
    raise RuntimeError(
        "Safety alignment training requires pseudo-label datasets generated on "
        "the development machine. Set dry_run=true for local validation."
    )


def _require_training_dependency(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"{package_name} is required for this training stage.") from exc
