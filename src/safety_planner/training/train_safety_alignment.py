from __future__ import annotations

from typing import Any

from ..utils import load_config
from .engine import TrainingConfig, train_safety_model


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
    train_manifest = config.get("train_manifest")
    output_dir = config.get("output_dir")
    if not train_manifest or not output_dir:
        raise ValueError("Safety alignment requires train_manifest and output_dir.")
    metrics = train_safety_model(
        train_manifest,
        output_dir,
        model_config=config.get("model", {}),
        training_config=TrainingConfig(**config.get("training", {})),
    )
    return {**manifest, "status": "completed", "metrics": metrics}


def _require_training_dependency(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"{package_name} is required for this training stage.") from exc
