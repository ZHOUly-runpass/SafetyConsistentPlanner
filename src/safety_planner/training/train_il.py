from __future__ import annotations

from typing import Any

from ..utils import load_config
from .engine import TrainingConfig, train_il_model


def train_il_baseline(config_path: str, dry_run: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = {
        "stage": "il_baseline",
        "config_path": config_path,
        "dry_run": dry_run,
        "model": "VectorSceneEncoder + ILHead",
        "losses": ["expert_imitation_loss"],
        "metrics": ["ADE", "FDE", "yaw_error", "velocity_error"],
    }
    if dry_run or config.get("dry_run", True):
        manifest["status"] = "validated_entrypoint"
        return manifest

    _require_training_dependency("torch", "PyTorch")
    data = config.get("data", {})
    output_dir = config.get("output_dir")
    if not data.get("train_manifest") or not data.get("val_manifest") or not output_dir:
        raise ValueError("Real IL training requires train_manifest, val_manifest, and output_dir.")
    train_cfg = TrainingConfig(**config.get("training", {}))
    metrics = train_il_model(
        data["train_manifest"],
        data["val_manifest"],
        output_dir,
        model_config=config.get("model", {}),
        training_config=train_cfg,
    )
    return {**manifest, "status": "completed", "metrics": metrics}


def _require_training_dependency(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"{package_name} is required for this training stage.") from exc
