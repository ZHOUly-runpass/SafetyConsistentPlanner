from __future__ import annotations

from typing import Any

from ..utils import load_config


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
    _require_training_dependency("nuplan", "nuPlan devkit")
    raise RuntimeError(
        "Real IL training requires configured nuPlan database paths, dataloader "
        "splits, checkpoint output paths, and a Linux training environment. "
        "Set dry_run=true for local validation."
    )


def _require_training_dependency(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"{package_name} is required for this training stage.") from exc
