from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets.shards import NpzShardDataset
from ..models import SafetyConsistentPlannerModel
from .losses import (
    cbf_feasibility_loss,
    correction_prediction_loss,
    expert_imitation_loss,
    feasibility_classification_loss,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_il_model(
    train_manifest: str | Path,
    val_manifest: str | Path,
    output_dir: str | Path,
    model_config: dict | None = None,
    training_config: TrainingConfig | None = None,
) -> dict:
    cfg = training_config or TrainingConfig()
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    model = SafetyConsistentPlannerModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_loader = _loader(train_manifest, cfg, shuffle=True)
    val_loader = _loader(val_manifest, cfg, shuffle=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss = _run_il_epoch(model, train_loader, device, optimizer, cfg.gradient_clip_norm)
        with torch.no_grad():
            val_loss = _run_il_epoch(model, val_loader, device, None, cfg.gradient_clip_norm)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        checkpoint = _checkpoint(model, optimizer, cfg, model_config or {}, history)
        torch.save(checkpoint, output / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, output / "best.pt")
    metrics = {"best_val_loss": best_loss, "history": history, "config": asdict(cfg)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_safety_epoch(
    model: SafetyConsistentPlannerModel,
    batches,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in batches:
        tensors = _move_batch(batch, device)
        output = model(tensors)
        h_loss = cbf_feasibility_loss(output["predicted_h_min"], tensors["target_h_min"])
        feasible_loss = feasibility_classification_loss(
            output["feasibility_logits"], tensors["target_feasible"]
        )
        correction_loss = correction_prediction_loss(
            output["predicted_correction"], tensors["target_correction"]
        )
        risk_loss = torch.mean((output["predicted_risk"] - tensors["target_risk"]) ** 2)
        loss = h_loss + feasible_loss + correction_loss + risk_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def train_safety_model(
    train_manifest: str | Path,
    output_dir: str | Path,
    model_config: dict | None = None,
    training_config: TrainingConfig | None = None,
) -> dict:
    cfg = training_config or TrainingConfig()
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    model = SafetyConsistentPlannerModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = _loader(train_manifest, cfg, shuffle=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    for epoch in range(cfg.epochs):
        loss = train_safety_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch + 1, "train_loss": loss})
        torch.save(_checkpoint(model, optimizer, cfg, model_config or {}, history), output / "last.pt")
    metrics = {"history": history, "config": asdict(cfg)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_il_epoch(model, loader, device, optimizer, gradient_clip_norm: float) -> float:
    model.train(optimizer is not None)
    values: list[float] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        prediction = model(batch)["base_trajectory"]
        target = batch["expert_future"].float()
        mask = batch["expert_future_mask"].bool()
        loss = expert_imitation_loss(prediction, target, mask)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values)) if values else 0.0


def _loader(path: str | Path, config: TrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        NpzShardDataset(path),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
        if name != "manifest"
    }


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _checkpoint(model, optimizer, config, model_config, history) -> dict:
    return {
        "schema_version": "1.0",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_config": asdict(config),
        "model_config": model_config,
        "history": history,
    }
