__all__ = [
    "expert_imitation_loss",
    "pseudo_label_loss",
    "cbf_feasibility_loss",
    "feasibility_classification_loss",
    "correction_prediction_loss",
    "smoothness_loss",
    "diversity_loss",
    "compute_total_loss",
    "train_il_baseline",
    "train_safety_alignment",
    "train_ranker",
]


def __getattr__(name: str):
    if name in {
        "expert_imitation_loss",
        "pseudo_label_loss",
        "cbf_feasibility_loss",
        "feasibility_classification_loss",
        "correction_prediction_loss",
        "smoothness_loss",
        "diversity_loss",
        "compute_total_loss",
    }:
        from . import losses

        return getattr(losses, name)
    if name == "train_il_baseline":
        from .train_il import train_il_baseline

        return train_il_baseline
    if name == "train_safety_alignment":
        from .train_safety_alignment import train_safety_alignment

        return train_safety_alignment
    if name == "train_ranker":
        from .train_ranker import train_ranker

        return train_ranker
    raise AttributeError(name)
from .engine import (
    TrainingConfig,
    seed_everything,
    train_il_model,
    train_safety_epoch,
    train_safety_model,
)
