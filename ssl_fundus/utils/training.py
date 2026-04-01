"""
Training utilities: early stopping, checkpointing, schedulers.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric
    stops improving.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    extra: dict = None,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    print(f"[CKPT] Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[CKPT] Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
):
    """Cosine annealing scheduler with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute inverse frequency class weights for handling imbalanced ODIR data.
    """
    all_labels = []
    for sample in dataset.samples:
        all_labels.append(sample["labels"])

    labels = torch.tensor(all_labels, dtype=torch.float32)
    pos_count = labels.sum(dim=0)
    neg_count = len(labels) - pos_count

    # Inverse frequency weighting
    weights = neg_count / (pos_count + 1e-6)

    # Clamp to prevent extreme weights
    weights = torch.clamp(weights, min=0.5, max=10.0)

    print(f"[INFO] Class weights: {dict(zip(['N','D','G','C','A','H','M','O'], weights.tolist()))}")
    return weights
