"""
Visualization utilities for training curves, ROC, and confusion matrices.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, List, Optional

from ..config import DISEASE_LABELS


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "AUC",
    save_path: str = "training_curves.png",
):
    """Plot training loss and validation metric curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", color="#FF5722", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric curve
    if val_metrics:
        axes[1].plot(val_metrics, label=f"Val {metric_name}", color="#4CAF50", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f"Validation {metric_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves saved to {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = "roc_curves.png",
):
    """Plot per-class ROC curves."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(DISEASE_LABELS)))

    for i, (label, color) in enumerate(zip(DISEASE_LABELS, colors)):
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ROC curves saved to {save_path}")


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "confusion_matrices.png",
):
    """Plot per-class confusion matrices in a grid."""
    n_classes = len(DISEASE_LABELS)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, (label, ax) in enumerate(zip(DISEASE_LABELS, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])

        # Annotate cells
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[row, col] > cm.max() / 2 else "black")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Confusion matrices saved to {save_path}")


def plot_ssl_loss(
    losses: Dict[str, List[float]],
    save_path: str = "ssl_pretraining_loss.png",
):
    """Plot SSL pretraining loss curves (can compare methods)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"simclr": "#2196F3", "byol": "#4CAF50", "dino": "#FF9800"}

    for method, loss_vals in losses.items():
        color = colors.get(method, "#9C27B0")
        ax.plot(loss_vals, label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SSL Pretraining Loss", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] SSL loss curves saved to {save_path}")
