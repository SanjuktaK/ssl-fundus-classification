"""
Evaluation metrics for multi-label fundus disease classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    multilabel_confusion_matrix,
)

from ..config import DISEASE_LABELS, NUM_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        y_true: Ground truth binary labels (N, C)
        y_pred: Predicted binary labels (N, C), thresholded
        y_prob: Predicted probabilities (N, C)

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # Overall metrics (macro-averaged)
    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["auc_weighted"] = roc_auc_score(y_true, y_prob, average="weighted")
    except ValueError:
        metrics["auc_macro"] = 0.0
        metrics["auc_weighted"] = 0.0

    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        metrics["ap_macro"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["ap_macro"] = 0.0

    # Exact match ratio (all labels correct for a sample)
    metrics["exact_match"] = np.mean(np.all(y_true == y_pred, axis=1))

    # Hamming accuracy
    metrics["hamming_accuracy"] = np.mean(y_true == y_pred)

    # Per-class metrics
    per_class = per_class_auc(y_true, y_prob)
    metrics["per_class_auc"] = per_class

    return metrics


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute AUC-ROC for each disease class.

    Returns:
        Dict mapping disease name to AUC score.
    """
    results = {}
    for i, label in enumerate(DISEASE_LABELS):
        try:
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            else:
                auc = float("nan")
        except ValueError:
            auc = float("nan")
        results[label] = auc
    return results


class DiceLoss(torch.nn.Module):
    """Soft Dice loss for multi-label classification."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=0)
        union = probs.sum(dim=0) + targets.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(torch.nn.Module):
    """
    Combined BCE + Dice loss for multi-label fundus classification.
    Addresses class imbalance common in ODIR dataset.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.bce = torch.nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
