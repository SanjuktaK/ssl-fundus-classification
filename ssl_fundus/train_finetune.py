"""
Fine-tuning Script for Multi-Label Disease Classification.

Loads a pretrained SSL backbone and trains a multi-label classifier
on the ODIR dataset for 8-disease prediction.

Usage:
    python -m ssl_fundus.train_finetune --backbone_path outputs/backbones/simclr_backbone.pt
    python -m ssl_fundus.train_finetune --backbone_path outputs/backbones/byol_backbone.pt --freeze_epochs 20
"""

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .config import get_config, TrainConfig, NUM_CLASSES, DISEASE_LABELS
from .data import FinetuneAugmentation
from .data.odir_dataset import ODIRDataset, get_dataloaders
from .models.backbone import get_backbone
from .models.classifier import MultiLabelClassifier
from .utils.metrics import compute_metrics, CombinedLoss
from .utils.training import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    get_cosine_schedule_with_warmup, compute_class_weights,
)
from .utils.visualization import (
    plot_training_curves, plot_roc_curves, plot_confusion_matrices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SSL backbone for disease classification")
    parser.add_argument("--backbone_path", type=str, required=True,
                        help="Path to pretrained backbone weights")
    parser.add_argument("--data_root", type=str, default="./data/ODIR")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=None,
                        help="Epochs to freeze backbone before full fine-tuning")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method_name", type=str, default="ssl",
                        help="Name tag for this run (e.g., simclr, byol, dino, ensemble)")
    return parser.parse_args()


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Run evaluation on a dataloader."""
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = (all_probs >= threshold).astype(np.float32)

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs, threshold)

    return avg_loss, metrics, all_labels, all_probs, all_preds


def train(config: TrainConfig, args):
    """Full fine-tuning pipeline."""
    cfg = config.finetune
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.learning_rate
    freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else cfg.freeze_backbone_epochs

    print(f"\n{'='*60}")
    print(f"  Multi-Label Fine-Tuning ({args.method_name.upper()})")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Backbone frozen for first {freeze_epochs} epochs")
    print(f"  Classes: {NUM_CLASSES} — {', '.join(DISEASE_LABELS)}")
    print(f"{'='*60}\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    train_transform = FinetuneAugmentation(config.data.image_size, is_train=True)
    val_transform = FinetuneAugmentation(config.data.image_size, is_train=False)

    loaders = get_dataloaders(
        args.data_root,
        finetune_train_transform=train_transform,
        finetune_val_transform=val_transform,
        mode="finetune",
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.data.seed,
    )

    # Compute class weights for imbalanced data
    full_dataset = ODIRDataset(args.data_root, mode="finetune")
    class_weights = compute_class_weights(full_dataset).to(device)

    # Model: load pretrained backbone + attach classifier
    backbone = get_backbone(config.backbone.name)
    backbone_state = torch.load(args.backbone_path, map_location=device)
    backbone.load_state_dict(backbone_state)
    print(f"[INFO] Loaded pretrained backbone from {args.backbone_path}")

    model = MultiLabelClassifier(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        dropout=cfg.dropout,
        freeze_backbone=(freeze_epochs > 0),
    ).to(device)

    # Loss
    criterion = CombinedLoss(
        bce_weight=cfg.bce_weight,
        dice_weight=cfg.dice_weight,
        label_smoothing=cfg.label_smoothing,
        class_weights=class_weights,
    )

    # Optimizer (only classifier params when backbone is frozen)
    if freeze_epochs > 0:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=cfg.weight_decay
        )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=epochs)
    scaler = GradScaler(enabled=config.mixed_precision)
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

    # Tracking
    train_losses, val_losses, val_aucs = [], [], []
    best_auc = 0.0

    for epoch in range(epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"\n[INFO] Unfreezing backbone at epoch {epoch+1}")
            model.unfreeze_backbone()
            # Re-create optimizer with all parameters and lower LR for backbone
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": lr * 0.1},
                {"params": model.classifier.parameters(), "lr": lr},
            ], weight_decay=cfg.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_epochs=3, total_epochs=epochs - epoch
            )

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(enabled=config.mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(loaders["train"])
        train_losses.append(avg_train_loss)

        # ── Validate ──
        val_loss, val_metrics, _, _, _ = evaluate(
            model, loaders["val"], criterion, device, cfg.threshold
        )
        val_losses.append(val_loss)
        val_auc = val_metrics.get("auc_macro", 0.0)
        val_aucs.append(val_auc)

        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[FT] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_path = os.path.join(args.output_dir, "checkpoints",
                                     f"best_{args.method_name}_classifier.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path,
                            extra={"val_auc": val_auc, "val_metrics": val_metrics})

        # Early stopping
        if early_stopper(val_auc):
            print(f"\n[STOP] Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
            break

    # ── Final Evaluation on Test Set ──
    print(f"\n{'='*60}")
    print(f"  Final Evaluation on Test Set")
    print(f"{'='*60}")

    # Load best model
    best_path = os.path.join(args.output_dir, "checkpoints",
                             f"best_{args.method_name}_classifier.pt")
    if os.path.exists(best_path):
        load_checkpoint(model, best_path, device=args.device)

    test_loss, test_metrics, y_true, y_prob, y_pred = evaluate(
        model, loaders["test"], criterion, device, cfg.threshold
    )

    print(f"\nTest Results ({args.method_name.upper()}):")
    print(f"  AUC (macro):     {test_metrics['auc_macro']:.4f}")
    print(f"  AUC (weighted):  {test_metrics['auc_weighted']:.4f}")
    print(f"  F1 (macro):      {test_metrics['f1_macro']:.4f}")
    print(f"  Precision:       {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:          {test_metrics['recall_macro']:.4f}")
    print(f"  Exact Match:     {test_metrics['exact_match']:.4f}")
    print(f"  mAP:             {test_metrics['ap_macro']:.4f}")

    print(f"\nPer-Class AUC:")
    for disease, auc_val in test_metrics["per_class_auc"].items():
        print(f"  {disease:20s}: {auc_val:.4f}" if not np.isnan(auc_val)
              else f"  {disease:20s}: N/A")

    # ── Save Results & Plots ──
    results_dir = os.path.join(args.output_dir, "results", args.method_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics JSON
    serializable_metrics = {
        k: v if not isinstance(v, dict) else {
            dk: float(dv) if not np.isnan(dv) else None for dk, dv in v.items()
        }
        for k, v in test_metrics.items()
    }
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Plots
    plot_training_curves(
        train_losses, val_losses, val_aucs, metric_name="AUC (macro)",
        save_path=os.path.join(results_dir, "training_curves.png"),
    )
    plot_roc_curves(y_true, y_prob,
                    save_path=os.path.join(results_dir, "roc_curves.png"))
    plot_confusion_matrices(y_true, y_pred,
                            save_path=os.path.join(results_dir, "confusion_matrices.png"))

    print(f"\n[DONE] Results saved to {results_dir}/")
    return test_metrics


def main():
    args = parse_args()
    config = get_config()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train(config, args)


if __name__ == "__main__":
    main()
