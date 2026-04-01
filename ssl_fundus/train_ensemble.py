"""
Ensemble Training Script.

Trains both SSL methods (SimCLR, BYOL), fine-tunes each,
then combines their predictions via late fusion for improved
multi-label disease classification.

Usage:
    python -m ssl_fundus ensemble --data_root ./data/ODIR
    python -m ssl_fundus ensemble --skip_pretrain  # if backbones already trained
"""

import argparse
import os
import json

import numpy as np
import torch

from .config import get_config, NUM_CLASSES, DISEASE_LABELS
from .data import FinetuneAugmentation
from .data.odir_dataset import ODIRDataset, get_dataloaders
from .models.backbone import get_backbone
from .models.classifier import MultiLabelClassifier
from .utils.metrics import compute_metrics
from .utils.visualization import plot_roc_curves, plot_confusion_matrices


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble SSL Training Pipeline")
    parser.add_argument("--data_root", type=str, default="./data/ODIR")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip SSL pretraining (use existing backbones)")
    parser.add_argument("--ssl_epochs", type=int, default=200)
    parser.add_argument("--ft_epochs", type=int, default=100)
    parser.add_argument("--fusion", type=str, default="average",
                        choices=["average", "weighted", "max"],
                        help="Ensemble fusion strategy")
    return parser.parse_args()


def get_ensemble_predictions(models, loader, device):
    """Get predictions from all models in the ensemble."""
    all_probs = {name: [] for name in models}
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            all_labels.append(labels.numpy())

            for name, model in models.items():
                model.eval()
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs[name].append(probs)

    all_labels = np.concatenate(all_labels, axis=0)
    for name in all_probs:
        all_probs[name] = np.concatenate(all_probs[name], axis=0)

    return all_labels, all_probs


def fuse_predictions(all_probs: dict, strategy: str = "average",
                     weights: dict = None) -> np.ndarray:
    """
    Fuse predictions from multiple models.

    Strategies:
        - average: Simple mean of probabilities
        - weighted: Weighted mean (weights from per-method val AUC)
        - max: Element-wise maximum
    """
    prob_arrays = list(all_probs.values())

    if strategy == "average":
        return np.mean(prob_arrays, axis=0)

    elif strategy == "weighted":
        if weights is None:
            return np.mean(prob_arrays, axis=0)
        total_w = sum(weights.values())
        fused = np.zeros_like(prob_arrays[0])
        for name, probs in all_probs.items():
            w = weights.get(name, 1.0) / total_w
            fused += w * probs
        return fused

    elif strategy == "max":
        return np.maximum.reduce(prob_arrays)

    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


def main():
    args = parse_args()
    config = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    methods = ["simclr", "byol"]

    # ── Step 1: SSL Pretraining ──
    if not args.skip_pretrain:
        print("\n" + "=" * 60)
        print("  PHASE 1: SSL Pretraining (SimCLR + BYOL)")
        print("=" * 60)

        from .train_ssl import train_simclr, train_byol

        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "backbones"), exist_ok=True)

        ssl_args = argparse.Namespace(
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.ssl_epochs,
            batch_size=None,
            lr=None,
            resume=None,
        )

        all_ssl_losses = {}
        for method, trainer in [("simclr", train_simclr), ("byol", train_byol)]:
            ssl_args.method = method
            losses = trainer(config, ssl_args)
            all_ssl_losses[method] = losses

        from .utils.visualization import plot_ssl_loss
        plot_ssl_loss(all_ssl_losses,
                      save_path=os.path.join(args.output_dir, "all_ssl_losses.png"))

    # ── Step 2: Fine-tune each backbone ──
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-Tuning Each Backbone")
    print("=" * 60)

    from .train_finetune import train as finetune_train

    method_metrics = {}
    for method in methods:
        backbone_path = os.path.join(args.output_dir, "backbones", f"{method}_backbone.pt")
        if not os.path.exists(backbone_path):
            print(f"[WARN] Backbone not found: {backbone_path}, skipping {method}")
            continue

        ft_args = argparse.Namespace(
            backbone_path=backbone_path,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.ft_epochs,
            batch_size=None,
            lr=None,
            freeze_epochs=None,
            method_name=method,
        )

        metrics = finetune_train(config, ft_args)
        method_metrics[method] = metrics

    # ── Step 3: Ensemble Evaluation ──
    print("\n" + "=" * 60)
    print("  PHASE 3: Ensemble Fusion & Evaluation")
    print("=" * 60)

    # Load best fine-tuned models
    ensemble_models = {}
    for method in methods:
        ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                 f"best_{method}_classifier.pt")
        if not os.path.exists(ckpt_path):
            continue

        backbone = get_backbone(config.backbone.name)
        model = MultiLabelClassifier(backbone=backbone, num_classes=NUM_CLASSES)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        ensemble_models[method] = model

    if len(ensemble_models) < 2:
        print("[WARN] Need at least 2 models for ensemble. Exiting.")
        return

    # Test dataloader
    val_transform = FinetuneAugmentation(config.data.image_size, is_train=False)
    loaders = get_dataloaders(
        args.data_root,
        finetune_val_transform=val_transform,
        finetune_train_transform=val_transform,
        mode="finetune",
        batch_size=config.finetune.batch_size,
        num_workers=config.data.num_workers,
    )

    y_true, all_probs = get_ensemble_predictions(ensemble_models, loaders["test"], device)

    # Compute weights from individual AUCs for weighted fusion
    fusion_weights = {}
    for method, metrics in method_metrics.items():
        fusion_weights[method] = metrics.get("auc_macro", 0.5)

    # Try all fusion strategies
    results_dir = os.path.join(args.output_dir, "results", "ensemble")
    os.makedirs(results_dir, exist_ok=True)

    for strategy in ["average", "weighted", "max"]:
        fused_probs = fuse_predictions(all_probs, strategy, fusion_weights)
        fused_preds = (fused_probs >= config.finetune.threshold).astype(np.float32)

        metrics = compute_metrics(y_true, fused_preds, fused_probs)

        print(f"\nEnsemble ({strategy.upper()}) Results:")
        print(f"  AUC (macro):  {metrics['auc_macro']:.4f}")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  mAP:          {metrics['ap_macro']:.4f}")
        print(f"  Exact Match:  {metrics['exact_match']:.4f}")

        # Save best fusion results
        if strategy == args.fusion:
            serializable = {
                k: (v if not isinstance(v, dict) else
                    {dk: float(dv) if not np.isnan(dv) else None
                     for dk, dv in v.items()})
                for k, v in metrics.items()
            }
            with open(os.path.join(results_dir, f"ensemble_{strategy}_metrics.json"), "w") as f:
                json.dump(serializable, f, indent=2)

            plot_roc_curves(y_true, fused_probs,
                            save_path=os.path.join(results_dir, "ensemble_roc_curves.png"))
            plot_confusion_matrices(y_true, fused_preds,
                                    save_path=os.path.join(results_dir, "ensemble_confusion.png"))

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  Summary: Individual vs Ensemble")
    print(f"{'='*60}")
    print(f"  {'Method':<15} {'AUC (macro)':<15} {'F1 (macro)':<15} {'mAP':<10}")
    print(f"  {'-'*55}")
    for method, metrics in method_metrics.items():
        print(f"  {method.upper():<15} "
              f"{metrics.get('auc_macro', 0):<15.4f} "
              f"{metrics.get('f1_macro', 0):<15.4f} "
              f"{metrics.get('ap_macro', 0):<10.4f}")

    fused_probs = fuse_predictions(all_probs, args.fusion, fusion_weights)
    fused_preds = (fused_probs >= config.finetune.threshold).astype(np.float32)
    ens_metrics = compute_metrics(y_true, fused_preds, fused_probs)
    print(f"  {'ENSEMBLE':<15} "
          f"{ens_metrics['auc_macro']:<15.4f} "
          f"{ens_metrics['f1_macro']:<15.4f} "
          f"{ens_metrics['ap_macro']:<10.4f}")

    print(f"\n[DONE] Full ensemble pipeline complete!")
    print(f"  Results: {results_dir}/")


if __name__ == "__main__":
    main()
