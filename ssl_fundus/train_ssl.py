"""
SSL Pretraining Script.

Trains a ResNet-50 backbone using SimCLR or BYOL
on the ODIR fundus image dataset (unlabeled).

Usage:
    python -m ssl_fundus pretrain --method simclr --epochs 200 --batch_size 256
    python -m ssl_fundus pretrain --method byol
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .config import get_config, TrainConfig
from .data import SimCLRAugmentation, BYOLAugmentation
from .data.odir_dataset import ODIRDataset, get_dataloaders
from .ssl_methods.simclr import SimCLR
from .ssl_methods.byol import BYOL
from .utils.training import save_checkpoint, get_cosine_schedule_with_warmup
from .utils.visualization import plot_ssl_loss


def parse_args():
    parser = argparse.ArgumentParser(description="SSL Pretraining for Fundus Images")
    parser.add_argument("--method", type=str, default="simclr",
                        choices=["simclr", "byol"],
                        help="SSL pretraining method")
    parser.add_argument("--data_root", type=str, default="./data/ODIR",
                        help="Path to ODIR dataset")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def train_simclr(config: TrainConfig, args):
    """Train using SimCLR."""
    cfg = config.simclr
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.learning_rate

    print(f"\n{'='*60}")
    print(f"  SimCLR Pretraining")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Temperature: {cfg.temperature}")
    print(f"{'='*60}\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    augmentation = SimCLRAugmentation(image_size=config.data.image_size)
    loaders = get_dataloaders(
        args.data_root, ssl_transform=augmentation, mode="ssl",
        batch_size=batch_size, num_workers=config.data.num_workers,
    )
    train_loader = loaders["train"]

    # Model
    model = SimCLR(
        backbone_name=config.backbone.name,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        temperature=cfg.temperature,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, epochs)
    scaler = GradScaler(enabled=config.mixed_precision)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[RESUME] Starting from epoch {start_epoch}")

    # Training loop
    losses = []
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (view1, view2) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            with autocast(enabled=config.mixed_precision):
                loss, _, _ = model(view1, view2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[SimCLR] Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"simclr_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    # Save final
    final_path = os.path.join(args.output_dir, "checkpoints", "simclr_final.pt")
    save_checkpoint(model, optimizer, epochs - 1, avg_loss, final_path)

    # Save backbone only (for fine-tuning)
    backbone_path = os.path.join(args.output_dir, "backbones", "simclr_backbone.pt")
    os.makedirs(os.path.dirname(backbone_path), exist_ok=True)
    torch.save(model.backbone.state_dict(), backbone_path)
    print(f"[SimCLR] Backbone saved to {backbone_path}")

    return losses


def train_byol(config: TrainConfig, args):
    """Train using BYOL."""
    cfg = config.byol
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.learning_rate

    print(f"\n{'='*60}")
    print(f"  BYOL Pretraining")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  EMA Decay: {cfg.ema_decay} → {cfg.ema_decay_end}")
    print(f"{'='*60}\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    augmentation = BYOLAugmentation(image_size=config.data.image_size)
    loaders = get_dataloaders(
        args.data_root, ssl_transform=augmentation, mode="ssl",
        batch_size=batch_size, num_workers=config.data.num_workers,
    )
    train_loader = loaders["train"]

    # Model
    model = BYOL(
        backbone_name=config.backbone.name,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        prediction_dim=cfg.prediction_dim,
        ema_decay=cfg.ema_decay,
    ).to(device)

    # Only optimize online network parameters
    online_params = (
        list(model.online_backbone.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(online_params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, epochs)
    scaler = GradScaler(enabled=config.mixed_precision)

    total_steps = epochs * len(train_loader)
    global_step = 0

    # Training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (view1, view2) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            with autocast(enabled=config.mixed_precision):
                loss = model(view1, view2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(online_params, config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # Update target network with cosine EMA schedule
            decay = BYOL.cosine_schedule_ema(
                cfg.ema_decay, cfg.ema_decay_end, global_step, total_steps
            )
            model.update_target(decay)
            global_step += 1

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[BYOL] Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"EMA: {decay:.4f} | "
                  f"Time: {elapsed:.1f}s")

        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"byol_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    # Save final
    final_path = os.path.join(args.output_dir, "checkpoints", "byol_final.pt")
    save_checkpoint(model, optimizer, epochs - 1, avg_loss, final_path)

    backbone_path = os.path.join(args.output_dir, "backbones", "byol_backbone.pt")
    os.makedirs(os.path.dirname(backbone_path), exist_ok=True)
    torch.save(model.online_backbone.state_dict(), backbone_path)
    print(f"[BYOL] Backbone saved to {backbone_path}")

    return losses


def main():
    args = parse_args()
    config = get_config(args.method)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "backbones"), exist_ok=True)

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    trainers = {
        "simclr": train_simclr,
        "byol": train_byol,
    }

    losses = trainers[args.method](config, args)

    # Plot loss curve
    plot_ssl_loss(
        {args.method: losses},
        save_path=os.path.join(args.output_dir, f"{args.method}_pretraining_loss.png"),
    )

    print(f"\n[DONE] {args.method.upper()} pretraining complete!")
    print(f"  Backbone: {args.output_dir}/backbones/{args.method}_backbone.pt")


if __name__ == "__main__":
    main()
