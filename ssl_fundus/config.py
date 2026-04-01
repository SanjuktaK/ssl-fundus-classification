"""
Configuration for SSL Fundus Image Classification Pipeline.
Supports SimCLR and BYOL pretraining on ODIR dataset with
multi-label disease classification fine-tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ──────────────────────────────────────────────
# ODIR Disease Labels (4 selected categories)
# ──────────────────────────────────────────────
DISEASE_LABELS = [
    "Normal",
    "Diabetes",        # Diabetic Retinopathy
    "Glaucoma",
    "Hypertension",
]

NUM_CLASSES = len(DISEASE_LABELS)


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    data_root: str = "./data/ODIR"
    image_size: int = 224
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    # Augmentation
    color_jitter_strength: float = 0.5
    gaussian_blur_prob: float = 0.5
    random_crop_scale: tuple = (0.08, 1.0)


@dataclass
class BackboneConfig:
    """Backbone architecture configuration."""
    name: str = "resnet50"
    pretrained_imagenet: bool = False  # Start from scratch for SSL
    feature_dim: int = 2048           # ResNet-50 output dim


@dataclass
class SimCLRConfig:
    """SimCLR-specific hyperparameters."""
    projection_dim: int = 128
    hidden_dim: int = 2048
    temperature: float = 0.07
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10


@dataclass
class BYOLConfig:
    """BYOL-specific hyperparameters."""
    projection_dim: int = 256
    hidden_dim: int = 4096
    prediction_dim: int = 256
    ema_decay: float = 0.996
    ema_decay_end: float = 1.0
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10


@dataclass
class FinetuneConfig:
    """Fine-tuning (downstream classification) configuration."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 10  # freeze backbone for initial epochs
    dropout: float = 0.3
    label_smoothing: float = 0.1
    # Loss weights: Dice + BCE combination
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    # Scheduler
    scheduler: str = "cosine"        # cosine or plateau
    scheduler_patience: int = 10
    min_lr: float = 1e-6
    # Early stopping
    early_stopping_patience: int = 20
    # Threshold for multi-label prediction
    threshold: float = 0.5


@dataclass
class TrainConfig:
    """Overall training configuration."""
    # SSL method: "simclr", "byol", or "ensemble"
    ssl_method: str = "simclr"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    # Distributed training
    distributed: bool = False
    # Data
    data: DataConfig = field(default_factory=DataConfig)
    # Backbone
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    # SSL methods
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    byol: BYOLConfig = field(default_factory=BYOLConfig)
    # Fine-tuning
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


def get_config(ssl_method: str = "simclr") -> TrainConfig:
    """Get default configuration for a given SSL method."""
    config = TrainConfig(ssl_method=ssl_method)
    return config
