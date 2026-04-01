# SSL for Fundus Image Classification

Self-supervised learning pipeline for multi-label ocular disease classification from retinal fundus images.

**Author**: Sanjukta Biswas

## Overview

This project uses **self-supervised learning** (SSL) to pretrain a ResNet-50 backbone on unlabeled fundus images, then fine-tunes it for multi-label disease classification. Two SSL methods are implemented and their predictions are fused via late ensemble.

**SSL Methods**: SimCLR, BYOL

**Target Diseases** (from ODIR dataset):

| Label | Disease |
|-------|---------|
| N | Normal (healthy fundus) |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| H | Hypertension |

**Pipeline**:
1. SSL pretraining (SimCLR or BYOL) on unlabeled fundus images
2. Fine-tuning with multi-label classifier head (BCE + Dice loss)
3. Ensemble fusion of SimCLR + BYOL predictions

## Project Structure

```
ssl_fundus/
├── __main__.py            # CLI entry point
├── config.py              # All hyperparameters
├── data/
│   ├── augmentations.py   # Fundus-aware augmentations
│   └── odir_dataset.py    # ODIR dataset loader
├── models/
│   ├── backbone.py        # ResNet-50 feature extractor
│   └── classifier.py      # Multi-label classification head
├── ssl_methods/
│   ├── simclr.py          # SimCLR (contrastive)
│   └── byol.py            # BYOL (self-distillation)
├── utils/
│   ├── metrics.py         # AUC, F1, Dice loss, etc.
│   ├── training.py        # Checkpointing, schedulers
│   └── visualization.py   # ROC curves, confusion matrices
├── train_ssl.py           # SSL pretraining script
├── train_finetune.py      # Downstream fine-tuning
└── train_ensemble.py      # Full ensemble pipeline
```

## Quick Start (Google Colab)

Open `SSL_Fundus_Pipeline.ipynb` in Colab for a self-contained demo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## CLI Usage

### 1. Set up data

Download the [ODIR-5K dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) and place it like this:

```
data/
└── ODIR/
    ├── ODIR-5K_Training_Annotations.csv   # (or .xlsx)
    └── Training Images/
        ├── 0_left.jpg
        ├── 0_right.jpg
        └── ...
```

The loader auto-detects annotation format (short codes N/D/G/H, full names, or diagnostic keywords).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. SSL Pretraining

```bash
# SimCLR
python -m ssl_fundus pretrain --method simclr --data_root ./data/ODIR --epochs 200

# BYOL
python -m ssl_fundus pretrain --method byol --data_root ./data/ODIR --epochs 200
```

Backbones are saved to `outputs/backbones/`.

### 4. Fine-tune

```bash
python -m ssl_fundus finetune \
    --backbone_path outputs/backbones/simclr_backbone.pt \
    --data_root ./data/ODIR \
    --method_name simclr
```

### 5. Full Ensemble Pipeline

Run everything end-to-end (pretrain both methods, fine-tune, ensemble):

```bash
python -m ssl_fundus ensemble --data_root ./data/ODIR
```

Skip pretraining if backbones already exist:

```bash
python -m ssl_fundus ensemble --data_root ./data/ODIR --skip_pretrain
```

### Custom Data Path

All commands accept `--data_root` to point to your dataset:

```bash
python -m ssl_fundus pretrain --method simclr --data_root /path/to/your/ODIR
python -m ssl_fundus ensemble --data_root /path/to/your/ODIR
```

## Key Configuration

Edit `ssl_fundus/config.py` or pass CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | SSL pretraining epochs |
| `--batch_size` | 256 | Batch size (reduce for less GPU memory) |
| `--lr` | 3e-4 | Learning rate |
| `--device` | cuda | Device (cuda or cpu) |
| `--output_dir` | ./outputs | Where to save checkpoints and results |

## Evaluation Metrics

The pipeline reports per-class and macro-averaged metrics: AUC-ROC, F1 score, precision, recall, mAP, exact match ratio. ROC curves and confusion matrices are saved as PNG files.

## Roadmap

This pipeline currently classifies fundus images using standard ODIR data. The next planned extension is an **automated fundus image quality analyzer** — a preprocessing module that performs QA checks on raw fundus images before they enter the SSL pipeline. This would include detection of poor illumination, blur, artifacts, and field-of-view issues, ensuring only clinically usable images contribute to training and inference. The goal is to build a fully end-to-end system: raw capture to quality-checked to SSL-pretrained to disease prediction.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, scikit-learn, pandas, matplotlib, Pillow
