"""
Main entry point for the SSL Fundus Image Classification Pipeline.

Usage:
    # Pretrain with a single method
    python -m ssl_fundus pretrain --method simclr
    python -m ssl_fundus pretrain --method byol

    # Fine-tune with a pretrained backbone
    python -m ssl_fundus finetune --backbone_path outputs/backbones/simclr_backbone.pt

    # Run full ensemble pipeline (pretrain both + finetune both + fuse)
    python -m ssl_fundus ensemble --data_root ./data/ODIR

    # Skip pretraining if backbones already exist
    python -m ssl_fundus ensemble --skip_pretrain
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="SSL Fundus Image Classification Pipeline",
        usage="python -m ssl_fundus {pretrain,finetune,ensemble} [options]",
    )
    parser.add_argument("command", choices=["pretrain", "finetune", "ensemble"],
                        help="Pipeline stage to run")

    # Parse just the command
    args, remaining = parser.parse_known_args()

    if args.command == "pretrain":
        from .train_ssl import main as ssl_main
        sys.argv = [sys.argv[0]] + remaining
        ssl_main()

    elif args.command == "finetune":
        from .train_finetune import main as ft_main
        sys.argv = [sys.argv[0]] + remaining
        ft_main()

    elif args.command == "ensemble":
        from .train_ensemble import main as ens_main
        sys.argv = [sys.argv[0]] + remaining
        ens_main()


if __name__ == "__main__":
    main()
