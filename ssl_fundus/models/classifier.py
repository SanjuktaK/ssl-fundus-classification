"""
Multi-label classification head for downstream fine-tuning.
"""

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier on top of a frozen/unfrozen SSL backbone.

    Architecture:
        backbone → GlobalAvgPool → FC(feature_dim, 512) → BN → ReLU → Dropout
                                  → FC(512, 256) → BN → ReLU → Dropout
                                  → FC(256, num_classes) → Sigmoid
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim

        if freeze_backbone:
            self._freeze_backbone()

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits (apply sigmoid externally for multi-label).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
