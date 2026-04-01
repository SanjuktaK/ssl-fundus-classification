"""
BYOL: Bootstrap Your Own Latent — A New Approach to Self-Supervised Learning
Grill et al., NeurIPS 2020.

Learns representations without negative pairs using an online-target
architecture with exponential moving average (EMA) updates.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbone import get_backbone


class MLPHead(nn.Module):
    """MLP head used for both projection and prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL model with online and target networks.

    Architecture:
        Online:  backbone → projector → predictor
        Target:  backbone → projector (EMA of online)

    The target network is NOT trained by gradient descent but updated
    via exponential moving average of the online network.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        prediction_dim: int = 256,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.ema_decay = ema_decay

        # Online network
        self.online_backbone = get_backbone(backbone_name, pretrained_imagenet=False)
        feature_dim = self.online_backbone.feature_dim

        self.online_projector = MLPHead(feature_dim, hidden_dim, projection_dim)
        self.predictor = MLPHead(projection_dim, hidden_dim, prediction_dim)

        # Target network (EMA copy — no gradients)
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Disable gradient computation for target
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None):
        """
        Update target network parameters via EMA.
        τ_target = decay * τ_target + (1 - decay) * τ_online
        """
        decay = decay if decay is not None else self.ema_decay

        for online_params, target_params in zip(
            self.online_backbone.parameters(), self.target_backbone.parameters()
        ):
            target_params.data = decay * target_params.data + (1 - decay) * online_params.data

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = decay * target_params.data + (1 - decay) * online_params.data

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass with two augmented views.

        Returns:
            loss: BYOL regression loss (mean of both directions)
        """
        # Online forward
        h1_online = self.online_backbone(x1)
        h2_online = self.online_backbone(x2)

        z1_online = self.online_projector(h1_online)
        z2_online = self.online_projector(h2_online)

        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # Target forward (no gradients)
        with torch.no_grad():
            h1_target = self.target_backbone(x1)
            h2_target = self.target_backbone(x2)

            z1_target = self.target_projector(h1_target)
            z2_target = self.target_projector(h2_target)

            # Stop gradient on target (detach)
            z1_target = z1_target.detach()
            z2_target = z2_target.detach()

        # Symmetrized loss: predict view2 from view1, and vice versa
        loss = (
            self._regression_loss(p1, z2_target) +
            self._regression_loss(p2, z1_target)
        ) / 2.0

        return loss

    @staticmethod
    def _regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        BYOL loss: negative cosine similarity.
        L = 2 - 2 * <p, z> / (||p|| * ||z||)
        """
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2.0 - 2.0 * (p * z).sum(dim=1).mean()

    def get_backbone(self) -> nn.Module:
        """Return online backbone for downstream fine-tuning."""
        return self.online_backbone

    @staticmethod
    def cosine_schedule_ema(
        base_decay: float, final_decay: float, current_step: int, total_steps: int
    ) -> float:
        """Cosine schedule for EMA decay: starts at base, ends at final."""
        import math
        return final_decay - (final_decay - base_decay) * (
            math.cos(math.pi * current_step / total_steps) + 1
        ) / 2
