"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Chen et al., ICML 2020.

Learns representations by maximizing agreement between differently augmented
views of the same image via a contrastive loss (NT-Xent).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbone import get_backbone


class ProjectionHead(nn.Module):
    """MLP projection head: maps features to contrastive embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR model.

    Components:
        - Backbone encoder f(·)
        - Projection head g(·) for contrastive learning
        - NT-Xent loss

    During fine-tuning, discard the projection head and use backbone features.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Encoder
        self.backbone = get_backbone(backbone_name, pretrained_imagenet=False)
        feature_dim = self.backbone.feature_dim

        # Projection head
        self.projector = ProjectionHead(feature_dim, hidden_dim, projection_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass with two augmented views.

        Args:
            x1, x2: Two augmented views of shape (B, C, H, W)

        Returns:
            loss: NT-Xent contrastive loss
            z1, z2: Projected representations (for monitoring)
        """
        # Encode both views
        h1 = self.backbone(x1)  # (B, feature_dim)
        h2 = self.backbone(x2)  # (B, feature_dim)

        # Project
        z1 = self.projector(h1)  # (B, projection_dim)
        z2 = self.projector(h2)  # (B, projection_dim)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute NT-Xent loss
        loss = self.nt_xent_loss(z1, z2)

        return loss, z1, z2

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss.

        For each positive pair (z1_i, z2_i), all other 2(N-1) samples
        in the batch serve as negatives.
        """
        B = z1.shape[0]
        device = z1.device

        # Concatenate representations: [z1_0, ..., z1_N, z2_0, ..., z2_N]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_indices = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ])

        # Extract positive similarities
        pos_sim = sim[torch.arange(2 * B, device=device), pos_indices]

        # NT-Xent: -log(exp(pos) / sum(exp(all)))
        loss = -pos_sim + torch.logsumexp(sim, dim=1)

        return loss.mean()

    def get_backbone(self) -> nn.Module:
        """Return backbone for downstream fine-tuning."""
        return self.backbone
