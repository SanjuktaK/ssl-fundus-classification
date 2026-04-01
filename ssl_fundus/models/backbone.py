"""
Backbone feature extractors for SSL pretraining and fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_backbone(name: str = "resnet50", pretrained_imagenet: bool = False) -> nn.Module:
    """
    Get backbone network, returning the encoder without the final FC layer.

    Returns:
        nn.Module with .feature_dim attribute indicating output dimensionality.
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained_imagenet else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.feature_dim = feature_dim
        return model

    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained_imagenet else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.feature_dim = feature_dim
        return model

    elif name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained_imagenet else None
        model = models.efficientnet_b3(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        model.feature_dim = feature_dim
        return model

    else:
        raise ValueError(f"Unsupported backbone: {name}. Choose from: resnet50, resnet18, efficientnet_b3")
