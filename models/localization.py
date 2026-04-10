"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.
        """
        super().__init__()

        # Encoder
        self.encoder = VGG11(in_channels=in_channels)

        # Localization head
        self.localizer = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _, _, H, W = x.shape

        features = self.encoder(x)

        bbox = self.localizer(features)

        # safer version (no inplace ops)
        bbox = torch.stack([
            torch.sigmoid(bbox[:, 0]) * W,
            torch.sigmoid(bbox[:, 1]) * H,
            torch.sigmoid(bbox[:, 2]) * W,
            torch.sigmoid(bbox[:, 3]) * H
        ], dim=1)

        return bbox