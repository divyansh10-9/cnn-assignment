"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Outputs bounding boxes in (x_center, y_center, width, height) format
    in pixel space, scaled to the input image dimensions (e.g. 0-224).
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

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
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Bounding boxes [B, 4] as (cx, cy, w, h) in pixel space [0, image_size].
        """
        _, _, H, W = x.shape

        features = self.encoder(x)
        raw = self.localizer(features)

        bbox = torch.stack([
            torch.sigmoid(raw[:, 0]) * W,   # cx in [0, W]
            torch.sigmoid(raw[:, 1]) * H,   # cy in [0, H]
            torch.sigmoid(raw[:, 2]) * W,   # w  in [0, W]
            torch.sigmoid(raw[:, 3]) * H,   # h  in [0, H]
        ], dim=1)

        return bbox
