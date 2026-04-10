"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Outputs bounding boxes in (x_center, y_center, width, height) format,
    normalized to [0, 1] relative to image dimensions so predictions are
    image-size agnostic at inference time.
    """

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
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Bounding boxes [B, 4] as (x_center, y_center, width, height),
            each value normalized to [0, 1] relative to image size.
        """

        features = self.encoder(x)
        bbox = self.localizer(features)

        # Sigmoid to constrain all four values to (0, 1).
        # This keeps coordinates normalized regardless of input resolution,
        # which matches how the Oxford-IIIT Pet bboxes should be represented.
        bbox = torch.sigmoid(bbox)

        return bbox
