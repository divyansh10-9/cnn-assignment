"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth",
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        """

        super().__init__()

        import gdown

        # Download pretrained weights
        gdown.download(id="1lT0ZIbZg0wgnB9qZRQhovCARektOucsu", output=classifier_path, quiet=False)
        gdown.download(id="1BH_30lmZVuTTJRHqPuTi2-U5k69_QVlF", output=localizer_path, quiet=False)
        gdown.download(id="1PemA9Uy88wFPp9rrAzE1Pw1Xohc5We8N", output=unet_path, quiet=False)

        # Shared backbone
        self.encoder = VGG11(in_channels=in_channels)

        # Heads
        self.classifier = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels
        )

        self.localizer = VGG11Localizer(
            in_channels=in_channels
        )

        self.segmenter = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels
        )

        # Load weights
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location="cpu")
        )

        self.localizer.load_state_dict(
            torch.load(localizer_path, map_location="cpu")
        )

        self.segmenter.load_state_dict(
            torch.load(unet_path, map_location="cpu")
        )

        # Share encoder weights
        self.classifier.encoder = self.encoder
        self.localizer.encoder = self.encoder
        self.segmenter.encoder = self.encoder

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            dict:
            - classification
            - localization
            - segmentation
        """

        classification = self.classifier(x)
        localization = self.localizer(x)
        segmentation = self.segmenter(x)

        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }