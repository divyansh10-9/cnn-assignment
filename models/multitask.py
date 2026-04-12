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
        super().__init__()

        import gdown
        import os

        os.makedirs("checkpoints", exist_ok=True)

        
        gdown.download(id="1tMymLIc119clBxKj80h6eGI48p9beX14",
                       output=classifier_path, quiet=False)
        gdown.download(id="1RQnCkwjPoTEaPHONG68rq2iJe3eEznRt",
                       output=localizer_path, quiet=False)
        gdown.download(id="1Cp6Ci-J07vv-HjIMWh0RUO6lC3L30wGO",
                       output=unet_path, quiet=False)


        self.classifier = VGG11Classifier(
            num_classes=num_breeds, in_channels=in_channels
        )
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location="cpu")
        )

        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.localizer.load_state_dict(
            torch.load(localizer_path, map_location="cpu")
        )

        self.segmenter = VGG11UNet(
            num_classes=seg_classes, in_channels=in_channels
        )
        self.segmenter.load_state_dict(
            torch.load(unet_path, map_location="cpu")
        )

        
        self.encoder = self.classifier.encoder
        self.localizer.encoder = self.encoder
        self.segmenter.encoder = self.encoder

    def forward(self, x: torch.Tensor):
        """Single forward pass producing all three task outputs.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            dict with keys:
              - "classification": logits [B, num_breeds]
              - "localization":   bbox   [B, 4]  (cx, cy, w, h) in [0,1]
              - "segmentation":   logits [B, seg_classes, H, W]
        """
        return {
            "classification": self.classifier(x),
            "localization":   self.localizer(x),
            "segmentation":   self.segmenter(x),
        }