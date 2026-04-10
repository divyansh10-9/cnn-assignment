"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VGG11UNet(nn.Module):

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        # Decoder

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = ConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(256 + 512, 256)  # FIXED

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 256, 128)  # FIXED

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 128, 64)    # FIXED

        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.dropout = CustomDropout(dropout_p)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):

        bottleneck, features = self.encoder(x, return_features=True)

        x = self.up4(bottleneck)
        x = torch.cat([x, features["block5"]], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, features["block4"]], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, features["block3"]], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, features["block2"]], dim=1)
        x = self.dec1(x)

        x = self.up0(x)

        x = self.dropout(x)
        logits = self.final(x)

        return logits