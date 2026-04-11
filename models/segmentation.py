"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Bottleneck: 512ch, 7x7  (after pool5)
        # block5 skip: 512ch, 14x14 (before pool5)
        # block4 skip: 512ch, 28x28 (before pool4)
        # block3 skip: 256ch, 56x56 (before pool3)
        # block2 skip: 128ch, 112x112 (before pool2)
        # block1 skip: 64ch,  224x224 (before pool1)

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)   # 7->14
        self.dec4 = ConvBlock(512 + 512, 512)                   # cat block5(512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # 14->28
        self.dec3 = ConvBlock(256 + 512, 256)                   # cat block4(512)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)   # 28->56
        self.dec2 = ConvBlock(128 + 256, 128)                   # cat block3(256)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)    # 56->112
        self.dec1 = ConvBlock(64 + 128, 64)                     # cat block2(128)

        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)     # 112->224

        self.dropout = CustomDropout(dropout_p)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Remember input spatial size to guarantee output matches
        input_h, input_w = x.shape[2], x.shape[3]

        bottleneck, features = self.encoder(x, return_features=True)

        # Decode with skip connections
        d = self.up4(bottleneck)
        d = torch.cat([d, features["block5"]], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, features["block4"]], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, features["block3"]], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, features["block2"]], dim=1)
        d = self.dec1(d)

        d = self.up0(d)

        d = self.dropout(d)
        logits = self.final(d)

        # Guarantee output spatial size matches input
        if logits.shape[2] != input_h or logits.shape[3] != input_w:
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode='bilinear', align_corners=False
            )

        return logits
