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


        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)  
        self.dec4 = ConvBlock(512 + 512, 512)                  

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)   
        self.dec3 = ConvBlock(256 + 512, 256)                 

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)   
        self.dec2 = ConvBlock(128 + 256, 128)                 

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)    
        self.dec1 = ConvBlock(64 + 128, 64)                     

        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)    

        self.dropout = CustomDropout(dropout_p)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input_h, input_w = x.shape[2], x.shape[3]

        bottleneck, features = self.encoder(x, return_features=True)

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

        if logits.shape[2] != input_h or logits.shape[3] != input_w:
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode='bilinear', align_corners=False
            )

        return logits
