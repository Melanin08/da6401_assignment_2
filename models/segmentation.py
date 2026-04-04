"""Segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11
from .layers import CustomDropout


class DoubleConv(nn.Module):
    """Two convolution layers used in each decoder block."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11(in_channels)

        # Full symmetric decoder
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(512 + 512, 512, dropout_p)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512, dropout_p)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256, dropout_p)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128, dropout_p)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64, dropout_p)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["f1"]
        f2 = features["f2"]
        f3 = features["f3"]
        f4 = features["f4"]
        f5 = features["f5"]

        x = self.up5(bottleneck)
        if x.shape[-2:] != f5.shape[-2:]:
            x = F.interpolate(x, size=f5.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f5], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        if x.shape[-2:] != f4.shape[-2:]:
            x = F.interpolate(x, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        if x.shape[-2:] != f3.shape[-2:]:
            x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        return x
