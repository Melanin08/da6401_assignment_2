"""VGG11 backbone."""

import torch
import torch.nn as nn

from .layers import CustomDropout


class ConvBlock(nn.Module):
    """Single VGG-style conv block with optional BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0, use_batchnorm: bool = True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        if dropout_p > 0:
            layers.append(CustomDropout(dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG11(nn.Module):
    """VGG11 backbone with optional intermediate feature returns."""

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        super().__init__()

        self.block1 = ConvBlock(in_channels, 64, dropout_p=0.0, use_batchnorm=use_batchnorm)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = ConvBlock(64, 128, dropout_p=0.0, use_batchnorm=use_batchnorm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            ConvBlock(128, 256, dropout_p=0.0, use_batchnorm=use_batchnorm),
            ConvBlock(256, 256, dropout_p=0.1, use_batchnorm=use_batchnorm),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = nn.Sequential(
            ConvBlock(256, 512, dropout_p=0.0),
            ConvBlock(512, 512, dropout_p=0.1),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.block5 = nn.Sequential(
            ConvBlock(512, 512, dropout_p=0.0),
            ConvBlock(512, 512, dropout_p=0.2),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f1 = self.block1(x)
        x = self.pool1(f1)

        f2 = self.block2(x)
        x = self.pool2(f2)

        f3 = self.block3(x)
        x = self.pool3(f3)

        f4 = self.block4(x)
        x = self.pool4(f4)

        f5 = self.block5(x)
        bottleneck = self.pool5(f5)

        if return_features:
            return bottleneck, {
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "f5": f5,
            }

        return bottleneck
