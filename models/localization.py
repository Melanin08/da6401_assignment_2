"""Localization modules."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, use_batchnorm: bool = True):
        super().__init__()

        self.encoder = VGG11(in_channels, use_batchnorm=use_batchnorm)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regressor(x)

        x = torch.sigmoid(x)

        cx = x[:, 0] * 224.0
        cy = x[:, 1] * 224.0
        w  = x[:, 2] * 224.0
        h  = x[:, 3] * 224.0

        out = torch.stack([cx, cy, w, h], dim=1)
        return out