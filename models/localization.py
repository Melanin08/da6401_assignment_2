"""Localization modules"""

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 512),   # ✅ reduced
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 128),           # ✅ reduced
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(128, 4),             # ✅ correct output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regressor(x)

        xc = torch.sigmoid(x[:, 0]) * 224
        yc = torch.sigmoid(x[:, 1]) * 224

        w = torch.relu(x[:, 2])
        h = torch.relu(x[:, 3])

        x = torch.stack([xc, yc, w, h], dim=1)

        return x