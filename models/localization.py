"""Localization modules."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        # Shared VGG11 backbone
        self.encoder = VGG11(in_channels)

        # Regression head for [x_center, y_center, width, height]
        # in resized-image pixel space (224x224)
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
        """
        Return bounding boxes [B, 4] in
        (x_center, y_center, width, height) format.
        Output is in 224x224 image pixel space.
        """
        x = self.encoder(x)
        x = self.regressor(x)

        xc = torch.sigmoid(x[:, 0]) * 224.0
        yc = torch.sigmoid(x[:, 1]) * 224.0
        w = torch.relu(x[:, 2])
        h = torch.relu(x[:, 3])

        x = torch.stack([xc, yc, w, h], dim=1)
        return x
