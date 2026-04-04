"""Classification components."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(128, num_classes),  # ✅ FIXED
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x