"""Classification components."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """VGG11-based classifier."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels)

        # Classifier head follows the standard VGG design more closely.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x
