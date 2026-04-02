"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer (manual implementation)."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Dropout probability.
        """
        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout during training.

        Args:
            x: Input tensor (any shape)

        Returns:
            Tensor after dropout
        """
        if not self.training or self.p == 0:
            return x

        if self.p >= 1.0:
            return torch.zeros_like(x)

        # Create dropout mask
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout scaling
        return x * mask / (1 - self.p)