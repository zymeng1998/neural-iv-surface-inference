"""Loss functions for IV surface reconstruction.

All losses operate on the full batch but can be masked to compute loss
only on observed or unobserved subsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """MSE loss computed only on masked (observed) points."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : (batch, 1) — predicted IV
        target : (batch, 1) — ground truth IV
        mask : (batch,) bool — True = include in loss
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        diff = (pred[mask] - target[mask]) ** 2
        return diff.mean()


class MaskedMAELoss(nn.Module):
    """MAE (L1) loss computed only on masked (observed) points."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        diff = (pred[mask] - target[mask]).abs()
        return diff.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of MSE and MAE on observed points.

    loss = alpha * MSE + (1 - alpha) * MAE
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = MaskedMSELoss()
        self.mae = MaskedMAELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.alpha * self.mse(pred, target, mask)
            + (1 - self.alpha) * self.mae(pred, target, mask)
        )
