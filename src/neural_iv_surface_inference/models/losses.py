"""Loss functions for IV surface reconstruction.

All losses operate on the full batch but can be masked to compute loss
only on observed or unobserved subsets.

The W4 uncertainty heads (2D.2) introduce two additional masked losses:

* :func:`gaussian_nll_loss` for a heteroscedastic Gaussian head emitting
  ``(mu, sigma)``.
* :func:`pinball_loss` for a quantile head emitting one prediction per
  configured quantile level.

Both are written to share the same ``(B, Q)`` masking convention used by the
conditional training loop (``masked_query_mse``), so they can be dropped into
``train_conditional`` behind a head-kind dispatch without touching collation.
"""

from __future__ import annotations

from collections.abc import Sequence

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


# ── W4 heteroscedastic / quantile heads (2D.2) ────────────────────────────


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked Gaussian negative log-likelihood.

    Computes ``0.5 * (log(2*pi*sigma^2) + (target - mu)^2 / sigma^2)``,
    averaged over the True positions of ``mask``.

    Parameters
    ----------
    mu, sigma, target : (B, Q) float tensors
        ``sigma`` must already be strictly positive (the head applies softplus).
    mask : (B, Q) bool tensor
        True for real query positions; False for padded slots.
    eps : float
        Floor used when ``sigma`` numerically drops below ``eps``.
    """
    if mu.shape != sigma.shape or mu.shape != target.shape:
        raise ValueError(
            f"gaussian_nll_loss: shape mismatch — mu={tuple(mu.shape)} "
            f"sigma={tuple(sigma.shape)} target={tuple(target.shape)}"
        )
    sigma = sigma.clamp_min(eps)
    log_sigma2 = 2.0 * torch.log(sigma)
    per_point = 0.5 * (log_sigma2 + ((target - mu) ** 2) / (sigma ** 2))
    m = mask.to(per_point.dtype)
    denom = m.sum().clamp_min(1.0)
    return (per_point * m).sum() / denom


def pinball_loss(
    quantile_preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quantiles: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Masked pinball (quantile) loss.

    Parameters
    ----------
    quantile_preds : (B, Q, K) float tensor
        Predictions for ``K`` quantile levels.
    target : (B, Q) float tensor
        Ground-truth IV.
    mask : (B, Q) bool tensor
        True for real query positions.
    quantiles : sequence of K floats in ``(0, 1)`` or 1-D tensor
        Quantile levels matching the last dim of ``quantile_preds``.

    Returns
    -------
    Scalar tensor — mean over (masked points, K quantile levels).
    """
    if quantile_preds.dim() != 3:
        raise ValueError(
            f"pinball_loss: quantile_preds must be 3-D (B, Q, K); "
            f"got {tuple(quantile_preds.shape)}"
        )
    if isinstance(quantiles, torch.Tensor):
        q_t = quantiles.to(quantile_preds.device, dtype=quantile_preds.dtype)
    else:
        q_t = torch.tensor(
            list(quantiles), device=quantile_preds.device, dtype=quantile_preds.dtype
        )
    if q_t.numel() != quantile_preds.shape[-1]:
        raise ValueError(
            f"pinball_loss: got {q_t.numel()} quantile levels but "
            f"predictions have last-dim {quantile_preds.shape[-1]}"
        )
    diff = target.unsqueeze(-1) - quantile_preds  # (B, Q, K)
    q_view = q_t.view(1, 1, -1)
    per_point = torch.maximum(q_view * diff, (q_view - 1.0) * diff)
    m = mask.to(per_point.dtype).unsqueeze(-1)  # (B, Q, 1)
    denom = m.sum().clamp_min(1.0) * q_t.numel()
    return (per_point * m).sum() / denom
