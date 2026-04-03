"""S3.2 — PyTorch masked MLP baseline for IV surface reconstruction.

A global MLP trained across all dates on observed points only.
Input: (log_moneyness, tau) coordinates.
Output: predicted implied volatility.

The "masked" aspect: during training, loss is computed only on
observed points.  During evaluation, predictions are made at all
points (observed and unobserved) and compared against iv_clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple MLP baseline for IV surface reconstruction.

    Architecture:
        input (2) → [Linear → LayerNorm → SiLU] × n_layers → Linear → output (1)

    Uses SiLU (swish) activation and LayerNorm for stable training.
    Output is softplus-activated to ensure positive IV predictions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 2)
            Columns: [log_moneyness, tau]

        Returns
        -------
        torch.Tensor, shape (batch, 1) — predicted IV (always positive).
        """
        raw = self.net(x)
        # Softplus ensures output > 0 (IV must be positive)
        return nn.functional.softplus(raw)
