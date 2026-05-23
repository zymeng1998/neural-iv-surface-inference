"""Set encoder + coordinate decoder for the W3 conditional surface model.

Two ``nn.Module``s:

  ``SetEncoder``         : ``O_t`` (set of observed points) -> ``z_t`` (latent)
  ``CoordinateDecoder``  : ``(k, tau, z_t)`` -> ``sigma_hat``

The set encoder is a DeepSets-style permutation-invariant network: a
per-element MLP applied to each observed point, then a **masked mean pool**
over the real elements, yielding a fixed-size latent ``z_t`` regardless of how
many quotes the date has. The decoder concatenates ``z_t`` with a query
coordinate ``(k, tau)`` and predicts IV through an MLP with a softplus output
(positive IV), mirroring the baseline-MLP head conventions.

The pooling op is exposed via a constructor argument so a future story can
swap masked mean for attention-based pooling without rewriting the encoder.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    """Linear -> LayerNorm -> SiLU, matching baseline-MLP head style."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.SiLU(),
    )


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _masked_mean(
    x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean of ``x`` over the last-but-one (set) axis, ignoring masked positions.

    Parameters
    ----------
    x    : (B, N, D) tensor
    mask : (B, N) bool tensor — True for real, False for padded

    Returns
    -------
    (B, D) tensor.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * m).sum(dim=1)
    counts = m.sum(dim=1).clamp_min(eps)
    return summed / counts


class SetEncoder(nn.Module):
    """Permutation-invariant set encoder (DeepSets-style).

    Per-element MLP -> masked pool -> optional post-pool MLP -> latent ``z_t``.

    Parameters
    ----------
    in_dim : int
        Per-element feature dim (e.g. 3 for ``(log_moneyness, tau, iv_input)``).
    hidden_dim : int
        Width of the per-element and post-pool MLPs.
    latent_dim : int
        Output dim of ``z_t``.
    n_elem_layers : int
        Number of per-element MLP blocks (>=1).
    n_post_layers : int
        Number of post-pool MLP blocks (>=0).
    pool : {"mean"}
        Pooling op. Only masked-mean is implemented; the argument exists so
        a future story can plug in attention-based pooling.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_elem_layers: int = 2,
        n_post_layers: int = 1,
        pool: Literal["mean"] = "mean",
    ) -> None:
        super().__init__()
        if n_elem_layers < 1:
            raise ValueError("n_elem_layers must be >= 1")
        if pool != "mean":
            raise ValueError(f"unsupported pool: {pool!r}")
        self.pool_name = pool

        elem_layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(n_elem_layers):
            elem_layers.append(_mlp_block(cur, hidden_dim))
            cur = hidden_dim
        self.elem_mlp = nn.Sequential(*elem_layers)

        post_layers: list[nn.Module] = []
        cur = hidden_dim
        for _ in range(n_post_layers):
            post_layers.append(_mlp_block(cur, hidden_dim))
            cur = hidden_dim
        post_layers.append(nn.Linear(cur, latent_dim))
        self.post_mlp = nn.Sequential(*post_layers)

        _init_linear(self)

    def forward(
        self, context: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context : (B, N, in_dim) float tensor — observed points, padded.
        mask    : (B, N) bool tensor — True for real rows, False for padded.

        Returns
        -------
        (B, latent_dim) tensor — ``z_t``.
        """
        h = self.elem_mlp(context)
        # Force padded outputs to zero so they cannot leak into the pool.
        h = h * mask.unsqueeze(-1).to(h.dtype)
        pooled = _masked_mean(h, mask)
        return self.post_mlp(pooled)


class CoordinateDecoder(nn.Module):
    """Concatenate ``z_t`` with each query coordinate and predict IV.

    Output is softplus-activated to guarantee positive IV.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = []
        cur = latent_dim + coord_dim
        for _ in range(n_layers):
            layers.append(_mlp_block(cur, hidden_dim))
            cur = hidden_dim
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)
        _init_linear(self)

    def forward(
        self, z: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z     : (B, latent_dim) float tensor.
        query : (B, N, coord_dim) float tensor — query coordinates.

        Returns
        -------
        (B, N) tensor — strictly positive predicted IV.
        """
        if z.dim() != 2:
            raise ValueError(f"z must be 2D (B, latent_dim); got {tuple(z.shape)}")
        if query.dim() != 3:
            raise ValueError(
                f"query must be 3D (B, N, coord_dim); got {tuple(query.shape)}"
            )
        n_query = query.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, n_query, -1)
        inp = torch.cat([z_exp, query], dim=-1)
        raw = self.net(inp).squeeze(-1)
        return nn.functional.softplus(raw)


class ConditionalSurfaceModel(nn.Module):
    """Thin wrapper composing ``SetEncoder`` + ``CoordinateDecoder``.

    Forward signature mirrors the batch contract from
    ``collate_conditional``: ``(context, context_mask, query) -> sigma_hat``.
    """

    def __init__(
        self,
        context_dim: int = 3,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_elem_layers: int = 2,
        n_post_layers: int = 1,
        n_decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = SetEncoder(
            in_dim=context_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_elem_layers=n_elem_layers,
            n_post_layers=n_post_layers,
        )
        self.decoder = CoordinateDecoder(
            latent_dim=latent_dim,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_layers=n_decoder_layers,
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(context, context_mask)
        return self.decoder(z, query)
