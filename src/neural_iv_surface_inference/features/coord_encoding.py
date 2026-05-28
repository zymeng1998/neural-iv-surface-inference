"""Coordinate-encoding modules for the conditional surface decoder (3A.2).

Two ``nn.Module``s share a minimal interface:

  ``RawCoordEncoding``     : pass-through ``(B, N, coord_dim) -> (B, N, coord_dim)``.
  ``FourierCoordEncoding`` : sinusoidal Fourier features per Tancik et al.
                             (NeurIPS 2020), log-spaced frequency bands.

Each module exposes an integer ``encoded_dim`` property so a downstream
decoder can size its trunk-input layer without knowing which encoding is
in use. The frequency bands are registered as a non-persistent buffer so
``.to(device)`` moves them with the module while keeping checkpoints
backward-compatible.

A small builder, :func:`build_coord_encoding`, resolves a config dict to
the right module. Default Fourier hyperparameters are locked here:
``num_bands=8``, ``max_freq=10.0``, ``include_input=True``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


DEFAULT_NUM_BANDS: int = 8
DEFAULT_MAX_FREQ: float = 10.0
DEFAULT_INCLUDE_INPUT: bool = True


class RawCoordEncoding(nn.Module):
    """Identity encoding — returns the input coordinate tensor unchanged."""

    def __init__(self, coord_dim: int) -> None:
        super().__init__()
        if coord_dim < 1:
            raise ValueError(f"coord_dim must be >= 1; got {coord_dim}")
        self.coord_dim = int(coord_dim)

    @property
    def encoded_dim(self) -> int:
        return self.coord_dim

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError(
                f"query must be 3D (B, N, coord_dim); got {tuple(query.shape)}"
            )
        if query.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected last-dim {self.coord_dim}; got {query.shape[-1]}"
            )
        return query


class FourierCoordEncoding(nn.Module):
    """Sinusoidal Fourier-feature encoding with log-spaced bands.

    For each coordinate ``x_i`` and each frequency band ``f_b`` the encoding
    emits ``[sin(2π f_b x_i), cos(2π f_b x_i)]``. With ``include_input=True``
    the raw coordinates are concatenated as well. The frequency bands are
    log-spaced in ``[1, max_freq]`` (Tancik et al., NeurIPS 2020).

    Output dimensionality::

        encoded_dim = (coord_dim if include_input else 0) + 2 * coord_dim * num_bands
    """

    def __init__(
        self,
        coord_dim: int,
        num_bands: int = DEFAULT_NUM_BANDS,
        max_freq: float = DEFAULT_MAX_FREQ,
        include_input: bool = DEFAULT_INCLUDE_INPUT,
    ) -> None:
        super().__init__()
        if coord_dim < 1:
            raise ValueError(f"coord_dim must be >= 1; got {coord_dim}")
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1; got {num_bands}")
        if max_freq <= 0.0:
            raise ValueError(f"max_freq must be > 0; got {max_freq}")
        self.coord_dim = int(coord_dim)
        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)
        self.include_input = bool(include_input)

        # Log-spaced frequencies in [1, max_freq], inclusive on both ends.
        if num_bands == 1:
            freqs = torch.tensor([1.0], dtype=torch.float32)
        else:
            log_min = math.log(1.0)
            log_max = math.log(self.max_freq)
            freqs = torch.exp(
                torch.linspace(log_min, log_max, self.num_bands, dtype=torch.float32)
            )
        # Pre-multiply by 2π so forward stays a single matmul-free elementwise op.
        self.register_buffer(
            "_omegas", 2.0 * math.pi * freqs, persistent=False
        )

    @property
    def encoded_dim(self) -> int:
        base = self.coord_dim if self.include_input else 0
        return base + 2 * self.coord_dim * self.num_bands

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError(
                f"query must be 3D (B, N, coord_dim); got {tuple(query.shape)}"
            )
        if query.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected last-dim {self.coord_dim}; got {query.shape[-1]}"
            )
        # query: (B, N, C); _omegas: (K,)
        # scaled: (B, N, C, K) = query.unsqueeze(-1) * _omegas
        scaled = query.unsqueeze(-1) * self._omegas
        sin_feats = torch.sin(scaled)
        cos_feats = torch.cos(scaled)
        # Flatten the (C, K) axes into a single feature axis of size 2*C*K.
        b, n, c, k = scaled.shape
        sincos = torch.stack((sin_feats, cos_feats), dim=-1).reshape(b, n, c * k * 2)
        if self.include_input:
            return torch.cat((query, sincos), dim=-1)
        return sincos


def build_coord_encoding(cfg: dict[str, Any] | None, coord_dim: int) -> nn.Module:
    """Construct a coordinate encoding module from a config dict.

    Accepts:
      ``None`` or ``{"kind": "raw"}`` → :class:`RawCoordEncoding`.
      ``{"kind": "fourier", "num_bands": int, "max_freq": float,
         "include_input": bool}`` → :class:`FourierCoordEncoding`.

    Unknown ``kind`` raises ``ValueError``.
    """
    resolved = dict(cfg or {"kind": "raw"})
    kind = str(resolved.get("kind", "raw"))
    if kind == "raw":
        return RawCoordEncoding(coord_dim=coord_dim)
    if kind == "fourier":
        return FourierCoordEncoding(
            coord_dim=coord_dim,
            num_bands=int(resolved.get("num_bands", DEFAULT_NUM_BANDS)),
            max_freq=float(resolved.get("max_freq", DEFAULT_MAX_FREQ)),
            include_input=bool(resolved.get("include_input", DEFAULT_INCLUDE_INPUT)),
        )
    raise ValueError(
        f"unknown coord_encoding kind: {kind!r} (expected 'raw' or 'fourier')"
    )


__all__ = [
    "DEFAULT_NUM_BANDS",
    "DEFAULT_MAX_FREQ",
    "DEFAULT_INCLUDE_INPUT",
    "RawCoordEncoding",
    "FourierCoordEncoding",
    "build_coord_encoding",
]
