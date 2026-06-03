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

from typing import Any, Literal

import torch
import torch.nn as nn

from neural_iv_surface_inference.data.conditional_loaders import (
    DEFAULT_FEATURE_SET,
    feature_set_context_dim,
)
from neural_iv_surface_inference.features.coord_encoding import (
    build_coord_encoding,
)
from neural_iv_surface_inference.models.anp_decoder import ANPDecoder

# Default quantile triplet used by the W4 quantile head (2D.2).
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.5, 0.95)


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
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_elements: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        context : (B, N, in_dim) float tensor — observed points, padded.
        mask    : (B, N) bool tensor — True for real rows, False for padded.
        return_elements : bool
            If True, also return the pre-pool per-element embeddings ``H``
            (shape ``(B, N, hidden_dim)``) for downstream cross-attention.
            Padded rows are already zeroed in ``H``.

        Returns
        -------
        (B, latent_dim) ``z_t`` — or, if ``return_elements`` is True,
        a tuple ``(z_t, H)`` where ``H`` has shape ``(B, N, hidden_dim)``.
        """
        h = self.elem_mlp(context)
        # Force padded outputs to zero so they cannot leak into the pool.
        h = h * mask.unsqueeze(-1).to(h.dtype)
        pooled = _masked_mean(h, mask)
        z = self.post_mlp(pooled)
        if return_elements:
            return z, h
        return z


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


class MultiOutputDecoder(nn.Module):
    """Coordinate decoder with a configurable W4 head (2D.2).

    Shares the trunk MLP shape used by :class:`CoordinateDecoder` and replaces
    the final ``Linear(hidden, 1)`` with a head-specific output layer.

    Supported heads
    ---------------
    ``gaussian``
        Two output channels: ``mu`` (softplus → positive IV) and a raw scalar
        passed through softplus to give a strictly positive ``sigma``. The
        derived ``log_sigma2 = 2 * log(sigma)`` is also returned for downstream
        logging / NLL computation that prefers the log-variance form.
    ``quantile``
        ``K`` output channels (one per configured quantile level), each passed
        through softplus to guarantee positive IV. During ``eval`` the
        ``quantiles`` tensor is sorted along the level axis to enforce
        monotonicity at inference; during training the raw (un-sorted) tensor
        is returned so the pinball loss receives the per-level prediction it
        was paired with at construction.

    The point head is implemented by :class:`CoordinateDecoder` directly, which
    :class:`ConditionalSurfaceModel` keeps untouched so ``head.kind: point``
    runs are bit-for-bit equivalent to the 2C baseline.
    """

    def __init__(
        self,
        latent_dim: int,
        coord_dim: int,
        hidden_dim: int,
        n_layers: int,
        head_kind: Literal["gaussian", "quantile"],
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        sigma_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if head_kind not in {"gaussian", "quantile"}:
            raise ValueError(f"unsupported head_kind: {head_kind!r}")
        self.head_kind = head_kind
        self.sigma_eps = float(sigma_eps)

        trunk: list[nn.Module] = []
        cur = latent_dim + coord_dim
        for _ in range(n_layers):
            trunk.append(_mlp_block(cur, hidden_dim))
            cur = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        if head_kind == "gaussian":
            self.head = nn.Linear(cur, 2)
            self.register_buffer(
                "_quantiles", torch.tensor([], dtype=torch.float32), persistent=False
            )
        else:  # quantile
            qs = tuple(float(q) for q in quantiles)
            if not qs:
                raise ValueError("quantile head requires at least one level")
            if any(not (0.0 < q < 1.0) for q in qs):
                raise ValueError(f"quantiles must be strictly in (0,1); got {qs}")
            if list(qs) != sorted(qs):
                raise ValueError(
                    f"quantiles must be supplied in ascending order; got {qs}"
                )
            self.head = nn.Linear(cur, len(qs))
            self.register_buffer(
                "_quantiles",
                torch.tensor(qs, dtype=torch.float32),
                persistent=False,
            )

        _init_linear(self)

    @property
    def quantiles(self) -> tuple[float, ...]:
        return tuple(float(q) for q in self._quantiles.tolist())

    def forward(
        self, z: torch.Tensor, query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(f"z must be 2D (B, latent_dim); got {tuple(z.shape)}")
        if query.dim() != 3:
            raise ValueError(
                f"query must be 3D (B, N, coord_dim); got {tuple(query.shape)}"
            )
        n_query = query.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, n_query, -1)
        h = self.trunk(torch.cat([z_exp, query], dim=-1))
        raw = self.head(h)  # (B, N, out_dim)

        if self.head_kind == "gaussian":
            mu = nn.functional.softplus(raw[..., 0])
            sigma = nn.functional.softplus(raw[..., 1]) + self.sigma_eps
            log_sigma2 = 2.0 * torch.log(sigma)
            return {"mu": mu, "sigma": sigma, "log_sigma2": log_sigma2}

        # quantile head
        q_raw = nn.functional.softplus(raw)  # (B, N, K), positive IVs
        if not self.training:
            q_out, _ = torch.sort(q_raw, dim=-1)
        else:
            q_out = q_raw
        k_mid = q_out.shape[-1] // 2
        return {
            "mu": q_out[..., k_mid],
            "quantiles": q_out,
            "quantile_levels": self._quantiles,
        }


class ConditionalSurfaceModel(nn.Module):
    """Thin wrapper composing ``SetEncoder`` + a head-specific decoder.

    With ``head["kind"] == "point"`` (the default and the 2C baseline) the
    decoder is exactly :class:`CoordinateDecoder` and ``forward`` returns a
    dict ``{"mu": Tensor}``. For ``gaussian`` and ``quantile`` heads (2D.2) the
    decoder is replaced with :class:`MultiOutputDecoder` and ``forward``
    returns the head-specific keys (``sigma`` / ``log_sigma2`` for Gaussian,
    ``quantiles`` / ``quantile_levels`` for quantile) alongside ``mu``.

    The ``head`` config is also persisted on the module (``self.head_cfg``) so
    downstream adapters and checkpoint loaders can rebuild it from a stored
    config blob.
    """

    def __init__(
        self,
        context_dim: int | None = None,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_elem_layers: int = 2,
        n_post_layers: int = 1,
        n_decoder_layers: int = 3,
        head: dict[str, Any] | None = None,
        coord_encoding: dict[str, Any] | None = None,
        decoder_kind: Literal["deepsets", "anp"] = "deepsets",
        anp: dict[str, Any] | None = None,
        feature_set: str = DEFAULT_FEATURE_SET,
    ) -> None:
        super().__init__()
        # ADR 0008: ``feature_set`` is authoritative for the per-quote context
        # dimensionality. ``context_dim`` is retained for backward compat; if
        # passed it must agree with ``feature_set`` (catches mis-config where
        # someone flips to micro_v1 but leaves a stale context_dim=3).
        self.feature_set = str(feature_set)
        resolved_context_dim = feature_set_context_dim(self.feature_set)
        if context_dim is not None and int(context_dim) != resolved_context_dim:
            raise ValueError(
                f"context_dim={context_dim} conflicts with "
                f"feature_set={self.feature_set!r} (expected {resolved_context_dim})"
            )
        context_dim = resolved_context_dim
        self.head_cfg: dict[str, Any] = dict(head or {"kind": "point"})
        kind = str(self.head_cfg.get("kind", "point"))
        if kind not in {"point", "gaussian", "quantile"}:
            raise ValueError(
                f"unsupported head.kind: {kind!r} (expected point|gaussian|quantile)"
            )
        self.head_kind = kind

        if decoder_kind not in {"deepsets", "anp"}:
            raise ValueError(
                f"unsupported decoder_kind: {decoder_kind!r} (expected deepsets|anp)"
            )
        self.decoder_kind = decoder_kind
        self.anp_cfg: dict[str, Any] = dict(anp or {})

        self.coord_encoding_cfg: dict[str, Any] = dict(
            coord_encoding or {"kind": "raw"}
        )
        self.coord_encoding = build_coord_encoding(
            self.coord_encoding_cfg, coord_dim=coord_dim
        )
        decoder_coord_dim = int(self.coord_encoding.encoded_dim)

        self.encoder = SetEncoder(
            in_dim=context_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_elem_layers=n_elem_layers,
            n_post_layers=n_post_layers,
        )

        if decoder_kind == "anp":
            quantiles_anp = tuple(
                self.head_cfg.get("quantiles", DEFAULT_QUANTILES)
            ) if kind == "quantile" else DEFAULT_QUANTILES
            n_heads = int(self.anp_cfg.get("n_heads", 4))
            d_head = self.anp_cfg.get("d_head", None)
            mlp_hidden = int(self.anp_cfg.get("mlp_hidden", max(128, hidden_dim)))
            include_z = bool(self.anp_cfg.get("include_z_in_decoder", True))
            self.decoder = ANPDecoder(
                d_h=hidden_dim,
                d_z=latent_dim,
                d_query=decoder_coord_dim,
                n_heads=n_heads,
                d_head=int(d_head) if d_head is not None else None,
                mlp_hidden=mlp_hidden,
                head_kind=kind,
                quantiles=quantiles_anp,
                include_z_in_decoder=include_z,
            )
            if kind == "quantile":
                self.head_cfg["quantiles"] = list(quantiles_anp)
        elif kind == "point":
            self.decoder = CoordinateDecoder(
                latent_dim=latent_dim,
                coord_dim=decoder_coord_dim,
                hidden_dim=hidden_dim,
                n_layers=n_decoder_layers,
            )
        else:
            quantiles = tuple(
                self.head_cfg.get("quantiles", DEFAULT_QUANTILES)
            ) if kind == "quantile" else DEFAULT_QUANTILES
            self.decoder = MultiOutputDecoder(
                latent_dim=latent_dim,
                coord_dim=decoder_coord_dim,
                hidden_dim=hidden_dim,
                n_layers=n_decoder_layers,
                head_kind=kind,
                quantiles=quantiles,
            )
            # Normalise the persisted config so checkpoints carry the resolved
            # quantile levels even when the YAML supplied the defaults.
            if kind == "quantile":
                self.head_cfg["quantiles"] = list(quantiles)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_query = self.coord_encoding(query)
        if self.decoder_kind == "anp":
            z, H = self.encoder(context, context_mask, return_elements=True)
            return self.decoder(H=H, mask=context_mask, z=z, q=encoded_query)
        z = self.encoder(context, context_mask)
        if self.head_kind == "point":
            mu = self.decoder(z, encoded_query)
            return {"mu": mu}
        return self.decoder(z, encoded_query)
