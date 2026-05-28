"""Attentive Neural Process (ANP) cross-attention decoder (3B.2).

Composes with :class:`SetEncoder` from
``conditional_surface``: the encoder must emit both the pooled latent ``z``
and the pre-pool per-element embeddings ``H``. This decoder cross-attends
each query into ``H`` (with the context-row mask applied), forms a query-
specific context vector ``c_q``, and routes ``[z, c_q, q]`` through the
same three head-kind dispatch (``gaussian`` / ``quantile`` / ``point``)
used by :class:`MultiOutputDecoder` so loss code is unchanged.

Reference: Kim et al., *Attentive Neural Processes*, ICLR 2019.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default quantile triplet matches MultiOutputDecoder / 2D.2.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.5, 0.95)

HeadKind = Literal["gaussian", "quantile", "point"]


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
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


class ANPDecoder(nn.Module):
    """Multi-head cross-attention decoder for ANP.

    Parameters
    ----------
    d_h : int
        Per-element embedding dim from the encoder (key/value source).
    d_z : int
        Global summary dim (``z``).
    d_query : int
        Coord-encoded query dim (raw ``(k, τ)`` → 2; Fourier → larger).
    n_heads : int
        Number of attention heads. ``d_attn`` must be divisible by ``n_heads``.
    d_head : int | None
        Per-head dim. If ``None`` defaults to ``d_h // n_heads``.
    mlp_hidden : int
        Width of the post-attention decoder MLP.
    head_kind : {"gaussian", "quantile", "point"}
        Output head selection (matches :class:`MultiOutputDecoder`).
    quantiles : tuple[float, ...]
        Quantile levels (ascending, strict). Used only for ``head_kind="quantile"``.
    include_z_in_decoder : bool
        If False, decoder MLP sees only ``[c_q, q]``.
    sigma_eps : float
        Floor added to softplus σ to keep NLL finite.
    """

    def __init__(
        self,
        *,
        d_h: int,
        d_z: int,
        d_query: int,
        n_heads: int = 4,
        d_head: int | None = None,
        mlp_hidden: int = 128,
        head_kind: HeadKind = "gaussian",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        include_z_in_decoder: bool = True,
        sigma_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if head_kind not in {"gaussian", "quantile", "point"}:
            raise ValueError(f"unsupported head_kind: {head_kind!r}")
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")

        d_per_head = int(d_head) if d_head is not None else max(1, d_h // n_heads)
        d_attn = d_per_head * n_heads

        self.d_h = int(d_h)
        self.d_z = int(d_z)
        self.d_query = int(d_query)
        self.d_attn = int(d_attn)
        self.n_heads = int(n_heads)
        self.head_kind = head_kind
        self.include_z_in_decoder = bool(include_z_in_decoder)
        self.sigma_eps = float(sigma_eps)

        # Projections into the attention space.
        self.q_proj = nn.Linear(d_query, d_attn)
        self.kv_proj = nn.Linear(d_h, 2 * d_attn)
        self.out_proj = nn.Linear(d_attn, d_h)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads,
            batch_first=True,
            bias=True,
        )

        # Decoder MLP: [z?, c_q, q] -> head logits.
        mlp_in = d_h + d_query + (d_z if include_z_in_decoder else 0)
        self.decoder_mlp = nn.Sequential(
            _mlp_block(mlp_in, mlp_hidden),
            _mlp_block(mlp_hidden, mlp_hidden),
        )

        if head_kind == "gaussian":
            self.head = nn.Linear(mlp_hidden, 2)
            self.register_buffer(
                "_quantiles",
                torch.tensor([], dtype=torch.float32),
                persistent=False,
            )
        elif head_kind == "quantile":
            qs = tuple(float(q) for q in quantiles)
            if not qs:
                raise ValueError("quantile head requires at least one level")
            if any(not (0.0 < q < 1.0) for q in qs):
                raise ValueError(f"quantiles must be strictly in (0,1); got {qs}")
            if list(qs) != sorted(qs):
                raise ValueError(
                    f"quantiles must be supplied in ascending order; got {qs}"
                )
            self.head = nn.Linear(mlp_hidden, len(qs))
            self.register_buffer(
                "_quantiles",
                torch.tensor(qs, dtype=torch.float32),
                persistent=False,
            )
        else:  # point
            self.head = nn.Linear(mlp_hidden, 1)
            self.register_buffer(
                "_quantiles",
                torch.tensor([], dtype=torch.float32),
                persistent=False,
            )

        _init_linear(self)

    @property
    def quantiles(self) -> tuple[float, ...]:
        return tuple(float(q) for q in self._quantiles.tolist())

    def forward(
        self,
        *,
        H: torch.Tensor,
        mask: torch.Tensor,
        z: torch.Tensor,
        q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        H    : (B, N_ctx, d_h)   per-element encoder embeddings.
        mask : (B, N_ctx) bool   True for real rows, False for padded.
        z    : (B, d_z)          pooled global summary.
        q    : (B, N_q, d_query) coord-encoded query points.

        Returns
        -------
        Dict with keys depending on ``head_kind`` (matches MultiOutputDecoder).
        """
        if H.dim() != 3:
            raise ValueError(f"H must be 3D (B, N_ctx, d_h); got {tuple(H.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must be 2D (B, N_ctx); got {tuple(mask.shape)}")
        if z.dim() != 2:
            raise ValueError(f"z must be 2D (B, d_z); got {tuple(z.shape)}")
        if q.dim() != 3:
            raise ValueError(f"q must be 3D (B, N_q, d_query); got {tuple(q.shape)}")
        if H.shape[0] != mask.shape[0] or H.shape[1] != mask.shape[1]:
            raise ValueError(
                f"H/mask shape mismatch: {tuple(H.shape)} vs {tuple(mask.shape)}"
            )

        n_q = q.shape[1]

        # Project queries and keys/values.
        q_proj = self.q_proj(q)                         # (B, N_q, d_attn)
        kv = self.kv_proj(H)                            # (B, N_ctx, 2 d_attn)
        k_proj, v_proj = kv.chunk(2, dim=-1)            # each (B, N_ctx, d_attn)

        # MultiheadAttention `key_padding_mask`: True = ignore.
        key_padding_mask = ~mask  # (B, N_ctx)

        # Guard against rows with zero real elements: nn.MultiheadAttention
        # produces NaNs when an entire key row is masked. Force at least one
        # un-masked position per batch element by un-masking position 0 and
        # zeroing its value vector — softmax then puts all weight on a
        # zero-value vector → c_q = 0, finite output.
        all_pad = key_padding_mask.all(dim=1)           # (B,)
        if bool(all_pad.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
            v_proj = v_proj.clone()
            v_proj[all_pad, 0, :] = 0.0

        c_attn, _ = self.attn(
            query=q_proj,
            key=k_proj,
            value=v_proj,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                               # (B, N_q, d_attn)
        c_q = self.out_proj(c_attn)                     # (B, N_q, d_h)

        # Decoder MLP input.
        parts: list[torch.Tensor] = []
        if self.include_z_in_decoder:
            parts.append(z.unsqueeze(1).expand(-1, n_q, -1))
        parts.append(c_q)
        parts.append(q)
        mlp_in = torch.cat(parts, dim=-1)
        trunk = self.decoder_mlp(mlp_in)
        raw = self.head(trunk)                          # (B, N_q, K)

        if self.head_kind == "gaussian":
            mu = F.softplus(raw[..., 0])
            sigma = F.softplus(raw[..., 1]) + self.sigma_eps
            log_sigma2 = 2.0 * torch.log(sigma)
            return {"mu": mu, "sigma": sigma, "log_sigma2": log_sigma2}

        if self.head_kind == "quantile":
            q_raw = F.softplus(raw)                     # (B, N_q, K)
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

        # point head
        mu = F.softplus(raw.squeeze(-1))
        return {"mu": mu}
