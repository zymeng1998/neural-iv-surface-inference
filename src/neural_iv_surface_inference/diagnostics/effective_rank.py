"""Latent-spectrum diagnostic (Story 2E.2).

Pure-numpy SVD spectrum of a stacked latent matrix ``Z`` of shape
``[N, latent_dim]``. Produces the summary scalars and arrays that frame the
"how many directions does ``Z`` actually span?" question — effective rank
(entropy form over the variance distribution), stable rank (Frobenius² /
spectral²), k95 / k99, per-PC variance share, cumulative variance, dead-PC
count, and the full PC loading matrix (right singular vectors in the raw-dim
basis).

The causal "how much does the decoder actually use each direction?" question
is the job of the companion :mod:`contribution` module; this file only
describes how ``Z`` is *distributed*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

_LOG = logging.getLogger(__name__)

_DEAD_PC_VARIANCE_THRESHOLD: Final[float] = 1e-4


@dataclass(frozen=True)
class RankReport:
    """Summary of the SVD spectrum of a centred latent matrix.

    Attributes
    ----------
    singular_values
        Singular values of the mean-centred ``Z``, length ``latent_dim``,
        sorted descending.
    variance_ratio
        Per-PC variance share — ``sigma_i**2 / sum_j sigma_j**2`` —
        non-increasing.
    cumulative_variance
        Cumulative sum of ``variance_ratio``; ``cumulative_variance[-1] == 1``.
    eff_rank_entropy
        ``exp(H(p))`` where ``p_i = sigma_i**2 / sum_j sigma_j**2`` and ``H``
        is the Shannon entropy in nats. Equals 1 for a rank-1 ``Z`` and
        approaches ``latent_dim`` for an isotropic Gaussian ``Z``.
    stable_rank
        ``sum_i sigma_i**2 / sigma_max**2`` — the standard "stable rank"; a
        scale-invariant lower bound on effective rank that is insensitive
        to a fat tail of tiny singular values.
    k95, k99
        Smallest ``k`` such that the top-``k`` PCs explain ≥ 95% / 99% of
        the total variance.
    dead_pcs
        Count of PCs with ``variance_ratio < 1e-4``.
    pc_loadings
        Right singular vectors of the centred ``Z``, shape
        ``[latent_dim, latent_dim]``. Column ``k`` is PC ``k`` expressed in
        the original raw-dim basis; sign is arbitrary (SVD convention).
    """

    singular_values: NDArray[np.float64]
    variance_ratio: NDArray[np.float64]
    cumulative_variance: NDArray[np.float64]
    eff_rank_entropy: float
    stable_rank: float
    k95: int
    k99: int
    dead_pcs: int
    pc_loadings: NDArray[np.float64]


def analyze(Z: NDArray[np.floating]) -> RankReport:
    """Compute the SVD-based :class:`RankReport` of a latent matrix.

    Parameters
    ----------
    Z
        Latent matrix, shape ``[N, latent_dim]``. Must be 2-D and finite.

    Returns
    -------
    RankReport
        Frozen summary, see class docstring for fields.

    Raises
    ------
    ValueError
        If ``Z`` is not 2-D, not finite, or has zero rows.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, latent_dim); got shape {Z.shape!r}")
    n, d = Z.shape
    if n == 0 or d == 0:
        raise ValueError(f"Z must be non-empty; got shape {Z.shape!r}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains non-finite values (NaN or inf)")

    Z64 = np.asarray(Z, dtype=np.float64)
    Zc = Z64 - Z64.mean(axis=0, keepdims=True)

    # full_matrices=False -> singular_values has length min(N, d); we pad to
    # latent_dim with zeros so callers always see a length-d spectrum.
    _u, sigma, vt = np.linalg.svd(Zc, full_matrices=False)
    if sigma.shape[0] < d:
        pad = np.zeros(d - sigma.shape[0], dtype=sigma.dtype)
        sigma = np.concatenate([sigma, pad])
        vt_pad = np.zeros((d - vt.shape[0], d), dtype=vt.dtype)
        vt = np.concatenate([vt, vt_pad], axis=0)

    variances = sigma**2
    total = float(variances.sum())
    if total <= 0.0:
        _LOG.warning("Z has zero total variance; returning a degenerate report")
        variance_ratio = np.zeros(d, dtype=np.float64)
        variance_ratio[0] = 1.0
        cumulative = np.cumsum(variance_ratio)
        return RankReport(
            singular_values=sigma.astype(np.float64),
            variance_ratio=variance_ratio,
            cumulative_variance=cumulative,
            eff_rank_entropy=1.0,
            stable_rank=1.0,
            k95=1,
            k99=1,
            dead_pcs=int(d - 1),
            pc_loadings=vt.T.astype(np.float64),
        )

    variance_ratio = variances / total
    cumulative = np.cumsum(variance_ratio)

    # Entropy in nats over the variance distribution; ignore zero-weight terms.
    nonzero = variance_ratio[variance_ratio > 0.0]
    entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))
    eff_rank_entropy = float(np.exp(entropy_nats))

    sigma_max_sq = float(variances[0])
    stable_rank = float(total / sigma_max_sq) if sigma_max_sq > 0.0 else 0.0

    # k95 / k99: smallest k (1-indexed) hitting the threshold. searchsorted on
    # a non-decreasing cumulative array gives the right answer for ties.
    k95 = int(np.searchsorted(cumulative, 0.95) + 1)
    k99 = int(np.searchsorted(cumulative, 0.99) + 1)
    k95 = min(k95, d)
    k99 = min(k99, d)

    dead_pcs = int(np.sum(variance_ratio < _DEAD_PC_VARIANCE_THRESHOLD))

    pc_loadings = vt.T.astype(np.float64)

    return RankReport(
        singular_values=sigma.astype(np.float64),
        variance_ratio=variance_ratio.astype(np.float64),
        cumulative_variance=cumulative.astype(np.float64),
        eff_rank_entropy=eff_rank_entropy,
        stable_rank=stable_rank,
        k95=k95,
        k99=k99,
        dead_pcs=dead_pcs,
        pc_loadings=pc_loadings,
    )


__all__ = ["RankReport", "analyze"]
