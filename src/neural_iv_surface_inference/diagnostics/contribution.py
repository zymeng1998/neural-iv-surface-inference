"""Per-dimension and per-PC ablation utilities for latent diagnostics (Story 2E.2).

Companion to :mod:`effective_rank`. Where the spectrum module describes how
the latent matrix ``Z`` is *distributed*, this module measures how each
direction is *used* by the downstream decoder — by editing ``Z`` and
re-running the decoder, then comparing the loss to baseline.

The API is decoder-agnostic: callers pass a ``loss_fn`` that maps a latent
batch ``[B, latent_dim]`` to per-sample losses ``[B]``. The runner script
binds this callable to (decoder, queries, targets, head_kind). Keeping the
ablation utilities independent of the model wrapper makes them unit-testable
against synthetic linear decoders with analytically predictable rankings.

Ablation conventions
--------------------
- **Mean substitution, not hard zero.** ``ablate_dim`` replaces the targeted
  dimension with its mean over ``Z``. Zeroing would drag the activation off
  the learned manifold and inflate the ΔNLL beyond the dim's actual
  information content; mean substitution removes the *signal* while
  preserving the *marginal*.
- **PC-basis edits are exact round-trips.** Loadings are orthonormal (right
  singular vectors), so projecting into PC coordinates, editing, and
  projecting back is information-preserving everywhere except at the edited
  coordinates.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Callable signature for the per-batch loss function.

Receives ``(Z_batch: Tensor[B, latent_dim], row_indices: Tensor[B] int64)``
and returns a per-sample loss tensor of shape ``[B]``. The row-indices
argument lets the caller look up the right per-row payload (e.g. query
coordinates and target IV) that pairs with each latent row. The production
runner extracts ``Z`` once over the whole loader and caches per-row payloads
in a parallel structure indexed by ``row_indices``.
"""

_LOG = logging.getLogger(__name__)


def _batched_mean_loss(loss_fn: LossFn, Z: torch.Tensor, batch_size: int) -> float:
    """Average ``loss_fn(Z[i:j], idx)`` over rows in fixed-size batches."""
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, latent_dim); got shape {tuple(Z.shape)}")
    n = Z.shape[0]
    if n == 0:
        raise ValueError("Z must be non-empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")

    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            indices = torch.arange(start, stop, dtype=torch.int64)
            losses = loss_fn(Z[start:stop], indices)
            if losses.ndim != 1 or losses.shape[0] != stop - start:
                raise ValueError(
                    "loss_fn must return a 1-D tensor with one entry per row; "
                    f"got shape {tuple(losses.shape)} for batch size {stop - start}"
                )
            total += float(losses.sum().item())
            seen += stop - start
    return total / seen


def baseline_loss(
    loss_fn: LossFn, Z: torch.Tensor, batch_size: int = 1024
) -> float:
    """Mean per-sample loss on the unmodified latent matrix."""
    return _batched_mean_loss(loss_fn, Z, batch_size)


def ablate_dim(
    loss_fn: LossFn,
    Z: torch.Tensor,
    dim: int,
    batch_size: int = 1024,
) -> float:
    """Replace ``Z[:, dim]`` with its column mean and return the mean loss.

    Mean substitution (not hard zero) — see module docstring for rationale.
    """
    if dim < 0 or dim >= Z.shape[1]:
        raise ValueError(f"dim {dim} out of range for latent_dim {Z.shape[1]}")
    Z_edit = Z.clone()
    Z_edit[:, dim] = Z[:, dim].mean()
    return _batched_mean_loss(loss_fn, Z_edit, batch_size)


def project_to_pc_basis(
    Z: torch.Tensor, pc_loadings: torch.Tensor, Z_mean: torch.Tensor
) -> torch.Tensor:
    """Project rows of ``Z`` onto the PC basis.

    Parameters
    ----------
    Z
        Latent matrix, ``[N, latent_dim]``.
    pc_loadings
        Right singular vectors as columns, ``[latent_dim, latent_dim]`` —
        i.e. the ``pc_loadings`` field of :class:`~.effective_rank.RankReport`
        wrapped into a torch tensor.
    Z_mean
        Per-column mean of the original ``Z``, ``[latent_dim]``.

    Returns
    -------
    torch.Tensor
        PC coordinates of each row, ``[N, latent_dim]``.
    """
    if pc_loadings.shape[0] != pc_loadings.shape[1]:
        raise ValueError(
            f"pc_loadings must be square; got shape {tuple(pc_loadings.shape)}"
        )
    if Z.shape[1] != pc_loadings.shape[0]:
        raise ValueError(
            f"latent_dim mismatch: Z has {Z.shape[1]}, pc_loadings has "
            f"{pc_loadings.shape[0]}"
        )
    return (Z - Z_mean) @ pc_loadings


def reconstruct_from_pc_basis(
    P: torch.Tensor, pc_loadings: torch.Tensor, Z_mean: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`project_to_pc_basis` (PC coordinates back to raw)."""
    return Z_mean + P @ pc_loadings.T


def ablate_pc(
    loss_fn: LossFn,
    Z: torch.Tensor,
    pc_idx: int,
    pc_loadings: torch.Tensor,
    Z_mean: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    """Zero a single PC coordinate, reconstruct, and return the mean loss.

    Replaces PC ``pc_idx`` with its mean (which is exactly 0 in the centred
    PC basis), reconstructs raw latents, evaluates.
    """
    d = Z.shape[1]
    if pc_idx < 0 or pc_idx >= d:
        raise ValueError(f"pc_idx {pc_idx} out of range for latent_dim {d}")
    P = project_to_pc_basis(Z, pc_loadings, Z_mean)
    P_edit = P.clone()
    P_edit[:, pc_idx] = 0.0
    Z_edit = reconstruct_from_pc_basis(P_edit, pc_loadings, Z_mean)
    return _batched_mean_loss(loss_fn, Z_edit, batch_size)


def topk_pc_reconstruction(
    loss_fn: LossFn,
    Z: torch.Tensor,
    k: int,
    pc_loadings: torch.Tensor,
    Z_mean: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    """Keep only the top-``k`` PCs of ``Z``, reconstruct, and return the loss.

    PCs are assumed to be ordered most-to-least-important in the columns of
    ``pc_loadings`` (the convention produced by
    :func:`~.effective_rank.analyze`). All but the leading ``k`` PC
    coordinates are set to 0 before reconstruction.
    """
    d = Z.shape[1]
    if k <= 0 or k > d:
        raise ValueError(f"k must be in [1, {d}]; got {k}")
    P = project_to_pc_basis(Z, pc_loadings, Z_mean)
    if k < d:
        P_edit = P.clone()
        P_edit[:, k:] = 0.0
    else:
        P_edit = P
    Z_edit = reconstruct_from_pc_basis(P_edit, pc_loadings, Z_mean)
    return _batched_mean_loss(loss_fn, Z_edit, batch_size)


__all__ = [
    "LossFn",
    "ablate_dim",
    "ablate_pc",
    "baseline_loss",
    "project_to_pc_basis",
    "reconstruct_from_pc_basis",
    "topk_pc_reconstruction",
]
