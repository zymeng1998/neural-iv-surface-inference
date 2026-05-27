"""Tests for the per-dimension / per-PC ablation utilities (Story 2E.2).

Synthetic torch tensors + a tiny linear decoder with known weights so the
ranking and monotonicity properties of the ablation API can be predicted
analytically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neural_iv_surface_inference.diagnostics.contribution import (
    ablate_dim,
    ablate_pc,
    baseline_loss,
    project_to_pc_basis,
    reconstruct_from_pc_basis,
    topk_pc_reconstruction,
)
from neural_iv_surface_inference.diagnostics.effective_rank import analyze


pytestmark = pytest.mark.unit


def _rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260526)


def _build_linear_problem(
    n: int, d: int, weight_scale: np.ndarray, latent_std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Construct a synthetic regression problem.

    Returns
    -------
    Z, y, loss_fn
        ``Z`` is the latent matrix, ``y`` the target, and ``loss_fn`` maps
        a (possibly modified) latent batch to a per-sample squared-error.
    """
    rng = _rng()
    Z_np = rng.standard_normal((n, d)) * latent_std
    w_np = weight_scale
    y_np = Z_np @ w_np + 0.5
    Z = torch.tensor(Z_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)
    w = torch.tensor(w_np, dtype=torch.float64)

    def loss_fn(Zb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        pred = Zb @ w + 0.5
        return (pred - y[idx]) ** 2

    return Z, y, loss_fn


def test_baseline_loss_matches_direct_computation() -> None:
    Z, y, loss_fn = _build_linear_problem(
        n=500, d=4, weight_scale=np.array([1.0, 0.5, -2.0, 0.0]),
        latent_std=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    baseline = baseline_loss(loss_fn, Z, batch_size=128)
    direct = float(loss_fn(Z, torch.arange(Z.shape[0])).mean().item())
    assert baseline == pytest.approx(direct, rel=1e-10)


def test_ablate_dim_ranking_matches_w_times_std() -> None:
    # Predicted importance of dim i = |w_i| * std(Z[:, i]).
    d = 8
    weights = np.array([3.0, 0.1, -2.5, 1.0, 0.0, 0.7, -1.5, 0.3])
    stds = np.array([1.0, 1.5, 0.5, 2.0, 1.0, 1.0, 0.8, 1.2])
    Z, _, loss_fn = _build_linear_problem(
        n=2_000, d=d, weight_scale=weights, latent_std=stds,
    )
    baseline = baseline_loss(loss_fn, Z)

    deltas = np.array([
        ablate_dim(loss_fn, Z, dim=i) - baseline for i in range(d)
    ])
    predicted = np.abs(weights) * stds
    rho, _ = spearmanr(deltas, predicted)
    assert rho >= 0.95


def test_ablate_dim_on_zero_weight_has_zero_effect() -> None:
    d = 5
    weights = np.array([1.0, 0.0, -1.0, 0.0, 2.0])
    Z, _, loss_fn = _build_linear_problem(
        n=500, d=d, weight_scale=weights, latent_std=np.ones(d),
    )
    baseline = baseline_loss(loss_fn, Z)

    for dim_with_zero_w in (1, 3):
        delta = ablate_dim(loss_fn, Z, dim=dim_with_zero_w) - baseline
        assert abs(delta) < 1e-10


def test_project_and_reconstruct_round_trip() -> None:
    rng = _rng()
    n, d = 300, 6
    Z = torch.tensor(rng.standard_normal((n, d)) * np.arange(1, d + 1))
    report = analyze(Z.numpy())
    loadings = torch.tensor(report.pc_loadings)
    Z_mean = Z.mean(dim=0)

    P = project_to_pc_basis(Z, loadings, Z_mean)
    Z_back = reconstruct_from_pc_basis(P, loadings, Z_mean)
    assert torch.allclose(Z, Z_back, atol=1e-9)


def test_topk_pc_reconstruction_equals_baseline_when_k_equals_d() -> None:
    d = 6
    weights = np.array([2.0, 1.0, -1.5, 0.5, -0.7, 1.2])
    Z, _, loss_fn = _build_linear_problem(
        n=500, d=d, weight_scale=weights,
        latent_std=np.array([2.0, 1.5, 1.0, 0.8, 0.5, 0.3]),
    )
    baseline = baseline_loss(loss_fn, Z)
    report = analyze(Z.numpy())
    loadings = torch.tensor(report.pc_loadings)
    Z_mean = Z.mean(dim=0)

    full = topk_pc_reconstruction(loss_fn, Z, k=d, pc_loadings=loadings, Z_mean=Z_mean)
    assert full == pytest.approx(baseline, rel=1e-8, abs=1e-10)


def test_topk_pc_reconstruction_is_non_decreasing_loss_as_k_drops() -> None:
    d = 6
    weights = np.array([2.0, 1.0, -1.5, 0.5, -0.7, 1.2])
    Z, _, loss_fn = _build_linear_problem(
        n=1_000, d=d, weight_scale=weights,
        latent_std=np.array([3.0, 2.0, 1.0, 0.5, 0.3, 0.1]),
    )
    report = analyze(Z.numpy())
    loadings = torch.tensor(report.pc_loadings)
    Z_mean = Z.mean(dim=0)

    losses = [
        topk_pc_reconstruction(loss_fn, Z, k=k, pc_loadings=loadings, Z_mean=Z_mean)
        for k in range(1, d + 1)
    ]
    # As k increases, reconstruction is better, so loss should not increase.
    assert all(losses[i] >= losses[i + 1] - 1e-9 for i in range(d - 1))


def test_ablate_pc_zero_eigen_direction_has_no_effect() -> None:
    # Direction with effectively no variance carries no signal; removing it
    # cannot change the loss.
    rng = _rng()
    n, d_live, d_dead = 500, 3, 2
    weights = np.array([1.0, -1.0, 2.0, 0.5, 0.5])
    live = rng.standard_normal((n, d_live)) * np.array([2.0, 1.0, 0.5])
    dead = np.zeros((n, d_dead))
    Z_np = np.concatenate([live, dead], axis=1)
    Z = torch.tensor(Z_np, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    y = Z @ w + 0.5

    def loss_fn(Zb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return ((Zb @ w + 0.5) - y[idx]) ** 2

    report = analyze(Z.numpy())
    loadings = torch.tensor(report.pc_loadings)
    Z_mean = Z.mean(dim=0)
    baseline = baseline_loss(loss_fn, Z)

    # The bottom PCs carry no variance; ablating them must not change loss.
    bottom_pc = d_live + d_dead - 1
    delta = ablate_pc(
        loss_fn, Z, pc_idx=bottom_pc, pc_loadings=loadings, Z_mean=Z_mean,
    ) - baseline
    assert abs(delta) < 1e-9
