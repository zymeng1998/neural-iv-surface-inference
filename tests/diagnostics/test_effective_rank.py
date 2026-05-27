"""Tests for the latent-spectrum diagnostic (Story 2E.2).

Pure-numpy synthetic matrices — no model, no data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neural_iv_surface_inference.diagnostics.effective_rank import (
    RankReport,
    analyze,
)


pytestmark = pytest.mark.unit


def _rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260526)


def test_rank_one_matrix_collapses_to_one_dimension() -> None:
    rng = _rng()
    n, d = 200, 8
    u = rng.standard_normal(n)
    v = rng.standard_normal(d)
    Z = np.outer(u, v)

    report = analyze(Z)

    assert isinstance(report, RankReport)
    assert report.singular_values.shape == (d,)
    assert report.variance_ratio.shape == (d,)
    assert report.cumulative_variance.shape == (d,)
    assert report.pc_loadings.shape == (d, d)
    # One direction carries (very nearly) all the variance.
    assert report.eff_rank_entropy == pytest.approx(1.0, abs=1e-3)
    assert report.stable_rank == pytest.approx(1.0, abs=1e-3)
    assert report.k95 == 1
    assert report.k99 == 1
    assert report.variance_ratio[0] == pytest.approx(1.0, abs=1e-6)


def test_isotropic_gaussian_uses_full_rank() -> None:
    rng = _rng()
    n, d = 20_000, 8
    Z = rng.standard_normal((n, d))

    report = analyze(Z)

    # Entropy of a near-uniform distribution over d outcomes is ~d.
    assert 7.5 <= report.eff_rank_entropy <= 8.0
    assert 7.0 <= report.stable_rank <= 8.0
    assert report.k95 in (7, 8)
    assert report.k99 == 8
    assert report.dead_pcs == 0


def test_singular_values_and_cumulative_are_monotonic() -> None:
    rng = _rng()
    Z = rng.standard_normal((500, 12))

    report = analyze(Z)

    assert np.all(np.diff(report.singular_values) <= 1e-10)
    assert np.all(np.diff(report.cumulative_variance) >= -1e-12)
    assert report.cumulative_variance[-1] == pytest.approx(1.0, abs=1e-10)
    assert report.k95 <= report.k99


def test_dead_pcs_counts_zero_variance_columns() -> None:
    rng = _rng()
    n, d_live, d_dead = 500, 5, 3
    live = rng.standard_normal((n, d_live)) * np.array([3.0, 2.0, 1.0, 0.5, 0.25])
    dead = np.zeros((n, d_dead))
    Z = np.concatenate([live, dead], axis=1)

    report = analyze(Z)

    assert report.singular_values.shape == (d_live + d_dead,)
    # The last d_dead PCs carry zero variance.
    assert report.dead_pcs >= d_dead
    assert report.variance_ratio[-1] < 1e-6


def test_pc_loadings_recover_known_direction() -> None:
    # Inject a single dominant direction along the (1, 1, 0, 0) axis.
    rng = _rng()
    n, d = 1_000, 4
    direction = np.array([1.0, 1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    weights = rng.standard_normal(n) * 5.0
    noise = rng.standard_normal((n, d)) * 0.01
    Z = np.outer(weights, direction) + noise

    report = analyze(Z)

    top_pc = report.pc_loadings[:, 0]
    # Sign of an SVD vector is arbitrary; align before comparing.
    if np.dot(top_pc, direction) < 0:
        top_pc = -top_pc
    assert np.allclose(top_pc, direction, atol=1e-2)


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        analyze(np.zeros(10))
    with pytest.raises(ValueError):
        analyze(np.zeros((2, 2, 2)))


def test_rejects_non_finite_input() -> None:
    Z = np.ones((10, 3))
    Z[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyze(Z)
