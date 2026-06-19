"""Tests for the Phase 4A residual-target builder + `target_mode` loader flag.

Covers ADR 0010 (4A.2):
  - `rbf_predict_at_queries` reuses the 3X.6 RBF baseline verbatim.
  - `residual_target = iv_clean - rbf_pred`.
  - `add_rbf_pred_column` materialises both columns, finite.
  - loader `target_mode="absolute"` (default) is byte-for-byte unchanged.
  - loader `target_mode="residual"` yields `iv_clean - rbf_pred`, and the
    on-the-fly path agrees with a pre-materialised `rbf_pred` column.
  - invalid `target_mode` raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_iv_surface_inference.data import residual_targets as rt
from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
)
from neural_iv_surface_inference.models.interpolation import interpolate_surface_date


def _toy_df(n_dates: int = 2, seed: int = 0) -> pd.DataFrame:
    """A small valid benchmark frame: ≥3 observed rows per date so RBF fits."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        # 8 rows/date; first 5 observed (RBF needs ≥3).
        for i in range(8):
            k = float(rng.uniform(-0.4, 0.4))
            tau = float(rng.uniform(0.05, 1.0))
            iv_clean = 0.2 + 0.5 * k * k + 0.1 * tau
            iv_input = iv_clean + float(rng.normal(0, 0.01))
            rows.append({
                "date": date,
                "log_moneyness": k,
                "tau": tau,
                "implied_volatility": iv_input,
                "iv_clean": iv_clean,
                "observed": i < 5,
            })
    return pd.DataFrame(rows)


# ── residual_targets module ───────────────────────────────────────────────

def test_rbf_predict_matches_interpolation_baseline():
    df = _toy_df()
    date0 = df[df["date"] == df["date"].iloc[0]]
    got = rt.rbf_predict_at_queries(date0)
    expect = interpolate_surface_date(date0, method="rbf")
    np.testing.assert_array_equal(got, expect)


def test_residual_equals_iv_minus_rbf():
    df = _toy_df()
    date0 = df[df["date"] == df["date"].iloc[0]]
    r = rt.residual_targets_for_date(date0)
    expect = date0["iv_clean"].to_numpy(dtype=float) - rt.rbf_predict_at_queries(date0)
    np.testing.assert_array_equal(r, expect)


def test_add_rbf_pred_column_finite_and_consistent():
    out = rt.add_rbf_pred_column(_toy_df(n_dates=3))
    assert rt.RBF_PRED_COLUMN in out.columns
    assert rt.RESIDUAL_COLUMN in out.columns
    assert np.isfinite(out[rt.RBF_PRED_COLUMN]).all()
    np.testing.assert_allclose(
        out[rt.RESIDUAL_COLUMN].to_numpy(),
        out["iv_clean"].to_numpy() - out[rt.RBF_PRED_COLUMN].to_numpy(),
    )


# ── loader target_mode flag ───────────────────────────────────────────────

def _targets_concat(ds: ConditionalIVSurfaceDataset) -> np.ndarray:
    return np.concatenate([ds[i]["target"].numpy() for i in range(len(ds))])


def test_loader_absolute_mode_is_legacy_iv_clean():
    df = _toy_df()
    ds = ConditionalIVSurfaceDataset(df)  # default target_mode="absolute"
    assert ds.target_mode == "absolute"
    # Targets equal iv_clean (date-grouped, sorted) — the legacy behaviour.
    expect = (
        df.sort_values("date", kind="stable")["iv_clean"].to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(_targets_concat(ds), expect)


def test_loader_residual_mode_subtracts_rbf():
    df = _toy_df()
    ds = ConditionalIVSurfaceDataset(df, target_mode="residual")
    got = _targets_concat(ds)
    assert np.isfinite(got).all()
    # Reconstruct expected residual per date.
    parts = []
    for date in sorted(df["date"].unique()):
        g = df[df["date"] == date]
        parts.append(
            g["iv_clean"].to_numpy(dtype=np.float32)
            - rt.rbf_predict_at_queries(g).astype(np.float32)
        )
    np.testing.assert_allclose(got, np.concatenate(parts), rtol=0, atol=1e-6)


def test_loader_residual_precomputed_column_matches_onthefly():
    df = _toy_df()
    on_the_fly = _targets_concat(ConditionalIVSurfaceDataset(df, target_mode="residual"))
    pre = rt.add_rbf_pred_column(df)  # adds rbf_pred column
    precomputed = _targets_concat(
        ConditionalIVSurfaceDataset(pre, target_mode="residual")
    )
    np.testing.assert_allclose(on_the_fly, precomputed, rtol=0, atol=1e-6)


def test_loader_rejects_unknown_target_mode():
    with pytest.raises(ValueError, match="target_mode"):
        ConditionalIVSurfaceDataset(_toy_df(), target_mode="bogus")
