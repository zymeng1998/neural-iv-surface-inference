"""Tests for the Phase 4B sparsity-sweep harness (story 4B.2).

The harness produces, per regime, a ladder of progressively sparser *observed
contexts* while holding the *query/target set fixed*. These tests pin the
invariants the downstream diagnostic (4B.3-4B.5) relies on:

- fixed-query invariant (only the context shrinks, never the eval target set);
- monotone (nested) shrinkage along the intensity ladder;
- determinism under a fixed seed;
- dense rung (intensity 0) == full observed context;
- combined regime == intersection of fewer-quotes and thinned-wings;
- manifest is JSON-serializable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neural_iv_surface_inference.diagnostics.sparsity_sweep import (
    REGIMES,
    SparsitySweep,
    build_all_regimes,
    build_sweep,
    sweep_manifest,
)

LADDER = (0.0, 0.25, 0.5, 0.75)


def _frame(seed: int = 0) -> pd.DataFrame:
    """One date: a moneyness x maturity grid, ~70% marked observed (context)."""
    rng = np.random.default_rng(seed)
    moneyness = np.linspace(-0.4, 0.4, 9)
    taus = np.array([0.05, 0.15, 0.35, 0.6, 1.0])
    rows = []
    for exp_id, tau in enumerate(taus):
        for k in moneyness:
            rows.append((exp_id, float(tau), float(k)))
    df = pd.DataFrame(rows, columns=["expiration", "tau", "log_moneyness"])
    df["date"] = pd.Timestamp("2024-01-02")
    df["iv_true"] = 0.2 + 0.5 * df["log_moneyness"] ** 2 + 0.1 * df["tau"]
    # ~70% observed; the remainder is the fixed query/target set.
    df["observed"] = rng.random(len(df)) < 0.7
    return df


def test_regimes_constant():
    assert set(REGIMES) == {
        "fewer_quotes",
        "thin_wings",
        "missing_maturities",
        "combined_quotes_wings",
    }


@pytest.mark.parametrize("regime", REGIMES)
def test_fixed_query_invariant(regime):
    df = _frame()
    base_observed = df["observed"].to_numpy()
    query = ~base_observed
    sweep = build_sweep(df, regime, LADDER, seed=7)

    # The query mask is exactly the complement of the original observed set...
    np.testing.assert_array_equal(sweep.query_mask, query)
    for rung in sweep.rungs:
        # ...and no query row is ever recruited into the observed context...
        assert not np.any(rung.retained_observed & query)
        # ...and retained context is always a subset of the original observed.
        assert np.all(rung.retained_observed <= base_observed)


@pytest.mark.parametrize("regime", REGIMES)
def test_monotone_nested_shrinkage(regime):
    sweep = build_sweep(_frame(), regime, LADDER, seed=7)
    counts = [int(r.retained_observed.sum()) for r in sweep.rungs]
    # Non-increasing context size as intensity rises.
    assert counts == sorted(counts, reverse=True)
    # Strict nesting: each rung's context is a subset of the previous rung's.
    for prev, cur in zip(sweep.rungs, sweep.rungs[1:]):
        assert np.all(cur.retained_observed <= prev.retained_observed)


@pytest.mark.parametrize("regime", REGIMES)
def test_dense_rung_is_full_context(regime):
    df = _frame()
    base_observed = df["observed"].to_numpy()
    sweep = build_sweep(df, regime, LADDER, seed=7)
    # Intensity 0 (first rung) must reproduce the full observed context, so the
    # dense end of the ladder matches the 4A full-context comparison.
    assert sweep.rungs[0].intensity == 0.0
    np.testing.assert_array_equal(sweep.rungs[0].retained_observed, base_observed)


@pytest.mark.parametrize("regime", REGIMES)
def test_determinism_same_seed(regime):
    a = build_sweep(_frame(), regime, LADDER, seed=11)
    b = build_sweep(_frame(), regime, LADDER, seed=11)
    for ra, rb in zip(a.rungs, b.rungs):
        np.testing.assert_array_equal(ra.retained_observed, rb.retained_observed)


def test_thin_wings_is_seed_independent():
    # Thinning by |log-moneyness| quantile is deterministic regardless of seed.
    a = build_sweep(_frame(), "thin_wings", LADDER, seed=1)
    b = build_sweep(_frame(), "thin_wings", LADDER, seed=999)
    for ra, rb in zip(a.rungs, b.rungs):
        np.testing.assert_array_equal(ra.retained_observed, rb.retained_observed)


def test_fewer_quotes_seed_sensitive():
    a = build_sweep(_frame(), "fewer_quotes", LADDER, seed=1)
    b = build_sweep(_frame(), "fewer_quotes", LADDER, seed=2)
    # At a non-trivial sparsity rung the random retention differs across seeds.
    assert not np.array_equal(a.rungs[2].retained_observed, b.rungs[2].retained_observed)


def test_combined_is_intersection():
    df = _frame()
    seed = 5
    fq = build_sweep(df, "fewer_quotes", LADDER, seed=seed)
    tw = build_sweep(df, "thin_wings", LADDER, seed=seed)
    comb = build_sweep(df, "combined_quotes_wings", LADDER, seed=seed)
    for rc, rfq, rtw in zip(comb.rungs, fq.rungs, tw.rungs):
        np.testing.assert_array_equal(
            rc.retained_observed, rfq.retained_observed & rtw.retained_observed
        )


def test_missing_maturities_drops_whole_expirations():
    df = _frame()
    sweep = build_sweep(df, "missing_maturities", LADDER, seed=3)
    exp = df["expiration"].to_numpy()
    base_observed = df["observed"].to_numpy()
    base_maturities = set(np.unique(exp[base_observed]))
    # A sparser rung drops at least one whole observed expiration entirely.
    sparse_maturities = set(np.unique(exp[sweep.rungs[-1].retained_observed]))
    assert sparse_maturities < base_maturities
    # Whatever survives, survives as a *whole* expiration among observed rows:
    for rung in sweep.rungs:
        kept = set(np.unique(exp[rung.retained_observed]))
        for m in kept:
            obs_in_m = base_observed & (exp == m)
            np.testing.assert_array_equal(
                rung.retained_observed[obs_in_m], np.ones(int(obs_in_m.sum()), bool)
            )


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_intensity_validation(bad):
    with pytest.raises(ValueError):
        build_sweep(_frame(), "fewer_quotes", (0.0, bad), seed=0)


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        build_sweep(_frame(), "not_a_regime", LADDER, seed=0)


def test_build_all_regimes_and_manifest():
    df = _frame()
    ladders = {r: LADDER for r in REGIMES}
    sweeps = build_all_regimes(df, ladders, seed=7)
    assert set(sweeps) == set(REGIMES)
    assert all(isinstance(s, SparsitySweep) for s in sweeps.values())

    manifest = sweep_manifest(sweeps)
    # Round-trips through JSON (no numpy scalars leak in).
    text = json.dumps(manifest)
    assert "fewer_quotes" in text
    rec = manifest["regimes"]["fewer_quotes"]
    assert rec["n_query"] == int((~df["observed"].to_numpy()).sum())
    assert len(rec["rungs"]) == len(LADDER)
    # Counts in the manifest match the arrays.
    for spec_rung, obj_rung in zip(rec["rungs"], sweeps["fewer_quotes"].rungs):
        assert spec_rung["n_retained"] == int(obj_rung.retained_observed.sum())
