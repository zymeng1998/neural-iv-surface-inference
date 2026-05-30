"""Tests for ``src/data/05_build_otm_surface.py``.

Synthetic-fixture tests for the OTM-restricted surface builder (Phase
3X, ADR 0006, D5 + D7). No real SPY data needed — fixtures cover:

* OTM puts and OTM calls (non-ATM convention)
* near-ATM tie-break on tighter relative spread
* exact-ATM (``log_m == 0``) fallback to ``put``
* economically-equivalent same-type duplicates → deterministic drop
* non-equivalent same-type duplicates → quality select
* manifest hash stability for a fixed input
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# The builder script lives at src/data/05_build_otm_surface.py — its
# filename starts with a digit, so it cannot be imported via the normal
# ``import`` machinery. Load it explicitly.
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
_BUILDER_PATH = REPO_ROOT / "src" / "data" / "05_build_otm_surface.py"
_spec = importlib.util.spec_from_file_location("otm_surface_builder", _BUILDER_PATH)
otm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = otm
_spec.loader.exec_module(otm)


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _row(
    *,
    date: str,
    expiration: str,
    strike: float,
    type_: str,
    log_m: float,
    tau: float,
    iv: float,
    bid: float = 1.0,
    ask: float = 1.1,
    mid: float | None = None,
    volume: float = 100.0,
    open_interest: float = 100.0,
) -> dict:
    if mid is None:
        mid = (bid + ask) / 2.0
    return dict(
        date=date,
        expiration=expiration,
        strike=strike,
        type=type_,
        log_moneyness=log_m,
        tau=tau,
        implied_volatility=iv,
        bid=bid,
        ask=ask,
        mid=mid,
        volume=volume,
        open_interest=open_interest,
    )


def _write_parquet(rows: list[dict], path: Path) -> Path:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    # Sort by date for the per-date streaming iterator
    df = df.sort_values(["date", "expiration", "strike", "type"]).reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    return path


@pytest.fixture
def fixture_basic(tmp_path: Path) -> Path:
    """Mixed fixture covering the OTM rule, ATM tie-break, residuals."""
    date = "2024-01-02"
    exp = "2024-02-16"
    tau = 0.12
    rows = [
        # ── OTM put side: log_m < -atm_band ───────────────────────────
        # Both legs present at (date,exp,K=350); we expect ONLY put kept.
        _row(date=date, expiration=exp, strike=350.0, type_="call",
             log_m=-0.10, tau=tau, iv=0.30),
        _row(date=date, expiration=exp, strike=350.0, type_="put",
             log_m=-0.10, tau=tau, iv=0.22),

        # ── OTM call side: log_m > +atm_band ──────────────────────────
        _row(date=date, expiration=exp, strike=450.0, type_="call",
             log_m=+0.10, tau=tau, iv=0.18),
        _row(date=date, expiration=exp, strike=450.0, type_="put",
             log_m=+0.10, tau=tau, iv=0.40),

        # ── Near-ATM tie-break on tighter spread ──────────────────────
        # call has tighter spread → call wins.
        _row(date=date, expiration=exp, strike=400.0, type_="call",
             log_m=0.001, tau=tau, iv=0.20, bid=1.00, ask=1.02),
        _row(date=date, expiration=exp, strike=400.0, type_="put",
             log_m=0.001, tau=tau, iv=0.25, bid=1.00, ask=1.20),

        # ── Exact-ATM (log_m == 0): tie on spread → fallback to put ──
        _row(date=date, expiration=exp, strike=399.0, type_="call",
             log_m=0.0, tau=tau, iv=0.21, bid=1.00, ask=1.05),
        _row(date=date, expiration=exp, strike=399.0, type_="put",
             log_m=0.0, tau=tau, iv=0.27, bid=1.00, ask=1.05),
    ]
    return _write_parquet(rows, tmp_path / "strict_fixture.parquet")


@pytest.fixture
def fixture_residual_equivalent(tmp_path: Path) -> Path:
    """Two OTM-side rows collide at the rounded (date,log_m,tau) key
    AND are economically equivalent → drop one deterministically."""
    date = "2024-01-02"
    exp = "2024-02-16"
    tau = 0.12
    # Two distinct strikes but the OTM convention keeps both as PUTS
    # because both have log_m < -atm_band. Crafted so log_m rounds to
    # the same 10-dp value → same-type residual collision.
    rows = [
        _row(date=date, expiration=exp, strike=350.0, type_="put",
             log_m=-0.10, tau=tau, iv=0.22,
             bid=1.0, ask=1.1, volume=100, open_interest=100),
        _row(date=date, expiration=exp, strike=350.5, type_="put",
             log_m=-0.10, tau=tau, iv=0.22,
             bid=1.0, ask=1.1, volume=100, open_interest=100),
    ]
    return _write_parquet(rows, tmp_path / "strict_residual_eq.parquet")


@pytest.fixture
def fixture_residual_quality(tmp_path: Path) -> Path:
    """Same-type residual collision with non-equivalent economics →
    quality-select winner (tighter spread; tie-break OI)."""
    date = "2024-01-02"
    exp = "2024-02-16"
    tau = 0.12
    rows = [
        # Looser spread, lower OI
        _row(date=date, expiration=exp, strike=350.0, type_="put",
             log_m=-0.10, tau=tau, iv=0.22,
             bid=1.0, ask=1.20, volume=10, open_interest=5),
        # Tighter spread, higher OI — should win
        _row(date=date, expiration=exp, strike=350.5, type_="put",
             log_m=-0.10, tau=tau, iv=0.24,
             bid=1.0, ask=1.05, volume=20, open_interest=500),
    ]
    return _write_parquet(rows, tmp_path / "strict_residual_q.parquet")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def _read_output(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def test_otm_rule_drops_wrong_side(fixture_basic, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    manifest = otm.build_otm(
        input_path=fixture_basic,
        output_path=out_path,
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    df = _read_output(out_path)

    # OTM put side: strike 350.0 → put kept, call dropped.
    sel = df[(df["strike"] == 350.0)]
    assert len(sel) == 1
    assert sel.iloc[0]["type"] == "put"

    # OTM call side: strike 450.0 → call kept, put dropped.
    sel = df[(df["strike"] == 450.0)]
    assert len(sel) == 1
    assert sel.iloc[0]["type"] == "call"

    # 2 wrong-side rows dropped (the call at 350 + the put at 450).
    assert manifest.dropped_wrong_type == 2


def test_atm_tiebreak_picks_tighter_spread(fixture_basic, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    otm.build_otm(
        input_path=fixture_basic,
        output_path=out_path,
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    df = _read_output(out_path)

    # K=400, log_m=0.001, call has spread 0.02 vs put 0.20 → call wins.
    sel = df[df["strike"] == 400.0]
    assert len(sel) == 1
    assert sel.iloc[0]["type"] == "call"


def test_exact_atm_falls_back_to_put(fixture_basic, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    otm.build_otm(
        input_path=fixture_basic,
        output_path=out_path,
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    df = _read_output(out_path)

    # K=399, log_m=0.0, equal spreads → declared fallback "put".
    sel = df[df["strike"] == 399.0]
    assert len(sel) == 1
    assert sel.iloc[0]["type"] == "put"


def test_output_single_valued_at_coord_key(fixture_basic, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    otm.build_otm(
        input_path=fixture_basic,
        output_path=out_path,
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    df = _read_output(out_path)

    df["_lm"] = df["log_moneyness"].round(10)
    df["_tau"] = df["tau"].round(10)
    dup = df.groupby(["date", "_lm", "_tau"]).size()
    assert (dup > 1).sum() == 0, f"single-valued invariant violated: {dup[dup > 1]}"


def test_atm_band_sensitivity_counts(fixture_basic, tmp_path: Path) -> None:
    manifest = otm.build_otm(
        input_path=fixture_basic,
        output_path=tmp_path / "out.parquet",
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    sens = manifest.atm_sensitivity
    # All four bands must be present.
    expected_keys = {
        "atm_rows_at_band_1e-12",
        "atm_rows_at_band_0.001",
        "atm_rows_at_band_0.0025",
        "atm_rows_at_band_0.005",
    }
    assert expected_keys.issubset(sens.keys())

    # band=1e-12 catches only the exact-ATM pair (2 rows at log_m=0).
    assert sens["atm_rows_at_band_1e-12"] == 2
    # band=0.001 catches exact-ATM (2) + the |log_m|=0.001 pair (2) = 4.
    assert sens["atm_rows_at_band_0.001"] == 4
    # band=0.0025 same as 0.001 in this fixture (4).
    assert sens["atm_rows_at_band_0.0025"] == 4


def test_residual_equivalent_drops_one(
    fixture_residual_equivalent, tmp_path: Path
) -> None:
    residual_csv = tmp_path / "residual.csv"
    manifest = otm.build_otm(
        input_path=fixture_residual_equivalent,
        output_path=tmp_path / "out.parquet",
        atm_band=0.0025,
        residual_csv=residual_csv,
        verbose=False,
    )
    df = _read_output(tmp_path / "out.parquet")
    # Two equivalent puts → exactly one kept.
    assert len(df) == 1
    assert manifest.residual_groups == 1
    assert manifest.residual_rows_in == 2
    assert manifest.residual_rows_dropped_equivalent == 1
    # Residual CSV must list both colliding rows for audit.
    assert residual_csv.exists()
    csv_df = pd.read_csv(residual_csv)
    assert len(csv_df) == 2


def test_residual_quality_picks_better(
    fixture_residual_quality, tmp_path: Path
) -> None:
    manifest = otm.build_otm(
        input_path=fixture_residual_quality,
        output_path=tmp_path / "out.parquet",
        atm_band=0.0025,
        residual_csv=tmp_path / "residual.csv",
        verbose=False,
    )
    df = _read_output(tmp_path / "out.parquet")
    assert len(df) == 1
    # Tighter-spread, higher-OI row was at strike 350.5.
    assert df.iloc[0]["strike"] == 350.5
    assert manifest.residual_rows_quality_picked == 1


def test_manifest_hash_stable(fixture_basic, tmp_path: Path) -> None:
    m1 = otm.build_otm(
        input_path=fixture_basic,
        output_path=tmp_path / "out1.parquet",
        atm_band=0.0025,
        residual_csv=tmp_path / "residual1.csv",
        verbose=False,
    )
    m2 = otm.build_otm(
        input_path=fixture_basic,
        output_path=tmp_path / "out2.parquet",
        atm_band=0.0025,
        residual_csv=tmp_path / "residual2.csv",
        verbose=False,
    )
    assert m1.input_sha256 == m2.input_sha256
    assert m1.output_sha256 == m2.output_sha256
    # Strict per-counter equality across runs (deterministic build).
    d1 = m1.to_dict()
    d2 = m2.to_dict()
    for k in d1:
        if k in ("output_path", "elapsed_seconds"):
            continue
        assert d1[k] == d2[k], f"non-deterministic field: {k}"


def test_dry_run_writes_no_parquet(fixture_basic, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    manifest = otm.build_otm(
        input_path=fixture_basic,
        output_path=out_path,
        atm_band=0.0025,
        residual_csv=None,
        dry_run=True,
        verbose=False,
    )
    assert not out_path.exists()
    assert manifest.rows_out > 0
    assert manifest.output_sha256 == ""
