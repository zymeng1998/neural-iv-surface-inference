"""Tests for the ``paired_coord`` flag on apply_mask (3X.3 / ADR 0006 D4).

Two contracts:

1. ``paired_coord=False`` (default) must reproduce the legacy
   per-row behaviour byte-for-byte under a fixed seed.
2. ``paired_coord=True`` must produce zero cross-coordinate leakage:
   for every rounded ``(date, log_m, tau)`` key, all rows share the
   same ``observed`` flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.data.masking import apply_mask  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _paired_df(n_keys: int = 40, legs_per_key: int = 2) -> pd.DataFrame:
    """Synthetic surface with `legs_per_key` rows at each rounded coord.

    Mimics the duplicate-coordinate structure of the dirty surface (one
    call leg + one put leg sharing ``(date, log_m, tau)`` after the
    audit's 10-dp rounding).
    """
    rng = np.random.default_rng(0)
    log_m = rng.uniform(-0.3, 0.3, size=n_keys)
    tau = rng.uniform(0.02, 1.0, size=n_keys)
    dates = pd.to_datetime(["2024-01-02"] * n_keys)

    rows = []
    for d, lm, t in zip(dates, log_m, tau):
        for leg in range(legs_per_key):
            rows.append(dict(
                date=d, log_moneyness=lm, tau=t, leg=leg,
            ))
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        ("random", {"keep_frac": 0.4}),
        ("realistic", {"keep_frac": 0.4}),
        ("drop_short_maturity", {}),
        ("drop_wings", {}),
    ],
)
def test_default_off_is_byte_identical(strategy: str, kwargs: dict) -> None:
    df = _paired_df()
    legacy = apply_mask(df, strategy=strategy, seed=123, **kwargs)
    explicit_off = apply_mask(
        df, strategy=strategy, seed=123, paired_coord=False, **kwargs
    )
    assert legacy["observed"].equals(explicit_off["observed"]), strategy


@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        ("random", {"keep_frac": 0.4}),
        ("realistic", {"keep_frac": 0.4}),
    ],
)
def test_paired_on_yields_zero_cross_key_leakage(
    strategy: str, kwargs: dict
) -> None:
    df = _paired_df()
    out = apply_mask(
        df, strategy=strategy, seed=7, paired_coord=True, **kwargs
    )

    # Every rounded (date, log_m, tau) key must have homogeneous observed.
    key = pd.DataFrame({
        "date": out["date"].to_numpy(),
        "lm_r": np.round(out["log_moneyness"].to_numpy(dtype=float), 10),
        "tau_r": np.round(out["tau"].to_numpy(dtype=float), 10),
        "observed": out["observed"].to_numpy(),
    })
    per_key_unique = key.groupby(["date", "lm_r", "tau_r"])["observed"].nunique()
    assert (per_key_unique == 1).all(), strategy


def test_paired_off_on_unique_coords_matches_paired_on() -> None:
    # When no row shares a coordinate, paired_coord is a no-op (every
    # group has exactly one row).
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 50),
        "log_moneyness": rng.uniform(-0.3, 0.3, size=50),
        "tau": rng.uniform(0.02, 1.0, size=50),
    })
    off = apply_mask(df, strategy="random", keep_frac=0.5, seed=4)
    on = apply_mask(
        df, strategy="random", keep_frac=0.5, seed=4, paired_coord=True
    )
    assert off["observed"].equals(on["observed"])


def test_paired_on_no_date_column_still_works() -> None:
    # The masking API does not require a date column. With paired_coord,
    # we should fall back to grouping on (log_m, tau) only.
    df = _paired_df().drop(columns=["date"])
    out = apply_mask(
        df, strategy="random", keep_frac=0.4, seed=11, paired_coord=True
    )
    key = pd.DataFrame({
        "lm_r": np.round(out["log_moneyness"].to_numpy(dtype=float), 10),
        "tau_r": np.round(out["tau"].to_numpy(dtype=float), 10),
        "observed": out["observed"].to_numpy(),
    })
    per_key_unique = key.groupby(["lm_r", "tau_r"])["observed"].nunique()
    assert (per_key_unique == 1).all()
