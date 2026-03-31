#!/usr/bin/env python3
"""
Step 3: Build the processed SPY surface-point table.

Loads raw options + underlying, joins, computes derived columns,
applies cleaning rules, produces two outputs:
  - spy_surface_points.parquet        (conservative clean)
  - spy_surface_points_strict.parquet (strict modeling subset)

Usage:
    python src/data/03_build_spy_surface_table.py
"""

import sys
import time

import numpy as np
import pandas as pd

from config import (
    RAW_OPTIONS_FILE, RAW_UNDERLYING_FILE,
    PROCESSED_DIR, SURFACE_POINTS_FILE, SURFACE_POINTS_STRICT_FILE,
    REQUIRED_OPTIONS_COLS, REQUIRED_UNDERLYING_COLS,
    OPTIONAL_OPTIONS_COLS, OPTIONAL_UNDERLYING_COLS,
    MIN_IV, MAX_IV, MIN_TAU, MAX_TAU,
    STRICT_MIN_IV, STRICT_MAX_IV, STRICT_MIN_TAU, STRICT_MAX_TAU,
    STRICT_MIN_LOG_MONEYNESS, STRICT_MAX_LOG_MONEYNESS,
    WIDE_SPREAD_THRESHOLD, SMALL_EPS,
    REPORTS_DIR,
)


def load_and_validate(path, required_cols: list[str], label: str) -> pd.DataFrame:
    """Load Parquet and assert required columns exist."""
    print(f"[load] {label}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}. Run step 01 first.")

    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")

    return df


def build_surface_points(opts: pd.DataFrame, und: pd.DataFrame) -> pd.DataFrame:
    """Join options with underlying, compute derived columns."""

    # ── Normalize date columns ───────────────────────────────────────
    print("[process] Normalizing dates...")
    opts["date"] = pd.to_datetime(opts["date"])
    opts["expiration"] = pd.to_datetime(opts["expiration"])
    und["date"] = pd.to_datetime(und["date"])

    # ── Join on date (SPY-only, so symbol join is implicit) ──────────
    # Keep only the columns we need from underlying
    und_cols = ["date", "close"]
    for col in OPTIONAL_UNDERLYING_COLS:
        if col in und.columns and col not in und_cols:
            und_cols.append(col)

    # Deduplicate underlying (shouldn't happen, but defensive)
    und_sub = und[und_cols].drop_duplicates(subset=["date"])

    print(f"[process] Joining options ({len(opts):,}) with "
          f"underlying ({len(und_sub):,}) on date...")
    n_before = len(opts)
    df = opts.merge(und_sub, on="date", how="inner", suffixes=("", "_und"))
    print(f"  Joined: {len(df):,} rows (dropped {n_before - len(df):,} "
          f"with no underlying match)")

    # ── Compute derived columns ──────────────────────────────────────
    print("[process] Computing derived columns...")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["days_to_expiry"] = (df["expiration"] - df["date"]).dt.days
    df["tau"] = df["days_to_expiry"] / 365.0
    df["spot"] = df["close"]
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])

    return df


def apply_hard_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are clearly invalid."""
    n0 = len(df)
    drop_reasons = {}

    # Null in critical fields
    critical = ["date", "expiration", "strike", "type", "implied_volatility", "close"]
    for col in critical:
        mask = df[col].isna()
        drop_reasons[f"null_{col}"] = mask.sum()
        df = df[~mask]

    # Negative bid/ask
    mask_neg_bid = df["bid"] < 0
    drop_reasons["bid < 0"] = mask_neg_bid.sum()
    df = df[~mask_neg_bid]

    mask_neg_ask = df["ask"] < 0
    drop_reasons["ask < 0"] = mask_neg_ask.sum()
    df = df[~mask_neg_ask]

    # Crossed quotes
    mask_crossed = df["ask"] < df["bid"]
    drop_reasons["ask < bid"] = mask_crossed.sum()
    df = df[~mask_crossed]

    # IV out of range
    mask_iv_low = df["implied_volatility"] <= MIN_IV
    drop_reasons[f"IV <= {MIN_IV}"] = mask_iv_low.sum()
    df = df[~mask_iv_low]

    mask_iv_high = df["implied_volatility"] > MAX_IV
    drop_reasons[f"IV > {MAX_IV}"] = mask_iv_high.sum()
    df = df[~mask_iv_high]

    # Expired or nonsensical tau
    mask_tau = df["tau"] <= MIN_TAU
    drop_reasons[f"tau <= {MIN_TAU}"] = mask_tau.sum()
    df = df[~mask_tau]

    mask_tau_far = df["tau"] > MAX_TAU
    drop_reasons[f"tau > {MAX_TAU}"] = mask_tau_far.sum()
    df = df[~mask_tau_far]

    n1 = len(df)
    print(f"[clean] Hard cleaning: {n0:,} → {n1:,} "
          f"(dropped {n0 - n1:,} = {(n0 - n1) / n0 * 100:.1f}%)")
    for reason, count in drop_reasons.items():
        if count > 0:
            print(f"  {reason}: {count:,}")

    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for questionable-but-not-fatal conditions."""
    df = df.copy()
    df["flag_zero_bid"] = df["bid"] == 0
    df["flag_zero_volume"] = df["volume"] == 0
    df["flag_zero_oi"] = df["open_interest"] == 0
    df["flag_wide_spread"] = (
        (df["ask"] - df["bid"]) / np.maximum(df["mid"], SMALL_EPS)
        > WIDE_SPREAD_THRESHOLD
    )
    return df


def build_strict_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a tighter subset for modeling."""
    mask = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["mid"] > 0)
        & (df["implied_volatility"] >= STRICT_MIN_IV)
        & (df["implied_volatility"] <= STRICT_MAX_IV)
        & (df["tau"] >= STRICT_MIN_TAU)
        & (df["tau"] <= STRICT_MAX_TAU)
        & (df["log_moneyness"] >= STRICT_MIN_LOG_MONEYNESS)
        & (df["log_moneyness"] <= STRICT_MAX_LOG_MONEYNESS)
    )
    strict = df[mask].copy()
    print(f"[strict] {len(df):,} → {len(strict):,} "
          f"(kept {len(strict) / len(df) * 100:.1f}%)")
    return strict


def write_build_report(df: pd.DataFrame, strict: pd.DataFrame) -> None:
    """Write a short build report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "spy_build_report.md"

    lines = [
        "# SPY Surface Build Report\n",
        "## Conservative Dataset\n",
        f"- Rows: {len(df):,}",
        f"- Date range: {df['date'].min().date()} → {df['date'].max().date()}",
        f"- Unique dates: {df['date'].nunique():,}",
        f"- Call rows: {(df['type'].str.lower() == 'call').sum():,}",
        f"- Put rows: {(df['type'].str.lower() == 'put').sum():,}",
        f"- IV range: [{df['implied_volatility'].min():.4f}, "
        f"{df['implied_volatility'].max():.4f}]",
        f"- Tau range: [{df['tau'].min():.4f}, {df['tau'].max():.4f}]",
        f"- Log-moneyness range: [{df['log_moneyness'].min():.4f}, "
        f"{df['log_moneyness'].max():.4f}]",
        "",
        "### Quality flag summary\n",
        f"- zero_bid: {df['flag_zero_bid'].sum():,} "
        f"({df['flag_zero_bid'].mean() * 100:.1f}%)",
        f"- zero_volume: {df['flag_zero_volume'].sum():,} "
        f"({df['flag_zero_volume'].mean() * 100:.1f}%)",
        f"- zero_oi: {df['flag_zero_oi'].sum():,} "
        f"({df['flag_zero_oi'].mean() * 100:.1f}%)",
        f"- wide_spread: {df['flag_wide_spread'].sum():,} "
        f"({df['flag_wide_spread'].mean() * 100:.1f}%)",
        "",
        "## Strict Modeling Subset\n",
        f"- Rows: {len(strict):,}",
        f"- IV range: [{STRICT_MIN_IV}, {STRICT_MAX_IV}]",
        f"- Tau range: [{STRICT_MIN_TAU:.4f}, {STRICT_MAX_TAU}]",
        f"- Log-moneyness range: [{STRICT_MIN_LOG_MONEYNESS}, "
        f"{STRICT_MAX_LOG_MONEYNESS}]",
    ]

    path.write_text("\n".join(lines))
    print(f"[report] Written to {path}")


def main():
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────
    opts = load_and_validate(
        RAW_OPTIONS_FILE, REQUIRED_OPTIONS_COLS, "SPY Options"
    )
    und = load_and_validate(
        RAW_UNDERLYING_FILE, REQUIRED_UNDERLYING_COLS, "SPY Underlying"
    )

    # ── Build ────────────────────────────────────────────────────────
    df = build_surface_points(opts, und)

    # ── Clean ────────────────────────────────────────────────────────
    df = apply_hard_cleaning(df)
    df = add_quality_flags(df)

    # ── Save conservative ────────────────────────────────────────────
    print(f"\n[save] {SURFACE_POINTS_FILE}")
    df.to_parquet(SURFACE_POINTS_FILE, index=False)
    print(f"  {len(df):,} rows, "
          f"{SURFACE_POINTS_FILE.stat().st_size / 1e6:.1f} MB")

    # ── Build and save strict subset ─────────────────────────────────
    strict = build_strict_subset(df)
    print(f"[save] {SURFACE_POINTS_STRICT_FILE}")
    strict.to_parquet(SURFACE_POINTS_STRICT_FILE, index=False)
    print(f"  {len(strict):,} rows, "
          f"{SURFACE_POINTS_STRICT_FILE.stat().st_size / 1e6:.1f} MB")

    # ── Report ───────────────────────────────────────────────────────
    write_build_report(df, strict)

    elapsed = time.time() - t0
    print(f"\n✓ Build complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Build failed: {e}", file=sys.stderr)
        raise
