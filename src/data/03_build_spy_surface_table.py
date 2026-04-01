#!/usr/bin/env python3
"""
Step 3: Build the processed SPY surface-point table.

Refactored to use year-by-year partitioned processing to avoid OOM.
See: docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md

Produces:
  data_processed/spy/partitions/spy_surface_YYYY.parquet  (per-year)
  data_processed/spy/spy_surface_points.parquet            (conservative)
  data_processed/spy/spy_surface_points_strict.parquet     (strict subset)

Usage:
    python src/data/03_build_spy_surface_table.py              # full run
    python src/data/03_build_spy_surface_table.py --test-year  # one year only
"""

import argparse
import gc
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

PARTITIONS_DIR = PROCESSED_DIR / "partitions"
TEST_YEAR = 2024  # default single-year test


# ── Helpers ──────────────────────────────────────────────────────────

def load_underlying() -> pd.DataFrame:
    """Load and prepare the small underlying table (fits in memory easily)."""
    print(f"[load] Underlying: {RAW_UNDERLYING_FILE}")
    if not RAW_UNDERLYING_FILE.exists():
        raise FileNotFoundError(
            f"Underlying not found at {RAW_UNDERLYING_FILE}. Run step 01 first."
        )

    und = pd.read_parquet(RAW_UNDERLYING_FILE)
    print(f"  {len(und):,} rows")

    missing = [c for c in REQUIRED_UNDERLYING_COLS if c not in und.columns]
    if missing:
        raise ValueError(f"Underlying missing required columns: {missing}")

    und["date"] = pd.to_datetime(und["date"])

    # Select columns we need
    keep_cols = ["date", "close"]
    for col in OPTIONAL_UNDERLYING_COLS:
        if col in und.columns and col not in keep_cols:
            keep_cols.append(col)

    und = und[keep_cols].drop_duplicates(subset=["date"])
    print(f"  {len(und):,} unique dates after dedup")
    return und


def discover_years() -> list[int]:
    """Read the options Parquet metadata to find available years without loading data."""
    print(f"[scan] Discovering years in {RAW_OPTIONS_FILE}...")
    # Read only the date column to find year range
    dates = pd.read_parquet(RAW_OPTIONS_FILE, columns=["date"])
    dates["date"] = pd.to_datetime(dates["date"])
    years = sorted(dates["date"].dt.year.unique().tolist())
    del dates
    gc.collect()
    print(f"  Found {len(years)} years: {years[0]}–{years[-1]}")
    return years


def load_options_year(year: int) -> pd.DataFrame:
    """Load one year of options data using row-group filtering."""
    # Read full file but filter immediately — PyArrow will handle predicate pushdown
    # for row groups where possible
    df = pd.read_parquet(RAW_OPTIONS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].copy()

    # Prune to only needed columns immediately
    keep_cols = []
    for col in REQUIRED_OPTIONS_COLS + OPTIONAL_OPTIONS_COLS:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    # Always keep expiration
    if "expiration" not in keep_cols:
        keep_cols.append("expiration")
    df = df[keep_cols]

    return df


def load_options_year_scan(year: int) -> pd.DataFrame:
    """Load one year of options using chunked scan to reduce peak memory."""
    pf = pq.ParquetFile(RAW_OPTIONS_FILE)

    # Determine which columns to read
    available = set(pf.schema_arrow.names)
    read_cols = []
    for col in REQUIRED_OPTIONS_COLS + OPTIONAL_OPTIONS_COLS + ["expiration"]:
        if col in available and col not in read_cols:
            read_cols.append(col)

    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=read_cols):
        chunk = batch.to_pandas()
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"].dt.year == year]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=read_cols)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def process_partition(opts: pd.DataFrame, und: pd.DataFrame, year: int) -> pd.DataFrame:
    """Process one year: join, derive, clean, flag."""

    # ── Normalize dates ──────────────────────────────────────────────
    opts["expiration"] = pd.to_datetime(opts["expiration"])

    # ── Join with underlying ─────────────────────────────────────────
    n_before = len(opts)
    df = opts.merge(und, on="date", how="inner", suffixes=("", "_und"))
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    join: dropped {n_dropped:,} rows with no underlying match")

    # ── Derive columns ───────────────────────────────────────────────
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["days_to_expiry"] = (df["expiration"] - df["date"]).dt.days
    df["tau"] = df["days_to_expiry"] / 365.0
    df["spot"] = df["close"]
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])

    # ── Hard cleaning ────────────────────────────────────────────────
    n0 = len(df)

    # Null critical fields
    for col in ["date", "expiration", "strike", "type", "implied_volatility", "close"]:
        df = df[df[col].notna()]

    # Negative / crossed quotes
    df = df[df["bid"] >= 0]
    df = df[df["ask"] >= 0]
    df = df[df["ask"] >= df["bid"]]

    # IV range
    df = df[df["implied_volatility"] > MIN_IV]
    df = df[df["implied_volatility"] <= MAX_IV]

    # Tau range
    df = df[df["tau"] > MIN_TAU]
    df = df[df["tau"] <= MAX_TAU]

    n1 = len(df)
    print(f"    clean: {n0:,} → {n1:,} (dropped {n0 - n1:,})")

    # ── Quality flags ────────────────────────────────────────────────
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
    return df[mask].copy()


def write_build_report(total_rows: int, strict_rows: int,
                       year_stats: list[dict]) -> None:
    """Write build report from accumulated stats."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "spy_build_report.md"

    lines = [
        "# SPY Surface Build Report\n",
        "## Conservative Dataset\n",
        f"- Total rows: {total_rows:,}",
        f"- Years processed: {len(year_stats)}",
        "",
        "### Per-year row counts\n",
        "| Year | Rows | Dropped |",
        "|------|-----:|--------:|",
    ]
    for ys in year_stats:
        lines.append(f"| {ys['year']} | {ys['rows_after']:,} | {ys['rows_dropped']:,} |")

    lines += [
        "",
        "## Strict Modeling Subset\n",
        f"- Rows: {strict_rows:,}",
        f"- Retention: {strict_rows / max(total_rows, 1) * 100:.1f}%",
        f"- IV range: [{STRICT_MIN_IV}, {STRICT_MAX_IV}]",
        f"- Tau range: [{STRICT_MIN_TAU:.4f}, {STRICT_MAX_TAU}]",
        f"- Log-moneyness range: [{STRICT_MIN_LOG_MONEYNESS}, "
        f"{STRICT_MAX_LOG_MONEYNESS}]",
    ]

    path.write_text("\n".join(lines))
    print(f"[report] Written to {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SPY surface table")
    parser.add_argument(
        "--test-year", action="store_true",
        help=f"Process only {TEST_YEAR} as a scalability test"
    )
    args = parser.parse_args()

    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load underlying (small, stays in memory) ─────────────────────
    und = load_underlying()

    # ── Discover years ───────────────────────────────────────────────
    all_years = discover_years()
    years = [TEST_YEAR] if args.test_year else all_years

    if args.test_year:
        print(f"\n*** TEST MODE: processing only {TEST_YEAR} ***\n")

    # ── Process year by year ─────────────────────────────────────────
    year_stats = []
    for i, year in enumerate(years):
        yt0 = time.time()
        print(f"\n[{i + 1}/{len(years)}] Processing {year}...")

        # Load one year via chunked scan
        opts = load_options_year_scan(year)
        n_raw = len(opts)
        print(f"    loaded: {n_raw:,} rows")

        if n_raw == 0:
            print(f"    skipped: no data for {year}")
            continue

        # Process
        df = process_partition(opts, und, year)
        del opts
        gc.collect()

        # Save partition
        part_path = PARTITIONS_DIR / f"spy_surface_{year}.parquet"
        df.to_parquet(part_path, index=False)
        elapsed = time.time() - yt0
        print(f"    saved: {len(df):,} rows → {part_path.name} "
              f"({part_path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s)")

        year_stats.append({
            "year": year,
            "rows_raw": n_raw,
            "rows_after": len(df),
            "rows_dropped": n_raw - len(df),
        })

        del df
        gc.collect()

    # ── Concatenate partitions ───────────────────────────────────────
    print(f"\n[concat] Merging {len(year_stats)} partitions...")
    partition_files = sorted(PARTITIONS_DIR.glob("spy_surface_*.parquet"))
    chunks = [pd.read_parquet(p) for p in partition_files]
    full = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    total_rows = len(full)
    print(f"  Total: {total_rows:,} rows")

    # ── Save conservative ────────────────────────────────────────────
    print(f"[save] {SURFACE_POINTS_FILE}")
    full.to_parquet(SURFACE_POINTS_FILE, index=False)
    print(f"  {total_rows:,} rows, "
          f"{SURFACE_POINTS_FILE.stat().st_size / 1e6:.1f} MB")

    # ── Build and save strict subset ─────────────────────────────────
    strict = build_strict_subset(full)
    strict_rows = len(strict)
    print(f"[strict] {total_rows:,} → {strict_rows:,} "
          f"(kept {strict_rows / total_rows * 100:.1f}%)")

    print(f"[save] {SURFACE_POINTS_STRICT_FILE}")
    strict.to_parquet(SURFACE_POINTS_STRICT_FILE, index=False)
    print(f"  {strict_rows:,} rows, "
          f"{SURFACE_POINTS_STRICT_FILE.stat().st_size / 1e6:.1f} MB")

    del full, strict
    gc.collect()

    # ── Report ───────────────────────────────────────────────────────
    write_build_report(total_rows, strict_rows, year_stats)

    elapsed = time.time() - t0
    print(f"\n✓ Build complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Build failed: {e}", file=sys.stderr)
        raise
