#!/usr/bin/env python3
"""
Step 4: Build benchmark task datasets from processed surface points.

Takes the output of Step 3 (spy_surface_points_strict.parquet) and produces
one or more benchmark datasets, each with:
  - observation mask (which points the model "sees")
  - noise injection on observed points
  - time-based train/val/test split
  - provenance metadata

Benchmark variants are defined in configs/benchmark_tasks.yaml.

Usage:
    python src/data/04_build_benchmark_tasks.py                  # all variants
    python src/data/04_build_benchmark_tasks.py --variant random20_noise0
    python src/data/04_build_benchmark_tasks.py --list            # list variants
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# Allow imports from both src/data and the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, SURFACE_POINTS_STRICT_FILE

from neural_iv_surface_inference.data.masking import apply_mask
from neural_iv_surface_inference.data.noise import inject_noise, NOISE_REGIMES
from neural_iv_surface_inference.data.splits import (
    time_split,
    split_summary,
    benchmark_name,
    save_benchmark,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmark_tasks.yaml"
BENCHMARK_DIR = PROCESSED_DIR / "benchmarks"


def load_config(path: Path) -> dict:
    """Load benchmark task configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_one_variant(
    df,
    variant: dict,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """Build a single benchmark variant from a variant spec dict.

    Expected keys in *variant*:
        strategy, keep_frac, noise_regime,
        heteroscedastic (optional, default False),
        seed (optional, default 42)
    """
    strategy = variant["strategy"]
    keep_frac = variant["keep_frac"]
    noise_regime = variant["noise_regime"]
    hetero = variant.get("heteroscedastic", False)
    seed = variant.get("seed", 42)

    name = benchmark_name(strategy, keep_frac, noise_regime, hetero)

    if verbose:
        print(f"\n[build] {name}")
        print(f"  strategy={strategy}  keep_frac={keep_frac}  "
              f"noise={noise_regime}  hetero={hetero}  seed={seed}")

    t0 = time.time()

    # 1. Apply masking
    df_masked = apply_mask(df, strategy=strategy, keep_frac=keep_frac, seed=seed)
    n_obs = df_masked["observed"].sum()
    n_total = len(df_masked)
    if verbose:
        print(f"  mask: {n_obs:,} / {n_total:,} observed "
              f"({n_obs / n_total * 100:.1f}%)")

    # 2. Inject noise on observed points
    df_noisy = inject_noise(
        df_masked,
        regime=noise_regime,
        heteroscedastic=hetero,
        seed=seed + 1,  # different seed from masking
    )

    # 3. Time-based split
    df_split = time_split(df_noisy)
    if verbose:
        summary = split_summary(df_split)
        for _, row in summary.iterrows():
            print(f"  {row['split']:5s}: {row['n_dates']:,} dates, "
                  f"{row['n_rows']:,} rows  "
                  f"[{row['date_min']} → {row['date_max']}]")

    # 4. Save
    metadata = {
        "strategy": strategy,
        "keep_frac": str(keep_frac),
        "noise_regime": noise_regime,
        "noise_sigma": str(NOISE_REGIMES.get(noise_regime, "custom")),
        "heteroscedastic": str(hetero),
        "seed": str(seed),
        "source_file": str(SURFACE_POINTS_STRICT_FILE),
        "n_total": str(n_total),
        "n_observed": str(n_obs),
    }

    path = save_benchmark(df_split, output_dir, name, metadata=metadata)
    elapsed = time.time() - t0
    size_mb = path.stat().st_size / 1e6

    if verbose:
        print(f"  saved: {path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build benchmark task datasets from processed surface points"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Build only this variant (by short name, e.g. random20_noise0)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured variants and exit",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to benchmark_tasks.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    variants = cfg.get("variants", [])

    if args.list:
        print("Configured benchmark variants:")
        for v in variants:
            name = benchmark_name(
                v["strategy"], v["keep_frac"], v["noise_regime"],
                v.get("heteroscedastic", False),
            )
            print(f"  {name}")
        return

    # Load source data
    print(f"[load] Source: {SURFACE_POINTS_STRICT_FILE}")
    if not SURFACE_POINTS_STRICT_FILE.exists():
        print(
            f"✗ Source file not found: {SURFACE_POINTS_STRICT_FILE}\n"
            f"  Run Step 3 first (03_build_spy_surface_table.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(SURFACE_POINTS_STRICT_FILE)
    print(f"  {len(df):,} rows, {df['date'].nunique():,} dates")

    # Determine which variants to build
    if args.variant:
        matched = [
            v for v in variants
            if args.variant in benchmark_name(
                v["strategy"], v["keep_frac"], v["noise_regime"],
                v.get("heteroscedastic", False),
            )
        ]
        if not matched:
            print(f"✗ No variant matching '{args.variant}'", file=sys.stderr)
            sys.exit(1)
        variants = matched

    output_dir = Path(cfg.get("output_dir", str(BENCHMARK_DIR)))

    t0 = time.time()
    paths = []
    for v in variants:
        p = build_one_variant(df, v, output_dir)
        paths.append(p)

    elapsed = time.time() - t0
    print(f"\n✓ Built {len(paths)} benchmark(s) in {elapsed:.1f}s")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Failed: {e}", file=sys.stderr)
        raise
