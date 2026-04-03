#!/usr/bin/env python3
"""
Step 4: Build benchmark task datasets from processed surface points.

Takes the output of Step 3 (spy_surface_points_strict.parquet) and produces
one or more benchmark datasets, each with:
  - observation mask (which points the model "sees")
  - noise injection on observed points
  - time-based train/val/test split
  - provenance metadata

**Memory-safe**: instead of loading the full 21M-row strict file at once,
this script streams through the data using PyArrow iter_batches.  A global
date→split mapping is precomputed once from partition metadata, so the
chronological split is consistent across chunks.

Benchmark variants are defined in configs/benchmark_tasks.yaml.

Usage:
    python src/data/04_build_benchmark_tasks.py                  # all variants
    python src/data/04_build_benchmark_tasks.py --variant random20_noise0
    python src/data/04_build_benchmark_tasks.py --list            # list variants
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Allow imports from both src/data and the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, SURFACE_POINTS_STRICT_FILE

from neural_iv_surface_inference.data.masking import apply_mask
from neural_iv_surface_inference.data.noise import inject_noise, NOISE_REGIMES
from neural_iv_surface_inference.data.splits import (
    compute_date_split_map,
    benchmark_name,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmark_tasks.yaml"
BENCHMARK_DIR = PROCESSED_DIR / "benchmarks"
PARTITION_DIR = PROCESSED_DIR / "partitions"

# Batch size for PyArrow iter_batches (~500k rows ≈ manageable memory)
BATCH_SIZE = 500_000


def load_config(path: Path) -> dict:
    """Load benchmark task configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_split_column(df: pd.DataFrame, date_split_map: dict) -> pd.DataFrame:
    """Add a 'split' column using the precomputed date→split mapping."""
    df["date"] = pd.to_datetime(df["date"])
    df["split"] = df["date"].map(date_split_map)
    # Any unmapped dates (shouldn't happen) default to test
    df["split"] = df["split"].fillna("test")
    return df


def build_one_variant_streaming(
    source_path: Path,
    variant: dict,
    output_dir: Path,
    date_split_map: dict,
    verbose: bool = True,
) -> Path:
    """Build a single benchmark variant by streaming through the source file.

    Reads the strict parquet in chunks via iter_batches, applies masking,
    noise, and split assignment per chunk, then writes to a single output
    Parquet file using ParquetWriter.
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.parquet"

    pf = pq.ParquetFile(source_path)
    writer = None
    total_rows = 0
    total_observed = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    batch_num = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        chunk = batch.to_pandas()
        batch_num += 1

        if verbose and batch_num == 1:
            print(f"  streaming {pf.metadata.num_rows:,} total rows "
                  f"in batches of {BATCH_SIZE:,}")

        # 1. Masking — use batch-local seed offset for reproducibility
        chunk_masked = apply_mask(
            chunk, strategy=strategy, keep_frac=keep_frac,
            seed=seed + batch_num,
        )

        # 2. Noise injection on observed points
        chunk_noisy = inject_noise(
            chunk_masked,
            regime=noise_regime,
            heteroscedastic=hetero,
            seed=seed + 1 + batch_num,
        )

        # 3. Split assignment via precomputed map
        chunk_split = _apply_split_column(chunk_noisy, date_split_map)

        # Track stats
        n_obs = chunk_split["observed"].sum()
        total_rows += len(chunk_split)
        total_observed += n_obs
        for s in split_counts:
            split_counts[s] += (chunk_split["split"] == s).sum()

        # 4. Write to Parquet
        table = pa.Table.from_pandas(chunk_split, preserve_index=False)
        if writer is None:
            # Attach provenance metadata to schema
            meta = {
                "benchmark_name": name,
                "strategy": strategy,
                "keep_frac": str(keep_frac),
                "noise_regime": noise_regime,
                "noise_sigma": str(NOISE_REGIMES.get(noise_regime, "custom")),
                "heteroscedastic": str(hetero),
                "seed": str(seed),
                "source_file": str(source_path),
            }
            existing_meta = table.schema.metadata or {}
            existing_meta.update({k.encode(): v.encode() for k, v in meta.items()})
            schema = table.schema.with_metadata(existing_meta)
            writer = pq.ParquetWriter(out_path, schema)

        writer.write_table(table)

        # Free chunk memory
        del chunk, chunk_masked, chunk_noisy, chunk_split, table
        gc.collect()

        if verbose:
            print(f"  batch {batch_num}: {total_rows:,} rows processed")

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6

    if verbose:
        print(f"  total: {total_rows:,} rows, "
              f"{total_observed:,} observed ({total_observed / max(total_rows, 1) * 100:.1f}%)")
        for s in ["train", "val", "test"]:
            print(f"  {s:5s}: {split_counts[s]:,} rows")
        print(f"  saved: {out_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    return out_path


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
    split_cfg = cfg.get("split", {})
    train_frac = split_cfg.get("train_frac", 0.70)
    val_frac = split_cfg.get("val_frac", 0.15)

    if args.list:
        print("Configured benchmark variants:")
        for v in variants:
            name = benchmark_name(
                v["strategy"], v["keep_frac"], v["noise_regime"],
                v.get("heteroscedastic", False),
            )
            print(f"  {name}")
        return

    # Validate source file
    if not SURFACE_POINTS_STRICT_FILE.exists():
        print(
            f"✗ Source file not found: {SURFACE_POINTS_STRICT_FILE}\n"
            f"  Run Step 3 first (03_build_spy_surface_table.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Precompute global date→split mapping from partition files
    print(f"[split] Precomputing date→split map from partitions in {PARTITION_DIR}")
    if PARTITION_DIR.exists() and any(PARTITION_DIR.glob("spy_surface_*.parquet")):
        date_split_map = compute_date_split_map(
            PARTITION_DIR,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    else:
        # Fallback: read only the date column from the strict file
        print(f"  (no partitions found, reading date column from strict file)")
        dates_table = pq.read_table(SURFACE_POINTS_STRICT_FILE, columns=["date"])
        dates = pd.to_datetime(dates_table["date"].to_pandas()).unique()
        sorted_dates = sorted(dates)
        n = len(sorted_dates)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        date_split_map = {}
        for i, d in enumerate(sorted_dates):
            if i < n_train:
                date_split_map[d] = "train"
            elif i < n_train + n_val:
                date_split_map[d] = "val"
            else:
                date_split_map[d] = "test"
        del dates_table, dates, sorted_dates
        gc.collect()

    n_dates = len(date_split_map)
    n_train_dates = sum(1 for v in date_split_map.values() if v == "train")
    n_val_dates = sum(1 for v in date_split_map.values() if v == "val")
    n_test_dates = sum(1 for v in date_split_map.values() if v == "test")
    print(f"  {n_dates:,} dates: {n_train_dates} train, "
          f"{n_val_dates} val, {n_test_dates} test")

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
        p = build_one_variant_streaming(
            SURFACE_POINTS_STRICT_FILE, v, output_dir, date_split_map,
        )
        paths.append(p)
        gc.collect()

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
