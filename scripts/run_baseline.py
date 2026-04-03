#!/usr/bin/env python3
"""Train and evaluate baseline models on IV surface benchmark data.

Runs both the interpolation baseline (S3.1) and the masked MLP baseline
(S3.2) on a benchmark dataset, computes evaluation metrics, and prints
a comparison summary.

Usage:
    python scripts/run_baseline.py --config configs/baseline.yaml
    python scripts/run_baseline.py --config configs/baseline.yaml --benchmark path/to/benchmark.parquet
    python scripts/run_baseline.py --config configs/baseline.yaml --interp-only
    python scripts/run_baseline.py --config configs/baseline.yaml --mlp-only
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from neural_iv_surface_inference.utils.io import load_config
from neural_iv_surface_inference.utils.seed import set_seed
from neural_iv_surface_inference.data.loaders import load_benchmark_splits
from neural_iv_surface_inference.models.interpolation import run_interpolation_baseline
from neural_iv_surface_inference.training.train import train_mlp, predict_mlp
from neural_iv_surface_inference.training.eval import (
    evaluate_predictions,
    metrics_to_dataframe,
    print_evaluation,
)


def run_interpolation(data: dict, config: dict) -> pd.DataFrame:
    """Run interpolation baseline on all splits and return metrics."""
    method = config.get("interpolation", {}).get("method", "rbf")
    all_results = []

    for split in ["train", "val", "test"]:
        df_split = data[f"{split}_df"]
        print(f"\n[interp] Running on {split} ({len(df_split):,} points)...")
        t0 = time.time()

        df_pred = run_interpolation_baseline(df_split, method=method, verbose=False)
        elapsed = time.time() - t0

        results = evaluate_predictions(df_pred)
        print_evaluation(results, model_name=f"Interpolation ({method}) — {split}")
        print(f"  Time: {elapsed:.1f}s")

        summary = metrics_to_dataframe(results, model_name=f"interp_{method}")
        summary["split"] = split
        all_results.append(summary)

    return pd.concat(all_results, ignore_index=True)


def run_mlp(data: dict, config: dict) -> pd.DataFrame:
    """Train MLP and evaluate on all splits."""
    import torch

    mlp_config = config.get("mlp", {})
    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoint_dir", "artifacts/checkpoints"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[mlp] Training on device: {device}")

    # Train
    model, history = train_mlp(
        data["train_loader"],
        data["val_loader"],
        config=mlp_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Evaluate on all splits
    all_results = []
    for split in ["train", "val", "test"]:
        df_split = data[f"{split}_df"].copy()
        print(f"\n[mlp] Evaluating on {split} ({len(df_split):,} points)...")

        preds = predict_mlp(model, data[f"{split}_loader"], device=device)
        df_split["iv_pred"] = preds

        results = evaluate_predictions(df_split)
        print_evaluation(results, model_name=f"MLP — {split}")

        summary = metrics_to_dataframe(results, model_name="mlp")
        summary["split"] = split
        all_results.append(summary)

    return pd.concat(all_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml",
        help="Path to baseline config YAML",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Override benchmark file path",
    )
    parser.add_argument("--interp-only", action="store_true")
    parser.add_argument("--mlp-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Resolve benchmark path
    benchmark_path = Path(
        args.benchmark or config["data"]["benchmark_file"]
    )
    if not benchmark_path.exists():
        print(f"Benchmark file not found: {benchmark_path}", file=sys.stderr)
        print("Run Step 3 and Step 4 first.", file=sys.stderr)
        sys.exit(1)

    print(f"[data] Loading benchmark: {benchmark_path}")
    data = load_benchmark_splits(
        benchmark_path,
        batch_size=config["data"].get("batch_size", 256),
        num_workers=config["data"].get("num_workers", 0),
    )
    for split in ["train", "val", "test"]:
        df = data[f"{split}_df"]
        n_obs = df["observed"].sum()
        print(f"  {split}: {len(df):,} points, {n_obs:,} observed "
              f"({n_obs / len(df) * 100:.1f}%)")

    # Run baselines
    results_dfs = []

    if not args.mlp_only:
        interp_results = run_interpolation(data, config)
        results_dfs.append(interp_results)

    if not args.interp_only:
        mlp_results = run_mlp(data, config)
        results_dfs.append(mlp_results)

    # Save combined results
    if results_dfs:
        combined = pd.concat(results_dfs, ignore_index=True)
        results_dir = Path(config.get("paths", {}).get("results_dir", "artifacts/results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "baseline_results.csv"
        combined.to_csv(results_path, index=False)
        print(f"\n[save] Results saved to {results_path}")

        # Print comparison table
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY (test split)")
        print("=" * 70)
        test_rows = combined[combined["split"] == "test"]
        if len(test_rows) > 0:
            cols = ["model", "overall_mae", "overall_rmse",
                    "observed_mae", "unobserved_mae"]
            available_cols = [c for c in cols if c in test_rows.columns]
            print(test_rows[available_cols].to_string(index=False))
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        raise
