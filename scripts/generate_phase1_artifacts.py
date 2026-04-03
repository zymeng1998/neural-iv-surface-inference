#!/usr/bin/env python3
"""Generate Phase 1 S4.3 artifacts from baseline outputs.

Outputs:
- artifacts/figures/baseline_test_mae_comparison.png
- artifacts/figures/error_vs_noise_sweep.png
- artifacts/figures/region_error_heatmap_interp_test.png
- artifacts/figures/reference_surface_sample.png           (if benchmark file exists)
- artifacts/figures/sparse_observed_sample.png            (if benchmark file exists)
- artifacts/figures/reconstructed_surface_sample.png       (if benchmark file exists)
- artifacts/tables/phase1_summary_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dirs(fig_dir: Path, table_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)


def _save_baseline_comparison(results_df: pd.DataFrame, fig_dir: Path) -> None:
    test_df = results_df[results_df["split"] == "test"].copy()
    if test_df.empty:
        return

    plt.figure(figsize=(7, 4))
    x = np.arange(len(test_df))
    width = 0.35
    plt.bar(x - width / 2, test_df["overall_mae"], width=width, label="overall_mae")
    plt.bar(x + width / 2, test_df["unobserved_mae"], width=width, label="unobserved_mae")
    plt.xticks(x, test_df["model"], rotation=20)
    plt.ylabel("Error")
    plt.title("Test Split: Baseline Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_test_mae_comparison.png", dpi=160)
    plt.close()


def _save_noise_sweep_plot(sweep_df: pd.DataFrame, fig_dir: Path) -> None:
    if sweep_df.empty:
        return

    def _noise_rank(name: str) -> int:
        if "noiselow" in name:
            return 1
        if "noisemed" in name:
            return 2
        if "noisehigh" in name:
            return 3
        return 0

    df = sweep_df.copy()
    df["noise_rank"] = df["variant"].map(_noise_rank)
    df = df.sort_values("noise_rank")

    plt.figure(figsize=(7, 4))
    plt.plot(df["noise_rank"], df["overall_mae"], marker="o", label="overall_mae")
    plt.plot(df["noise_rank"], df["unobserved_mae"], marker="o", label="unobserved_mae")
    plt.xticks([1, 2, 3], ["low", "medium", "high"])
    plt.xlabel("Noise regime")
    plt.ylabel("MAE")
    plt.title("Error vs Noise Regime (Sampled Test Sweep)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "error_vs_noise_sweep.png", dpi=160)
    plt.close()


def _save_region_heatmap(results_df: pd.DataFrame, fig_dir: Path) -> None:
    test_interp = results_df[
        (results_df["split"] == "test") & (results_df["model"].str.contains("interp"))
    ]
    if test_interp.empty:
        return
    row = test_interp.iloc[0]

    maturity = ["short", "medium", "long"]
    moneyness = ["deep_itm", "itm", "atm", "otm", "deep_otm"]
    mat_vals = [row.get(f"by_maturity_{m}_mae", np.nan) for m in maturity]
    mon_vals = [row.get(f"by_moneyness_{m}_mae", np.nan) for m in moneyness]

    heat = np.vstack([mat_vals + [np.nan, np.nan], mon_vals[:3] + mon_vals[3:5]])
    # Shape into 2x5 for plotting consistency.
    heat = np.array([
        [mat_vals[0], mat_vals[1], mat_vals[2], np.nan, np.nan],
        [mon_vals[0], mon_vals[1], mon_vals[2], mon_vals[3], mon_vals[4]],
    ])

    plt.figure(figsize=(8, 3.5))
    im = plt.imshow(heat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="MAE")
    plt.yticks([0, 1], ["maturity", "moneyness"])
    plt.xticks([0, 1, 2, 3, 4], ["1", "2", "3", "4", "5"])
    plt.title("Regional Error Heatmap (interp_rbf, test)")
    plt.tight_layout()
    plt.savefig(fig_dir / "region_error_heatmap_interp_test.png", dpi=160)
    plt.close()


def _save_surface_samples(benchmark_path: Path, fig_dir: Path) -> None:
    if not benchmark_path.exists():
        return

    df = pd.read_parquet(
        benchmark_path,
        columns=["date", "tau", "log_moneyness", "iv_clean", "implied_volatility", "observed", "split"],
    )
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        return
    sample_date = sorted(test_df["date"].unique())[0]
    sample = test_df[test_df["date"] == sample_date].copy()

    # Reference surface
    plt.figure(figsize=(6, 4))
    plt.scatter(sample["log_moneyness"], sample["tau"], c=sample["iv_clean"], s=12)
    plt.colorbar(label="iv_clean")
    plt.title(f"Reference Surface Sample ({sample_date})")
    plt.xlabel("log_moneyness")
    plt.ylabel("tau")
    plt.tight_layout()
    plt.savefig(fig_dir / "reference_surface_sample.png", dpi=160)
    plt.close()

    # Sparse/noisy observed points
    obs = sample[sample["observed"]]
    plt.figure(figsize=(6, 4))
    plt.scatter(obs["log_moneyness"], obs["tau"], c=obs["implied_volatility"], s=12)
    plt.colorbar(label="observed implied_volatility")
    plt.title(f"Sparse Observed Sample ({sample_date})")
    plt.xlabel("log_moneyness")
    plt.ylabel("tau")
    plt.tight_layout()
    plt.savefig(fig_dir / "sparse_observed_sample.png", dpi=160)
    plt.close()

    # Reconstructed sample with nearest interpolation as fast deterministic visual.
    from neural_iv_surface_inference.models.interpolation import run_interpolation_baseline

    sample["iv_pred"] = run_interpolation_baseline(sample, method="nearest", verbose=False)
    plt.figure(figsize=(6, 4))
    plt.scatter(sample["log_moneyness"], sample["tau"], c=sample["iv_pred"], s=12)
    plt.colorbar(label="iv_pred")
    plt.title(f"Reconstructed Surface Sample ({sample_date})")
    plt.xlabel("log_moneyness")
    plt.ylabel("tau")
    plt.tight_layout()
    plt.savefig(fig_dir / "reconstructed_surface_sample.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 artifacts")
    parser.add_argument("--results", default="artifacts/results/baseline_results.csv")
    parser.add_argument("--sweep", default="artifacts/results/interp_sweep_sampled_test.csv")
    parser.add_argument(
        "--benchmark",
        default="data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet",
    )
    parser.add_argument("--fig-dir", default="artifacts/figures")
    parser.add_argument("--table-dir", default="artifacts/tables")
    args = parser.parse_args()

    results_path = Path(args.results)
    sweep_path = Path(args.sweep)
    benchmark_path = Path(args.benchmark)
    fig_dir = Path(args.fig_dir)
    table_dir = Path(args.table_dir)

    _ensure_dirs(fig_dir, table_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results_df = pd.read_csv(results_path)
    sweep_df = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()

    # Summary table (test split only)
    test_cols = [c for c in [
        "model",
        "split",
        "overall_mae",
        "overall_rmse",
        "observed_mae",
        "unobserved_mae",
        "overall_n",
    ] if c in results_df.columns]
    summary = results_df[results_df["split"] == "test"][test_cols].copy()
    summary.to_csv(table_dir / "phase1_summary_table.csv", index=False)

    _save_baseline_comparison(results_df, fig_dir)
    _save_noise_sweep_plot(sweep_df, fig_dir)
    _save_region_heatmap(results_df, fig_dir)
    _save_surface_samples(benchmark_path, fig_dir)

    print(f"Saved summary table to {table_dir / 'phase1_summary_table.csv'}")
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
