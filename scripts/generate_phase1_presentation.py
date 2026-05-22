#!/usr/bin/env python3
"""Generate curated, presentation-quality Phase 1 figures.

Reads the committed result CSVs and renders a consistent, captioned figure set
via the tested ``neural_iv_surface_inference.viz`` module. Figures are written to
``artifacts/figures/presentation/`` (regenerable; gitignored) — this script is
the reproducible source of record.

Aggregate figures (from CSVs) need no data files. Per-date surface figures
(reference / observed / reconstructed triptych, spatial error, joint
maturity x moneyness heatmap) need the benchmark parquet and are produced only
when ``--benchmark`` points at an existing file (RunPod).

Usage:
    python scripts/generate_phase1_presentation.py
    python scripts/generate_phase1_presentation.py --benchmark path/to/bench.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Headless backend before pyplot is imported anywhere downstream.
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from neural_iv_surface_inference.viz import (
    apply_house_style,
    plot_baseline_comparison,
    plot_observed_vs_unobserved,
    plot_regional_error_bars,
    plot_joint_error_heatmap,
    plot_noise_sweep,
    plot_reconstruction_triptych,
    plot_spatial_error,
    plot_surface_scatter,
)
from neural_iv_surface_inference.training.eval import (
    evaluate_predictions_2d,
    evaluate_predictions_2d_counts,
)


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def generate_aggregate_figures(
    results_df: pd.DataFrame,
    sweep_df: pd.DataFrame | None,
    out_dir: Path,
) -> list[Path]:
    """Figures derived purely from the committed result CSVs."""
    written: list[Path] = []
    written.append(_save(
        plot_baseline_comparison(results_df, split="test"),
        out_dir, "fig01_baseline_comparison.png"))
    written.append(_save(
        plot_observed_vs_unobserved(results_df, split="test"),
        out_dir, "fig02_observed_vs_unobserved.png"))

    for model in sorted(results_df["model"].unique()):
        written.append(_save(
            plot_regional_error_bars(results_df, model=model,
                                     dimension="moneyness"),
            out_dir, f"fig03_{model}_error_by_moneyness.png"))
        written.append(_save(
            plot_regional_error_bars(results_df, model=model,
                                     dimension="maturity"),
            out_dir, f"fig04_{model}_error_by_maturity.png"))

    if sweep_df is not None and "variant" in sweep_df.columns:
        written.append(_save(
            plot_noise_sweep(sweep_df), out_dir, "fig05_noise_sweep.png"))
    return written


def generate_surface_figures(
    benchmark_path: Path,
    out_dir: Path,
    method: str = "rbf",
    n_dates: int = 1,
) -> list[Path]:
    """Per-date surface visuals + joint heatmap. Requires the benchmark parquet.

    Uses the interpolation baseline for the reconstruction so the figure is
    deterministic and needs no checkpoint.
    """
    from neural_iv_surface_inference.models.interpolation import (
        run_interpolation_baseline,
    )

    df = pd.read_parquet(
        benchmark_path,
        columns=["date", "tau", "log_moneyness", "iv_clean",
                 "implied_volatility", "observed", "split"],
    )
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        print("  benchmark has no test split; skipping surface figures")
        return []

    written: list[Path] = []
    sample_dates = sorted(test_df["date"].unique())[:n_dates]
    for i, date in enumerate(sample_dates):
        sample = test_df[test_df["date"] == date].copy()
        sample["iv_pred"] = run_interpolation_baseline(
            sample, method=method, verbose=False)
        label = str(pd.Timestamp(date).date())

        written.append(_save(
            plot_surface_scatter(sample, value_col="iv_clean",
                                 title=f"Reference surface — {label}"),
            out_dir, f"fig06_{i}_reference_surface.png"))
        written.append(_save(
            plot_reconstruction_triptych(sample, date_label=label),
            out_dir, f"fig07_{i}_reconstruction_triptych.png"))
        written.append(_save(
            plot_spatial_error(sample, title=f"Spatial abs error — {label}"),
            out_dir, f"fig08_{i}_spatial_error.png"))

    # Joint maturity x moneyness heatmap over the full test split.
    test_eval = test_df.copy()
    test_eval["iv_pred"] = run_interpolation_baseline(
        test_eval, method=method, verbose=False)
    grid = evaluate_predictions_2d(test_eval, metric="mae")
    counts = evaluate_predictions_2d_counts(test_eval)
    written.append(_save(
        plot_joint_error_heatmap(
            grid, counts,
            title=f"interp_{method} — joint MAE (maturity x moneyness, test)"),
        out_dir, "fig09_joint_error_heatmap.png"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 presentation figures")
    parser.add_argument("--results", default="artifacts/results/baseline_results.csv")
    parser.add_argument("--sweep", default="artifacts/results/interp_sweep_sampled_test.csv")
    parser.add_argument("--benchmark", default=None,
                        help="Benchmark parquet for surface figures (RunPod)")
    parser.add_argument("--out-dir", default="artifacts/figures/presentation")
    parser.add_argument("--method", default="rbf")
    args = parser.parse_args()

    apply_house_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results CSV not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    results_df = pd.read_csv(results_path)

    sweep_path = Path(args.sweep)
    sweep_df = pd.read_csv(sweep_path) if sweep_path.exists() else None

    print(f"[figures] aggregate -> {out_dir}")
    written = generate_aggregate_figures(results_df, sweep_df, out_dir)

    if args.benchmark:
        bench = Path(args.benchmark)
        if bench.exists():
            print(f"[figures] surface (from {bench}) -> {out_dir}")
            written += generate_surface_figures(bench, out_dir, method=args.method)
        else:
            print(f"[figures] benchmark not found ({bench}); "
                  "skipping surface figures (run on RunPod)")
    else:
        print("[figures] no --benchmark; skipping surface figures "
              "(run on RunPod for surface visuals)")

    print(f"\nDone: {len(written)} figures in {out_dir}")


if __name__ == "__main__":
    main()
