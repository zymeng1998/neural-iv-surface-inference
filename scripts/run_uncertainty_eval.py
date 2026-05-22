#!/usr/bin/env python3
"""Run the W1 uncertainty-evaluation pipeline end-to-end (story 2A.5).

Wires the model-agnostic predictor interface (2A.2), core uncertainty metrics
(2A.3), and abstention / risk–coverage curves (2A.4) into one runner that
evaluates a predictor on a benchmark and writes committed artifacts:

  - ``artifacts/results/uncertainty_eval_<tag>.csv``        (tidy metrics table)
  - ``artifacts/results/uncertainty_eval_<tag>_curve.csv``  (risk–coverage sweep)
  - ``artifacts/results/uncertainty_eval_<tag>_risk_coverage.png`` (figure)

Phase 1 baselines carry ``uncertainty=None``; interval coverage/width are
therefore reported as NaN and abstention uses an oracle (``-abs_error``)
confidence ranking — see ``eval/report.py`` for the documented convention.

Usage:
    # Real benchmark (only available where the parquet exists, e.g. RunPod):
    python scripts/run_uncertainty_eval.py --benchmark path/to/bench.parquet \
        --predictor interp --method rbf --tag spy_random40_noiselow

    # Tiny synthetic benchmark (local / CI, no data files needed):
    python scripts/run_uncertainty_eval.py --synthetic --predictor interp \
        --tag synthetic_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from neural_iv_surface_inference.eval.predictor import Predictor
from neural_iv_surface_inference.eval.adapters import InterpolationPredictor
from neural_iv_surface_inference.eval.report import (
    metrics_row,
    metrics_table,
    risk_coverage_table,
)

DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_synthetic_benchmark(
    n_dates: int = 20,
    n_points: int = 80,
    seed: int = 0,
) -> pd.DataFrame:
    """Tiny synthetic benchmark with the standard Step-4 columns.

    A smooth quadratic smile + small observed noise, chronological train/val/test
    split. Enough structure for interpolation to beat a constant and for the
    metrics/curve machinery to exercise its full path — not a research dataset.
    """
    rng = np.random.default_rng(seed)
    n = n_dates * n_points

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    date_col = np.repeat(dates, n_points)

    tau = rng.uniform(1 / 365, 2.0, size=n)
    log_m = rng.uniform(-0.5, 0.5, size=n)
    iv_clean = np.clip(0.2 + 0.15 * log_m**2 + 0.02 * tau, 0.01, 3.0)

    observed = rng.random(n) < 0.4
    iv_noisy = iv_clean.copy()
    iv_noisy[observed] += rng.normal(0, 0.01, size=n)[observed]
    iv_noisy = np.clip(iv_noisy, 1e-4, 10.0)

    split = np.array(["test"] * n, dtype=object)
    n_train = int(n_dates * 0.7)
    n_val = int(n_dates * 0.85)
    split[date_col < dates[n_train]] = "train"
    split[(date_col >= dates[n_train]) & (date_col < dates[n_val])] = "val"

    return pd.DataFrame({
        "date": date_col,
        "tau": tau,
        "log_moneyness": log_m,
        "iv_clean": iv_clean,
        "implied_volatility": iv_noisy,
        "observed": observed,
        "split": split,
        "noise_sigma": np.full(n, 0.01),
    })


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {split_name: frame} for the splits present in ``df``."""
    return {
        s: df[df["split"] == s].reset_index(drop=True)
        for s in df["split"].unique()
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictor(
    predictor: Predictor,
    df_splits: dict[str, pd.DataFrame],
    model_name: str,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    n_curve_points: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``predictor`` on each available split; return (metrics, curve) tables.

    Splits not present in ``df_splits`` are skipped.
    """
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []

    for split in splits:
        df = df_splits.get(split)
        if df is None or len(df) == 0:
            continue
        result = predictor.predict(df)
        y_true = df["iv_clean"].to_numpy()
        observed = df["observed"].to_numpy()

        rows.append(metrics_row(
            result, y_true, observed, model_name=model_name, split=split,
            n_curve_points=n_curve_points,
        ))
        curves.append(risk_coverage_table(
            result, y_true, model_name=model_name, split=split,
            n_points=n_curve_points,
        ))

    metrics_df = metrics_table(rows)
    curve_df = (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    )
    return metrics_df, curve_df


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_risk_coverage_figure(curve_df: pd.DataFrame, path: Path) -> None:
    """Save a risk–coverage figure (one line per split) to ``path``."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if not curve_df.empty:
        for split, grp in curve_df.groupby("split"):
            grp = grp.sort_values("coverage")
            ax.plot(grp["coverage"], grp["retained_mae"], marker="o",
                    markersize=3, label=str(split))
        source = curve_df["confidence_source"].iloc[0]
        ax.set_title(f"Risk–coverage ({curve_df['model'].iloc[0]}, "
                     f"confidence={source})")
    ax.set_xlabel("Coverage (retained fraction)")
    ax.set_ylabel("Retained MAE")
    ax.grid(True, alpha=0.3)
    ax.legend(title="split")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_artifacts(
    metrics_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> dict[str, Path]:
    """Write metrics CSV, curve CSV, and risk–coverage figure; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"uncertainty_eval_{tag}.csv"
    curve_path = out_dir / f"uncertainty_eval_{tag}_curve.csv"
    fig_path = out_dir / f"uncertainty_eval_{tag}_risk_coverage.png"

    metrics_df.to_csv(metrics_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    save_risk_coverage_figure(curve_df, fig_path)

    return {"metrics": metrics_path, "curve": curve_path, "figure": fig_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_predictor(name: str, method: str) -> tuple[Predictor, str]:
    """Construct a predictor adapter and its model label.

    Only the interpolation baseline is wired for the no-data / CI path. The MLP
    baseline needs a trained checkpoint; running it through this CLI is left to a
    later wiring step (this story does not run expensive training).
    """
    if name == "interp":
        return InterpolationPredictor(method=method), f"interp_{method}"
    raise ValueError(
        f"predictor '{name}' not supported by this runner yet (use 'interp')"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run W1 uncertainty evaluation")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path to a benchmark Parquet (Step 4 output)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a tiny synthetic benchmark (no data files)")
    parser.add_argument("--predictor", type=str, default="interp",
                        choices=["interp"])
    parser.add_argument("--method", type=str, default="rbf",
                        help="Interpolation method (linear|cubic|rbf|nearest)")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to evaluate")
    parser.add_argument("--out-dir", type=str, default="artifacts/results")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Risk–coverage curve resolution")
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    # Load data.
    if args.synthetic:
        print("[data] Using synthetic benchmark")
        df = make_synthetic_benchmark()
        df_splits = split_frames(df)
    elif args.benchmark:
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.exists():
            print(f"Benchmark not found: {benchmark_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[data] Loading benchmark: {benchmark_path}")
        df = pd.read_parquet(benchmark_path)
        df_splits = split_frames(df)
    else:
        print("Provide --benchmark PATH or --synthetic", file=sys.stderr)
        sys.exit(1)

    predictor, model_name = build_predictor(args.predictor, args.method)
    print(f"[run] predictor={model_name}, splits={splits}")

    metrics_df, curve_df = evaluate_predictor(
        predictor, df_splits, model_name=model_name,
        splits=splits, n_curve_points=args.n_points,
    )

    paths = write_artifacts(metrics_df, curve_df, Path(args.out_dir), args.tag)

    print("\n[metrics]")
    print(metrics_df.to_string(index=False))
    print("\n[artifacts]")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
