#!/usr/bin/env python3
"""Story 3A.4 — evaluate the two 3A.3 decoder-only retrains (Fourier vs raw).

Thin orchestrator over the W1 evaluator (story 2A.5). Reads the committed
prediction parquets at ``artifacts/runs/3A/{fourier,raw}/predictions_{val,test}.parquet``,
constructs ``PredictionResult`` objects from the heteroscedastic Gaussian
head outputs (``mu``, ``sigma``), runs the W1 metrics + abstention machinery,
and writes:

    results/3/<dataset>/3a_fourier/metrics_summary.csv
    results/3/<dataset>/3a_fourier/calibration_table.csv
    results/3/<dataset>/3a_fourier/abstention_curve.csv
    results/3/<dataset>/3a_raw/...        (same three files)
    results/3/<dataset>/3a_compare/comparison.csv

The comparison CSV is long-format with one row per (variant, split, metric),
including the Phase 2D.9 RBF and 2D.7 calibrated-Gaussian rows for reference
(sourced read-only from ``results/2D/<dataset>/{interpolation,conditional_calibrated}/metrics_summary.csv``).

Local-only, CPU-only, read-only on ``artifacts/runs/3A/``. No training.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from neural_iv_surface_inference.eval.calibration import gaussian_z
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.eval.report import (
    metrics_row,
    metrics_table,
    risk_coverage_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = "spy_phase1_random40_noiselow"
DEFAULT_RUNS_DIR = REPO_ROOT / "artifacts" / "runs" / "3A"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "3"
DEFAULT_BASELINE_DIR = REPO_ROOT / "results" / "2D"

CALIBRATION_ALPHAS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SPLITS: tuple[str, ...] = ("val", "test")

VARIANTS: tuple[str, ...] = ("fourier", "raw")


# ─────────────────────────────────────────────────────────────────────────────
# Load 3A.3 predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_variant_predictions(runs_dir: Path, variant: str) -> dict[str, pd.DataFrame]:
    """Load val + test prediction parquets for a single 3A.3 variant."""
    out: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = runs_dir / variant / f"predictions_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing prediction parquet: {path}")
        out[split] = pd.read_parquet(path)
    return out


def build_prediction_result(df: pd.DataFrame, alpha: float = 0.9) -> PredictionResult:
    """Build a W1 ``PredictionResult`` from a 3A.3 prediction parquet.

    Bands are the **uncalibrated** Gaussian band at the nominal ``alpha``
    target (``mu ± z * sigma``). 3A.3 did not refit a calibrator on top of
    the decoder-only retrain — the calibration question is reported as
    empirical coverage vs nominal in ``calibration_table.csv``.
    """
    mu = df["mu"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    z = gaussian_z(alpha)
    return PredictionResult(
        pred=mu,
        uncertainty=sigma,
        lower=mu - z * sigma,
        upper=mu + z * sigma,
        meta={"nominal_alpha": alpha},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Mean per-point Gaussian NLL with the full ``log(2π)`` constant.

    NLL = 0.5 * (log(2π) + 2 log σ + (y - μ)² / σ²).
    """
    sigma = np.clip(sigma, 1e-12, None)
    return float(np.mean(
        0.5 * (math.log(2.0 * math.pi)
               + 2.0 * np.log(sigma)
               + ((y - mu) / sigma) ** 2)
    ))


def empirical_coverage_at(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, alpha: float
) -> float:
    z = gaussian_z(alpha)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return float(np.mean((y >= lower) & (y <= upper)))


def calibration_table(
    df: pd.DataFrame,
    variant: str,
    split: str,
    alphas: tuple[float, ...] = CALIBRATION_ALPHAS,
) -> pd.DataFrame:
    """Empirical vs nominal coverage at a sweep of nominal targets."""
    mu = df["mu"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    y = df["iv_clean"].to_numpy(dtype=float)
    rows = [
        {
            "variant": variant,
            "split": split,
            "nominal_coverage": a,
            "empirical_coverage": empirical_coverage_at(mu, sigma, y, a),
            "n": int(len(y)),
        }
        for a in alphas
    ]
    return pd.DataFrame(rows)


def variant_metrics_and_curves(
    variant: str,
    frames: dict[str, pd.DataFrame],
    nominal_alpha: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the W1 evaluator + Gaussian NLL + calibration sweep for one variant.

    Returns (metrics_df, calibration_df, abstention_curve_df).
    """
    metric_rows: list[dict] = []
    calib_rows: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []

    model_name = f"3a_{variant}_gaussian"

    for split in SPLITS:
        df = frames[split]
        result = build_prediction_result(df, alpha=nominal_alpha)
        y_clean = df["iv_clean"].to_numpy(dtype=float)
        observed = df["observed"].to_numpy(dtype=bool)

        row = metrics_row(
            result, y_clean, observed,
            model_name=model_name, split=split,
        )
        row["gaussian_nll"] = gaussian_nll(
            df["mu"].to_numpy(dtype=float),
            df["sigma"].to_numpy(dtype=float),
            y_clean,
        )
        row["nominal_alpha"] = nominal_alpha
        metric_rows.append(row)

        calib_rows.append(calibration_table(df, variant=variant, split=split))
        curve_rows.append(risk_coverage_table(
            result, y_clean, model_name=model_name, split=split,
        ))

    metrics_df = metrics_table(metric_rows)
    calib_df = pd.concat(calib_rows, ignore_index=True)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    return metrics_df, calib_df, curve_df


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (Phase 2D.9 committed metrics)
# ─────────────────────────────────────────────────────────────────────────────

def load_2d9_baseline_rows(baseline_dir: Path, dataset: str) -> pd.DataFrame:
    """Load reference rows from Phase 2D.9 committed metrics_summary.csv files.

    Returns a long-format frame: columns ``variant``, ``split``, ``metric``,
    ``value``. The 2D.9 numbers are sourced from a 10-date capped slice
    (see phase2_result_memo.md §3); they are **reference rows**, not
    head-to-head on the same row count as 3A.3 (full fold).
    """
    refs = {
        "baseline_rbf_2d9": baseline_dir / dataset / "interpolation" / "metrics_summary.csv",
        "baseline_2d7_gaussian_calibrated_2d9":
            baseline_dir / dataset / "conditional_calibrated" / "metrics_summary.csv",
    }
    rows: list[dict] = []
    for variant, path in refs.items():
        if not path.exists():
            print(f"[warn] missing baseline file: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            split = r["split"]
            if split not in SPLITS:
                continue
            for col in ("mae", "coverage_90", "mean_width",
                        "hi_conf_mae_keep0.8"):
                if col in df.columns:
                    rows.append({
                        "variant": variant,
                        "split": split,
                        "metric": col,
                        "value": float(r[col]),
                        "source": str(path.relative_to(REPO_ROOT)),
                        "n": int(r["n"]),
                    })
    return pd.DataFrame(rows)


def variant_to_long(metrics_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Convert a wide W1 metrics frame to the comparison long format."""
    keep = [
        ("overall_mae", "mae"),
        ("interval_coverage", "coverage_90"),
        ("mean_interval_width", "mean_width"),
        ("hc_mae_keep0.8", "hi_conf_mae_keep0.8"),
        ("gaussian_nll", "gaussian_nll"),
    ]
    rows: list[dict] = []
    for _, r in metrics_df.iterrows():
        for src, out in keep:
            if src in metrics_df.columns and pd.notna(r[src]):
                rows.append({
                    "variant": f"3a_{variant}",
                    "split": r["split"],
                    "metric": out,
                    "value": float(r[src]),
                    "source": f"artifacts/runs/3A/{variant}/predictions_{r['split']}.parquet",
                    "n": int(r["n"]),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(
    runs_dir: Path,
    results_dir: Path,
    baseline_dir: Path,
    dataset: str,
) -> None:
    out_root = results_dir / dataset
    per_variant_metrics: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        print(f"[3A.4] evaluating variant={variant}")
        frames = load_variant_predictions(runs_dir, variant)
        metrics_df, calib_df, curve_df = variant_metrics_and_curves(
            variant=variant, frames=frames,
        )
        per_variant_metrics[variant] = metrics_df

        out_dir = out_root / f"3a_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
        calib_df.to_csv(out_dir / "calibration_table.csv", index=False)
        curve_df.to_csv(out_dir / "abstention_curve.csv", index=False)
        print(f"  wrote: {out_dir.relative_to(REPO_ROOT)}")

        print(metrics_df[["model", "split", "n", "overall_mae",
                          "interval_coverage", "mean_interval_width",
                          "hc_mae_keep0.8", "gaussian_nll"]].to_string(index=False))

    # Paired comparison.
    long_rows = [variant_to_long(per_variant_metrics[v], v) for v in VARIANTS]
    baseline_long = load_2d9_baseline_rows(baseline_dir, dataset)
    if not baseline_long.empty:
        long_rows.append(baseline_long)
    comparison = pd.concat(long_rows, ignore_index=True)

    compare_dir = out_root / "3a_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(compare_dir / "comparison.csv", index=False)
    print(f"  wrote: {(compare_dir / 'comparison.csv').relative_to(REPO_ROOT)}")

    # Print headline delta.
    pivot = (
        comparison[comparison["variant"].str.startswith("3a_")]
        .pivot_table(index=["split", "metric"], columns="variant", values="value")
    )
    print("\n[headline] 3A Fourier vs raw")
    print(pivot.to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        runs_dir=args.runs_dir,
        results_dir=args.results_dir,
        baseline_dir=args.baseline_dir,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
