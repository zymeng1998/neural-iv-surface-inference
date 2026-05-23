#!/usr/bin/env python3
"""Run the W2 structure-diagnostics pipeline end-to-end (story 2B.5).

Wires the masking-sensitivity harness (2B.2), no-arbitrage diagnostics (2B.3),
and risk-flag synthesis + region heatmaps (2B.4) into one runner that evaluates
a predictor on a benchmark and writes committed artifacts:

  - ``artifacts/results/structure_diagnostics_<tag>.csv``         (per-date summary)
  - ``artifacts/results/structure_diagnostics_<tag>_regions.csv`` (region grid)
  - ``artifacts/results/structure_diagnostics_<tag>_<split>_risk.png``
  - ``artifacts/results/structure_diagnostics_<tag>_<split>_instability.png``

Diagnostics are computed on the predictor's *predicted* surface, per date. The
synthetic benchmark is a smooth arbitrage-free grid, so structural violations
are ~zero by construction — the synthetic run validates wiring + artifact shape,
not a research result. The real benchmark run happens on RunPod.

Usage:
    # Tiny synthetic grid benchmark (local / CI, no data files needed):
    python scripts/run_structure_diagnostics.py --synthetic --predictor interp \
        --tag synthetic_demo

    # Real benchmark (where the parquet exists, e.g. RunPod):
    python scripts/run_structure_diagnostics.py --benchmark path/to/bench.parquet \
        --predictor interp --method rbf --tag spy_random40_noiselow
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
from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
    InterpolationPredictor,
)
from neural_iv_surface_inference.diagnostics.report import (
    diagnose_date,
    diagnostics_summary_row,
    diagnostics_summary_table,
    region_table,
)
from neural_iv_surface_inference.diagnostics.risk_flags import RiskFlagConfig
from neural_iv_surface_inference.viz.diagnostic_plots import (
    plot_instability_heatmap,
    plot_risk_region_heatmap,
)
from neural_iv_surface_inference.viz.style import apply_house_style

DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_synthetic_benchmark(
    n_dates: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Tiny synthetic **grid** benchmark with the standard Step-4 columns.

    Unlike the W1 runner's random-scatter synthetic, this lays points on a shared
    ``(log_moneyness, tau)`` grid per date so the no-arbitrage checks (which group
    by exact coordinate) have evaluable pairs/triples. The smile is smooth and
    arbitrage-free, so structural violations are ~zero — the run validates the
    full predictor → diagnostics → artifacts path, not a research finding.
    """
    rng = np.random.default_rng(seed)
    ks = np.linspace(-0.3, 0.3, 7)
    taus = np.array([7, 30, 90, 180, 365], dtype=float) / 365.0
    grid_k, grid_t = np.meshgrid(ks, taus)
    grid_k = grid_k.ravel()
    grid_t = grid_t.ravel()
    per_date = len(grid_k)

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    frames = []
    for d in dates:
        iv_clean = np.clip(0.2 + 0.15 * grid_k**2 + 0.02 * grid_t, 0.01, 3.0)
        observed = rng.random(per_date) < 0.5
        iv_noisy = iv_clean.copy()
        iv_noisy[observed] += rng.normal(0, 0.01, size=per_date)[observed]
        iv_noisy = np.clip(iv_noisy, 1e-4, 10.0)
        frames.append(pd.DataFrame({
            "date": d,
            "tau": grid_t,
            "log_moneyness": grid_k,
            "iv_clean": iv_clean,
            "implied_volatility": iv_noisy,
            "observed": observed,
            "noise_sigma": np.full(per_date, 0.01),
        }))
    df = pd.concat(frames, ignore_index=True)

    split = np.array(["test"] * len(df), dtype=object)
    n_train = dates[int(n_dates * 0.7)]
    n_val = dates[int(n_dates * 0.85)]
    split[df["date"].to_numpy() < n_train] = "train"
    split[(df["date"].to_numpy() >= n_train) & (df["date"].to_numpy() < n_val)] = "val"
    df["split"] = split
    return df


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {split_name: frame} for the splits present in ``df``."""
    return {
        s: df[df["split"] == s].reset_index(drop=True)
        for s in df["split"].unique()
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(
    predictor: Predictor,
    df_splits: dict[str, pd.DataFrame],
    model_name: str,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    keep_fraction: float = 0.8,
    n_draws: int = 20,
    seed: int = 0,
    config: RiskFlagConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the W2 stack per date; return (per-date summary, region grid) tables."""
    summary_rows: list[dict] = []
    region_frames: list[pd.DataFrame] = []

    for split in splits:
        df = df_splits.get(split)
        if df is None or len(df) == 0:
            continue

        pooled_risk: list[np.ndarray] = []
        pooled_inst: list[np.ndarray] = []
        pooled_flag: list[np.ndarray] = []
        pooled_k: list[np.ndarray] = []
        pooled_t: list[np.ndarray] = []

        for date, df_date in df.groupby("date"):
            df_date = df_date.reset_index(drop=True)
            instab, diag, flags, _ = diagnose_date(
                predictor, df_date, keep_fraction, n_draws, seed, config
            )
            summary_rows.append(
                diagnostics_summary_row(model_name, split, date, instab, diag, flags)
            )
            pooled_risk.append(flags.risk_score)
            pooled_inst.append(instab.std)
            pooled_flag.append(flags.no_arb_risk_flag)
            pooled_k.append(df_date["log_moneyness"].to_numpy())
            pooled_t.append(df_date["tau"].to_numpy())

        region_frames.append(region_table(
            model_name, split,
            np.concatenate(pooled_risk),
            np.concatenate(pooled_inst),
            np.concatenate(pooled_flag),
            np.concatenate(pooled_k),
            np.concatenate(pooled_t),
        ))

    summary_df = diagnostics_summary_table(summary_rows)
    region_df = (
        pd.concat(region_frames, ignore_index=True)
        if region_frames else pd.DataFrame()
    )
    return summary_df, region_df


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def _region_pivot(region_df: pd.DataFrame, split: str, value: str) -> pd.DataFrame:
    """Pivot the long region table back to a (maturity × moneyness) grid."""
    from neural_iv_surface_inference.training.eval import (
        MONEYNESS_BUCKETS,
        TAU_BUCKETS,
    )
    sub = region_df[region_df["split"] == split]
    grid = sub.pivot(index="maturity", columns="moneyness", values=value)
    return grid.reindex(index=list(TAU_BUCKETS), columns=list(MONEYNESS_BUCKETS))


def write_artifacts(
    summary_df: pd.DataFrame,
    region_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> dict[str, Path]:
    """Write summary CSV, region CSV, and per-split region heatmaps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_house_style()

    summary_path = out_dir / f"structure_diagnostics_{tag}.csv"
    region_path = out_dir / f"structure_diagnostics_{tag}_regions.csv"
    summary_df.to_csv(summary_path, index=False)
    region_df.to_csv(region_path, index=False)

    paths: dict[str, Path] = {"summary": summary_path, "regions": region_path}
    if not region_df.empty:
        for split in region_df["split"].unique():
            risk_grid = _region_pivot(region_df, split, "mean_risk_score")
            inst_grid = _region_pivot(region_df, split, "mean_instability")
            risk_fig = plot_risk_region_heatmap(
                risk_grid, title=f"No-arb risk ({split})"
            )
            inst_fig = plot_instability_heatmap(
                inst_grid, title=f"Masking instability ({split})"
            )
            risk_p = out_dir / f"structure_diagnostics_{tag}_{split}_risk.png"
            inst_p = out_dir / f"structure_diagnostics_{tag}_{split}_instability.png"
            risk_fig.savefig(risk_p, dpi=120)
            inst_fig.savefig(inst_p, dpi=120)
            import matplotlib.pyplot as plt
            plt.close(risk_fig)
            plt.close(inst_fig)
            paths[f"risk_{split}"] = risk_p
            paths[f"instability_{split}"] = inst_p
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_predictor(
    name: str, method: str, checkpoint: str | None = None
) -> tuple[Predictor, str]:
    """Construct a predictor adapter and its model label.

    Interpolation baseline for the no-data / CI path; conditional surface
    predictor (2C.5) via a 2C.4 checkpoint. The MLP baseline still requires
    explicit checkpoint loading and is not wired through this CLI.
    """
    if name == "interp":
        return InterpolationPredictor(method=method), f"interp_{method}"
    if name == "conditional":
        if not checkpoint:
            raise ValueError(
                "predictor 'conditional' requires --checkpoint <path to best_conditional.pt>"
            )
        return (
            ConditionalSurfacePredictor.from_checkpoint(checkpoint),
            "conditional_surface",
        )
    raise ValueError(
        f"predictor '{name}' not supported by this runner yet (use 'interp' or 'conditional')"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run W2 structure diagnostics")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path to a benchmark Parquet (Step 4 output)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a tiny synthetic grid benchmark (no data files)")
    parser.add_argument("--predictor", type=str, default="interp",
                        choices=["interp", "conditional"])
    parser.add_argument("--method", type=str, default="rbf",
                        help="Interpolation method (linear|cubic|rbf|nearest)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a conditional-model checkpoint "
                             "(required when --predictor conditional)")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to evaluate")
    parser.add_argument("--out-dir", type=str, default="artifacts/results")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--keep-fraction", type=float, default=0.8,
                        help="Masking-sensitivity observed keep fraction")
    parser.add_argument("--n-draws", type=int, default=20,
                        help="Masking-sensitivity mask draws per date")
    parser.add_argument("--instability-threshold", type=float, default=float("inf"),
                        help="Flag a point when instability exceeds this")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Cap dates per split for runtime (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    if args.synthetic:
        print("[data] Using synthetic grid benchmark")
        df = make_synthetic_benchmark()
    elif args.benchmark:
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.exists():
            print(f"Benchmark not found: {benchmark_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[data] Loading benchmark: {benchmark_path}")
        df = pd.read_parquet(benchmark_path)
    else:
        print("Provide --benchmark PATH or --synthetic", file=sys.stderr)
        sys.exit(1)

    df_splits = split_frames(df)
    if args.max_dates is not None:
        for s, frame in df_splits.items():
            keep_dates = frame["date"].drop_duplicates().head(args.max_dates)
            df_splits[s] = frame[frame["date"].isin(keep_dates)].reset_index(drop=True)

    predictor, model_name = build_predictor(
        args.predictor, args.method, checkpoint=args.checkpoint
    )
    config = RiskFlagConfig(instability_threshold=args.instability_threshold)
    print(f"[run] predictor={model_name}, splits={splits}, "
          f"keep_fraction={args.keep_fraction}, n_draws={args.n_draws}")

    summary_df, region_df = run_diagnostics(
        predictor, df_splits, model_name=model_name, splits=splits,
        keep_fraction=args.keep_fraction, n_draws=args.n_draws,
        seed=args.seed, config=config,
    )

    paths = write_artifacts(summary_df, region_df, Path(args.out_dir), args.tag)

    print("\n[summary]")
    print(summary_df.to_string(index=False))
    print("\n[artifacts]")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
