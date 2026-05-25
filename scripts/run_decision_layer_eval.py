#!/usr/bin/env python3
"""Decision-layer evaluation runner (story 2D.6 — local skeleton).

Wires the W1 uncertainty evaluator (2A.5), the W2 structure diagnostics
(2B.5), the calibrated conditional predictor (2D.4), and the decision layer
(2D.5) into a single command that scores one or more
``(dataset, predictor)`` pairs and writes auditable artifacts under
``<results_root>/<dataset>/<predictor>/`` plus a top-level
``<results_root>/comparison_summary.csv``.

This is the **skeleton** that 2D.9 will swap real AV benchmarks and
calibrated checkpoints into. The synthetic smoke
(``tests/test_decision_layer_runner.py``) drives the importable helpers
below with in-process objects and asserts every expected file is written.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

# Ensure src/ is importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml

from neural_iv_surface_inference.eval.predictor import Predictor, PredictionResult
from neural_iv_surface_inference.eval.decision_layer import (
    DecisionConfig,
    DecisionResult,
    apply_decision_layer,
)
from neural_iv_surface_inference.eval.uncertainty_metrics import (
    interval_coverage,
    mean_interval_width,
    high_confidence_mae,
)
from neural_iv_surface_inference.eval.abstention import risk_coverage_curve
from neural_iv_surface_inference.diagnostics.report import diagnose_date
from neural_iv_surface_inference.diagnostics.risk_flags import RiskFlagConfig
from neural_iv_surface_inference.diagnostics.no_arbitrage import ViolationResult
from neural_iv_surface_inference.diagnostics.risk_flags import bin_to_regions
from neural_iv_surface_inference.training.eval import mae as _mae

DEFAULT_NOMINAL_COVERAGE = 0.90

# Columns of the top-level comparison summary CSV (single source of truth).
COMPARISON_COLUMNS: tuple[str, ...] = (
    "dataset",
    "predictor",
    "test_mae",
    "hi_conf_mae",
    "coverage_90",
    "mean_width",
    "abstain_rate",
    "mean_tradability",
    "n_forbidden_flag_violations",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsBudget:
    keep_fraction: float = 0.8
    n_draws: int = 10
    seed: int = 0
    max_dates_per_split: int | None = 20


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    predictor: str
    benchmark_path: str | None = None
    checkpoint: str | None = None
    calibrator_path: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    results_root: Path
    decision_config_path: Path
    diagnostics: DiagnosticsBudget
    nominal_coverage: float
    pairs: tuple[PairSpec, ...]


def load_eval_config(path: str | Path) -> EvalConfig:
    """Parse the runner YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    diag_raw = raw.get("diagnostics", {}) or {}
    diag = DiagnosticsBudget(
        keep_fraction=float(diag_raw.get("keep_fraction", 0.8)),
        n_draws=int(diag_raw.get("n_draws", 10)),
        seed=int(diag_raw.get("seed", 0)),
        max_dates_per_split=diag_raw.get("max_dates_per_split"),
    )
    pairs = tuple(
        PairSpec(
            dataset=p["dataset"],
            predictor=p["predictor"],
            benchmark_path=p.get("benchmark_path"),
            checkpoint=p.get("checkpoint"),
            calibrator_path=p.get("calibrator_path"),
            extra={
                k: v for k, v in p.items()
                if k not in {"dataset", "predictor", "benchmark_path",
                             "checkpoint", "calibrator_path"}
            },
        )
        for p in (raw.get("pairs") or [])
    )
    return EvalConfig(
        results_root=Path(raw.get("results_root", "results/2D")),
        decision_config_path=Path(
            raw.get("decision_config", "configs/decision_layer.yaml")
        ),
        diagnostics=diag,
        nominal_coverage=float(raw.get("nominal_coverage", DEFAULT_NOMINAL_COVERAGE)),
        pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{split: frame}`` for the splits present in ``df``."""
    return {
        s: df[df["split"] == s].reset_index(drop=True)
        for s in df["split"].unique()
    }


def _cap_dates(df: pd.DataFrame, max_dates: int | None) -> pd.DataFrame:
    if max_dates is None:
        return df
    keep = df["date"].drop_duplicates().head(max_dates)
    return df[df["date"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Default no-arb flag computation (used in production; tests inject a stub).
# ---------------------------------------------------------------------------


FlagsFn = Callable[[Predictor, pd.DataFrame, DiagnosticsBudget], Mapping[str, np.ndarray]]


def compute_no_arb_flags(
    predictor: Predictor,
    df_split: pd.DataFrame,
    budget: DiagnosticsBudget,
) -> dict[str, np.ndarray]:
    """Run the W2 stack date-by-date and project per-check point masks to the split.

    Returns a mapping with the three structural-flag arrays expected by
    :func:`apply_decision_layer`:

      - ``calendar_violation``
      - ``monotonicity_violation``
      - ``convexity_violation``

    Plus the union ``no_arb_risk_flag`` for completeness.
    """
    n = len(df_split)
    flags = {
        "calendar_violation": np.zeros(n, dtype=bool),
        "monotonicity_violation": np.zeros(n, dtype=bool),
        "convexity_violation": np.zeros(n, dtype=bool),
        "no_arb_risk_flag": np.zeros(n, dtype=bool),
    }
    if n == 0:
        return flags
    # Recover row indices per date via cumulative offsets along the (already
    # date-grouped within frame) split. We rely on df_split's own index.
    config = RiskFlagConfig()
    for date, df_date in df_split.groupby("date", sort=False):
        idx = df_date.index.to_numpy()
        df_date_r = df_date.reset_index(drop=True)
        _, diag, flags_res, _ = diagnose_date(
            predictor, df_date_r,
            budget.keep_fraction, budget.n_draws, budget.seed, config,
        )
        for name, key in (
            ("calendar_violation", "calendar"),
            ("monotonicity_violation", "monotonicity"),
            ("convexity_violation", "convexity"),
        ):
            r: ViolationResult = diag[key]
            flags[name][idx] = r.point_mask
        flags["no_arb_risk_flag"][idx] = flags_res.no_arb_risk_flag
    return flags


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairResult:
    dataset: str
    predictor: str
    out_dir: Path
    per_row: pd.DataFrame
    metrics_summary: pd.DataFrame
    region_table: pd.DataFrame
    comparison_row: dict


def evaluate_pair(
    predictor: Predictor,
    df_splits: dict[str, pd.DataFrame],
    dataset: str,
    predictor_name: str,
    decision_config: DecisionConfig,
    results_root: Path,
    diagnostics_budget: DiagnosticsBudget | None = None,
    flags_fn: FlagsFn | None = None,
    nominal_coverage: float = DEFAULT_NOMINAL_COVERAGE,
) -> PairResult:
    """Run W1 + W2 + 2D.5 on one ``(dataset, predictor)`` pair, write artifacts.

    The ``flags_fn`` hook lets the synthetic smoke supply stubbed risk flags
    without invoking the full W2 stack; production callers can leave it
    ``None`` to use :func:`compute_no_arb_flags`.
    """
    budget = diagnostics_budget or DiagnosticsBudget()
    flags_fn = flags_fn or compute_no_arb_flags
    out_dir = Path(results_root) / dataset / predictor_name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    region_frames: list[pd.DataFrame] = []
    comparison_row = {
        "dataset": dataset,
        "predictor": predictor_name,
        "test_mae": float("nan"),
        "hi_conf_mae": float("nan"),
        "coverage_90": float("nan"),
        "mean_width": float("nan"),
        "abstain_rate": float("nan"),
        "mean_tradability": float("nan"),
        "n_forbidden_flag_violations": 0,
    }

    for split, df in df_splits.items():
        if df is None or len(df) == 0:
            continue
        df = _cap_dates(df, budget.max_dates_per_split)
        result = predictor.predict(df)
        y_true = df["iv_clean"].to_numpy(dtype=float)
        abs_err = np.abs(np.asarray(result.pred, dtype=float) - y_true)

        no_arb_flags = flags_fn(predictor, df, budget)
        decision = apply_decision_layer(result, no_arb_flags, decision_config)

        per_row = _per_row_frame(
            df, result, decision, no_arb_flags,
            split=split, dataset=dataset, predictor_name=predictor_name,
        )
        per_row_frames.append(per_row)

        metrics_rows.append(_metrics_row(
            split=split, y_true=y_true, abs_err=abs_err,
            result=result, decision=decision, no_arb_flags=no_arb_flags,
            nominal_coverage=nominal_coverage,
        ))

        region_frames.append(_region_frame(
            split=split, dataset=dataset, predictor_name=predictor_name,
            df=df, decision=decision,
        ))

        if split == "test":
            comparison_row.update(_comparison_update(
                y_true=y_true, abs_err=abs_err,
                result=result, decision=decision, no_arb_flags=no_arb_flags,
                nominal_coverage=nominal_coverage,
            ))

    per_row_df = (
        pd.concat(per_row_frames, ignore_index=True)
        if per_row_frames else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metrics_rows)
    region_df = (
        pd.concat(region_frames, ignore_index=True)
        if region_frames else pd.DataFrame()
    )

    # Persist.
    per_row_path = out_dir / "predictions_decisions.csv"
    metrics_path = out_dir / "metrics_summary.csv"
    region_path = out_dir / "region_tradability.csv"
    per_row_df.to_csv(per_row_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    region_df.to_csv(region_path, index=False)

    _save_abstention_curve(per_row_df, out_dir / "abstention_curve.png")
    _save_calibration_plot(per_row_df, out_dir / "calibration_plot.png",
                           nominal_coverage=nominal_coverage)

    return PairResult(
        dataset=dataset,
        predictor=predictor_name,
        out_dir=out_dir,
        per_row=per_row_df,
        metrics_summary=metrics_df,
        region_table=region_df,
        comparison_row=comparison_row,
    )


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------


def _per_row_frame(
    df: pd.DataFrame,
    result: PredictionResult,
    decision: DecisionResult,
    no_arb_flags: Mapping[str, np.ndarray],
    split: str,
    dataset: str,
    predictor_name: str,
) -> pd.DataFrame:
    n = len(df)
    confidence = np.asarray(result.meta.get("confidence_score"), dtype=float)
    base = {
        "dataset": dataset,
        "predictor": predictor_name,
        "split": split,
        "date": df["date"].to_numpy(),
        "log_moneyness": df["log_moneyness"].to_numpy(),
        "tau": df["tau"].to_numpy(),
        "iv_true": df["iv_clean"].to_numpy(dtype=float),
        "iv_pred": np.asarray(result.pred, dtype=float),
        "uncertainty": (
            np.asarray(result.uncertainty, dtype=float)
            if result.uncertainty is not None else np.full(n, np.nan)
        ),
        "lower": (
            np.asarray(result.lower, dtype=float)
            if result.lower is not None else np.full(n, np.nan)
        ),
        "upper": (
            np.asarray(result.upper, dtype=float)
            if result.upper is not None else np.full(n, np.nan)
        ),
        "confidence_score": confidence,
        "abstain_flag": decision.abstain_flag,
        "tradability_score": decision.tradability_score,
        "decision_reason": decision.decision_reason,
    }
    for name in ("calendar_violation", "monotonicity_violation",
                 "convexity_violation"):
        arr = no_arb_flags.get(name)
        base[name] = (
            np.asarray(arr, dtype=bool)
            if arr is not None else np.zeros(n, dtype=bool)
        )
    return pd.DataFrame(base)


def _metrics_row(
    split: str,
    y_true: np.ndarray,
    abs_err: np.ndarray,
    result: PredictionResult,
    decision: DecisionResult,
    no_arb_flags: Mapping[str, np.ndarray],
    nominal_coverage: float,
) -> dict:
    confidence = np.asarray(result.meta.get("confidence_score"), dtype=float)
    n = len(y_true)
    cov = (
        interval_coverage(y_true, result.lower, result.upper)
        if result.lower is not None and result.upper is not None
        else float("nan")
    )
    width = (
        mean_interval_width(result.lower, result.upper)
        if result.lower is not None and result.upper is not None
        else float("nan")
    )
    hi_conf = high_confidence_mae(abs_err, confidence, keep_fraction=0.8)
    forbidden_n = 0
    for name in ("calendar_violation", "convexity_violation"):
        a = no_arb_flags.get(name)
        if a is not None:
            forbidden_n += int(np.asarray(a, dtype=bool).sum())
    return {
        "split": split,
        "n": int(n),
        "mae": float(_mae(np.asarray(result.pred, dtype=float), y_true)),
        f"coverage_{int(nominal_coverage * 100)}": cov,
        "mean_width": width,
        "hi_conf_mae_keep0.8": hi_conf,
        "abstain_rate": float(np.mean(decision.abstain_flag)),
        "mean_tradability": float(np.mean(decision.tradability_score)),
        "n_forbidden_flag_violations": forbidden_n,
    }


def _region_frame(
    split: str,
    dataset: str,
    predictor_name: str,
    df: pd.DataFrame,
    decision: DecisionResult,
) -> pd.DataFrame:
    """Aggregate ``tradability_score`` + abstain rate on the canonical region grid."""
    log_m = df["log_moneyness"].to_numpy(dtype=float)
    tau = df["tau"].to_numpy(dtype=float)
    trad = bin_to_regions(decision.tradability_score, log_m, tau, agg="mean")
    abstain = bin_to_regions(
        np.asarray(decision.abstain_flag, dtype=float), log_m, tau,
        agg="fraction",
    )
    n_grid = bin_to_regions(np.ones(len(log_m)), log_m, tau, agg="sum")
    rows = []
    for mat in trad.index:
        for mon in trad.columns:
            rows.append({
                "dataset": dataset,
                "predictor": predictor_name,
                "split": split,
                "maturity": mat,
                "moneyness": mon,
                "mean_tradability": trad.loc[mat, mon],
                "abstain_rate": abstain.loc[mat, mon],
                "n_points": n_grid.loc[mat, mon],
            })
    return pd.DataFrame(rows)


def _comparison_update(
    y_true: np.ndarray,
    abs_err: np.ndarray,
    result: PredictionResult,
    decision: DecisionResult,
    no_arb_flags: Mapping[str, np.ndarray],
    nominal_coverage: float,
) -> dict:
    confidence = np.asarray(result.meta.get("confidence_score"), dtype=float)
    forbidden_n = 0
    for name in ("calendar_violation", "convexity_violation"):
        a = no_arb_flags.get(name)
        if a is not None:
            forbidden_n += int(np.asarray(a, dtype=bool).sum())
    return {
        "test_mae": float(_mae(np.asarray(result.pred, dtype=float), y_true)),
        "hi_conf_mae": high_confidence_mae(abs_err, confidence, keep_fraction=0.8),
        "coverage_90": (
            interval_coverage(y_true, result.lower, result.upper)
            if result.lower is not None and result.upper is not None
            else float("nan")
        ),
        "mean_width": (
            mean_interval_width(result.lower, result.upper)
            if result.lower is not None and result.upper is not None
            else float("nan")
        ),
        "abstain_rate": float(np.mean(decision.abstain_flag)),
        "mean_tradability": float(np.mean(decision.tradability_score)),
        "n_forbidden_flag_violations": forbidden_n,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_abstention_curve(per_row: pd.DataFrame, path: Path) -> None:
    """Plot risk–coverage by split using the calibrated confidence score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if not per_row.empty:
        for split, grp in per_row.groupby("split"):
            ae = np.abs(grp["iv_pred"].to_numpy() - grp["iv_true"].to_numpy())
            curve = risk_coverage_curve(
                ae, grp["confidence_score"].to_numpy(), n_points=30,
            )
            ax.plot(curve.coverage, curve.retained_mae, marker="o",
                    markersize=3, label=str(split))
    ax.set_xlabel("Coverage (retained fraction)")
    ax.set_ylabel("Retained MAE")
    ax.set_title("Abstention curve (confidence = 2D.4)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="split")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_calibration_plot(
    per_row: pd.DataFrame, path: Path, nominal_coverage: float,
) -> None:
    """Reliability diagram: empirical coverage of bands at sweep of widths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if not per_row.empty:
        for split, grp in per_row.groupby("split"):
            y = grp["iv_true"].to_numpy()
            mu = grp["iv_pred"].to_numpy()
            lo = grp["lower"].to_numpy()
            hi = grp["upper"].to_numpy()
            if not np.isfinite(lo).any() or not np.isfinite(hi).any():
                continue
            half = (hi - lo) / 2.0
            sigma = np.maximum(half, 1e-9)
            z = (y - mu) / sigma
            # Sweep nominal coverage from 0.05 to 0.99 -> empirical share inside.
            nominal = np.linspace(0.05, 0.99, 25)
            from scipy.stats import norm
            emp = []
            for q in nominal:
                z_q = norm.ppf(0.5 + q / 2.0)
                emp.append(float(np.mean(np.abs(z) <= z_q)))
            ax.plot(nominal, emp, marker="o", markersize=3, label=str(split))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.axvline(nominal_coverage, color="grey", alpha=0.3, linewidth=1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration (reliability diagram)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="split")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def write_comparison_summary(rows: list[dict], path: Path) -> Path:
    df = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_from_config(
    eval_config: EvalConfig,
    results_root: Path | None = None,
    predictor_factory: Callable[[PairSpec], tuple[Predictor, dict[str, pd.DataFrame]]] | None = None,
) -> list[PairResult]:
    """Iterate over `pairs` and write all artifacts. Returns per-pair results.

    ``predictor_factory`` must yield ``(predictor, df_splits)`` for each pair.
    The default implementation requires the pair to point at a real benchmark
    Parquet and a supported predictor name — only the local skeleton is
    exercised here; story 2D.9 supplies the real adapters.
    """
    if predictor_factory is None:
        raise ValueError(
            "run_from_config requires a predictor_factory; 2D.9 wires the real "
            "adapters. Pass --synthetic in 2D.6 smoke tests to bypass this."
        )

    decision_config = DecisionConfig.from_yaml(eval_config.decision_config_path)
    root = Path(results_root or eval_config.results_root)
    root.mkdir(parents=True, exist_ok=True)

    pair_results: list[PairResult] = []
    comparison_rows: list[dict] = []
    for spec in eval_config.pairs:
        predictor, df_splits = predictor_factory(spec)
        pr = evaluate_pair(
            predictor=predictor,
            df_splits=df_splits,
            dataset=spec.dataset,
            predictor_name=spec.predictor,
            decision_config=decision_config,
            results_root=root,
            diagnostics_budget=eval_config.diagnostics,
            nominal_coverage=eval_config.nominal_coverage,
        )
        pair_results.append(pr)
        comparison_rows.append(pr.comparison_row)

    write_comparison_summary(comparison_rows, root / "comparison_summary.csv")
    return pair_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run decision-layer evaluation (W1 + W2 + 2D.5)"
    )
    parser.add_argument("--config", type=str,
                        default="configs/decision_layer_eval.yaml")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Override `results_root` from the config "
                             "(synthetic smoke writes to a temp dir).")
    args = parser.parse_args()

    eval_config = load_eval_config(args.config)
    print(f"[config] pairs={len(eval_config.pairs)} "
          f"decision_config={eval_config.decision_config_path}")
    if not eval_config.pairs:
        print("[config] No pairs configured — nothing to do. "
              "Fill `pairs:` in the config (story 2D.9).")
        return
    run_from_config(eval_config, results_root=Path(args.results_root)
                    if args.results_root else None)


if __name__ == "__main__":
    main()
