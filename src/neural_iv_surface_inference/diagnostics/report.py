"""Assemble W2 structure-diagnostic tables (Phase 2, story 2B.5).

Pure table-building over the W2 diagnostic outputs — no I/O. The runner
(``scripts/run_structure_diagnostics.py``) calls these and owns persistence
(CSV / figures).

The diagnostics are computed on a predictor's **predicted surface** (structural
validity is a property of the reconstruction, not the noisy observations), one
date at a time, since both the masking-sensitivity harness (2B.2) and the
no-arbitrage checks (2B.3) operate per single-date surface.

Two tables:
  - :func:`diagnostics_summary_table` — one tidy row per ``split × date`` with
    violation counts / rates / severities for each check, mean masking
    instability, and the risk-flag rate.
  - :func:`region_table` — long-form ``(maturity × moneyness)`` region grid
    (one row per cell) carrying mean risk score, mean instability, flag
    fraction, and point count, suitable for the heatmaps and downstream use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neural_iv_surface_inference.diagnostics.masking_sensitivity import (
    MaskingSensitivityResult,
    masking_sensitivity,
)
from neural_iv_surface_inference.diagnostics.no_arbitrage import no_arb_diagnostics
from neural_iv_surface_inference.diagnostics.risk_flags import (
    RiskFlagConfig,
    RiskFlagResult,
    bin_to_regions,
    derive_risk_flags,
)
from neural_iv_surface_inference.eval.predictor import Predictor

_CHECKS = ("calendar", "monotonicity", "convexity")
_PRED_COL = "iv_pred"

SUMMARY_COLUMNS = [
    "model", "split", "date", "n_points",
    "calendar_n_eval", "calendar_violations", "calendar_rate", "calendar_severity",
    "monotonicity_n_eval", "monotonicity_violations", "monotonicity_rate",
    "monotonicity_severity",
    "convexity_n_eval", "convexity_violations", "convexity_rate", "convexity_severity",
    "mean_instability", "risk_flag_rate",
]

REGION_COLUMNS = [
    "model", "split", "maturity", "moneyness",
    "mean_risk_score", "mean_instability", "flag_fraction", "n_points",
]


def diagnose_date(
    predictor: Predictor,
    df_date: pd.DataFrame,
    keep_fraction: float = 0.8,
    n_draws: int = 20,
    seed: int = 0,
    config: RiskFlagConfig | None = None,
) -> tuple[MaskingSensitivityResult, dict, RiskFlagResult, np.ndarray]:
    """Run the full W2 stack on one single-date surface.

    Returns ``(instability, diagnostics, flags, pred)`` where ``pred`` is the
    predictor's surface used for the structural checks. The no-arbitrage checks
    score the *predicted* surface (written to an ``iv_pred`` column), and the
    risk flags fold in masking instability via the per-point ``std``.
    """
    instab = masking_sensitivity(predictor, df_date, keep_fraction, n_draws, seed)
    pred = np.asarray(predictor.predict(df_date).pred, dtype=float)
    df_struct = df_date.copy()
    df_struct[_PRED_COL] = pred
    diag = no_arb_diagnostics(df_struct, iv_col=_PRED_COL)
    flags = derive_risk_flags(diag, instability=instab.std, config=config)
    return instab, diag, flags, pred


def diagnostics_summary_row(
    model: str,
    split: str,
    date,
    instab: MaskingSensitivityResult,
    diag: dict,
    flags: RiskFlagResult,
) -> dict:
    """Build one summary row for a single date's diagnostics."""
    row: dict = {
        "model": model,
        "split": split,
        "date": date,
        "n_points": diag[_CHECKS[0]].n_points,
    }
    for check in _CHECKS:
        r = diag[check]
        row[f"{check}_n_eval"] = r.n_evaluated
        row[f"{check}_violations"] = r.n_violations
        row[f"{check}_rate"] = r.rate
        row[f"{check}_severity"] = r.severity
    with np.errstate(invalid="ignore"):
        row["mean_instability"] = float(np.nanmean(instab.std)) if len(instab) else float("nan")
    row["risk_flag_rate"] = (
        float(np.mean(flags.no_arb_risk_flag)) if flags.n_points else float("nan")
    )
    return row


def diagnostics_summary_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble the per-date summary rows into a tidy DataFrame."""
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def region_table(
    model: str,
    split: str,
    risk_score: np.ndarray,
    instability: np.ndarray,
    flag: np.ndarray,
    log_moneyness: np.ndarray,
    tau: np.ndarray,
) -> pd.DataFrame:
    """Long-form (maturity × moneyness) region grid for one split.

    Aggregates the per-point arrays (pooled across all dates in the split) onto
    the canonical bucket grid: mean risk score, mean instability, flag fraction,
    and point count, one row per cell.
    """
    g_risk = bin_to_regions(risk_score, log_moneyness, tau, agg="mean")
    g_inst = bin_to_regions(instability, log_moneyness, tau, agg="mean")
    g_flag = bin_to_regions(
        np.asarray(flag, dtype=float), log_moneyness, tau, agg="fraction"
    )
    g_n = bin_to_regions(
        np.ones(len(risk_score)), log_moneyness, tau, agg="sum"
    )

    rows: list[dict] = []
    for mat in g_risk.index:
        for mon in g_risk.columns:
            n_cell = g_n.loc[mat, mon]
            rows.append({
                "model": model,
                "split": split,
                "maturity": mat,
                "moneyness": mon,
                "mean_risk_score": g_risk.loc[mat, mon],
                "mean_instability": g_inst.loc[mat, mon],
                "flag_fraction": g_flag.loc[mat, mon],
                "n_points": 0 if np.isnan(n_cell) else int(n_cell),
            })
    return pd.DataFrame(rows)[REGION_COLUMNS]
