"""Assemble uncertainty-evaluation metric tables (Phase 2 W1, story 2A.5).

Pure table-building over a :class:`PredictionResult` plus truth arrays — no I/O.
The runner (``scripts/run_uncertainty_eval.py``) calls these and owns
persistence (CSV / figure).

Two tables:
  - :func:`metrics_table` — one tidy row per predictor × split, combining point
    error, uncertainty/calibration metrics (or ``nan`` when the predictor
    carries no uncertainty), and abstention summaries (AURC + high-confidence
    MAE at fixed keep fractions).
  - :func:`risk_coverage_table` — long-form risk–coverage sweep, one row per
    coverage level, suitable for plotting.

Confidence convention
----------------------
Metrics that rank points by confidence (abstention, high-confidence MAE) need a
per-point confidence score (higher = more confident). We derive it from the
prediction:

- if ``result.uncertainty`` is present → ``confidence = -uncertainty``;
- otherwise → an **oracle** ranking ``confidence = -abs_error``.

The oracle case is labelled ``confidence_source="oracle_error"``. It is not a
real model signal — it characterises the *best achievable* abstention curve
given perfect ranking, a useful reference until W4 supplies real uncertainty.
Phase 1 baselines (interpolation, MLP) carry ``uncertainty=None``, so they use
the oracle source; interval coverage/width are reported as ``nan`` for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neural_iv_surface_inference.training.eval import mae, rmse
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.eval.uncertainty_metrics import (
    interval_coverage,
    mean_interval_width,
    error_uncertainty_correlation,
    high_confidence_mae,
)
from neural_iv_surface_inference.eval.abstention import (
    risk_coverage_curve,
    area_under_risk_coverage,
)

# Keep fractions reported as standalone high-confidence-MAE columns.
DEFAULT_KEEP_FRACTIONS: tuple[float, ...] = (0.5, 0.8)


def confidence_from_result(
    result: PredictionResult,
    abs_error: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Derive a per-point confidence score and its source label.

    Returns ``(confidence, source)`` where higher confidence = more reliable.
    See module docstring for the convention.
    """
    if result.uncertainty is not None:
        return -np.asarray(result.uncertainty, dtype=float), "uncertainty"
    return -np.asarray(abs_error, dtype=float), "oracle_error"


def metrics_row(
    result: PredictionResult,
    y_true: np.ndarray,
    observed: np.ndarray,
    model_name: str,
    split: str,
    keep_fractions: tuple[float, ...] = DEFAULT_KEEP_FRACTIONS,
    n_curve_points: int = 50,
) -> dict:
    """Build one tidy metrics row for a predictor on a single split.

    Columns: identifiers (``model``, ``split``, ``n``), point error
    (``overall_mae/rmse``, ``observed_mae``, ``unobserved_mae``), calibration
    (``interval_coverage``, ``mean_interval_width`` — ``nan`` without bounds),
    error–uncertainty correlation (``err_unc_pearson/spearman`` — ``nan``
    without an uncertainty signal), and abstention (``confidence_source``,
    ``aurc``, ``hc_mae_keep{frac}``).
    """
    pred = np.asarray(result.pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    abs_error = np.abs(pred - y_true)

    row: dict = {
        "model": model_name,
        "split": split,
        "n": int(len(pred)),
        "overall_mae": mae(pred, y_true),
        "overall_rmse": rmse(pred, y_true),
        "observed_mae": (
            mae(pred[observed], y_true[observed]) if observed.any() else float("nan")
        ),
        "unobserved_mae": (
            mae(pred[~observed], y_true[~observed]) if (~observed).any()
            else float("nan")
        ),
    }

    # Calibration metrics require interval bounds (None for Phase 1 baselines).
    if result.lower is not None and result.upper is not None:
        row["interval_coverage"] = interval_coverage(
            y_true, result.lower, result.upper
        )
        row["mean_interval_width"] = mean_interval_width(
            result.lower, result.upper
        )
    else:
        row["interval_coverage"] = float("nan")
        row["mean_interval_width"] = float("nan")

    # Error–uncertainty correlation requires a raw uncertainty signal.
    if result.uncertainty is not None:
        corr = error_uncertainty_correlation(abs_error, result.uncertainty)
    else:
        corr = {"pearson": float("nan"), "spearman": float("nan")}
    row["err_unc_pearson"] = corr["pearson"]
    row["err_unc_spearman"] = corr["spearman"]

    # Abstention summary.
    confidence, source = confidence_from_result(result, abs_error)
    row["confidence_source"] = source
    curve = risk_coverage_curve(abs_error, confidence, n_points=n_curve_points)
    row["aurc"] = area_under_risk_coverage(curve)
    for frac in keep_fractions:
        key = f"hc_mae_keep{frac:g}"
        row[key] = high_confidence_mae(abs_error, confidence, keep_fraction=frac)

    return row


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Stack metric rows (from :func:`metrics_row`) into a tidy DataFrame."""
    return pd.DataFrame(rows)


def risk_coverage_table(
    result: PredictionResult,
    y_true: np.ndarray,
    model_name: str,
    split: str,
    n_points: int = 50,
) -> pd.DataFrame:
    """Long-form risk–coverage sweep for one predictor × split (for plotting).

    Columns: ``model``, ``split``, ``confidence_source``, ``coverage``,
    ``retained_mae``, ``n_retained``. Empty (zero rows) if no finite points.
    """
    pred = np.asarray(result.pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    abs_error = np.abs(pred - y_true)

    confidence, source = confidence_from_result(result, abs_error)
    curve = risk_coverage_curve(abs_error, confidence, n_points=n_points)

    return pd.DataFrame({
        "model": model_name,
        "split": split,
        "confidence_source": source,
        "coverage": curve.coverage,
        "retained_mae": curve.retained_mae,
        "n_retained": curve.n_retained,
    })
