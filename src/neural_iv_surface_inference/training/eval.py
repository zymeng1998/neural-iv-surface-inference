"""Evaluation utilities for IV surface models.

Computes metrics on predictions vs ground truth (iv_clean), split by:
  - overall
  - observed vs unobserved points
  - maturity buckets (short / medium / long)
  - moneyness buckets (ITM / ATM / OTM)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def mape(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute percentage error (clamp target away from zero)."""
    denom = np.maximum(np.abs(target), 1e-6)
    return float((np.abs(pred - target) / denom).mean() * 100)


# ---------------------------------------------------------------------------
# Regional bucketing
# ---------------------------------------------------------------------------

TAU_BUCKETS = {
    "short": (0, 30 / 365),       # < 30 days
    "medium": (30 / 365, 180 / 365),  # 30 days – 6 months
    "long": (180 / 365, float("inf")),
}

MONEYNESS_BUCKETS = {
    "deep_itm": (-float("inf"), -0.2),
    "itm": (-0.2, -0.05),
    "atm": (-0.05, 0.05),
    "otm": (0.05, 0.2),
    "deep_otm": (0.2, float("inf")),
}


def _bucket_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
    buckets: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Compute metrics within each bucket of *values*."""
    results = {}
    for name, (lo, hi) in buckets.items():
        mask = (values >= lo) & (values < hi)
        n = mask.sum()
        if n == 0:
            results[name] = {"mae": float("nan"), "rmse": float("nan"), "n": 0}
        else:
            results[name] = {
                "mae": mae(pred[mask], target[mask]),
                "rmse": rmse(pred[mask], target[mask]),
                "n": int(n),
            }
    return results


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(df: pd.DataFrame) -> dict:
    """Compute full evaluation metrics on a DataFrame with predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``iv_pred``, ``iv_clean``, ``observed``,
        ``log_moneyness``, ``tau``.

    Returns
    -------
    dict with nested metric groups:
        overall, observed, unobserved, by_maturity, by_moneyness
    """
    pred = df["iv_pred"].values
    target = df["iv_clean"].values
    obs_mask = df["observed"].values
    tau = df["tau"].values
    log_m = df["log_moneyness"].values

    results = {
        "overall": {
            "mae": mae(pred, target),
            "rmse": rmse(pred, target),
            "mape": mape(pred, target),
            "n": len(pred),
        },
    }

    # Observed vs unobserved
    for subset_name, mask in [("observed", obs_mask), ("unobserved", ~obs_mask)]:
        n = mask.sum()
        if n > 0:
            results[subset_name] = {
                "mae": mae(pred[mask], target[mask]),
                "rmse": rmse(pred[mask], target[mask]),
                "mape": mape(pred[mask], target[mask]),
                "n": int(n),
            }
        else:
            results[subset_name] = {
                "mae": float("nan"), "rmse": float("nan"),
                "mape": float("nan"), "n": 0,
            }

    # By maturity
    results["by_maturity"] = _bucket_metrics(pred, target, tau, TAU_BUCKETS)

    # By moneyness
    results["by_moneyness"] = _bucket_metrics(pred, target, log_m, MONEYNESS_BUCKETS)

    return results


def evaluate_predictions_2d(
    df: pd.DataFrame,
    metric: str = "mae",
) -> pd.DataFrame:
    """Joint maturity x moneyness error grid (for a true 2D heatmap).

    ``evaluate_predictions`` reports maturity and moneyness as independent
    marginals; this computes the *joint* cell error so a real
    (maturity x moneyness) heatmap can be plotted.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``iv_pred``, ``iv_clean``, ``tau``, ``log_moneyness``.
    metric : str
        ``"mae"`` or ``"rmse"``.

    Returns
    -------
    pd.DataFrame
        Indexed by maturity bucket (rows, in ``TAU_BUCKETS`` order), columns =
        moneyness buckets (in ``MONEYNESS_BUCKETS`` order). Each value is the
        cell error; empty cells are ``nan``. A companion count grid is available
        via :func:`evaluate_predictions_2d_counts`.
    """
    if metric not in ("mae", "rmse"):
        raise ValueError(f"metric must be 'mae' or 'rmse', got {metric!r}")
    err_fn = mae if metric == "mae" else rmse

    pred = df["iv_pred"].values
    target = df["iv_clean"].values
    tau = df["tau"].values
    log_m = df["log_moneyness"].values

    grid = np.full((len(TAU_BUCKETS), len(MONEYNESS_BUCKETS)), np.nan)
    for i, (_, (t_lo, t_hi)) in enumerate(TAU_BUCKETS.items()):
        t_mask = (tau >= t_lo) & (tau < t_hi)
        for j, (_, (m_lo, m_hi)) in enumerate(MONEYNESS_BUCKETS.items()):
            mask = t_mask & (log_m >= m_lo) & (log_m < m_hi)
            if mask.any():
                grid[i, j] = err_fn(pred[mask], target[mask])

    return pd.DataFrame(
        grid,
        index=list(TAU_BUCKETS.keys()),
        columns=list(MONEYNESS_BUCKETS.keys()),
    )


def evaluate_predictions_2d_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Point counts per joint (maturity x moneyness) cell.

    Same shape/orientation as :func:`evaluate_predictions_2d`; useful to flag
    sparsely-populated cells when reading a heatmap.
    """
    tau = df["tau"].values
    log_m = df["log_moneyness"].values

    grid = np.zeros((len(TAU_BUCKETS), len(MONEYNESS_BUCKETS)), dtype=int)
    for i, (_, (t_lo, t_hi)) in enumerate(TAU_BUCKETS.items()):
        t_mask = (tau >= t_lo) & (tau < t_hi)
        for j, (_, (m_lo, m_hi)) in enumerate(MONEYNESS_BUCKETS.items()):
            grid[i, j] = int((t_mask & (log_m >= m_lo) & (log_m < m_hi)).sum())

    return pd.DataFrame(
        grid,
        index=list(TAU_BUCKETS.keys()),
        columns=list(MONEYNESS_BUCKETS.keys()),
    )


def metrics_to_dataframe(results: dict, model_name: str = "") -> pd.DataFrame:
    """Flatten evaluation results into a single-row summary DataFrame."""
    flat = {"model": model_name}

    for group in ["overall", "observed", "unobserved"]:
        if group in results:
            for metric, val in results[group].items():
                flat[f"{group}_{metric}"] = val

    for bucket_group in ["by_maturity", "by_moneyness"]:
        if bucket_group in results:
            for bucket, metrics in results[bucket_group].items():
                for metric, val in metrics.items():
                    flat[f"{bucket_group}_{bucket}_{metric}"] = val

    return pd.DataFrame([flat])


def print_evaluation(results: dict, model_name: str = ""):
    """Pretty-print evaluation results."""
    header = f"  Evaluation: {model_name}" if model_name else "  Evaluation"
    print(header)
    print("  " + "=" * 50)

    for group in ["overall", "observed", "unobserved"]:
        if group in results:
            r = results[group]
            print(f"  {group:12s}  MAE={r['mae']:.6f}  RMSE={r['rmse']:.6f}  "
                  f"MAPE={r.get('mape', 0):.2f}%  n={r['n']}")

    print()
    for bucket_group, label in [("by_maturity", "Maturity"), ("by_moneyness", "Moneyness")]:
        if bucket_group in results:
            print(f"  {label} breakdown:")
            for bucket, r in results[bucket_group].items():
                print(f"    {bucket:12s}  MAE={r['mae']:.6f}  "
                      f"RMSE={r['rmse']:.6f}  n={r['n']}")
            print()
