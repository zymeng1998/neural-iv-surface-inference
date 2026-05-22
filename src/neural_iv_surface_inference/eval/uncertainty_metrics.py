"""Core model-agnostic uncertainty-evaluation metrics (Phase 2 W1, story 2A.3).

Pure functions over numpy arrays — no model, no I/O. These score the
reliability outputs that a :class:`PredictionResult` carries and form the
measurement foundation every downstream reliability claim depends on.

Conventions
-----------
- ``abs_error`` is the absolute point error ``|y_pred - y_true|`` (non-negative).
- ``confidence`` is a per-point score where **higher means more confident**.
  A predictor's raw ``uncertainty`` (higher = less confident) is converted to a
  confidence by the caller, e.g. ``confidence = -uncertainty``. Keeping these
  functions on a generic confidence score keeps them model-agnostic.
- ``uncertainty`` (in :func:`error_uncertainty_correlation`) is the raw signal
  where higher = more uncertain; a well-calibrated model has error rising with
  uncertainty (positive correlation).

NaN / empty handling
--------------------
Every function drops rows where any relevant input is NaN (pairwise / rowwise),
then computes on the finite remainder. If nothing finite remains, the documented
empty-return is produced (``nan`` floats, ``0`` counts). Inputs are never
silently coerced — a length mismatch raises ``ValueError``.

Abstention / risk–coverage curves are intentionally not here; they are 2A.4.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _check_same_length(*arrays: np.ndarray) -> None:
    lengths = {len(a) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"inputs must share length; got {sorted(lengths)}")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Row mask that is True only where every array is finite."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage: fraction of ``y_true`` inside ``[lower, upper]``.

    Compared against the interval's nominal level (e.g. 0.9 for a 90% interval)
    to assess calibration. Rows with any NaN among the three are dropped.
    Returns ``nan`` if no finite rows remain.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    _check_same_length(y_true, lower, upper)

    mask = _finite_mask(y_true, lower, upper)
    if not mask.any():
        return float("nan")

    inside = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    return float(inside.mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width ``upper - lower``.

    Sharpness companion to :func:`interval_coverage`: narrower is better at a
    fixed coverage. Rows with any NaN are dropped; ``nan`` if none remain.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    _check_same_length(lower, upper)

    mask = _finite_mask(lower, upper)
    if not mask.any():
        return float("nan")

    return float((upper[mask] - lower[mask]).mean())


def error_uncertainty_correlation(
    abs_error: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, float]:
    """Pearson and Spearman correlation between absolute error and uncertainty.

    A reliable uncertainty signal correlates positively with realized error.
    Returns ``{"pearson": nan, "spearman": nan}`` when fewer than two finite
    pairs remain or when either array is constant (correlation undefined).
    """
    abs_error = np.asarray(abs_error, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    _check_same_length(abs_error, uncertainty)

    mask = _finite_mask(abs_error, uncertainty)
    ae = abs_error[mask]
    un = uncertainty[mask]

    if len(ae) < 2 or ae.std() == 0 or un.std() == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}

    pearson = float(stats.pearsonr(ae, un).statistic)
    spearman = float(stats.spearmanr(ae, un).statistic)
    return {"pearson": pearson, "spearman": spearman}


def confidence_bucket_metrics(
    abs_error: np.ndarray,
    confidence: np.ndarray,
    n_buckets: int = 10,
) -> list[dict[str, float]]:
    """Mean absolute error stratified into confidence-quantile buckets.

    Points are split into ``n_buckets`` groups by confidence quantile. Bucket 0
    is the **lowest**-confidence group, the last bucket the **highest**. A
    reliable model shows mean error decreasing as bucket index rises.

    Returns one dict per non-empty bucket with keys ``bucket`` (index),
    ``conf_lo`` / ``conf_hi`` (confidence range, inclusive lo / exclusive hi
    except the top bucket which is inclusive), ``mae`` (mean abs error), and
    ``n`` (count). Rows with any NaN are dropped; an empty list is returned if
    no finite rows remain.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    abs_error = np.asarray(abs_error, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    _check_same_length(abs_error, confidence)

    mask = _finite_mask(abs_error, confidence)
    ae = abs_error[mask]
    conf = confidence[mask]
    if len(ae) == 0:
        return []

    # Quantile edges over the confidence distribution.
    edges = np.quantile(conf, np.linspace(0.0, 1.0, n_buckets + 1))
    # np.digitize on interior edges; clip into [0, n_buckets-1].
    idx = np.digitize(conf, edges[1:-1], right=False)
    idx = np.clip(idx, 0, n_buckets - 1)

    results: list[dict[str, float]] = []
    for b in range(n_buckets):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        results.append({
            "bucket": b,
            "conf_lo": float(edges[b]),
            "conf_hi": float(edges[b + 1]),
            "mae": float(ae[sel].mean()),
            "n": n,
        })
    return results


def high_confidence_mae(
    abs_error: np.ndarray,
    confidence: np.ndarray,
    keep_fraction: float = 0.5,
) -> float:
    """MAE on the top-``keep_fraction`` most-confident retained subset.

    Sorts by confidence descending and keeps the top ``keep_fraction`` of finite
    points (at least one), returning their mean absolute error. This is the
    "retained error" an abstention policy would achieve at that coverage; the
    full risk–coverage sweep is 2A.4. Rows with any NaN are dropped. Returns
    ``nan`` if no finite rows remain.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

    abs_error = np.asarray(abs_error, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    _check_same_length(abs_error, confidence)

    mask = _finite_mask(abs_error, confidence)
    ae = abs_error[mask]
    conf = confidence[mask]
    if len(ae) == 0:
        return float("nan")

    k = max(1, int(round(keep_fraction * len(ae))))
    # Indices of the k highest-confidence points.
    keep_idx = np.argsort(conf)[::-1][:k]
    return float(ae[keep_idx].mean())
