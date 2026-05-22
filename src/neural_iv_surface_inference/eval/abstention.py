"""Abstention / selective-prediction curves (Phase 2 W1, story 2A.4).

The risk–coverage curve sweeps an abstention threshold from "keep every point"
down to "keep only the most-confident point", reporting the mean absolute error
of the retained subset at each coverage level. Lower retained error at lower
coverage means a usable confidence signal — this is the measurement W5 (2D)
later uses to justify an ``abstain_flag``.

Pure functions over numpy arrays — no model, no I/O, no threshold/policy choice
(that is W5), no disk artifacts (the runner, 2A.5, owns persistence).

Conventions (shared with ``uncertainty_metrics``)
-------------------------------------------------
- ``abs_error`` is the absolute point error ``|y_pred - y_true|`` (non-negative).
- ``confidence`` is a per-point score where **higher means more confident**.
  Abstention keeps the highest-confidence points first.
- Rows with any NaN among the inputs are dropped before computing; a length
  mismatch raises ``ValueError``. An all-NaN / empty input yields an empty
  curve and a ``nan`` area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# np.trapezoid (NumPy >= 2.0) renames the deprecated np.trapz; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _check_same_length(*arrays: np.ndarray) -> None:
    lengths = {len(a) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"inputs must share length; got {sorted(lengths)}")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Aligned risk–coverage sweep.

    Arrays are ordered by **increasing coverage** (most-abstaining first,
    keep-all last). All three share the same length.

    Fields
    ------
    coverage : np.ndarray
        Retained fraction at each point, ``n_retained / n_finite`` in (0, 1].
        The final entry is exactly ``1.0`` (keep all).
    retained_mae : np.ndarray
        Mean absolute error over the retained (most-confident) subset at the
        corresponding coverage. The final entry equals the overall MAE.
    n_retained : np.ndarray
        Integer count of retained points at each coverage level.
    """

    coverage: np.ndarray
    retained_mae: np.ndarray
    n_retained: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.coverage) == len(self.retained_mae) == len(self.n_retained)):
            raise ValueError("RiskCoverageCurve arrays must share length")

    def __len__(self) -> int:
        return len(self.coverage)


def risk_coverage_curve(
    abs_error: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 20,
) -> RiskCoverageCurve:
    """Sweep retained MAE as the abstention threshold varies.

    Points are ranked by confidence (descending). For each sampled coverage
    level ``c`` we retain the top ``k = round(c * n)`` most-confident points and
    record their MAE. Coverage is reported as the *actual* ``k / n`` so the
    keep-all endpoint is exactly ``1.0`` and ``retained_mae[-1]`` equals the
    overall MAE.

    Parameters
    ----------
    abs_error, confidence : np.ndarray
        Equal-length arrays. NaN rows are dropped.
    n_points : int
        Number of coverage levels to sample (>= 1). The realized number may be
        smaller when ``n_points`` exceeds the finite sample size, since coverage
        levels collapse to distinct integer retain-counts.

    Returns
    -------
    RiskCoverageCurve
        Empty arrays when no finite rows remain.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    abs_error = np.asarray(abs_error, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    _check_same_length(abs_error, confidence)

    mask = _finite_mask(abs_error, confidence)
    ae = abs_error[mask]
    conf = confidence[mask]
    n = len(ae)
    if n == 0:
        empty_f = np.empty(0, dtype=float)
        return RiskCoverageCurve(empty_f, empty_f.copy(), np.empty(0, dtype=int))

    # Rank by confidence descending (most-confident retained first).
    order = np.argsort(conf, kind="stable")[::-1]
    ae_sorted = ae[order]

    # Cumulative mean of abs error over the top-k most-confident points.
    cum_mae = np.cumsum(ae_sorted) / np.arange(1, n + 1)

    # Sample retain-counts k in [1, n]; dedupe (keeps keep-all endpoint k=n).
    ks = np.unique(
        np.clip(np.round(np.linspace(1, n, n_points)).astype(int), 1, n)
    )

    coverage = ks / n
    retained_mae = cum_mae[ks - 1]
    return RiskCoverageCurve(
        coverage=coverage,
        retained_mae=retained_mae,
        n_retained=ks,
    )


def area_under_risk_coverage(curve: RiskCoverageCurve) -> float:
    """Area under the risk–coverage curve (AURC); lower is better.

    Trapezoidal integral of ``retained_mae`` against ``coverage``. A confidence
    signal that ranks errors well pushes retained error down at low coverage,
    shrinking the area. Returns ``nan`` for an empty curve, and the single MAE
    value for a one-point curve (degenerate, zero-width integral).
    """
    if len(curve) == 0:
        return float("nan")
    if len(curve) == 1:
        return float(curve.retained_mae[0])
    return float(_trapezoid(curve.retained_mae, curve.coverage))
