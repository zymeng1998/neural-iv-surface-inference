"""Risk-flag synthesis + region binning (Phase 2 W2, story 2B.4).

Turns the two W2 diagnostic signals into a decision-ready per-point risk
surface:

- **Structural** violations from the no-arbitrage diagnostics (2B.3) — each
  ``ViolationResult`` carries a per-point ``point_mask`` marking the rows it
  touches.
- **Instability** from the masking-sensitivity harness (2B.2) — an optional
  per-point spread (e.g. ``MaskingSensitivityResult.std``).

``derive_risk_flags`` combines them into a boolean ``no_arb_risk_flag`` and a
continuous ``risk_score``; ``bin_to_regions`` aggregates any per-point value
onto the canonical ``(tau, log_moneyness)`` bucket grid so region heatmaps
align with the W1 joint-heatmap convention.

These are **diagnostic** signals, not trading decisions: no tradability
threshold or ``abstain_flag`` policy is chosen here (that is W5 / 2D). The
combination rule is configurable so downstream stages can tune it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from neural_iv_surface_inference.diagnostics.no_arbitrage import ViolationResult
from neural_iv_surface_inference.training.eval import (
    MONEYNESS_BUCKETS,
    TAU_BUCKETS,
)

_CHECK_KEYS = ("calendar", "monotonicity", "convexity")


@dataclass(frozen=True)
class RiskFlagConfig:
    """Knobs for the per-point risk-flag combination rule.

    Fields
    ------
    instability_threshold : float
        A point is flagged when its instability strictly exceeds this value.
        Defaults to ``inf`` — i.e. instability never flags on its own unless a
        finite threshold is supplied. Lower threshold -> more flags (monotone).
    struct_weight, instability_weight : float
        Non-negative weights for the continuous ``risk_score`` =
        ``struct_weight * struct_fraction + instability_weight * instab_norm``,
        where ``struct_fraction`` is the share of checks (0..1) flagging the
        point and ``instab_norm`` is the instability rescaled to ``[0, 1]`` by
        its finite maximum (0 when no instability is supplied or all-zero).
    """

    instability_threshold: float = float("inf")
    struct_weight: float = 1.0
    instability_weight: float = 1.0


@dataclass(frozen=True)
class RiskFlagResult:
    """Per-point risk synthesis aligned to the scored surface's row order.

    Fields
    ------
    no_arb_risk_flag : np.ndarray (bool, shape (n,))
        ``True`` where a structural violation touches the point OR its
        instability exceeds ``config.instability_threshold``.
    risk_score : np.ndarray (float, shape (n,))
        Continuous combined score (see :class:`RiskFlagConfig`).
    struct_flag : np.ndarray (bool, shape (n,))
        Structural-only flag (OR of the 2B.3 per-check ``point_mask``s).
    struct_count : np.ndarray (int, shape (n,))
        Number of structural checks (0..3) flagging each point.
    n_points : int
        Surface row count.
    meta : dict
        Free-form details (weights, whether instability was used, ...).
    """

    no_arb_risk_flag: np.ndarray
    risk_score: np.ndarray
    struct_flag: np.ndarray
    struct_count: np.ndarray
    n_points: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.n_points


def derive_risk_flags(
    diagnostics: dict,
    instability: np.ndarray | None = None,
    config: RiskFlagConfig | None = None,
) -> RiskFlagResult:
    """Combine 2B.3 structural diagnostics (+ optional 2B.2 instability).

    Parameters
    ----------
    diagnostics : dict
        Output of :func:`no_arbitrage.no_arb_diagnostics` — must carry the
        ``calendar`` / ``monotonicity`` / ``convexity`` :class:`ViolationResult`
        entries (each with a per-point ``point_mask`` of equal length).
    instability : np.ndarray | None
        Optional per-point instability (e.g. masking-sensitivity ``std``), same
        length as the surface. ``nan`` entries are treated as 0 for scoring and
        never trigger the threshold flag.
    config : RiskFlagConfig | None
        Combination knobs; defaults to :class:`RiskFlagConfig` (instability does
        not flag unless a finite threshold is set).

    Returns
    -------
    RiskFlagResult
    """
    config = config or RiskFlagConfig()
    results: list[ViolationResult] = [diagnostics[k] for k in _CHECK_KEYS]

    n_points = results[0].n_points
    for r in results:
        if r.n_points != n_points:
            raise ValueError(
                f"check '{r.check}' has n_points {r.n_points}, expected {n_points}"
            )

    struct_count = np.zeros(n_points, dtype=int)
    for r in results:
        struct_count += r.point_mask.astype(int)
    struct_flag = struct_count > 0
    struct_fraction = struct_count / len(_CHECK_KEYS)

    if instability is not None:
        instability = np.asarray(instability, dtype=float)
        if len(instability) != n_points:
            raise ValueError(
                f"instability length {len(instability)} != n_points {n_points}"
            )
        instab_flag = np.where(
            np.isfinite(instability),
            instability > config.instability_threshold,
            False,
        )
        finite = instability[np.isfinite(instability)]
        max_i = finite.max() if finite.size and finite.max() > 0 else 0.0
        instab_norm = (
            np.nan_to_num(instability, nan=0.0) / max_i if max_i > 0 else
            np.zeros(n_points)
        )
        used_instability = True
    else:
        instab_flag = np.zeros(n_points, dtype=bool)
        instab_norm = np.zeros(n_points)
        used_instability = False

    risk_score = (
        config.struct_weight * struct_fraction
        + config.instability_weight * instab_norm
    )
    no_arb_risk_flag = struct_flag | instab_flag

    meta = {
        "used_instability": used_instability,
        "instability_threshold": config.instability_threshold,
        "struct_weight": config.struct_weight,
        "instability_weight": config.instability_weight,
    }
    return RiskFlagResult(
        no_arb_risk_flag=no_arb_risk_flag,
        risk_score=risk_score,
        struct_flag=struct_flag,
        struct_count=struct_count,
        n_points=n_points,
        meta=meta,
    )


def bin_to_regions(
    values: np.ndarray,
    log_moneyness: np.ndarray,
    tau: np.ndarray,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate a per-point value onto the canonical (tau x moneyness) grid.

    Rows follow ``TAU_BUCKETS`` order, columns follow ``MONEYNESS_BUCKETS``
    order, matching the W1 joint-heatmap convention. Bucket membership uses the
    same half-open ``[lo, hi)`` rule as ``training.eval``. Empty cells are
    ``nan``.

    Parameters
    ----------
    values : np.ndarray
        Per-point quantity to aggregate (e.g. ``risk_score`` or instability).
        ``nan`` entries are ignored within each cell.
    log_moneyness, tau : np.ndarray
        Per-point coordinates, same length as ``values``.
    agg : {"mean", "sum", "max", "fraction"}
        Cell aggregation. ``"fraction"`` treats ``values`` as boolean and
        returns the share of flagged points per cell.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(TAU_BUCKETS), len(MONEYNESS_BUCKETS))``.
    """
    values = np.asarray(values, dtype=float)
    log_moneyness = np.asarray(log_moneyness, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if not (len(values) == len(log_moneyness) == len(tau)):
        raise ValueError("values, log_moneyness, tau must share length")

    reducers = {
        "mean": np.nanmean,
        "sum": np.nansum,
        "max": np.nanmax,
        "fraction": lambda v: float(np.mean(v)) if len(v) else float("nan"),
    }
    if agg not in reducers:
        raise ValueError(f"unknown agg '{agg}'; choose {sorted(reducers)}")
    reduce = reducers[agg]

    grid = np.full((len(TAU_BUCKETS), len(MONEYNESS_BUCKETS)), np.nan)
    for i, (t_lo, t_hi) in enumerate(TAU_BUCKETS.values()):
        t_sel = (tau >= t_lo) & (tau < t_hi)
        for j, (m_lo, m_hi) in enumerate(MONEYNESS_BUCKETS.values()):
            cell = t_sel & (log_moneyness >= m_lo) & (log_moneyness < m_hi)
            if not cell.any():
                continue
            cell_vals = values[cell]
            if agg != "fraction":
                cell_vals = cell_vals[np.isfinite(cell_vals)]
                if cell_vals.size == 0:
                    continue
            grid[i, j] = reduce(cell_vals)

    return pd.DataFrame(
        grid,
        index=list(TAU_BUCKETS.keys()),
        columns=list(MONEYNESS_BUCKETS.keys()),
    )
