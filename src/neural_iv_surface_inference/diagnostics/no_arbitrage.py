"""No-arbitrage diagnostics (Phase 2 W2, story 2B.3).

Quantify the **structural validity** of a single-date implied-volatility
surface. These are *diagnostics, not constraints*: we count and score
violations of the classic static no-arbitrage conditions so a later stage
(W5 / Phase 3) can decide whether to penalize them. The functions are pure
over a surface frame, so they score any predictor's output identically.

Three checks (each defined precisely below):

- **Calendar (term-structure):** total implied variance
  ``w(k, tau) = sigma(k, tau)^2 * tau`` must be non-decreasing in ``tau`` at a
  fixed log-moneyness ``k``. A drop in ``w`` as ``tau`` grows is a calendar-spread
  arbitrage.
- **Monotonicity:** the undiscounted call price must be non-increasing in
  strike ``K`` at a fixed ``tau``. We price each point with the Black formula
  (forward ``F = 1``, ``K = exp(k)``, vol ``= sigma``, maturity ``= tau``) and
  test that ``C(K)`` does not increase as ``K`` increases.
- **Convexity:** the undiscounted call price must be convex in strike ``K`` at a
  fixed ``tau`` (butterfly arbitrage if not). Tested via the second divided
  difference of ``C`` along the strike axis; a negative value is a violation.

Conventions
-----------
- A "fixed-``k``" / "fixed-``tau``" group is formed by exact equality of the held
  coordinate. Constructed grids align exactly; on irregular real surfaces a
  group with fewer than the required number of distinct points simply
  contributes nothing (documented per check).
- ``iv_col`` selects which column carries the surface vols to score (default
  ``"implied_volatility"``). Rows whose vol or coordinates are non-finite are
  dropped per group before differencing (NaN-safe).
- Severity is always a non-negative magnitude of the breached inequality
  (``0.0`` when there are no violations), summed across all evaluated
  pairs/triples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

_TOL = 1e-9


@dataclass(frozen=True)
class ViolationResult:
    """Outcome of one no-arbitrage check over a single-date surface.

    Fields
    ------
    check : str
        Check name (``"calendar"`` / ``"monotonicity"`` / ``"convexity"``).
    mask : np.ndarray
        Boolean array over the **evaluated** units (consecutive pairs for
        calendar/monotonicity, interior triples for convexity), ``True`` where
        the inequality is breached. Length equals ``n_evaluated``.
    n_evaluated : int
        Number of pairs/triples actually tested across all groups.
    n_violations : int
        Number of breached units (``mask.sum()``).
    rate : float
        ``n_violations / n_evaluated``; ``nan`` when nothing could be evaluated.
    severity : float
        Sum of the non-negative breach magnitudes (units depend on the check:
        total-variance units for calendar, undiscounted-price units for
        monotonicity/convexity). ``0.0`` when there are no violations.
    meta : dict
        Free-form details (e.g. ``n_groups``, ``iv_col``).
    """

    check: str
    mask: np.ndarray
    n_evaluated: int
    n_violations: int
    rate: float
    severity: float
    meta: dict = field(default_factory=dict)


def _black_call_undiscounted(k: np.ndarray, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Undiscounted Black call price with forward ``F = 1`` and strike ``exp(k)``.

    ``C = N(d1) - exp(k) * N(d2)`` where
    ``d1 = (-k + 0.5 sigma^2 tau) / (sigma sqrt(tau))`` and ``d2 = d1 - sigma sqrt(tau)``.
    At the degenerate limit ``sigma sqrt(tau) -> 0`` the price is the intrinsic
    value ``max(1 - exp(k), 0)``.
    """
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    std = sigma * np.sqrt(tau)
    price = np.maximum(1.0 - np.exp(k), 0.0)
    valid = std > _TOL
    if np.any(valid):
        kv, stdv = k[valid], std[valid]
        d1 = (-kv + 0.5 * stdv**2) / stdv
        d2 = d1 - stdv
        price = price.copy()
        price[valid] = norm.cdf(d1) - np.exp(kv) * norm.cdf(d2)
    return price


def _finite_group_sort(values: np.ndarray, *arrays: np.ndarray):
    """Drop non-finite rows (across ``values`` + ``arrays``) and sort by ``values``."""
    finite = np.isfinite(values)
    for a in arrays:
        finite &= np.isfinite(a)
    values = values[finite]
    arrays = [a[finite] for a in arrays]
    order = np.argsort(values, kind="stable")
    return values[order], [a[order] for a in arrays]


def calendar_violations(
    df_date: pd.DataFrame, iv_col: str = "implied_volatility"
) -> ViolationResult:
    """Calendar (term-structure) check: ``w = sigma^2 * tau`` non-decreasing in ``tau``.

    Groups points by exact ``log_moneyness``; within each group, sorts by ``tau``
    and tests consecutive pairs. A pair is a violation when total variance drops
    (``w[i+1] < w[i]``). Severity sums the drop magnitudes. Groups with fewer than
    two distinct finite points contribute no evaluated pairs.
    """
    k = df_date["log_moneyness"].to_numpy(dtype=float)
    tau = df_date["tau"].to_numpy(dtype=float)
    sigma = df_date[iv_col].to_numpy(dtype=float)

    masks: list[np.ndarray] = []
    severity = 0.0
    n_groups = 0
    for kv in np.unique(k):
        sel = k == kv
        tau_s, (sigma_s,) = _finite_group_sort(tau[sel], sigma[sel])
        if len(tau_s) < 2:
            continue
        n_groups += 1
        w = sigma_s**2 * tau_s
        diff = np.diff(w)
        viol = diff < -_TOL
        masks.append(viol)
        severity += float(np.sum(-diff[viol]))

    return _assemble("calendar", masks, severity, {"n_groups": n_groups, "iv_col": iv_col})


def monotonicity_violations(
    df_date: pd.DataFrame, iv_col: str = "implied_volatility"
) -> ViolationResult:
    """Monotonicity check: undiscounted call price non-increasing in strike.

    Groups points by exact ``tau``; within each group, prices every point with
    the Black formula (forward ``1``, strike ``exp(k)``), sorts by strike, and
    tests consecutive pairs. A pair is a violation when the call price increases
    with strike (``C[i+1] > C[i]``). Severity sums the positive increments.
    Groups with fewer than two distinct finite points contribute nothing.
    """
    k = df_date["log_moneyness"].to_numpy(dtype=float)
    tau = df_date["tau"].to_numpy(dtype=float)
    sigma = df_date[iv_col].to_numpy(dtype=float)

    masks: list[np.ndarray] = []
    severity = 0.0
    n_groups = 0
    for tv in np.unique(tau):
        sel = tau == tv
        k_s, (sigma_s,) = _finite_group_sort(k[sel], sigma[sel])
        if len(k_s) < 2:
            continue
        n_groups += 1
        c = _black_call_undiscounted(k_s, np.full_like(k_s, tv), sigma_s)
        diff = np.diff(c)
        viol = diff > _TOL
        masks.append(viol)
        severity += float(np.sum(diff[viol]))

    return _assemble("monotonicity", masks, severity, {"n_groups": n_groups, "iv_col": iv_col})


def convexity_violations(
    df_date: pd.DataFrame, iv_col: str = "implied_volatility"
) -> ViolationResult:
    """Convexity check: undiscounted call price convex in strike (no butterflies).

    Groups points by exact ``tau``; within each group, prices every point with
    the Black formula, sorts by strike ``K = exp(k)``, and evaluates the second
    divided difference at each interior triple. A triple is a violation when the
    second difference is negative (concave). Severity sums the magnitudes of the
    negative second differences. Groups with fewer than three distinct finite
    points contribute no evaluated triples.
    """
    k = df_date["log_moneyness"].to_numpy(dtype=float)
    tau = df_date["tau"].to_numpy(dtype=float)
    sigma = df_date[iv_col].to_numpy(dtype=float)

    masks: list[np.ndarray] = []
    severity = 0.0
    n_groups = 0
    for tv in np.unique(tau):
        sel = tau == tv
        k_s, (sigma_s,) = _finite_group_sort(k[sel], sigma[sel])
        if len(k_s) < 3:
            continue
        n_groups += 1
        strike = np.exp(k_s)
        c = _black_call_undiscounted(k_s, np.full_like(k_s, tv), sigma_s)
        # Second divided difference at interior points i (1..n-2).
        left = (c[1:-1] - c[:-2]) / (strike[1:-1] - strike[:-2])
        right = (c[2:] - c[1:-1]) / (strike[2:] - strike[1:-1])
        second = (right - left) / (0.5 * (strike[2:] - strike[:-2]))
        viol = second < -_TOL
        masks.append(viol)
        severity += float(np.sum(-second[viol]))

    return _assemble("convexity", masks, severity, {"n_groups": n_groups, "iv_col": iv_col})


def _assemble(check: str, masks: list[np.ndarray], severity: float, meta: dict) -> ViolationResult:
    mask = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
    n_eval = int(mask.size)
    n_viol = int(mask.sum())
    rate = float(n_viol / n_eval) if n_eval else float("nan")
    return ViolationResult(
        check=check,
        mask=mask,
        n_evaluated=n_eval,
        n_violations=n_viol,
        rate=rate,
        severity=float(severity),
        meta=meta,
    )


def no_arb_diagnostics(
    df_date: pd.DataFrame, iv_col: str = "implied_volatility"
) -> dict:
    """Run all three checks on a single-date surface and aggregate the summary.

    Returns
    -------
    dict
        ``{"calendar": ViolationResult, "monotonicity": ViolationResult,
        "convexity": ViolationResult, "summary": {...}}`` where ``summary``
        carries the total counts, the total evaluated, an overall rate
        (``total_violations / total_evaluated``; ``nan`` when nothing was
        evaluated), and the total severity across checks. Severities are summed
        despite differing units — treat the aggregate as a coarse overall
        signal; per-check severities remain available for finer use.
    """
    results = {
        "calendar": calendar_violations(df_date, iv_col),
        "monotonicity": monotonicity_violations(df_date, iv_col),
        "convexity": convexity_violations(df_date, iv_col),
    }
    total_eval = sum(r.n_evaluated for r in results.values())
    total_viol = sum(r.n_violations for r in results.values())
    summary = {
        "total_evaluated": total_eval,
        "total_violations": total_viol,
        "overall_rate": float(total_viol / total_eval) if total_eval else float("nan"),
        "total_severity": float(sum(r.severity for r in results.values())),
        "per_check_severity": {name: r.severity for name, r in results.items()},
    }
    return {**results, "summary": summary}
