"""Sparsity-sweep harness (Phase 4B, story 4B.2).

Builds, for a single date, a ladder of progressively sparser **observed
contexts** while holding the **query/target set fixed**. This is the
construction the Phase 4B accuracy-survival diagnostic (ADR 0011) runs RBF and
the 4A RBF-prior hybrid over: as the observed chain thins, does the hybrid's
edge over RBF grow (accuracy story lives) or stay flat (~0-2 %, pivot to
reliability)?

Design invariants (pinned by ``tests/test_sparsity_sweep.py``)
-------------------------------------------------------------
- **Fixed query.** The query/target set is the complement of the *original*
  observed set and never changes across rungs or regimes. The harness only ever
  turns observed -> unobserved (it shrinks context); it never recruits a query
  point into the context, so per-rung MAE deltas stay comparable.
- **Monotone nesting.** Along an ascending intensity ladder, each rung's
  retained context is a subset of the previous rung's. ``intensity = 0`` keeps
  the full observed context (the dense rung == the 4A full-context comparison);
  ``intensity -> 1`` is maximally sparse.
- **Determinism.** A fixed ``seed`` reproduces every rung. ``fewer_quotes`` and
  ``missing_maturities`` use the seed (random retention / drop order);
  ``thin_wings`` is a pure |log-moneyness| quantile cut and is seed-independent.

Regimes
-------
- ``fewer_quotes`` — random retention of observed quotes (keep_frac = 1 - i).
- ``thin_wings`` — drop observed quotes beyond the (1 - i) |log-moneyness|
  quantile, so the deep-wing context must be inferred.
- ``missing_maturities`` — drop whole observed expirations (in a seeded order).
- ``combined_quotes_wings`` — joint stress: the intersection of ``fewer_quotes``
  and ``thin_wings`` at matched intensity (operator-requested).

This story is **construction only** — no RBF, no model, no full-dataset build.
Per-rung RBF inputs are materialized in 4B.3; the hybrid forward pass is 4B.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REGIMES: Tuple[str, ...] = (
    "fewer_quotes",
    "thin_wings",
    "missing_maturities",
    "combined_quotes_wings",
)

# Fixed per-component sub-seeds so a component computed standalone and the same
# component computed inside ``combined_quotes_wings`` use identical randomness
# (this is what makes combined == fewer_quotes & thin_wings exactly).
_COMPONENT_SEED: Mapping[str, int] = {"quotes": 1, "wings": 2, "maturity": 3}


@dataclass(frozen=True)
class SparsityRung:
    """One rung of a sparsity ladder for a single date.

    Attributes
    ----------
    regime : str
        One of :data:`REGIMES`.
    rung_index : int
        Position on the ladder (0 = densest).
    intensity : float
        Canonical sparsity intensity in ``[0, 1)`` (0 = full observed context).
    retained_observed : np.ndarray of bool, shape (n,)
        Which rows remain in the observed context at this rung. Always a subset
        of the original observed set.
    params : Mapping[str, float]
        Regime-specific descriptors (JSON-serializable scalars).
    """

    regime: str
    rung_index: int
    intensity: float
    retained_observed: np.ndarray
    params: Mapping[str, float]


@dataclass(frozen=True)
class SparsitySweep:
    """A full intensity ladder for one regime on one date."""

    regime: str
    query_mask: np.ndarray  # bool, (n,) — fixed across rungs
    base_observed: np.ndarray  # bool, (n,) — original observed context
    rungs: Tuple[SparsityRung, ...]


def _component_rng(seed: int, component: str) -> np.random.Generator:
    return np.random.default_rng(
        np.array([int(seed), _COMPONENT_SEED[component]], dtype=np.uint64)
    )


def _validate_intensities(intensities: Sequence[float]) -> Tuple[float, ...]:
    out = tuple(float(x) for x in intensities)
    for x in out:
        if not (0.0 <= x < 1.0):
            raise ValueError(f"intensity must be in [0, 1), got {x}")
    return out


def _fewer_quotes(
    base_observed: np.ndarray, intensity: float, seed: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    keep_frac = 1.0 - intensity
    u = _component_rng(seed, "quotes").random(base_observed.shape[0])
    retained = base_observed & (u < keep_frac)
    return retained, {"keep_frac": keep_frac}


def _thin_wings(
    base_observed: np.ndarray, abs_lm: np.ndarray, intensity: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    keep_quantile = 1.0 - intensity
    if not base_observed.any():
        return np.zeros_like(base_observed), {
            "abs_logm_keep_quantile": keep_quantile,
            "abs_logm_threshold": float("nan"),
        }
    threshold = float(np.quantile(abs_lm[base_observed], keep_quantile))
    retained = base_observed & (abs_lm <= threshold)
    return retained, {
        "abs_logm_keep_quantile": keep_quantile,
        "abs_logm_threshold": threshold,
    }


def _missing_maturities(
    base_observed: np.ndarray, maturities: np.ndarray, intensity: float, seed: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    observed_maturities = np.unique(maturities[base_observed])
    m_total = int(observed_maturities.shape[0])
    order = _component_rng(seed, "maturity").permutation(observed_maturities)
    n_drop = int(np.rint(intensity * m_total))
    n_drop = min(n_drop, m_total)
    dropped = set(order[:n_drop].tolist())
    if dropped:
        keep_mat = ~np.isin(maturities, list(dropped))
    else:
        keep_mat = np.ones_like(base_observed)
    retained = base_observed & keep_mat
    return retained, {
        "n_maturities_dropped": float(n_drop),
        "n_maturities_total": float(m_total),
    }


def build_sweep(
    df_date: pd.DataFrame,
    regime: str,
    intensities: Sequence[float],
    *,
    seed: int,
    observed_col: str = "observed",
    moneyness_col: str = "log_moneyness",
    maturity_col: str = "expiration",
) -> SparsitySweep:
    """Build a sparsity ladder for one ``regime`` on a single-date frame.

    Parameters
    ----------
    df_date : pd.DataFrame
        One date's rows. Must contain ``observed_col`` (bool context flag),
        ``moneyness_col`` (log K/S), and ``maturity_col`` (expiration id).
    regime : str
        One of :data:`REGIMES`.
    intensities : sequence of float
        Ascending ladder of sparsity intensities in ``[0, 1)``. ``0`` keeps the
        full observed context; larger = sparser. Ascending order yields nested
        rungs.
    seed : int
        Seed for the random components (``fewer_quotes`` retention and
        ``missing_maturities`` drop order). ``thin_wings`` ignores it.

    Returns
    -------
    SparsitySweep
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    intensities = _validate_intensities(intensities)

    base_observed = np.asarray(df_date[observed_col].to_numpy(), dtype=bool)
    query_mask = ~base_observed
    abs_lm = np.abs(np.asarray(df_date[moneyness_col].to_numpy(), dtype=float))
    maturities = df_date[maturity_col].to_numpy()

    rungs = []
    for i, intensity in enumerate(intensities):
        if regime == "fewer_quotes":
            retained, params = _fewer_quotes(base_observed, intensity, seed)
        elif regime == "thin_wings":
            retained, params = _thin_wings(base_observed, abs_lm, intensity)
        elif regime == "missing_maturities":
            retained, params = _missing_maturities(
                base_observed, maturities, intensity, seed
            )
        else:  # combined_quotes_wings
            r_q, p_q = _fewer_quotes(base_observed, intensity, seed)
            r_w, p_w = _thin_wings(base_observed, abs_lm, intensity)
            retained = r_q & r_w
            params = {**p_q, **p_w}
        # Defensive: never escape the original observed set.
        retained = retained & base_observed
        rungs.append(SparsityRung(regime, i, intensity, retained, params))

    return SparsitySweep(regime, query_mask, base_observed, tuple(rungs))


def build_all_regimes(
    df_date: pd.DataFrame,
    ladders: Mapping[str, Sequence[float]],
    *,
    seed: int,
    observed_col: str = "observed",
    moneyness_col: str = "log_moneyness",
    maturity_col: str = "expiration",
) -> Dict[str, SparsitySweep]:
    """Build every regime in ``ladders`` (keys must be valid :data:`REGIMES`)."""
    return {
        regime: build_sweep(
            df_date,
            regime,
            ladder,
            seed=seed,
            observed_col=observed_col,
            moneyness_col=moneyness_col,
            maturity_col=maturity_col,
        )
        for regime, ladder in ladders.items()
    }


def sweep_manifest(
    sweeps: Union[SparsitySweep, Mapping[str, SparsitySweep]],
) -> Dict[str, object]:
    """Return a JSON-serializable manifest for one or many sweeps."""
    if isinstance(sweeps, SparsitySweep):
        sweeps = {sweeps.regime: sweeps}

    regimes: Dict[str, object] = {}
    for regime, sweep in sweeps.items():
        n_points = int(sweep.base_observed.shape[0])
        n_base = int(sweep.base_observed.sum())
        regimes[regime] = {
            "n_points": n_points,
            "n_base_observed": n_base,
            "n_query": int(sweep.query_mask.sum()),
            "rungs": [
                {
                    "rung_index": int(r.rung_index),
                    "intensity": float(r.intensity),
                    "n_retained": int(r.retained_observed.sum()),
                    "retained_frac_of_observed": (
                        float(r.retained_observed.sum() / n_base) if n_base else 0.0
                    ),
                    "params": {k: float(v) for k, v in r.params.items()},
                }
                for r in sweep.rungs
            ],
        }
    return {"regimes": regimes}
