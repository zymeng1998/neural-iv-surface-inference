"""Masking-sensitivity harness (Phase 2 W2, story 2B.2).

Measures **prediction stability under perturbation of the observed set** for a
single surface (one date). The idea: hold the surface fixed, repeatedly draw a
random subset of the points currently treated as "observed", re-run the
predictor on the full set of query points, and quantify how much each query
point's prediction moves across draws.

- High per-point spread (std) -> a fragile, low-trust region.
- Low spread -> a stable region.

The signal is model-agnostic: anything satisfying the ``Predictor`` protocol
(2A.2) works. We never retrain — this is purely inference-time re-masking, so it
applies to predictors that need no fit (interpolation). It never invents
observations; it only subsamples the points already marked ``observed``.

Scope (per the 2B.2 spec): produce the raw per-point instability signal only.
Risk flags and heatmaps are 2B.4; the end-to-end runner / artifacts are 2B.5.

Conventions
-----------
- A "mask" is a boolean array of length ``len(df_date)`` selecting which rows
  are treated as observed for one draw. It is always a subset of the rows whose
  original ``observed`` is ``True``.
- Predictions are collected at **all** query points (every row of ``df_date``),
  so under a well-behaved predictor every point is queried on every draw.
- NaN-safe: per-point statistics use ``nanmean`` / ``nanstd`` over draws, and
  ``n_draws`` counts only the finite predictions for that point. A point that is
  never finite across any draw yields ``mean = std = nan`` and ``n_draws = 0``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from neural_iv_surface_inference.eval.predictor import Predictor


@dataclass(frozen=True)
class MaskingSensitivityResult:
    """Per-point masking-sensitivity statistics for one surface.

    All per-point arrays share length ``n = len(df_date)`` and are aligned to
    the input frame's row order.

    Fields
    ------
    mean : np.ndarray, shape (n,)
        Per-point mean prediction across draws (NaN-safe). ``nan`` where a point
        was never finite.
    std : np.ndarray, shape (n,)
        Per-point standard deviation across draws — the **instability** signal
        (population std, ``ddof=0``; NaN-safe). ``nan`` where a point was never
        finite; ``0.0`` where every draw agreed.
    n_draws : np.ndarray, shape (n,)
        Count of finite predictions contributing to each point (``<= n_draws``).
    predictions : np.ndarray, shape (n_draws, n)
        Raw stacked predictions, one row per mask draw. Retained so downstream
        consumers (2B.4) can compute alternative spread measures.
    keep_fraction : float
        Fraction of the observed set retained per draw (as requested).
    seed : int
        RNG seed used to generate the masks (for reproducibility).
    meta : dict
        Free-form metadata (e.g. predictor model name, n_observed).
    """

    mean: np.ndarray
    std: np.ndarray
    n_draws: np.ndarray
    predictions: np.ndarray
    keep_fraction: float
    seed: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mean)


def mask_resample(
    df_date: pd.DataFrame,
    keep_fraction: float,
    n_draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Draw ``n_draws`` random observed-subsets of a single surface.

    Each returned mask is a boolean array of length ``len(df_date)`` that is a
    subset of the rows whose original ``observed`` is ``True``. We subsample the
    observed set — we never mark an originally-unobserved row as observed.

    Parameters
    ----------
    df_date : pd.DataFrame
        All surface points for one date. Must carry a boolean ``observed``
        column.
    keep_fraction : float
        Fraction of the observed points to retain per draw, in ``(0, 1]``. The
        per-draw keep count is ``round(keep_fraction * n_observed)``, clamped to
        at least 1 when any points are observed.
    n_draws : int
        Number of independent masks to draw (``>= 1``).
    seed : int
        Seed for a ``numpy.random.default_rng`` instance.

    Returns
    -------
    list[np.ndarray]
        ``n_draws`` boolean arrays, each of length ``len(df_date)``.

    Notes
    -----
    Edge case — too few observed points: if no rows are observed, every mask is
    all-``False`` (the predictor then sees an empty observed set and falls back
    to whatever it does in that case). If exactly one point is observed and
    ``keep_fraction`` rounds to 0, the keep count is clamped to 1, so every draw
    is identical and instability is 0.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(
            f"keep_fraction must be in (0, 1]; got {keep_fraction}"
        )
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1; got {n_draws}")
    if "observed" not in df_date.columns:
        raise ValueError("df_date must carry a boolean 'observed' column")

    n = len(df_date)
    observed_idx = np.flatnonzero(df_date["observed"].to_numpy(dtype=bool))
    n_obs = len(observed_idx)
    rng = np.random.default_rng(seed)

    if n_obs == 0:
        return [np.zeros(n, dtype=bool) for _ in range(n_draws)]

    n_keep = max(1, int(round(keep_fraction * n_obs)))
    n_keep = min(n_keep, n_obs)

    masks: list[np.ndarray] = []
    for _ in range(n_draws):
        chosen = rng.choice(observed_idx, size=n_keep, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True
        masks.append(mask)
    return masks


def masking_sensitivity(
    predictor: Predictor,
    df_date: pd.DataFrame,
    keep_fraction: float = 0.8,
    n_draws: int = 20,
    seed: int = 0,
) -> MaskingSensitivityResult:
    """Quantify per-point prediction instability under observed-set re-masking.

    For each of ``n_draws`` masks (see :func:`mask_resample`), build a per-draw
    copy of ``df_date`` whose ``observed`` column reflects the subset, call
    ``predictor.predict`` over all query points, and stack the results. Return
    per-point mean and std (instability) across draws.

    Parameters
    ----------
    predictor : Predictor
        Anything satisfying the ``Predictor`` protocol (2A.2). Called once per
        draw; never retrained.
    df_date : pd.DataFrame
        All surface points for one date (single ``date`` value expected).
    keep_fraction : float, default 0.8
        Fraction of observed points retained per draw.
    n_draws : int, default 20
        Number of mask draws.
    seed : int, default 0
        RNG seed (forwarded to :func:`mask_resample`).

    Returns
    -------
    MaskingSensitivityResult
        Per-point ``mean``, ``std`` (instability), ``n_draws`` count, and the
        raw stacked ``predictions``.

    Notes
    -----
    The predictor's prediction is realigned to ``df_date`` row order implicitly:
    we pass a row-order-preserving copy and expect a length-``n`` result, as
    guaranteed by the ``Predictor`` contract.
    """
    masks = mask_resample(df_date, keep_fraction, n_draws, seed)
    n = len(df_date)

    preds = np.full((n_draws, n), np.nan, dtype=float)
    for i, mask in enumerate(masks):
        df_draw = df_date.copy()
        df_draw["observed"] = mask
        result = predictor.predict(df_draw)
        pred = np.asarray(result.pred, dtype=float)
        if len(pred) != n:
            raise ValueError(
                f"predictor returned {len(pred)} predictions, expected {n} "
                "to match df_date row order"
            )
        preds[i] = pred

    n_finite = np.isfinite(preds).sum(axis=0)
    # nanmean / nanstd warn on all-NaN columns; we intentionally produce nan
    # there (a never-finite point) and document that, so suppress the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(preds, axis=0)
        std = np.nanstd(preds, axis=0, ddof=0)

    n_obs = int(df_date["observed"].to_numpy(dtype=bool).sum())
    meta = {
        "n_observed": n_obs,
        "n_points": n,
        "keep_fraction": float(keep_fraction),
        "n_draws": int(n_draws),
    }

    return MaskingSensitivityResult(
        mean=mean,
        std=std,
        n_draws=n_finite,
        predictions=preds,
        keep_fraction=float(keep_fraction),
        seed=int(seed),
        meta=meta,
    )


def instability_summary(
    result: MaskingSensitivityResult,
    observed: np.ndarray | None = None,
    unobserved_only: bool = False,
) -> dict[str, float]:
    """Aggregate a :class:`MaskingSensitivityResult` into scalar summaries.

    Parameters
    ----------
    result : MaskingSensitivityResult
        Per-point instability produced by :func:`masking_sensitivity`.
    observed : np.ndarray | None
        Original boolean ``observed`` mask (length ``len(result)``). Required
        when ``unobserved_only`` is ``True``.
    unobserved_only : bool, default False
        If ``True``, restrict the summary to originally-unobserved points (the
        true reconstruction targets), using ``observed`` to identify them.

    Returns
    -------
    dict[str, float]
        ``mean_std`` and ``median_std`` (NaN-safe over the selected points),
        plus ``n_points`` (count of points with a finite std contributing to
        the summary). Empty selection yields ``nan`` statistics and ``0`` count.
    """
    std = result.std
    if unobserved_only:
        if observed is None:
            raise ValueError("observed mask required when unobserved_only=True")
        observed = np.asarray(observed, dtype=bool)
        if len(observed) != len(std):
            raise ValueError(
                f"observed length {len(observed)} != result length {len(std)}"
            )
        std = std[~observed]

    finite = std[np.isfinite(std)]
    if finite.size == 0:
        return {"mean_std": float("nan"), "median_std": float("nan"),
                "n_points": 0}
    return {
        "mean_std": float(np.mean(finite)),
        "median_std": float(np.median(finite)),
        "n_points": int(finite.size),
    }
