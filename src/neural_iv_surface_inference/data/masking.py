"""Sparse observation masking for IV surface benchmark tasks.

Given a full set of surface points for a date, produces binary masks
indicating which points are 'observed' (input to the model) vs
'unobserved' (held out for evaluation).

Masking strategies:
  - random_uniform:  keep each point independently with probability p
  - drop_maturity:   drop entire maturity (tau) slices
  - drop_moneyness:  drop moneyness wings or ATM region
  - realistic:       observation density decays away from ATM + short-dated,
                     mimicking real market liquidity patterns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core masking functions
# ---------------------------------------------------------------------------

def random_uniform_mask(
    n: int,
    keep_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a boolean mask of length *n* with ~keep_frac fraction True.

    Parameters
    ----------
    n : int
        Number of observation points.
    keep_frac : float
        Fraction of points to keep (0, 1].
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    np.ndarray of bool, shape (n,)
    """
    if not 0 < keep_frac <= 1:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    return rng.random(n) < keep_frac


def drop_maturity_mask(
    tau: np.ndarray,
    drop_region: str,
    rng: np.random.Generator,
    *,
    tau_short_threshold: float = 30 / 365,
    tau_long_threshold: float = 1.0,
) -> np.ndarray:
    """Drop points in a maturity region.

    Parameters
    ----------
    tau : np.ndarray
        Time-to-maturity in years for each point.
    drop_region : str
        Which region to drop: 'short' (tau < threshold),
        'long' (tau > threshold), or 'random_slice' (one randomly
        chosen maturity bucket).
    rng : np.random.Generator
        Seeded random generator.
    tau_short_threshold : float
        Boundary for 'short' region (default 30 days).
    tau_long_threshold : float
        Boundary for 'long' region (default 1 year).

    Returns
    -------
    np.ndarray of bool, shape (len(tau),) — True = keep, False = drop
    """
    keep = np.ones(len(tau), dtype=bool)

    if drop_region == "short":
        keep[tau < tau_short_threshold] = False
    elif drop_region == "long":
        keep[tau > tau_long_threshold] = False
    elif drop_region == "random_slice":
        # Discretise tau into ~10 buckets and drop one at random
        bucket_edges = np.linspace(tau.min(), tau.max(), 11)
        bucket_idx = np.digitize(tau, bucket_edges) - 1
        bucket_idx = np.clip(bucket_idx, 0, 9)
        drop_bucket = rng.integers(0, 10)
        keep[bucket_idx == drop_bucket] = False
    else:
        raise ValueError(f"Unknown drop_region: {drop_region}")

    return keep


def drop_moneyness_mask(
    log_moneyness: np.ndarray,
    drop_region: str,
    rng: np.random.Generator,
    *,
    wing_threshold: float = 0.3,
    atm_threshold: float = 0.05,
) -> np.ndarray:
    """Drop points in a moneyness region.

    Parameters
    ----------
    log_moneyness : np.ndarray
        log(K/S) for each point.
    drop_region : str
        'wings' (|log_m| > threshold), 'atm' (|log_m| < threshold),
        or 'left_wing' / 'right_wing'.
    rng : np.random.Generator
        Seeded random generator (unused for deterministic modes, kept
        for interface consistency).
    wing_threshold : float
        |log_moneyness| above which points are considered 'wings'.
    atm_threshold : float
        |log_moneyness| below which points are considered 'ATM'.

    Returns
    -------
    np.ndarray of bool — True = keep
    """
    abs_lm = np.abs(log_moneyness)
    keep = np.ones(len(log_moneyness), dtype=bool)

    if drop_region == "wings":
        keep[abs_lm > wing_threshold] = False
    elif drop_region == "atm":
        keep[abs_lm < atm_threshold] = False
    elif drop_region == "left_wing":
        keep[log_moneyness < -wing_threshold] = False
    elif drop_region == "right_wing":
        keep[log_moneyness > wing_threshold] = False
    else:
        raise ValueError(f"Unknown drop_region: {drop_region}")

    return keep


def realistic_liquidity_mask(
    log_moneyness: np.ndarray,
    tau: np.ndarray,
    base_keep_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mask that mimics real market observation density.

    Observation probability is highest near ATM + medium maturity and
    decays toward wings and very short/long maturities.

    Parameters
    ----------
    log_moneyness : np.ndarray
        log(K/S) per point.
    tau : np.ndarray
        Time-to-maturity in years per point.
    base_keep_frac : float
        Average fraction of points to keep.  Actual per-point probability
        is modulated by a liquidity score.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray of bool — True = keep
    """
    if not 0 < base_keep_frac <= 1:
        raise ValueError(f"base_keep_frac must be in (0, 1], got {base_keep_frac}")

    # Liquidity score: Gaussian centered at ATM (log_m=0) and moderate tau
    moneyness_score = np.exp(-0.5 * (log_moneyness / 0.2) ** 2)
    # Tau score peaks around 30-90 days, decays at extremes
    tau_days = tau * 365
    tau_score = np.exp(-0.5 * ((tau_days - 60) / 90) ** 2)

    raw_score = moneyness_score * tau_score
    # Normalise so that mean probability ≈ base_keep_frac
    mean_score = raw_score.mean()
    if mean_score < 1e-10:
        # Fallback to uniform if scores are degenerate
        return rng.random(len(log_moneyness)) < base_keep_frac

    prob = raw_score * (base_keep_frac / mean_score)
    prob = np.clip(prob, 0.02, 1.0)  # floor: always some chance to observe

    return rng.random(len(prob)) < prob


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def apply_mask(
    df: pd.DataFrame,
    strategy: str,
    keep_frac: float = 0.4,
    rng: np.random.Generator | None = None,
    seed: int = 42,
    paired_coord: bool = False,
    paired_coord_decimals: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """Add an ``observed`` boolean column to *df* using the chosen strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``log_moneyness`` and ``tau`` columns. When
        ``paired_coord=True`` and a ``date`` column is present, the
        rounded key also includes ``date``.
    strategy : str
        One of 'random', 'drop_short_maturity', 'drop_long_maturity',
        'drop_random_maturity', 'drop_wings', 'drop_atm', 'realistic'.
    keep_frac : float
        Target fraction of points to keep (for 'random' and 'realistic').
    rng : np.random.Generator, optional
        If None, one is created from *seed*.
    seed : int
        Seed for the random generator (used only if rng is None).
    paired_coord : bool, default False
        When True, rows sharing a rounded ``(date, log_m, tau)`` key
        flip observed/held-out together. The first row encountered for
        each key sets the value for the rest. Default-off path is
        byte-identical to the legacy behaviour (3X.3 / ADR 0006 D4).
    paired_coord_decimals : int, default 10
        Rounding tolerance for the paired key. Matches the audit's
        primary coordinate-rounding tolerance.
    **kwargs
        Forwarded to the underlying masking function.

    Returns
    -------
    pd.DataFrame with new column ``observed`` (bool).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n = len(df)
    tau = df["tau"].values
    log_m = df["log_moneyness"].values

    if strategy == "random":
        mask = random_uniform_mask(n, keep_frac, rng)
    elif strategy == "drop_short_maturity":
        mask = drop_maturity_mask(tau, "short", rng, **kwargs)
    elif strategy == "drop_long_maturity":
        mask = drop_maturity_mask(tau, "long", rng, **kwargs)
    elif strategy == "drop_random_maturity":
        mask = drop_maturity_mask(tau, "random_slice", rng, **kwargs)
    elif strategy == "drop_wings":
        mask = drop_moneyness_mask(log_m, "wings", rng, **kwargs)
    elif strategy == "drop_atm":
        mask = drop_moneyness_mask(log_m, "atm", rng, **kwargs)
    elif strategy == "realistic":
        mask = realistic_liquidity_mask(log_m, tau, keep_frac, rng)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")

    if paired_coord and n > 0:
        mask = _broadcast_to_paired_key(
            df, mask, decimals=paired_coord_decimals
        )

    df = df.copy()
    df["observed"] = mask
    return df


def _broadcast_to_paired_key(
    df: pd.DataFrame,
    mask: np.ndarray,
    decimals: int,
) -> np.ndarray:
    """Broadcast the first per-key mask value to every row in its key group.

    Key is the rounded ``(date, log_m, tau)`` triple; ``date`` is omitted
    if absent. Result is deterministic in row order (the first row
    encountered in *df* fixes the value for the rest in the same group).
    """
    n = len(df)
    lm_r = np.round(df["log_moneyness"].to_numpy(dtype=float), decimals)
    tau_r = np.round(df["tau"].to_numpy(dtype=float), decimals)
    key_cols: list[str] = ["lm_r", "tau_r"]
    key_frame: dict[str, np.ndarray] = {"lm_r": lm_r, "tau_r": tau_r}
    if "date" in df.columns:
        key_frame = {"date": df["date"].to_numpy(), **key_frame}
        key_cols = ["date", *key_cols]
    key_frame["__row"] = np.arange(n, dtype=np.int64)
    key_df = pd.DataFrame(key_frame)
    representative = (
        key_df.groupby(key_cols, sort=False)["__row"]
        .transform("min")
        .to_numpy()
    )
    return np.asarray(mask, dtype=bool)[representative]
