"""Noise injection for IV surface benchmark tasks.

Adds controlled perturbations to implied-volatility values on *observed*
points, simulating bid/ask noise, staleness, and measurement error.

Three pre-defined regimes (low / medium / high) plus a custom mode.
Noise is additive in IV space:  iv_noisy = iv_clean + ε,  ε ~ N(0, σ²).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Pre-defined noise regimes  (σ in IV units, e.g. 0.01 = 1 vol point)
NOISE_REGIMES: dict[str, float] = {
    "none": 0.0,
    "low": 0.01,       # ~1 vol point
    "medium": 0.03,    # ~3 vol points
    "high": 0.05,      # ~5 vol points
}


def add_gaussian_noise(
    iv: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to IV values.

    Parameters
    ----------
    iv : np.ndarray
        Clean implied-volatility values.
    sigma : float
        Standard deviation of the noise (in IV units).
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray — noisy IV values, floored at 1e-4 to stay positive.
    """
    if sigma <= 0:
        return iv.copy()
    noisy = iv + rng.normal(0, sigma, size=len(iv))
    return np.maximum(noisy, 1e-4)


def add_heteroscedastic_noise(
    iv: np.ndarray,
    log_moneyness: np.ndarray,
    tau: np.ndarray,
    base_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add noise that is larger in illiquid regions (wings, short-dated).

    The noise standard deviation scales as:
        σ_local = base_sigma * (1 + 2 * |log_moneyness| / 0.3) * (1 + 0.5 / max(tau*365, 1))

    This produces ~base_sigma near ATM / medium-maturity and 2-5x more
    in the wings or very short-dated, matching real market noise patterns.
    """
    moneyness_factor = 1.0 + 2.0 * np.abs(log_moneyness) / 0.3
    tau_days = np.maximum(tau * 365, 1.0)
    tau_factor = 1.0 + 0.5 / tau_days
    sigma_local = base_sigma * moneyness_factor * tau_factor
    noise = rng.normal(0, 1, size=len(iv)) * sigma_local
    return np.maximum(iv + noise, 1e-4)


def inject_noise(
    df: pd.DataFrame,
    regime: str = "low",
    sigma: float | None = None,
    heteroscedastic: bool = False,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Add noise to the ``implied_volatility`` of *observed* points.

    Unobserved points (``observed == False``) are left unchanged — they
    serve as clean ground truth for evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``implied_volatility``, ``observed``, and (for
        heteroscedastic mode) ``log_moneyness`` and ``tau``.
    regime : str
        One of 'none', 'low', 'medium', 'high', or 'custom'.
        Ignored if *sigma* is provided explicitly.
    sigma : float, optional
        Explicit noise σ.  Overrides *regime* if given.
    heteroscedastic : bool
        If True, use location-dependent noise instead of i.i.d.
    rng : np.random.Generator, optional
        If None, created from *seed*.
    seed : int
        Seed for the random generator.

    Returns
    -------
    pd.DataFrame with:
      - ``iv_clean`` : original implied_volatility (always preserved)
      - ``implied_volatility`` : noisy for observed, clean for unobserved
      - ``noise_sigma`` : the σ used (scalar, stored as column for provenance)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if sigma is None:
        if regime not in NOISE_REGIMES:
            raise ValueError(
                f"Unknown noise regime '{regime}'. "
                f"Choose from {list(NOISE_REGIMES)} or pass sigma explicitly."
            )
        sigma = NOISE_REGIMES[regime]

    df = df.copy()
    df["iv_clean"] = df["implied_volatility"].values.copy()
    df["noise_sigma"] = sigma

    obs_mask = df["observed"].values
    iv = df["implied_volatility"].values.copy()  # copy to avoid read-only arrays from Parquet

    if sigma > 0 and obs_mask.any():
        obs_idx = np.where(obs_mask)[0]
        iv_obs = iv[obs_idx]

        if heteroscedastic:
            iv_noisy = add_heteroscedastic_noise(
                iv_obs,
                df["log_moneyness"].values[obs_idx],
                df["tau"].values[obs_idx],
                sigma,
                rng,
            )
        else:
            iv_noisy = add_gaussian_noise(iv_obs, sigma, rng)

        iv[obs_idx] = iv_noisy
        df["implied_volatility"] = iv

    return df
