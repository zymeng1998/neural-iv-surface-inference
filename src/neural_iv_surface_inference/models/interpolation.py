"""S3.1 — Simple interpolation baseline for IV surface reconstruction.

Per-date interpolation: given the observed points on a single date,
interpolate to predict IV at all query points (both observed and
unobserved).  No training required.

Supports multiple interpolation methods:
  - 'linear'   : scipy griddata with linear interpolation
  - 'cubic'    : scipy griddata with cubic interpolation
  - 'rbf'      : scipy RBF (thin-plate spline) interpolation
  - 'nearest'  : scipy griddata with nearest-neighbor (fallback)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator


def interpolate_surface_date(
    df_date: pd.DataFrame,
    method: str = "rbf",
) -> np.ndarray:
    """Interpolate IV for a single date using observed points.

    Parameters
    ----------
    df_date : pd.DataFrame
        All surface points for one date.  Must contain:
        ``log_moneyness``, ``tau``, ``implied_volatility``, ``observed``.
    method : str
        Interpolation method: 'linear', 'cubic', 'rbf', or 'nearest'.

    Returns
    -------
    np.ndarray of shape (len(df_date),) — predicted IV for every point.
    """
    obs = df_date[df_date["observed"]].copy()
    if len(obs) < 3:
        # Not enough points — fall back to mean
        mean_iv = obs["implied_volatility"].mean() if len(obs) > 0 else 0.2
        return np.full(len(df_date), mean_iv)

    coords_obs = obs[["log_moneyness", "tau"]].values
    iv_obs = obs["implied_volatility"].values
    coords_all = df_date[["log_moneyness", "tau"]].values

    if method == "rbf":
        return _rbf_interpolate(coords_obs, iv_obs, coords_all)
    elif method in ("linear", "cubic", "nearest"):
        return _griddata_interpolate(coords_obs, iv_obs, coords_all, method)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _rbf_interpolate(
    coords_obs: np.ndarray,
    iv_obs: np.ndarray,
    coords_all: np.ndarray,
) -> np.ndarray:
    """RBF interpolation with thin-plate spline kernel."""
    try:
        rbf = RBFInterpolator(
            coords_obs, iv_obs,
            kernel="thin_plate_spline",
            smoothing=1e-3,
        )
        pred = rbf(coords_all)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to linear griddata if RBF fails
        pred = _griddata_interpolate(coords_obs, iv_obs, coords_all, "linear")
    return np.clip(pred, 1e-4, 10.0)


def _griddata_interpolate(
    coords_obs: np.ndarray,
    iv_obs: np.ndarray,
    coords_all: np.ndarray,
    method: str,
) -> np.ndarray:
    """Scipy griddata interpolation with nearest-neighbor gap fill."""
    pred = griddata(coords_obs, iv_obs, coords_all, method=method)

    # griddata returns NaN for extrapolation — fill with nearest
    nan_mask = np.isnan(pred)
    if nan_mask.any():
        nearest = griddata(coords_obs, iv_obs, coords_all, method="nearest")
        pred[nan_mask] = nearest[nan_mask]

    return np.clip(pred, 1e-4, 10.0)


def run_interpolation_baseline(
    df: pd.DataFrame,
    method: str = "rbf",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run interpolation baseline on all dates in a benchmark dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full benchmark dataset with ``date``, ``log_moneyness``, ``tau``,
        ``implied_volatility``, ``observed``, ``iv_clean``, ``split`` columns.
    method : str
        Interpolation method.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame — copy of *df* with added ``iv_pred`` column.
    """
    df = df.copy()
    df["iv_pred"] = np.nan

    dates = df["date"].unique()
    for i, date in enumerate(dates):
        mask = df["date"] == date
        df_date = df[mask]
        preds = interpolate_surface_date(df_date, method=method)
        df.loc[mask, "iv_pred"] = preds

        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(dates)}] dates interpolated")

    if verbose:
        print(f"  Interpolation complete: {len(dates)} dates, method={method}")

    return df
