"""Per-date IV-surface visuals (the most intuitive Phase 1 figures).

Functions take a single-date DataFrame (or arrays) and return a Matplotlib
``Figure``. They are data-source agnostic — unit-tested on synthetic surfaces,
run on the real benchmark parquet on RunPod. Expected columns on a date frame:
``log_moneyness``, ``tau``, ``iv_clean``, ``implied_volatility``, ``observed``
(and ``iv_pred`` for the reconstruction / error views).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_iv_surface_inference.viz.style import PALETTE


def plot_surface_scatter(
    df_date: pd.DataFrame,
    value_col: str = "iv_clean",
    title: str | None = None,
    cmap: str | None = None,
) -> plt.Figure:
    """Scatter the surface in (log_moneyness, tau) coloured by ``value_col``."""
    if value_col not in df_date.columns:
        raise ValueError(f"missing column {value_col!r}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(
        df_date["log_moneyness"], df_date["tau"],
        c=df_date[value_col], s=14, cmap=cmap or PALETTE["iv_cmap"],
    )
    fig.colorbar(sc, ax=ax, label=value_col)
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("tau (years)")
    ax.set_title(title or f"Surface — {value_col}")
    ax.grid(True, alpha=0.3)
    return fig


def plot_reconstruction_triptych(
    df_date: pd.DataFrame,
    pred_col: str = "iv_pred",
    date_label: str | None = None,
) -> plt.Figure:
    """Three panels on a shared IV scale: reference (clean), sparse observed
    input, and reconstructed prediction. The single clearest "what the model
    does" figure for the project.
    """
    for col in ("iv_clean", "implied_volatility", "observed", pred_col):
        if col not in df_date.columns:
            raise ValueError(f"missing column {col!r}")

    obs = df_date[df_date["observed"].astype(bool)]

    # Shared colour scale across panels for honest comparison.
    vmin = float(np.nanmin(df_date["iv_clean"]))
    vmax = float(np.nanmax(df_date["iv_clean"]))
    cmap = PALETTE["iv_cmap"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    panels = [
        (axes[0], df_date, "iv_clean", "Reference (clean)"),
        (axes[1], obs, "implied_volatility", "Sparse observed input"),
        (axes[2], df_date, pred_col, "Reconstructed"),
    ]
    sc = None
    for ax, data, col, sub in panels:
        sc = ax.scatter(data["log_moneyness"], data["tau"], c=data[col],
                        s=14, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("log-moneyness")
        ax.set_title(sub)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("tau (years)")
    fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="implied vol")

    suptitle = "Surface reconstruction"
    if date_label:
        suptitle += f" — {date_label}"
    fig.suptitle(suptitle, fontweight="bold")
    return fig


def plot_spatial_error(
    df_date: pd.DataFrame,
    pred_col: str = "iv_pred",
    truth_col: str = "iv_clean",
    title: str | None = None,
) -> plt.Figure:
    """Spatial map of absolute prediction error over (log_moneyness, tau).

    Reveals *where* a model fails (e.g. short maturity, deep wings) rather than
    just a scalar MAE.
    """
    for col in (pred_col, truth_col):
        if col not in df_date.columns:
            raise ValueError(f"missing column {col!r}")

    abs_err = np.abs(df_date[pred_col].to_numpy() - df_date[truth_col].to_numpy())

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    sc = ax.scatter(df_date["log_moneyness"], df_date["tau"], c=abs_err,
                    s=14, cmap=PALETTE["error_cmap"])
    fig.colorbar(sc, ax=ax, label="absolute error")
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("tau (years)")
    ax.set_title(title or "Spatial absolute error")
    ax.grid(True, alpha=0.3)
    return fig
