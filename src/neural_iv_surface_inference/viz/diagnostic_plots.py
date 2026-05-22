"""Region heatmaps for the W2 diagnostics (Phase 2, story 2B.4).

Visualizes the per-region risk and instability surfaces produced by
``diagnostics.risk_flags.bin_to_regions``. Each function takes an
already-binned ``(tau x moneyness)`` grid (a ``pd.DataFrame`` from
``bin_to_regions``) and returns a Matplotlib ``Figure`` — the caller owns
saving. Styling reuses the Phase 1 house palette so diagnostics figures sit
alongside the W1 result gallery.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_iv_surface_inference.viz.style import PALETTE


def _region_heatmap(
    grid: pd.DataFrame,
    cmap: str,
    cbar_label: str,
    title: str,
    fmt: str,
    annotate: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = grid.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=20)
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("moneyness")
    ax.set_ylabel("maturity")
    ax.set_title(title)
    ax.grid(False)

    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=8, color="white")
    return fig


def plot_risk_region_heatmap(
    grid: pd.DataFrame,
    title: str | None = None,
    cbar_label: str = "risk",
    annotate: bool = True,
) -> plt.Figure:
    """Heatmap of a binned risk grid (e.g. mean ``risk_score`` or flag fraction).

    ``grid`` is the output of ``bin_to_regions`` (rows = maturity buckets,
    columns = moneyness buckets).
    """
    return _region_heatmap(
        grid,
        cmap=PALETTE["error_cmap"],
        cbar_label=cbar_label,
        title=title or "No-arb risk by maturity x moneyness",
        fmt=".2f",
        annotate=annotate,
    )


def plot_instability_heatmap(
    grid: pd.DataFrame,
    title: str | None = None,
    cbar_label: str = "instability (std)",
    annotate: bool = True,
) -> plt.Figure:
    """Heatmap of a binned masking-instability grid (e.g. mean per-point std)."""
    return _region_heatmap(
        grid,
        cmap=PALETTE["iv_cmap"],
        cbar_label=cbar_label,
        title=title or "Masking instability by maturity x moneyness",
        fmt=".3f",
        annotate=annotate,
    )
