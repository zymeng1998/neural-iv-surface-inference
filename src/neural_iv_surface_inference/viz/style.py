"""Shared house style for Phase 1 figures.

Keeps a single, intentional look across the technical notebook and the
presentation figures: muted palette, clear hierarchy, light grids, readable
type. Import and call :func:`apply_house_style` once before plotting.
"""

from __future__ import annotations

# Semantic palette — assign by meaning, not decoration.
PALETTE: dict[str, str] = {
    "interp": "#2A6F97",     # deep teal-blue — interpolation baseline
    "mlp": "#E07A5F",        # terracotta — neural baseline
    "observed": "#3D405B",   # ink — observed points
    "unobserved": "#81B29A", # sage — unobserved points
    "accent": "#F2CC8F",     # warm sand — highlights
    "error_cmap": "magma",   # sequential, perceptually uniform for error maps
    "iv_cmap": "viridis",    # sequential for IV-level surfaces
    "grid": "#D9D9D9",
}


def apply_house_style() -> None:
    """Apply consistent Matplotlib rcParams. Idempotent; safe to call repeatedly."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.6,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": False,
    })
