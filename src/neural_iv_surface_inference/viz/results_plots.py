"""Aggregate metric figures from the committed Phase 1 result CSVs.

Each function takes already-loaded data (DataFrame / metric dict) and returns a
Matplotlib ``Figure``. No file I/O, no ``plt.show``. Bucket ordering follows the
canonical order in ``training.eval`` so figures stay consistent with the metric
definitions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_iv_surface_inference.training.eval import (
    TAU_BUCKETS,
    MONEYNESS_BUCKETS,
)
from neural_iv_surface_inference.viz.style import PALETTE

_MODEL_COLOR = {"interp": PALETTE["interp"], "mlp": PALETTE["mlp"]}


def _model_color(model_name: str) -> str:
    key = "mlp" if "mlp" in str(model_name).lower() else "interp"
    return _MODEL_COLOR[key]


def plot_baseline_comparison(
    results_df: pd.DataFrame,
    split: str = "test",
    metrics: tuple[str, ...] = ("overall_mae", "unobserved_mae"),
    title: str | None = None,
) -> plt.Figure:
    """Grouped bars comparing models on the chosen metrics for one split."""
    df = results_df[results_df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"no rows for split={split!r}")

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        offset = (k - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, df[metric], width=width,
               label=metric.replace("_", " "),
               color=PALETTE["accent"] if k else PALETTE["observed"],
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Error")
    ax.set_title(title or f"Baseline comparison — {split} split")
    ax.legend()
    return fig


def plot_observed_vs_unobserved(
    results_df: pd.DataFrame,
    split: str = "test",
    title: str | None = None,
) -> plt.Figure:
    """Per-model observed vs unobserved MAE — exposes whether a model
    generalizes to unseen points or merely fits observed ones."""
    df = results_df[results_df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"no rows for split={split!r}")

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, df["observed_mae"], width=width, label="observed",
           color=PALETTE["observed"], edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, df["unobserved_mae"], width=width, label="unobserved",
           color=PALETTE["unobserved"], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("MAE")
    ax.set_title(title or f"Observed vs unobserved MAE — {split} split")
    ax.legend()
    return fig


def plot_regional_error_bars(
    results_df: pd.DataFrame,
    model: str,
    dimension: str = "moneyness",
    split: str = "test",
    metric: str = "mae",
    title: str | None = None,
) -> plt.Figure:
    """Bar chart of error across maturity or moneyness buckets for one model."""
    if dimension not in ("maturity", "moneyness"):
        raise ValueError("dimension must be 'maturity' or 'moneyness'")
    buckets = list(
        TAU_BUCKETS if dimension == "maturity" else MONEYNESS_BUCKETS
    )

    row = results_df[
        (results_df["split"] == split) & (results_df["model"] == model)
    ]
    if row.empty:
        raise ValueError(f"no row for model={model!r}, split={split!r}")
    row = row.iloc[0]

    values = [row.get(f"by_{dimension}_{b}_{metric}", np.nan) for b in buckets]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(buckets, values, color=_model_color(model),
           edgecolor="white", linewidth=0.6)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel(dimension)
    ax.set_title(title or f"{model} — {metric.upper()} by {dimension} ({split})")
    ax.tick_params(axis="x", rotation=15)
    return fig


def plot_joint_error_heatmap(
    grid: pd.DataFrame,
    counts: pd.DataFrame | None = None,
    metric: str = "mae",
    title: str | None = None,
    annotate: bool = True,
) -> plt.Figure:
    """Heatmap of the joint maturity x moneyness error grid.

    ``grid`` is the output of ``eval.evaluate_predictions_2d`` (rows = maturity,
    columns = moneyness). If ``counts`` is given, cells are annotated with the
    error value and (small) point count.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = grid.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=PALETTE["error_cmap"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric.upper())

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=20)
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("moneyness")
    ax.set_ylabel("maturity")
    ax.set_title(title or f"Joint {metric.upper()} by maturity x moneyness")
    ax.grid(False)

    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                label = f"{val:.3f}"
                if counts is not None:
                    label += f"\nn={int(counts.iloc[i, j]):,}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, color="white")
    return fig


def plot_noise_sweep(
    sweep_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("overall_mae", "unobserved_mae"),
    title: str | None = None,
) -> plt.Figure:
    """Line plot of error across low/medium/high noise regimes.

    ``sweep_df`` must carry a ``variant`` column whose names contain
    ``noiselow`` / ``noisemed`` / ``noisehigh``.
    """
    rank = {"noiselow": 1, "noisemed": 2, "noisehigh": 3}

    def _rank(name: str) -> int:
        for k, v in rank.items():
            if k in str(name):
                return v
        return 0

    df = sweep_df.copy()
    df["noise_rank"] = df["variant"].map(_rank)
    df = df[df["noise_rank"] > 0].sort_values("noise_rank")
    if df.empty:
        raise ValueError("no noise-regime rows found in sweep_df")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for metric in metrics:
        if metric not in df.columns:
            continue
        ax.plot(df["noise_rank"], df[metric], marker="o",
                label=metric.replace("_", " "))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_xlabel("Noise regime")
    ax.set_ylabel("MAE")
    ax.set_title(title or "Error vs noise regime")
    ax.legend()
    return fig
