"""Visualization layer for Phase 1 baseline results and IV surfaces.

Pure plotting helpers that return Matplotlib ``Figure`` objects (no implicit
``plt.show``/``savefig`` — the caller owns persistence). Two groups:

  - :mod:`results_plots` — aggregate metric figures from the committed result
    CSVs / metric dicts (baseline comparison, observed-vs-unobserved gap,
    regional error bars, joint maturity x moneyness heatmap, noise sweep).
  - :mod:`surface_plots` — per-date IV-surface visuals (reference / observed /
    reconstructed triptych, spatial point-error map). These need the benchmark
    parquet (RunPod) at call time, but the functions themselves are data-source
    agnostic and unit-tested on synthetic surfaces.
"""

from neural_iv_surface_inference.viz.results_plots import (
    plot_baseline_comparison,
    plot_observed_vs_unobserved,
    plot_regional_error_bars,
    plot_joint_error_heatmap,
    plot_noise_sweep,
)
from neural_iv_surface_inference.viz.surface_plots import (
    plot_surface_scatter,
    plot_reconstruction_triptych,
    plot_spatial_error,
)
from neural_iv_surface_inference.viz.style import apply_house_style, PALETTE

__all__ = [
    "apply_house_style",
    "PALETTE",
    "plot_baseline_comparison",
    "plot_observed_vs_unobserved",
    "plot_regional_error_bars",
    "plot_joint_error_heatmap",
    "plot_noise_sweep",
    "plot_surface_scatter",
    "plot_reconstruction_triptych",
    "plot_spatial_error",
]
