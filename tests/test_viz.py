"""Tests for the visualization layer and the joint 2D evaluation grid.

Matplotlib runs headless (Agg). Figure correctness is checked structurally
(figure returned, expected axes/labels, no exceptions) on synthetic data — the
visual quality itself is reviewed by eye, not asserted here.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless before pyplot is imported anywhere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.training.eval import (
    evaluate_predictions_2d,
    evaluate_predictions_2d_counts,
    TAU_BUCKETS,
    MONEYNESS_BUCKETS,
)
from neural_iv_surface_inference.viz import (
    apply_house_style,
    plot_baseline_comparison,
    plot_observed_vs_unobserved,
    plot_regional_error_bars,
    plot_joint_error_heatmap,
    plot_noise_sweep,
    plot_surface_scatter,
    plot_reconstruction_triptych,
    plot_spatial_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def results_df():
    """Mimic artifacts/results/baseline_results.csv (subset of columns)."""
    rows = []
    for model in ("interp_rbf", "mlp"):
        for split in ("train", "val", "test"):
            row = {
                "model": model, "split": split,
                "overall_mae": 0.07, "overall_rmse": 0.11,
                "observed_mae": 0.06, "unobserved_mae": 0.08,
            }
            for b in TAU_BUCKETS:
                row[f"by_maturity_{b}_mae"] = 0.05
            for b in MONEYNESS_BUCKETS:
                row[f"by_moneyness_{b}_mae"] = 0.05
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def surface_date_df():
    rng = np.random.default_rng(0)
    n = 200
    log_m = rng.uniform(-0.5, 0.5, n)
    tau = rng.uniform(1 / 365, 1.5, n)
    iv_clean = np.clip(0.2 + 0.15 * log_m**2 + 0.02 * tau, 0.01, 3.0)
    observed = rng.random(n) < 0.4
    iv_noisy = iv_clean + rng.normal(0, 0.01, n)
    return pd.DataFrame({
        "log_moneyness": log_m, "tau": tau, "iv_clean": iv_clean,
        "implied_volatility": iv_noisy, "observed": observed,
        "iv_pred": iv_clean + rng.normal(0, 0.02, n),
    })


# ---------------------------------------------------------------------------
# 2D evaluation grid
# ---------------------------------------------------------------------------

class TestEvaluate2D:
    def test_grid_shape_and_orientation(self, surface_date_df):
        df = surface_date_df.copy()
        grid = evaluate_predictions_2d(df, metric="mae")
        assert grid.shape == (len(TAU_BUCKETS), len(MONEYNESS_BUCKETS))
        assert list(grid.index) == list(TAU_BUCKETS)
        assert list(grid.columns) == list(MONEYNESS_BUCKETS)

    def test_counts_sum_to_n(self, surface_date_df):
        counts = evaluate_predictions_2d_counts(surface_date_df)
        assert counts.to_numpy().sum() == len(surface_date_df)

    def test_perfect_pred_zero_error(self, surface_date_df):
        df = surface_date_df.copy()
        df["iv_pred"] = df["iv_clean"]
        grid = evaluate_predictions_2d(df, metric="mae")
        vals = grid.to_numpy()
        assert np.nanmax(vals) == pytest.approx(0.0, abs=1e-12)

    def test_invalid_metric(self, surface_date_df):
        with pytest.raises(ValueError):
            evaluate_predictions_2d(surface_date_df, metric="bogus")


# ---------------------------------------------------------------------------
# Results plots
# ---------------------------------------------------------------------------

class TestResultsPlots:
    def test_apply_house_style_idempotent(self):
        apply_house_style()
        apply_house_style()  # no error on repeat

    def test_baseline_comparison(self, results_df):
        fig = plot_baseline_comparison(results_df, split="test")
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "test" in ax.get_title()
        assert len(ax.get_xticklabels()) == 2  # two models

    def test_baseline_comparison_bad_split(self, results_df):
        with pytest.raises(ValueError):
            plot_baseline_comparison(results_df, split="nope")

    def test_observed_vs_unobserved(self, results_df):
        fig = plot_observed_vs_unobserved(results_df, split="test")
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "observed" in labels and "unobserved" in labels

    def test_regional_error_bars_moneyness(self, results_df):
        fig = plot_regional_error_bars(results_df, model="interp_rbf",
                                       dimension="moneyness", split="test")
        ax = fig.axes[0]
        assert len(ax.patches) == len(MONEYNESS_BUCKETS)

    def test_regional_error_bars_maturity(self, results_df):
        fig = plot_regional_error_bars(results_df, model="mlp",
                                       dimension="maturity", split="test")
        assert len(fig.axes[0].patches) == len(TAU_BUCKETS)

    def test_regional_error_bars_bad_dimension(self, results_df):
        with pytest.raises(ValueError):
            plot_regional_error_bars(results_df, model="mlp", dimension="x")

    def test_regional_error_bars_missing_model(self, results_df):
        with pytest.raises(ValueError):
            plot_regional_error_bars(results_df, model="ghost")

    def test_joint_heatmap(self, surface_date_df):
        grid = evaluate_predictions_2d(surface_date_df)
        counts = evaluate_predictions_2d_counts(surface_date_df)
        fig = plot_joint_error_heatmap(grid, counts)
        ax = fig.axes[0]
        assert len(ax.get_xticklabels()) == len(MONEYNESS_BUCKETS)
        assert len(ax.get_yticklabels()) == len(TAU_BUCKETS)

    def test_noise_sweep(self):
        sweep = pd.DataFrame({
            "variant": ["spy_phase1_random40_noiselow",
                        "spy_phase1_random40_noisemed",
                        "spy_phase1_random40_noisehigh"],
            "overall_mae": [0.070, 0.071, 0.073],
            "unobserved_mae": [0.079, 0.080, 0.081],
        })
        fig = plot_noise_sweep(sweep)
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["low", "medium", "high"]

    def test_noise_sweep_empty(self):
        with pytest.raises(ValueError):
            plot_noise_sweep(pd.DataFrame({"variant": ["x"], "overall_mae": [0.1]}))


# ---------------------------------------------------------------------------
# Surface plots
# ---------------------------------------------------------------------------

class TestSurfacePlots:
    def test_surface_scatter(self, surface_date_df):
        fig = plot_surface_scatter(surface_date_df, value_col="iv_clean")
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_xlabel() == "log-moneyness"

    def test_surface_scatter_missing_col(self, surface_date_df):
        with pytest.raises(ValueError):
            plot_surface_scatter(surface_date_df, value_col="nope")

    def test_triptych_three_panels(self, surface_date_df):
        fig = plot_reconstruction_triptych(surface_date_df, date_label="2020-01-02")
        # 3 scatter axes + 1 colorbar axis
        assert len(fig.axes) >= 3
        titles = [fig.axes[i].get_title() for i in range(3)]
        assert "Reference (clean)" in titles[0]

    def test_triptych_missing_col(self, surface_date_df):
        df = surface_date_df.drop(columns=["iv_pred"])
        with pytest.raises(ValueError):
            plot_reconstruction_triptych(df)

    def test_spatial_error(self, surface_date_df):
        fig = plot_spatial_error(surface_date_df)
        assert isinstance(fig, plt.Figure)
        assert "error" in fig.axes[0].get_title().lower()

    def test_spatial_error_missing_col(self, surface_date_df):
        with pytest.raises(ValueError):
            plot_spatial_error(surface_date_df.drop(columns=["iv_pred"]))
