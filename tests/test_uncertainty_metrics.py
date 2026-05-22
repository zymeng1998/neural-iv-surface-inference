"""Tests for core uncertainty-evaluation metrics (Story 2A.3).

Synthetic data with analytically known answers — no model or data files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.uncertainty_metrics import (
    interval_coverage,
    mean_interval_width,
    error_uncertainty_correlation,
    confidence_bucket_metrics,
    high_confidence_mae,
)


# ---------------------------------------------------------------------------
# interval_coverage
# ---------------------------------------------------------------------------

class TestIntervalCoverage:
    def test_gaussian_coverage_tracks_nominal(self):
        rng = np.random.default_rng(0)
        n = 50_000
        y = rng.standard_normal(n)
        # 90% central interval of N(0,1): +/- 1.6449
        z = 1.6449
        lower = np.full(n, -z)
        upper = np.full(n, z)
        cov = interval_coverage(y, lower, upper)
        assert abs(cov - 0.90) < 0.01

    def test_full_and_zero_coverage(self):
        y = np.array([0.0, 1.0, 2.0])
        assert interval_coverage(y, y - 1, y + 1) == 1.0
        assert interval_coverage(y, y + 10, y + 20) == 0.0

    def test_boundary_inclusive(self):
        y = np.array([0.0, 1.0])
        lower = np.array([0.0, 1.0])
        upper = np.array([0.0, 1.0])
        assert interval_coverage(y, lower, upper) == 1.0

    def test_nan_rows_dropped(self):
        y = np.array([0.0, np.nan, 5.0])
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        # row 0 inside, row 2 outside, row 1 dropped → 0.5
        assert interval_coverage(y, lower, upper) == 0.5

    def test_all_nan_returns_nan(self):
        y = np.array([np.nan, np.nan])
        assert np.isnan(interval_coverage(y, y, y))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            interval_coverage(np.zeros(3), np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------------------
# mean_interval_width
# ---------------------------------------------------------------------------

class TestMeanIntervalWidth:
    def test_known_width(self):
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])  # width 2 each
        assert mean_interval_width(lower, upper) == 2.0

    def test_nan_dropped(self):
        lower = np.array([0.0, np.nan])
        upper = np.array([1.0, 5.0])
        assert mean_interval_width(lower, upper) == 1.0

    def test_all_nan_returns_nan(self):
        assert np.isnan(mean_interval_width(np.array([np.nan]), np.array([np.nan])))


# ---------------------------------------------------------------------------
# error_uncertainty_correlation
# ---------------------------------------------------------------------------

class TestErrorUncertaintyCorrelation:
    def test_perfect_positive(self):
        u = np.linspace(0.1, 1.0, 50)
        ae = 2.0 * u  # monotone, linear → pearson and spearman ≈ 1
        out = error_uncertainty_correlation(ae, u)
        assert out["pearson"] == pytest.approx(1.0, abs=1e-9)
        assert out["spearman"] == pytest.approx(1.0, abs=1e-9)

    def test_monotone_nonlinear_spearman_one(self):
        u = np.linspace(0.1, 1.0, 50)
        ae = u**3  # monotone but nonlinear → spearman == 1, pearson < 1
        out = error_uncertainty_correlation(ae, u)
        assert out["spearman"] == pytest.approx(1.0, abs=1e-9)
        assert out["pearson"] < 1.0

    def test_negative_correlation(self):
        u = np.linspace(0.1, 1.0, 50)
        ae = -2.0 * u + 5.0
        out = error_uncertainty_correlation(ae, u)
        assert out["pearson"] == pytest.approx(-1.0, abs=1e-9)

    def test_constant_input_returns_nan(self):
        u = np.ones(10)
        ae = np.linspace(0, 1, 10)
        out = error_uncertainty_correlation(ae, u)
        assert np.isnan(out["pearson"])
        assert np.isnan(out["spearman"])

    def test_too_few_points_returns_nan(self):
        out = error_uncertainty_correlation(np.array([1.0]), np.array([2.0]))
        assert np.isnan(out["pearson"])

    def test_nan_pairs_dropped(self):
        u = np.array([0.1, 0.2, 0.3, np.nan, 0.5])
        ae = np.array([0.2, 0.4, 0.6, 5.0, 1.0])  # finite subset perfectly linear
        out = error_uncertainty_correlation(ae, u)
        assert out["pearson"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# confidence_bucket_metrics
# ---------------------------------------------------------------------------

class TestConfidenceBucketMetrics:
    def test_recovers_error_ordering(self):
        # confidence increasing; error decreasing with confidence.
        n = 1000
        conf = np.linspace(0.0, 1.0, n)
        ae = 1.0 - conf  # high confidence → low error
        buckets = confidence_bucket_metrics(ae, conf, n_buckets=5)
        assert len(buckets) == 5
        maes = [b["mae"] for b in buckets]
        # bucket 0 = lowest confidence = highest error; strictly decreasing
        assert all(maes[i] > maes[i + 1] for i in range(len(maes) - 1))
        # counts sum to n
        assert sum(b["n"] for b in buckets) == n

    def test_bucket_index_and_range_fields(self):
        conf = np.linspace(0, 1, 100)
        ae = np.ones(100)
        buckets = confidence_bucket_metrics(ae, conf, n_buckets=4)
        assert [b["bucket"] for b in buckets] == [0, 1, 2, 3]
        assert buckets[0]["conf_lo"] == pytest.approx(0.0)
        assert buckets[-1]["conf_hi"] == pytest.approx(1.0)

    def test_empty_returns_empty_list(self):
        out = confidence_bucket_metrics(
            np.array([np.nan]), np.array([np.nan]), n_buckets=3
        )
        assert out == []

    def test_nan_rows_dropped(self):
        ae = np.array([1.0, 2.0, np.nan, 4.0])
        conf = np.array([0.1, 0.2, 0.3, np.nan])
        buckets = confidence_bucket_metrics(ae, conf, n_buckets=2)
        assert sum(b["n"] for b in buckets) == 2

    def test_invalid_n_buckets(self):
        with pytest.raises(ValueError):
            confidence_bucket_metrics(np.zeros(3), np.zeros(3), n_buckets=0)


# ---------------------------------------------------------------------------
# high_confidence_mae
# ---------------------------------------------------------------------------

class TestHighConfidenceMae:
    def test_retains_low_error_subset(self):
        # high confidence aligned with low error
        conf = np.array([0.1, 0.2, 0.8, 0.9])
        ae = np.array([4.0, 3.0, 1.0, 0.0])  # top-2 confident → errors 1.0, 0.0
        assert high_confidence_mae(ae, conf, keep_fraction=0.5) == pytest.approx(0.5)

    def test_keep_all_equals_full_mae(self):
        conf = np.array([0.1, 0.5, 0.9])
        ae = np.array([1.0, 2.0, 3.0])
        assert high_confidence_mae(ae, conf, keep_fraction=1.0) == pytest.approx(2.0)

    def test_at_least_one_retained(self):
        conf = np.array([0.1, 0.2, 0.9])
        ae = np.array([5.0, 6.0, 0.0])
        # tiny fraction still keeps the single most-confident point
        assert high_confidence_mae(ae, conf, keep_fraction=0.01) == pytest.approx(0.0)

    def test_nan_dropped(self):
        conf = np.array([0.9, np.nan, 0.1])
        ae = np.array([0.0, 5.0, 4.0])
        # finite: conf [0.9, 0.1], ae [0.0, 4.0]; keep top-1 → 0.0
        assert high_confidence_mae(ae, conf, keep_fraction=0.5) == pytest.approx(0.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(
            high_confidence_mae(np.array([np.nan]), np.array([np.nan]))
        )

    def test_invalid_keep_fraction(self):
        with pytest.raises(ValueError):
            high_confidence_mae(np.zeros(3), np.zeros(3), keep_fraction=0.0)
        with pytest.raises(ValueError):
            high_confidence_mae(np.zeros(3), np.zeros(3), keep_fraction=1.5)
