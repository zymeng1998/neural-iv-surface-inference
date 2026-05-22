"""Tests for abstention / risk–coverage curves (Story 2A.4).

Synthetic data with analytically known answers — no model or data files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.abstention import (
    RiskCoverageCurve,
    risk_coverage_curve,
    area_under_risk_coverage,
)


# ---------------------------------------------------------------------------
# risk_coverage_curve
# ---------------------------------------------------------------------------

class TestRiskCoverageCurve:
    def test_keep_all_endpoint_equals_overall_mae(self):
        rng = np.random.default_rng(0)
        ae = np.abs(rng.standard_normal(500))
        conf = rng.standard_normal(500)
        curve = risk_coverage_curve(ae, conf, n_points=20)
        assert curve.coverage[-1] == pytest.approx(1.0)
        assert curve.retained_mae[-1] == pytest.approx(ae.mean())
        assert curve.n_retained[-1] == 500

    def test_perfect_ranking_monotone_decreasing(self):
        # confidence perfectly (inversely) ranks error: high conf → low error.
        n = 1000
        ae = np.linspace(0.0, 1.0, n)
        conf = -ae  # most confident = smallest error
        curve = risk_coverage_curve(ae, conf, n_points=50)
        # arrays ordered by increasing coverage → retained error non-decreasing
        diffs = np.diff(curve.retained_mae)
        assert np.all(diffs >= -1e-12)
        # retaining only the most-confident subset → strictly lower error than all
        assert curve.retained_mae[0] < curve.retained_mae[-1]

    def test_random_confidence_approximately_flat(self):
        rng = np.random.default_rng(42)
        n = 20_000
        ae = np.abs(rng.standard_normal(n))
        conf = rng.standard_normal(n)  # independent of error
        curve = risk_coverage_curve(ae, conf, n_points=20)
        overall = ae.mean()
        # ignore the very smallest coverage levels (tiny subset → noisy);
        # the bulk of the curve should hug overall MAE.
        bulk = curve.retained_mae[curve.coverage >= 0.2]
        assert np.allclose(bulk, overall, atol=0.05)

    def test_arrays_aligned_and_sorted(self):
        ae = np.array([0.5, 0.1, 0.9, 0.3])
        conf = np.array([0.0, 1.0, -1.0, 0.5])
        curve = risk_coverage_curve(ae, conf, n_points=4)
        assert len(curve.coverage) == len(curve.retained_mae) == len(curve.n_retained)
        assert np.all(np.diff(curve.coverage) > 0)  # strictly increasing coverage

    def test_known_small_case(self):
        # confidence order (desc): idx1(0.1), idx3(0.3), idx0(0.5), idx2(0.9)
        ae = np.array([0.5, 0.1, 0.9, 0.3])
        conf = np.array([0.0, 1.0, -1.0, 0.5])
        curve = risk_coverage_curve(ae, conf, n_points=4)
        # k=1 → just idx1 → 0.1; k=4 → mean(all)=0.45
        assert curve.retained_mae[0] == pytest.approx(0.1)
        assert curve.retained_mae[-1] == pytest.approx(0.45)

    def test_nan_rows_dropped(self):
        ae = np.array([0.1, np.nan, 0.3, 0.5])
        conf = np.array([1.0, 0.5, np.nan, 0.2])
        # finite rows: idx0 (0.1, conf1.0), idx3 (0.5, conf0.2)
        curve = risk_coverage_curve(ae, conf, n_points=2)
        assert curve.n_retained[-1] == 2
        assert curve.retained_mae[-1] == pytest.approx(0.3)  # mean(0.1, 0.5)
        assert curve.retained_mae[0] == pytest.approx(0.1)  # top conf only

    def test_empty_input(self):
        curve = risk_coverage_curve(np.array([np.nan]), np.array([np.nan]))
        assert len(curve) == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            risk_coverage_curve(np.zeros(3), np.zeros(2))

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            risk_coverage_curve(np.zeros(3), np.zeros(3), n_points=0)

    def test_curve_length_validation(self):
        with pytest.raises(ValueError):
            RiskCoverageCurve(
                coverage=np.zeros(3),
                retained_mae=np.zeros(2),
                n_retained=np.zeros(3, dtype=int),
            )


# ---------------------------------------------------------------------------
# area_under_risk_coverage
# ---------------------------------------------------------------------------

class TestAreaUnderRiskCoverage:
    def test_perfect_ranking_lower_than_random(self):
        n = 2000
        ae = np.abs(np.random.default_rng(1).standard_normal(n))
        # perfect ranking: confidence = -ae
        perfect = risk_coverage_curve(ae, -ae, n_points=50)
        # random ranking
        rand_conf = np.random.default_rng(2).standard_normal(n)
        rand = risk_coverage_curve(ae, rand_conf, n_points=50)
        assert area_under_risk_coverage(perfect) < area_under_risk_coverage(rand)

    def test_constant_error_area_equals_mae(self):
        # all errors equal → retained MAE constant = c → AURC over coverage in
        # [k_min/n, 1] approximates c * (width). For full [0,1] support the
        # trapz of a constant c against coverage spanning ~0..1 is ~c.
        n = 1000
        c = 0.3
        ae = np.full(n, c)
        conf = np.linspace(0, 1, n)
        curve = risk_coverage_curve(ae, conf, n_points=100)
        # constant integrand: trapz = c * (coverage[-1] - coverage[0])
        expected = c * (curve.coverage[-1] - curve.coverage[0])
        assert area_under_risk_coverage(curve) == pytest.approx(expected, abs=1e-9)

    def test_empty_curve_nan(self):
        curve = risk_coverage_curve(np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(area_under_risk_coverage(curve))

    def test_single_point_curve(self):
        curve = risk_coverage_curve(np.array([0.4]), np.array([1.0]))
        assert len(curve) == 1
        assert area_under_risk_coverage(curve) == pytest.approx(0.4)
