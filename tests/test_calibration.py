"""Unit tests for the W4 calibration + fusion layer (story 2D.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.adapters import CalibratedConditionalPredictor
from neural_iv_surface_inference.eval.calibration import (
    Calibrator,
    apply_monotone_scaling,
    fit_conformal_quantile,
    fit_monotone_scaling,
    fit_temperature_gaussian,
    fuse_uncertainty,
    gaussian_z,
)
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.eval.uncertainty_metrics import (
    error_uncertainty_correlation,
    interval_coverage,
)


# ---------------------------------------------------------------------------
# Temperature scaling — Gaussian head
# ---------------------------------------------------------------------------


def test_temperature_recovers_true_scale_when_sigma_underestimated():
    rng = np.random.default_rng(0)
    n = 20_000
    true_sigma = 1.0
    y = rng.normal(0.0, true_sigma, size=n)
    mu = np.zeros(n)
    # Predicted sigma is half the true scale; T should land near 2.0.
    sigma_hat = np.full(n, 0.5)
    T = fit_temperature_gaussian(y, mu, sigma_hat, alpha=0.9)
    assert 1.7 < T < 2.3
    z = gaussian_z(0.9)
    cov = float(np.mean(np.abs(y - mu) <= z * T * sigma_hat))
    assert abs(cov - 0.9) < 0.01


def test_temperature_gaussian_well_calibrated_returns_near_one():
    rng = np.random.default_rng(1)
    n = 20_000
    y = rng.normal(0.0, 1.0, size=n)
    T = fit_temperature_gaussian(y, np.zeros(n), np.ones(n), alpha=0.9)
    assert 0.9 < T < 1.1


# ---------------------------------------------------------------------------
# Conformal quantile adjustment
# ---------------------------------------------------------------------------


def test_conformal_widens_too_tight_interval_to_nominal():
    rng = np.random.default_rng(2)
    n = 10_000
    y = rng.normal(0.0, 1.0, size=n)
    # Raw 50% interval — should be widened so val coverage hits 90%.
    z = 0.6745
    q_lo = np.full(n, -z)
    q_hi = np.full(n, z)
    delta = fit_conformal_quantile(y, q_lo, q_hi, alpha=0.9)
    assert delta > 0
    cov = float(np.mean((y >= q_lo - delta) & (y <= q_hi + delta)))
    assert cov >= 0.9 - 0.005


def test_conformal_can_tighten_too_wide_interval():
    rng = np.random.default_rng(3)
    n = 10_000
    y = rng.normal(0.0, 1.0, size=n)
    # Raw 99.9% interval — should be tightened (delta < 0) for 90% nominal.
    q_lo = np.full(n, -3.3)
    q_hi = np.full(n, 3.3)
    delta = fit_conformal_quantile(y, q_lo, q_hi, alpha=0.9)
    assert delta < 0
    cov = float(np.mean((y >= q_lo - delta) & (y <= q_hi + delta)))
    assert abs(cov - 0.9) < 0.02


def test_conformal_asymmetric_fixture_meets_nominal():
    rng = np.random.default_rng(4)
    n = 8_000
    # Asymmetric noise — heavy upper tail.
    y = rng.standard_exponential(n) - 1.0
    q_lo = np.full(n, -0.5)
    q_hi = np.full(n, 1.0)
    delta = fit_conformal_quantile(y, q_lo, q_hi, alpha=0.9)
    cov = float(np.mean((y >= q_lo - delta) & (y <= q_hi + delta)))
    assert cov >= 0.88


# ---------------------------------------------------------------------------
# Monotone scaling + fusion
# ---------------------------------------------------------------------------


def test_monotone_scaling_recovers_linear_relationship():
    rng = np.random.default_rng(5)
    n = 5_000
    src = rng.uniform(0.0, 1.0, size=n)
    abs_err = 0.3 * src + 0.05 + rng.normal(0.0, 0.01, size=n)
    a, b = fit_monotone_scaling(abs_err, src)
    assert 0.2 < a < 0.4
    assert abs(b - 0.05) < 0.03


def test_monotone_scaling_clamps_negative_slope_to_zero():
    rng = np.random.default_rng(6)
    n = 1_000
    src = rng.uniform(0.0, 1.0, size=n)
    abs_err = -0.5 * src + 0.7 + rng.normal(0.0, 0.01, size=n)
    a, _ = fit_monotone_scaling(abs_err, src)
    assert a == 0.0


def test_apply_monotone_scaling_is_non_negative():
    out = apply_monotone_scaling(np.array([-1.0, 0.0, 2.0]), 0.5, -0.1)
    assert np.all(out >= 0.0)


def test_fusion_is_monotone_in_each_source():
    rng = np.random.default_rng(7)
    n = 200
    s_primary = rng.uniform(0.01, 1.0, size=n)
    s_ens = rng.uniform(0.01, 1.0, size=n)
    s_msk = rng.uniform(0.01, 1.0, size=n)
    u_base = fuse_uncertainty(s_primary, s_ens, s_msk)
    # Increasing any source non-negatively should never decrease u.
    u_more_primary = fuse_uncertainty(s_primary + 0.1, s_ens, s_msk)
    u_more_ens = fuse_uncertainty(s_primary, s_ens + 0.1, s_msk)
    u_more_msk = fuse_uncertainty(s_primary, s_ens, s_msk + 0.1)
    assert np.all(u_more_primary >= u_base - 1e-9)
    assert np.all(u_more_ens >= u_base - 1e-9)
    assert np.all(u_more_msk >= u_base - 1e-9)


# ---------------------------------------------------------------------------
# End-to-end Calibrator
# ---------------------------------------------------------------------------


def _gaussian_val_test(seed: int, n: int, true_sigma: float, sigma_hat: float):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, true_sigma, size=n)
    mu = np.zeros(n)
    sigma = np.full(n, sigma_hat)
    return y, mu, sigma


def test_calibrator_gaussian_meets_nominal_coverage_on_holdout():
    y_val, mu_val, sigma_val = _gaussian_val_test(0, 10_000, 1.0, 0.5)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=y_val,
        mu=mu_val,
        sigma=sigma_val,
        nominal_alpha=0.9,
    )
    y_test, mu_test, sigma_test = _gaussian_val_test(99, 10_000, 1.0, 0.5)
    out = cal.apply(mu=mu_test, sigma=sigma_test)
    cov = interval_coverage(y_test, out["lower"], out["upper"])
    assert abs(cov - 0.9) < 0.02


def test_calibrator_quantile_meets_nominal_coverage_on_holdout():
    rng = np.random.default_rng(0)
    n = 8_000
    y_val = rng.normal(0.0, 1.0, size=n)
    # Tighter-than-90% raw interval.
    q_lo_val = np.full(n, -0.8)
    q_hi_val = np.full(n, 0.8)
    cal = Calibrator.fit(
        head_kind="quantile",
        y_true=y_val,
        mu=np.zeros(n),
        q_lo=q_lo_val,
        q_hi=q_hi_val,
        nominal_alpha=0.9,
    )
    y_test = rng.normal(0.0, 1.0, size=n)
    out = cal.apply(
        mu=np.zeros(n),
        q_lo=np.full(n, -0.8),
        q_hi=np.full(n, 0.8),
    )
    cov = interval_coverage(y_test, out["lower"], out["upper"])
    assert abs(cov - 0.9) < 0.02


def test_calibrator_confidence_signal_correlates_with_error():
    rng = np.random.default_rng(1)
    n = 5_000
    # Heteroscedastic ground truth: noise scales with sigma_hat.
    sigma_true = rng.uniform(0.1, 1.0, size=n)
    y = rng.normal(0.0, sigma_true)
    mu = np.zeros(n)
    sigma_hat = 0.5 * sigma_true  # under-confident model
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=y,
        mu=mu,
        sigma=sigma_hat,
        nominal_alpha=0.9,
    )
    out = cal.apply(mu=mu, sigma=sigma_hat)
    corr = error_uncertainty_correlation(np.abs(y - mu), out["uncertainty"])
    assert corr["pearson"] > 0.5
    assert corr["spearman"] > 0.5


def test_calibrator_roundtrip_json(tmp_path):
    y, mu, sigma = _gaussian_val_test(7, 2_000, 1.0, 0.5)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=y,
        mu=mu,
        sigma=sigma,
        nominal_alpha=0.9,
    )
    path = tmp_path / "cal.json"
    cal.save(path)
    cal2 = Calibrator.load(path)
    assert cal2.fit_params == cal.fit_params
    out1 = cal.apply(mu=mu, sigma=sigma)
    out2 = cal2.apply(mu=mu, sigma=sigma)
    np.testing.assert_allclose(out1["upper"], out2["upper"])
    np.testing.assert_allclose(out1["confidence_score"], out2["confidence_score"])


# ---------------------------------------------------------------------------
# CalibratedConditionalPredictor wiring
# ---------------------------------------------------------------------------


class _MockGaussianBase:
    def __init__(self, mu, sigma):
        self._mu = np.asarray(mu, dtype=np.float32)
        self._sigma = np.asarray(sigma, dtype=np.float32)

    def predict(self, df):
        assert len(df) == len(self._mu)
        return PredictionResult(
            pred=self._mu.copy(),
            uncertainty=self._sigma.copy(),
            meta={"model": "mock_gaussian"},
        )


class _MockEnsemble:
    def __init__(self, mu, disagreement):
        self._mu = np.asarray(mu, dtype=np.float32)
        self._dis = np.asarray(disagreement, dtype=np.float32)

    def predict(self, df):
        return PredictionResult(
            pred=self._mu.copy(),
            uncertainty=self._dis.copy(),
            meta={"model": "mock_ensemble"},
        )


def test_calibrated_predictor_fills_all_four_slots_and_meta_score():
    rng = np.random.default_rng(123)
    n = 1_500
    sigma_true = rng.uniform(0.1, 1.0, size=n)
    y = rng.normal(0.0, sigma_true)
    mu = np.zeros(n)
    sigma_hat = 0.5 * sigma_true
    disagreement = 0.2 * sigma_true + rng.normal(0, 0.02, size=n)

    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=y,
        mu=mu,
        sigma=sigma_hat,
        disagreement=disagreement,
        nominal_alpha=0.9,
    )
    base = _MockGaussianBase(mu, sigma_hat)
    ens = _MockEnsemble(mu, disagreement)
    df = pd.DataFrame({"_x": np.arange(n)})

    pred = CalibratedConditionalPredictor(base, cal, ensemble=ens)
    res = pred.predict(df)

    assert isinstance(res, PredictionResult)
    assert res.uncertainty is not None and res.uncertainty.shape == (n,)
    assert res.lower is not None and res.upper is not None
    assert np.all(res.upper >= res.lower)
    conf = res.meta["confidence_score"]
    assert conf.shape == (n,)
    assert np.all((conf >= 0.0) & (conf <= 1.0))

    cov = interval_coverage(y, res.lower, res.upper)
    assert abs(cov - 0.9) < 0.03


def test_calibrated_predictor_raises_when_required_signal_missing():
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=np.array([0.1, -0.2, 0.3]),
        mu=np.zeros(3),
        sigma=np.array([0.5, 0.5, 0.5]),
        nominal_alpha=0.9,
    )

    class _BaseNoSigma:
        def predict(self, df):
            return PredictionResult(pred=np.zeros(len(df)), uncertainty=None)

    with pytest.raises(ValueError, match="sigma"):
        CalibratedConditionalPredictor(_BaseNoSigma(), cal).predict(
            pd.DataFrame({"_x": [0, 1, 2]})
        )
