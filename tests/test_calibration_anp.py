"""Unit tests for the ANP calibrator re-fit (story 3B.6).

3B.6 re-fits the 2D.4 calibrator on the ANP architecture's cached val
predictions. The fusion formula, temperature-scaling / split-conformal
paths, and the ``Calibrator`` / ``CalibratedConditionalPredictor`` classes
are unchanged from 2D.4 (covered by ``tests/test_calibration.py``); only the
input bundle differs. These tests exercise the *re-fit on an ANP-shaped
predictions bundle* end-to-end on a tiny in-test synthetic fixture, with no
dependency on the large cached CSVs or any Pod artifact.

Fixture schema mirrors the 3B.4 / 3B.5 evaluator output columns:
    gaussian: date, log_moneyness, tau, observed, iv_true, mu, sigma
    ensemble: ... disagreement_std
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.adapters import CalibratedConditionalPredictor
from neural_iv_surface_inference.eval.calibration import Calibrator
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.eval.uncertainty_metrics import interval_coverage


# ---------------------------------------------------------------------------
# Synthetic ANP-shaped predictions bundle
# ---------------------------------------------------------------------------


def _anp_gaussian_bundle(seed: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gaussian_frame, ensemble_frame) sharing the 3B.4/3B.5 columns.

    Heteroscedastic ground truth with an under-confident sigma_hat and an
    ensemble disagreement that tracks the true noise scale, so the fused
    confidence signal has genuine error-ranking content.
    """
    rng = np.random.default_rng(seed)
    sigma_true = rng.uniform(0.05, 0.6, size=n)
    mu = rng.normal(0.0, 0.1, size=n)
    iv_true = mu + rng.normal(0.0, sigma_true)
    sigma_hat = 0.5 * sigma_true  # model under-estimates its own spread
    disagreement = 0.3 * sigma_true + rng.normal(0.0, 0.01, size=n)

    keys = {
        "date": np.repeat("2020-01-02", n),
        "log_moneyness": rng.uniform(-0.3, 0.3, size=n),
        "tau": rng.uniform(0.02, 1.0, size=n),
        "observed": np.ones(n, dtype=int),
        "iv_true": iv_true,
    }
    gaussian = pd.DataFrame({**keys, "mu": mu, "sigma": sigma_hat})
    ensemble = pd.DataFrame({**keys, "mu": mu, "disagreement_std": disagreement})
    return gaussian, ensemble


def _anp_quantile_bundle(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    iv_true = rng.normal(0.0, 1.0, size=n)
    # Deliberately too-tight raw interval so the split-conformal offset is
    # non-trivially positive.
    q_lo = np.full(n, -0.8)
    q_hi = np.full(n, 0.8)
    return pd.DataFrame(
        {
            "date": np.repeat("2020-01-02", n),
            "log_moneyness": rng.uniform(-0.3, 0.3, size=n),
            "tau": rng.uniform(0.02, 1.0, size=n),
            "observed": np.ones(n, dtype=int),
            "iv_true": iv_true,
            "mu": 0.5 * (q_lo + q_hi),
            "q_lo": q_lo,
            "q_hi": q_hi,
        }
    )


# ---------------------------------------------------------------------------
# Fit on the ANP bundle
# ---------------------------------------------------------------------------


def test_anp_gaussian_fit_temperature_is_finite_and_positive():
    g, e = _anp_gaussian_bundle(seed=0, n=8_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float),
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9,
        extra={"source": "synthetic_anp_gaussian"},
    )
    fp = cal.fit_params
    assert fp.temperature is not None
    assert np.isfinite(fp.temperature) and fp.temperature > 0.0
    assert fp.conformal_delta is None
    assert fp.has_ensemble is True
    assert np.isfinite(fp.u0) and np.isfinite(fp.u_scale) and fp.u_scale > 0.0


def test_anp_quantile_fit_conformal_offset_is_finite():
    q = _anp_quantile_bundle(seed=1, n=6_000)
    cal = Calibrator.fit(
        head_kind="quantile",
        y_true=q["iv_true"].to_numpy(float),
        mu=q["mu"].to_numpy(float),
        q_lo=q["q_lo"].to_numpy(float),
        q_hi=q["q_hi"].to_numpy(float),
        nominal_alpha=0.9,
        extra={"source": "synthetic_anp_quantile"},
    )
    fp = cal.fit_params
    assert fp.temperature is None
    assert fp.conformal_delta is not None
    assert np.isfinite(fp.conformal_delta)


def test_anp_gaussian_fit_recovers_nominal_coverage_on_holdout():
    g, e = _anp_gaussian_bundle(seed=0, n=10_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float),
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9,
    )
    g2, _ = _anp_gaussian_bundle(seed=99, n=10_000)
    out = cal.apply(mu=g2["mu"].to_numpy(float), sigma=g2["sigma"].to_numpy(float))
    cov = interval_coverage(g2["iv_true"].to_numpy(float), out["lower"], out["upper"])
    assert abs(cov - 0.9) < 0.02


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_anp_calibrator_json_roundtrips(tmp_path):
    g, e = _anp_gaussian_bundle(seed=7, n=4_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float),
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9,
        extra={"source": "synthetic_anp_gaussian"},
    )
    path = tmp_path / "3b6_anp.json"
    cal.save(path)
    reloaded = Calibrator.load(path)
    assert reloaded.fit_params == cal.fit_params
    out1 = cal.apply(
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
    )
    out2 = reloaded.apply(
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
    )
    np.testing.assert_allclose(out1["upper"], out2["upper"])
    np.testing.assert_allclose(out1["lower"], out2["lower"])
    np.testing.assert_allclose(out1["confidence_score"], out2["confidence_score"])


# ---------------------------------------------------------------------------
# CalibratedConditionalPredictor wrapping a synthetic ANP base
# ---------------------------------------------------------------------------


class _MockANPGaussianBase:
    """Stand-in for an ANP-loaded ConditionalSurfacePredictor (gaussian head)."""

    def __init__(self, mu, sigma):
        self._mu = np.asarray(mu, dtype=np.float32)
        self._sigma = np.asarray(sigma, dtype=np.float32)

    def predict(self, df):
        assert len(df) == len(self._mu)
        return PredictionResult(
            pred=self._mu.copy(),
            uncertainty=self._sigma.copy(),
            meta={"model": "mock_anp_gaussian"},
        )


class _MockANPEnsemble:
    def __init__(self, mu, disagreement):
        self._mu = np.asarray(mu, dtype=np.float32)
        self._dis = np.asarray(disagreement, dtype=np.float32)

    def predict(self, df):
        return PredictionResult(
            pred=self._mu.copy(),
            uncertainty=self._dis.copy(),
            meta={"model": "mock_anp_ensemble"},
        )


def test_calibrated_anp_predictor_fills_all_four_fields():
    g, e = _anp_gaussian_bundle(seed=123, n=2_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float),
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9,
    )
    base = _MockANPGaussianBase(g["mu"].to_numpy(), g["sigma"].to_numpy())
    ens = _MockANPEnsemble(e["mu"].to_numpy(), e["disagreement_std"].to_numpy())
    df = g[["date", "log_moneyness", "tau", "observed"]].copy()

    pred = CalibratedConditionalPredictor(base, cal, ensemble=ens)
    res = pred.predict(df)

    n = len(g)
    assert isinstance(res, PredictionResult)
    assert res.pred is not None and res.pred.shape == (n,)
    assert res.uncertainty is not None and res.uncertainty.shape == (n,)
    assert res.lower is not None and res.lower.shape == (n,)
    assert res.upper is not None and res.upper.shape == (n,)
    assert np.all(np.isfinite(res.pred))
    assert np.all(np.isfinite(res.uncertainty))
    assert np.all(res.upper >= res.lower)
    conf = res.meta["confidence_score"]
    assert conf.shape == (n,)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_calibrated_anp_predictor_high_confidence_mae_below_overall():
    """Sanity check of the fusion: keeping the high-confidence half lowers MAE."""
    g, e = _anp_gaussian_bundle(seed=5, n=6_000)
    y = g["iv_true"].to_numpy(float)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=y,
        mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9,
    )
    base = _MockANPGaussianBase(g["mu"].to_numpy(), g["sigma"].to_numpy())
    ens = _MockANPEnsemble(e["mu"].to_numpy(), e["disagreement_std"].to_numpy())
    df = g[["date", "log_moneyness", "tau", "observed"]].copy()
    res = CalibratedConditionalPredictor(base, cal, ensemble=ens).predict(df)

    abs_err = np.abs(y - np.asarray(res.pred, dtype=float))
    conf = np.asarray(res.meta["confidence_score"], dtype=float)
    mae_all = float(np.mean(abs_err))
    hi = conf >= 0.5  # the 2D.5 decision-layer confidence threshold
    assert hi.any()
    mae_hi = float(np.mean(abs_err[hi]))
    assert mae_hi < mae_all
