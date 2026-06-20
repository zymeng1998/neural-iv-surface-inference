"""Unit tests for the RBF-prior hybrid calibrator re-fit (story 4A.6).

4A.6 re-fits the 2D.4 / 3X.11 calibrator on the residual-hybrid's cached val
predictions (4A.4 gaussian + 4A.5 ensemble disagreement). The fusion formula,
temperature-scaling path, and the ``Calibrator`` /
``CalibratedConditionalPredictor`` classes are unchanged (covered by
``tests/test_calibration*.py``); only the input bundle differs. The
distinguishing feature of the hybrid bundle is that ``mu`` is the *summed*
prediction ``sigma_hat = rbf_pred + f_theta`` (so it sits on the IV scale),
while ``sigma`` is the residual-scale spread (unchanged by the additive
shift). These tests exercise the re-fit on a tiny synthetic hybrid-shaped
bundle — no dependency on the large gitignored CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.calibration import Calibrator
from neural_iv_surface_inference.eval.uncertainty_metrics import interval_coverage


def _hybrid_gaussian_bundle(seed: int, n: int):
    """Residual-hybrid-shaped (gaussian, ensemble) frames.

    mu = rbf_pred + residual (on the IV scale); sigma = residual-scale spread
    (under-confident); disagreement tracks the true noise. Mirrors the 4A.4 /
    4A.5 evaluator columns.
    """
    rng = np.random.default_rng(seed)
    rbf_pred = rng.uniform(0.05, 0.30, size=n)        # RBF prior on the IV scale
    sigma_true = rng.uniform(0.004, 0.05, size=n)
    residual = rng.normal(0.0, 0.01, size=n)          # small structured residual
    mu = rbf_pred + residual                          # summed hybrid sigma_hat
    iv_true = mu + rng.normal(0.0, sigma_true)
    sigma_hat = 0.5 * sigma_true                       # under-estimates spread
    disagreement = 0.3 * sigma_true + rng.normal(0.0, 0.001, size=n)
    keys = {
        "date": np.repeat("2020-01-02", n),
        "log_moneyness": rng.uniform(-0.3, 0.3, size=n),
        "tau": rng.uniform(0.02, 1.0, size=n),
        "observed": np.ones(n, dtype=int),
        "iv_true": iv_true,
        "iv_clean": iv_true,
        "rbf_pred": rbf_pred,   # hybrid passthrough; ignored by the calibrator
    }
    import pandas as pd
    gaussian = pd.DataFrame({**keys, "mu": mu, "sigma": sigma_hat,
                             "log_sigma2": np.log(np.maximum(sigma_hat, 1e-6) ** 2)})
    ensemble = pd.DataFrame({**keys, "mu": mu, "disagreement_std": disagreement})
    return gaussian, ensemble


def test_hybrid_gaussian_fit_temperature_finite_positive():
    g, e = _hybrid_gaussian_bundle(seed=0, n=8_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float), mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float),
        nominal_alpha=0.9, extra={"source": "synthetic_hybrid_gaussian"},
    )
    fp = cal.fit_params
    assert fp.temperature is not None and np.isfinite(fp.temperature) and fp.temperature > 0.0
    assert fp.has_ensemble is True
    assert np.isfinite(fp.u0) and np.isfinite(fp.u_scale) and fp.u_scale > 0.0


def test_hybrid_gaussian_recovers_nominal_coverage_on_holdout():
    g, e = _hybrid_gaussian_bundle(seed=0, n=10_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float), mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float), nominal_alpha=0.9,
    )
    g2, _ = _hybrid_gaussian_bundle(seed=99, n=10_000)
    out = cal.apply(mu=g2["mu"].to_numpy(float), sigma=g2["sigma"].to_numpy(float))
    cov = interval_coverage(g2["iv_true"].to_numpy(float), out["lower"], out["upper"])
    assert abs(cov - 0.9) < 0.02


def test_hybrid_calibrator_json_roundtrips(tmp_path):
    g, e = _hybrid_gaussian_bundle(seed=7, n=4_000)
    cal = Calibrator.fit(
        head_kind="gaussian",
        y_true=g["iv_true"].to_numpy(float), mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float), nominal_alpha=0.9,
    )
    path = tmp_path / "4a6_hybrid.json"
    cal.save(path)
    reloaded = Calibrator.load(path)
    assert reloaded.fit_params == cal.fit_params


def test_hybrid_high_confidence_mae_below_overall():
    """The fusion ranks error: keeping the high-confidence subset lowers MAE."""
    from neural_iv_surface_inference.eval.adapters import CalibratedConditionalPredictor
    from neural_iv_surface_inference.eval.predictor import PredictionResult

    g, e = _hybrid_gaussian_bundle(seed=5, n=6_000)
    y = g["iv_true"].to_numpy(float)
    cal = Calibrator.fit(
        head_kind="gaussian", y_true=y, mu=g["mu"].to_numpy(float),
        sigma=g["sigma"].to_numpy(float),
        disagreement=e["disagreement_std"].to_numpy(float), nominal_alpha=0.9,
    )

    class _Base:
        def __init__(s, mu, sig): s._mu = np.asarray(mu, np.float32); s._s = np.asarray(sig, np.float32)
        def predict(s, df): return PredictionResult(pred=s._mu.copy(), uncertainty=s._s.copy(), meta={})

    class _Ens:
        def __init__(s, mu, d): s._mu = np.asarray(mu, np.float32); s._d = np.asarray(d, np.float32)
        def predict(s, df): return PredictionResult(pred=s._mu.copy(), uncertainty=s._d.copy(), meta={})

    base = _Base(g["mu"].to_numpy(), g["sigma"].to_numpy())
    ens = _Ens(e["mu"].to_numpy(), e["disagreement_std"].to_numpy())
    df = g[["date", "log_moneyness", "tau", "observed"]].copy()
    res = CalibratedConditionalPredictor(base, cal, ensemble=ens).predict(df)
    abs_err = np.abs(y - np.asarray(res.pred, float))
    conf = np.asarray(res.meta["confidence_score"], float)
    hi = conf >= 0.5
    assert hi.any()
    assert float(np.mean(abs_err[hi])) < float(np.mean(abs_err))
