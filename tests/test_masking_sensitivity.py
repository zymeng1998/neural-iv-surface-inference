"""Tests for the masking-sensitivity harness (Story 2B.2).

Synthetic surfaces + deterministic stub predictors — no model or data files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.diagnostics.masking_sensitivity import (
    MaskingSensitivityResult,
    instability_summary,
    mask_resample,
    masking_sensitivity,
)
from neural_iv_surface_inference.eval.predictor import (
    PredictionResult,
    Predictor,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_surface(n_obs: int = 8, n_query: int = 5, seed: int = 0) -> pd.DataFrame:
    """Single-date surface: n_obs observed rows followed by n_query unobserved."""
    rng = np.random.default_rng(seed)
    n = n_obs + n_query
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-02"),
            "log_moneyness": rng.uniform(-0.3, 0.3, n),
            "tau": rng.uniform(0.05, 1.0, n),
            "implied_volatility": rng.uniform(0.1, 0.4, n),
            "observed": np.array([True] * n_obs + [False] * n_query),
        }
    )


class ConstantPredictor:
    """Returns the same prediction regardless of the observed mask."""

    def __init__(self, value: float = 0.2):
        self.value = value

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        return PredictionResult(
            pred=np.full(len(df), self.value), meta={"model": "constant"}
        )


class NaNPredictor:
    """Returns NaN for the last row, finite elsewhere — never-finite point."""

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        pred = np.full(len(df), 0.2)
        pred[-1] = np.nan
        return PredictionResult(pred=pred, meta={"model": "nan"})


# ---------------------------------------------------------------------------
# mask_resample
# ---------------------------------------------------------------------------

class TestMaskResample:
    def test_subset_of_observed_and_correct_count(self):
        df = make_surface(n_obs=10, n_query=4)
        observed = df["observed"].to_numpy(dtype=bool)
        masks = mask_resample(df, keep_fraction=0.5, n_draws=6, seed=1)
        assert len(masks) == 6
        for m in masks:
            assert m.shape == (len(df),)
            assert m.dtype == bool
            # never select an originally-unobserved point
            assert not np.any(m & ~observed)
            assert m.sum() == 5  # round(0.5 * 10)

    def test_seed_reproducibility(self):
        df = make_surface()
        a = mask_resample(df, 0.7, 5, seed=42)
        b = mask_resample(df, 0.7, 5, seed=42)
        c = mask_resample(df, 0.7, 5, seed=43)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert not all(np.array_equal(x, y) for x, y in zip(a, c))

    def test_no_observed_points_yields_all_false(self):
        df = make_surface(n_obs=0, n_query=6)
        masks = mask_resample(df, 0.8, 3, seed=0)
        assert all(m.sum() == 0 for m in masks)

    def test_single_observed_clamped_to_one(self):
        df = make_surface(n_obs=1, n_query=4)
        masks = mask_resample(df, 0.1, 4, seed=0)  # rounds to 0 -> clamp 1
        assert all(m.sum() == 1 for m in masks)

    def test_invalid_args(self):
        df = make_surface()
        with pytest.raises(ValueError):
            mask_resample(df, keep_fraction=0.0, n_draws=3, seed=0)
        with pytest.raises(ValueError):
            mask_resample(df, keep_fraction=1.5, n_draws=3, seed=0)
        with pytest.raises(ValueError):
            mask_resample(df, keep_fraction=0.5, n_draws=0, seed=0)
        with pytest.raises(ValueError):
            mask_resample(df.drop(columns=["observed"]), 0.5, 3, 0)


# ---------------------------------------------------------------------------
# masking_sensitivity
# ---------------------------------------------------------------------------

class TestMaskingSensitivity:
    def test_protocol_conformance(self):
        assert isinstance(ConstantPredictor(), Predictor)
        assert isinstance(NaNPredictor(), Predictor)

    def test_shapes_and_length(self):
        df = make_surface(n_obs=8, n_query=5)
        res = masking_sensitivity(ConstantPredictor(), df, 0.8, n_draws=10, seed=0)
        assert isinstance(res, MaskingSensitivityResult)
        assert len(res) == len(df)
        assert res.mean.shape == (len(df),)
        assert res.std.shape == (len(df),)
        assert res.n_draws.shape == (len(df),)
        assert res.predictions.shape == (10, len(df))

    def test_deterministic_predictor_zero_instability(self):
        df = make_surface()
        res = masking_sensitivity(ConstantPredictor(0.25), df, 0.7, 15, seed=3)
        np.testing.assert_allclose(res.std, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.mean, 0.25)
        assert np.all(res.n_draws == 15)

    def test_which_points_sensitive_predictor_positive_instability(self):
        # predictor whose prediction depends on the identity of observed rows
        class SumIndexPredictor:
            def predict(self, df: pd.DataFrame) -> PredictionResult:
                obs = df["observed"].to_numpy(dtype=bool)
                val = float(np.flatnonzero(obs).sum())
                return PredictionResult(pred=np.full(len(df), val))

        df = make_surface(n_obs=12, n_query=4)
        res = masking_sensitivity(SumIndexPredictor(), df, 0.5, 30, seed=11)
        assert np.all(res.std > 0.0)
        # reproducible
        res2 = masking_sensitivity(SumIndexPredictor(), df, 0.5, 30, seed=11)
        np.testing.assert_allclose(res.std, res2.std)
        # different seed -> different draws -> generally different std
        res3 = masking_sensitivity(SumIndexPredictor(), df, 0.5, 30, seed=99)
        assert not np.allclose(res.predictions, res3.predictions)

    def test_never_finite_point_yields_nan_and_zero_count(self):
        df = make_surface(n_obs=8, n_query=5)
        res = masking_sensitivity(NaNPredictor(), df, 0.8, 10, seed=0)
        assert np.isnan(res.mean[-1])
        assert np.isnan(res.std[-1])
        assert res.n_draws[-1] == 0
        # other points are finite
        assert np.all(res.n_draws[:-1] == 10)
        assert np.all(np.isfinite(res.std[:-1]))

    def test_wrong_length_prediction_raises(self):
        class ShortPredictor:
            def predict(self, df: pd.DataFrame) -> PredictionResult:
                return PredictionResult(pred=np.zeros(len(df) - 1))

        df = make_surface()
        with pytest.raises(ValueError):
            masking_sensitivity(ShortPredictor(), df, 0.8, 5, seed=0)


# ---------------------------------------------------------------------------
# instability_summary
# ---------------------------------------------------------------------------

class TestInstabilitySummary:
    def test_basic_summary(self):
        df = make_surface()
        res = masking_sensitivity(ConstantPredictor(), df, 0.8, 10, seed=0)
        summ = instability_summary(res)
        assert summ["mean_std"] == pytest.approx(0.0)
        assert summ["median_std"] == pytest.approx(0.0)
        assert summ["n_points"] == len(df)

    def test_unobserved_only_restriction(self):
        df = make_surface(n_obs=8, n_query=5)
        observed = df["observed"].to_numpy(dtype=bool)
        res = masking_sensitivity(ConstantPredictor(), df, 0.8, 10, seed=0)
        summ = instability_summary(res, observed=observed, unobserved_only=True)
        assert summ["n_points"] == 5

    def test_unobserved_only_requires_observed(self):
        df = make_surface()
        res = masking_sensitivity(ConstantPredictor(), df, 0.8, 5, seed=0)
        with pytest.raises(ValueError):
            instability_summary(res, observed=None, unobserved_only=True)

    def test_length_mismatch_raises(self):
        df = make_surface()
        res = masking_sensitivity(ConstantPredictor(), df, 0.8, 5, seed=0)
        with pytest.raises(ValueError):
            instability_summary(
                res, observed=np.array([True, False]), unobserved_only=True
            )

    def test_all_nan_summary(self):
        # NaN std everywhere -> nan stats, zero count
        res = MaskingSensitivityResult(
            mean=np.array([np.nan, np.nan]),
            std=np.array([np.nan, np.nan]),
            n_draws=np.array([0, 0]),
            predictions=np.full((3, 2), np.nan),
            keep_fraction=0.8,
            seed=0,
        )
        summ = instability_summary(res)
        assert np.isnan(summ["mean_std"])
        assert np.isnan(summ["median_std"])
        assert summ["n_points"] == 0
