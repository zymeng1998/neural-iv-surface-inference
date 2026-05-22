"""Tests for the model-agnostic predictor interface (Story 2A.2).

Uses synthetic data — no real market data or benchmark files required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.eval import (
    PredictionResult,
    Predictor,
    InterpolationPredictor,
    MLPPredictor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Small single-date-ish benchmark frame with the standard columns."""
    rng = np.random.default_rng(0)
    n_dates = 3
    n_points = 40
    n = n_dates * n_points

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    date_col = np.repeat(dates, n_points)

    tau = rng.uniform(1 / 365, 2.0, size=n)
    log_m = rng.uniform(-0.5, 0.5, size=n)
    iv_clean = np.clip(0.2 + 0.15 * log_m**2 + 0.02 * tau, 0.01, 3.0)
    observed = rng.random(n) < 0.5
    iv_noisy = np.clip(iv_clean + rng.normal(0, 0.01, size=n), 1e-4, 10.0)

    return pd.DataFrame({
        "date": date_col,
        "tau": tau,
        "log_moneyness": log_m,
        "iv_clean": iv_clean,
        "implied_volatility": iv_noisy,
        "observed": observed,
        "split": "test",
        "noise_sigma": np.full(n, 0.01),
    })


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_construct_minimal(self):
        pr = PredictionResult(pred=np.zeros(5))
        assert len(pr) == 5
        assert pr.uncertainty is None
        assert pr.lower is None
        assert pr.upper is None
        assert pr.meta == {}

    def test_is_frozen(self):
        pr = PredictionResult(pred=np.zeros(3))
        with pytest.raises(Exception):
            pr.pred = np.ones(3)  # frozen dataclass → FrozenInstanceError

    def test_optional_arrays_same_length_ok(self):
        n = 4
        pr = PredictionResult(
            pred=np.zeros(n),
            uncertainty=np.ones(n),
            lower=np.zeros(n),
            upper=np.ones(n),
            meta={"model": "x"},
        )
        assert len(pr) == n

    @pytest.mark.parametrize("field_name", ["uncertainty", "lower", "upper"])
    def test_length_mismatch_rejected(self, field_name):
        with pytest.raises(ValueError):
            PredictionResult(pred=np.zeros(5), **{field_name: np.ones(4)})


# ---------------------------------------------------------------------------
# Protocol conformance + adapters
# ---------------------------------------------------------------------------

class TestAdapters:
    def test_interpolation_conforms_to_protocol(self):
        assert isinstance(InterpolationPredictor(), Predictor)

    def test_mlp_conforms_to_protocol(self):
        model = BaselineMLP(input_dim=2, hidden_dim=32, n_layers=2)
        assert isinstance(MLPPredictor(model), Predictor)

    def test_interpolation_predict(self, synthetic_df):
        result = InterpolationPredictor(method="linear").predict(synthetic_df)
        assert isinstance(result, PredictionResult)
        assert len(result) == len(synthetic_df)
        assert np.all(np.isfinite(result.pred))
        assert result.uncertainty is None
        assert result.meta["model"] == "interpolation"

    def test_mlp_predict(self, synthetic_df):
        model = BaselineMLP(input_dim=2, hidden_dim=32, n_layers=2)
        predictor = MLPPredictor(model, device=torch.device("cpu"))
        result = predictor.predict(synthetic_df)
        assert isinstance(result, PredictionResult)
        assert len(result) == len(synthetic_df)
        assert np.all(np.isfinite(result.pred))
        assert np.all(result.pred > 0)  # softplus output
        assert result.uncertainty is None
        assert result.meta["model"] == "baseline_mlp"
