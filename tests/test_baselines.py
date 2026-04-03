"""Tests for baseline models, training loop, and evaluation.

Uses synthetic data — no real market data or benchmark files required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.interpolation import (
    interpolate_surface_date,
    run_interpolation_baseline,
)
from neural_iv_surface_inference.models.losses import (
    MaskedMSELoss,
    MaskedMAELoss,
    CombinedLoss,
)
from neural_iv_surface_inference.data.loaders import IVSurfaceDataset
from neural_iv_surface_inference.training.train import (
    train_one_epoch,
    validate,
    train_mlp,
    predict_mlp,
)
from neural_iv_surface_inference.training.eval import (
    mae,
    rmse,
    mape,
    evaluate_predictions,
    metrics_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_benchmark_df():
    """Create a synthetic benchmark DataFrame (as if from Step 4)."""
    rng = np.random.default_rng(0)
    n_dates = 30
    n_points = 100
    n = n_dates * n_points

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    date_col = np.repeat(dates, n_points)

    tau = rng.uniform(1 / 365, 2.0, size=n)
    log_m = rng.uniform(-0.5, 0.5, size=n)

    # Generate IV with a known smile pattern for testability
    iv_clean = 0.2 + 0.15 * log_m**2 + 0.02 * tau
    iv_clean = np.clip(iv_clean, 0.01, 3.0)

    observed = rng.random(n) < 0.4
    noise = rng.normal(0, 0.01, size=n)
    iv_noisy = iv_clean.copy()
    iv_noisy[observed] += noise[observed]
    iv_noisy = np.clip(iv_noisy, 1e-4, 10.0)

    # Time-based split
    split = np.array(["test"] * n, dtype=object)
    split[date_col < dates[21]] = "train"
    split[(date_col >= dates[21]) & (date_col < dates[26])] = "val"

    return pd.DataFrame({
        "date": date_col,
        "tau": tau,
        "log_moneyness": log_m,
        "iv_clean": iv_clean,
        "implied_volatility": iv_noisy,
        "observed": observed,
        "split": split,
        "strike": 100 * np.exp(log_m),
        "spot": np.full(n, 100.0),
        "noise_sigma": np.full(n, 0.01),
    })


@pytest.fixture
def split_dfs(synthetic_benchmark_df):
    """Return train/val/test DataFrames."""
    df = synthetic_benchmark_df
    return {
        "train_df": df[df["split"] == "train"].reset_index(drop=True),
        "val_df": df[df["split"] == "val"].reset_index(drop=True),
        "test_df": df[df["split"] == "test"].reset_index(drop=True),
    }


@pytest.fixture
def train_loader(split_dfs):
    ds = IVSurfaceDataset(split_dfs["train_df"])
    return DataLoader(ds, batch_size=64, shuffle=True)


@pytest.fixture
def val_loader(split_dfs):
    ds = IVSurfaceDataset(split_dfs["val_df"])
    return DataLoader(ds, batch_size=64, shuffle=False)


@pytest.fixture
def test_loader(split_dfs):
    ds = IVSurfaceDataset(split_dfs["test_df"])
    return DataLoader(ds, batch_size=64, shuffle=False)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestBaselineMLP:
    def test_forward_shape(self):
        model = BaselineMLP(input_dim=2, hidden_dim=64, n_layers=2)
        x = torch.randn(32, 2)
        out = model(x)
        assert out.shape == (32, 1)

    def test_output_positive(self):
        """IV predictions must be positive (softplus output)."""
        model = BaselineMLP(input_dim=2, hidden_dim=64, n_layers=2)
        x = torch.randn(100, 2) * 5  # extreme inputs
        out = model(x)
        assert (out > 0).all()

    def test_different_configs(self):
        for n_layers in [1, 2, 4]:
            for hidden in [32, 128]:
                model = BaselineMLP(input_dim=2, hidden_dim=hidden, n_layers=n_layers)
                x = torch.randn(8, 2)
                out = model(x)
                assert out.shape == (8, 1)

    def test_with_dropout(self):
        model = BaselineMLP(input_dim=2, hidden_dim=64, n_layers=2, dropout=0.2)
        model.train()
        x = torch.randn(32, 2)
        out = model(x)
        assert out.shape == (32, 1)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_masked_mse(self):
        loss_fn = MaskedMSELoss()
        pred = torch.tensor([[0.2], [0.3], [0.4]])
        target = torch.tensor([[0.2], [0.5], [0.4]])
        mask = torch.tensor([True, True, False])
        loss = loss_fn(pred, target, mask)
        # MSE of (0.2-0.2)^2=0 and (0.3-0.5)^2=0.04 → mean=0.02
        assert abs(loss.item() - 0.02) < 1e-5

    def test_masked_mae(self):
        loss_fn = MaskedMAELoss()
        pred = torch.tensor([[0.2], [0.3]])
        target = torch.tensor([[0.3], [0.5]])
        mask = torch.tensor([True, True])
        loss = loss_fn(pred, target, mask)
        # MAE of |0.2-0.3|=0.1 and |0.3-0.5|=0.2 → mean=0.15
        assert abs(loss.item() - 0.15) < 1e-5

    def test_empty_mask_returns_zero(self):
        loss_fn = MaskedMSELoss()
        pred = torch.tensor([[0.2], [0.3]])
        target = torch.tensor([[0.5], [0.5]])
        mask = torch.tensor([False, False])
        loss = loss_fn(pred, target, mask)
        assert loss.item() == 0.0

    def test_combined_loss(self):
        loss_fn = CombinedLoss(alpha=0.5)
        pred = torch.tensor([[0.2], [0.4]])
        target = torch.tensor([[0.3], [0.5]])
        mask = torch.tensor([True, True])
        loss = loss_fn(pred, target, mask)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestIVSurfaceDataset:
    def test_length(self, split_dfs):
        ds = IVSurfaceDataset(split_dfs["train_df"])
        assert len(ds) == len(split_dfs["train_df"])

    def test_sample_keys(self, split_dfs):
        ds = IVSurfaceDataset(split_dfs["train_df"])
        sample = ds[0]
        assert set(sample.keys()) == {"features", "target", "iv_input", "observed", "date_idx"}
        assert sample["features"].shape == (2,)
        assert sample["target"].shape == ()

    def test_features_are_coords(self, split_dfs):
        ds = IVSurfaceDataset(split_dfs["train_df"])
        sample = ds[0]
        df_row = split_dfs["train_df"].iloc[0]
        assert abs(sample["features"][0].item() - df_row["log_moneyness"]) < 1e-5
        assert abs(sample["features"][1].item() - df_row["tau"]) < 1e-5


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_interpolate_single_date(self, synthetic_benchmark_df):
        date = synthetic_benchmark_df["date"].iloc[0]
        df_date = synthetic_benchmark_df[synthetic_benchmark_df["date"] == date].copy()
        preds = interpolate_surface_date(df_date, method="rbf")
        assert len(preds) == len(df_date)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_all_methods(self, synthetic_benchmark_df):
        date = synthetic_benchmark_df["date"].iloc[0]
        df_date = synthetic_benchmark_df[synthetic_benchmark_df["date"] == date].copy()
        for method in ["linear", "cubic", "rbf", "nearest"]:
            preds = interpolate_surface_date(df_date, method=method)
            assert len(preds) == len(df_date)

    def test_run_full_baseline(self, synthetic_benchmark_df):
        # Run on a small subset to keep test fast
        test_df = synthetic_benchmark_df[
            synthetic_benchmark_df["split"] == "test"
        ].reset_index(drop=True)
        iv_pred = run_interpolation_baseline(test_df, method="linear", verbose=False)
        assert isinstance(iv_pred, np.ndarray)
        assert len(iv_pred) == len(test_df)
        assert not np.isnan(iv_pred).any()

    def test_few_points_fallback(self):
        """When only 1-2 observed points exist, should return mean."""
        df = pd.DataFrame({
            "log_moneyness": [0.0, 0.1, 0.2],
            "tau": [0.1, 0.2, 0.3],
            "implied_volatility": [0.2, 0.3, 0.4],
            "observed": [True, False, False],
        })
        preds = interpolate_surface_date(df, method="rbf")
        assert len(preds) == 3
        # With 1 observed point, should get the mean (0.2) everywhere
        np.testing.assert_allclose(preds, 0.2, atol=1e-5)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_one_epoch(self, train_loader):
        model = BaselineMLP(input_dim=2, hidden_dim=32, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = MaskedMSELoss()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, torch.device("cpu"))
        assert loss > 0
        assert np.isfinite(loss)

    def test_validate(self, val_loader):
        model = BaselineMLP(input_dim=2, hidden_dim=32, n_layers=2)
        criterion = MaskedMSELoss()
        metrics = validate(model, val_loader, criterion, torch.device("cpu"))
        assert "obs_mse" in metrics
        assert "unobs_mae" in metrics
        assert metrics["obs_mse"] >= 0

    def test_train_mlp_short(self, train_loader, val_loader, tmp_path):
        config = {
            "hidden_dim": 32,
            "n_layers": 2,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "epochs": 3,
            "patience": 5,
        }
        model, history = train_mlp(
            train_loader, val_loader, config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
        )
        assert len(history) == 3
        assert (tmp_path / "best_mlp.pt").exists()
        # Loss should decrease or at least not explode
        assert all(np.isfinite(h["train_loss"]) for h in history)

    def test_predict_mlp(self, test_loader):
        model = BaselineMLP(input_dim=2, hidden_dim=32, n_layers=2)
        preds = predict_mlp(model, test_loader, device=torch.device("cpu"))
        assert len(preds) > 0
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)  # softplus ensures this


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_mae_rmse(self):
        pred = np.array([0.2, 0.3, 0.4])
        target = np.array([0.2, 0.5, 0.4])
        assert abs(mae(pred, target) - 0.2 / 3) < 1e-6
        assert rmse(pred, target) > 0

    def test_evaluate_predictions(self, synthetic_benchmark_df):
        # Add fake predictions
        df = synthetic_benchmark_df.copy()
        df["iv_pred"] = df["iv_clean"] + np.random.default_rng(0).normal(0, 0.01, len(df))
        results = evaluate_predictions(df)

        assert "overall" in results
        assert "observed" in results
        assert "unobserved" in results
        assert "by_maturity" in results
        assert "by_moneyness" in results
        assert results["overall"]["mae"] > 0
        assert results["overall"]["n"] == len(df)

    def test_metrics_to_dataframe(self, synthetic_benchmark_df):
        df = synthetic_benchmark_df.copy()
        df["iv_pred"] = df["iv_clean"]
        results = evaluate_predictions(df)
        summary = metrics_to_dataframe(results, model_name="test")
        assert len(summary) == 1
        assert summary["model"].iloc[0] == "test"
        # Perfect predictions → MAE should be ~0
        assert summary["overall_mae"].iloc[0] < 1e-6

    def test_mape(self):
        pred = np.array([0.22, 0.33])
        target = np.array([0.2, 0.3])
        result = mape(pred, target)
        assert result > 0
        assert result < 20  # ~10% error


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_interp_then_eval(self, synthetic_benchmark_df):
        """End-to-end: interpolate → evaluate → produce metrics."""
        test_df = synthetic_benchmark_df[
            synthetic_benchmark_df["split"] == "test"
        ].reset_index(drop=True)

        test_df = test_df.copy()
        test_df["iv_pred"] = run_interpolation_baseline(test_df, method="linear", verbose=False)
        metrics = evaluate_predictions(test_df)

        # The interpolation should do better than random on synthetic smile data
        assert metrics["overall"]["mae"] < 0.5  # generous threshold

    def test_mlp_then_eval(self, split_dfs):
        """End-to-end: train MLP → predict → evaluate."""
        train_ds = IVSurfaceDataset(split_dfs["train_df"])
        val_ds = IVSurfaceDataset(split_dfs["val_df"])
        test_ds = IVSurfaceDataset(split_dfs["test_df"])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        test_loader = DataLoader(test_ds, batch_size=64)

        config = {
            "hidden_dim": 32, "n_layers": 2, "dropout": 0.0,
            "learning_rate": 1e-3, "weight_decay": 0.0,
            "epochs": 5, "patience": 10,
        }
        model, _ = train_mlp(
            train_loader, val_loader, config,
            device=torch.device("cpu"),
        )

        preds = predict_mlp(model, test_loader, device=torch.device("cpu"))
        test_df = split_dfs["test_df"].copy()
        test_df["iv_pred"] = preds

        metrics = evaluate_predictions(test_df)
        assert metrics["overall"]["mae"] < 1.0  # generous — just checking it runs
