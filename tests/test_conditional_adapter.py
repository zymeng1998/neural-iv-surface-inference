"""Tests for the ConditionalSurfacePredictor adapter (2C.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
)
from neural_iv_surface_inference.eval.predictor import (
    PredictionResult,
    Predictor,
)
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)


def _synthetic_frame(n_dates: int, n_per_date: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        k = rng.uniform(-0.3, 0.3, n_per_date).astype(np.float32)
        tau = rng.uniform(0.05, 1.0, n_per_date).astype(np.float32)
        iv_clean = (0.2 + 0.5 * k**2 + 0.1 * tau).astype(np.float32)
        iv_noisy = iv_clean + rng.normal(0, 0.005, n_per_date).astype(np.float32)
        observed = np.ones(n_per_date, dtype=bool)
        observed[-max(1, n_per_date // 3) :] = False
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "log_moneyness": k,
                    "tau": tau,
                    "implied_volatility": iv_noisy,
                    "iv_clean": iv_clean,
                    "observed": observed,
                    "split": "train",
                    "noise_sigma": 0.005,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _train_tiny(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    train_df = _synthetic_frame(6, 9, seed=0)
    val_df = _synthetic_frame(3, 9, seed=1)
    ds_train = ConditionalIVSurfaceDataset(train_df)
    ds_val = ConditionalIVSurfaceDataset(val_df)
    train_loader = DataLoader(
        ds_train, batch_size=3, shuffle=False, collate_fn=collate_conditional
    )
    val_loader = DataLoader(
        ds_val, batch_size=3, shuffle=False, collate_fn=collate_conditional
    )
    config = {
        "context_dim": 3,
        "coord_dim": 2,
        "hidden_dim": 16,
        "latent_dim": 8,
        "n_elem_layers": 2,
        "n_post_layers": 1,
        "n_decoder_layers": 2,
        "learning_rate": 5e-3,
        "weight_decay": 1e-5,
        "epochs": 4,
        "patience": 10,
        "seed": 0,
    }
    train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        log_every=100,
    )
    return tmp_path / "best_conditional.pt"


def test_satisfies_predictor_protocol(tmp_path: Path):
    ckpt_path = _train_tiny(tmp_path)
    pred = ConditionalSurfacePredictor.from_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert isinstance(pred, Predictor)


def test_predict_returns_aligned_predictionresult(tmp_path: Path):
    ckpt_path = _train_tiny(tmp_path)
    pred = ConditionalSurfacePredictor.from_checkpoint(ckpt_path, device=torch.device("cpu"))

    df = _synthetic_frame(4, 7, seed=42)
    # Shuffle row order to test alignment preservation.
    df_shuf = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    result = pred.predict(df_shuf)
    assert isinstance(result, PredictionResult)
    assert result.uncertainty is None
    assert len(result) == len(df_shuf)
    assert result.pred.shape == (len(df_shuf),)
    # Predictions are positive (softplus) and finite
    assert np.all(np.isfinite(result.pred))
    assert np.all(result.pred > 0.0)
    # Permuting back to original order does NOT change the per-row mapping:
    # i.e. prediction for original row i is independent of df_shuf row order.
    # Verify by running predict on the original order and matching by index.
    result_orig = pred.predict(df)
    # Build a dict from (date, log_moneyness, tau) -> prediction for the shuffled run
    key_to_pred_shuf = {
        (r["date"], float(r["log_moneyness"]), float(r["tau"])): float(result.pred[i])
        for i, r in df_shuf.iterrows()
    }
    for i, r in df.iterrows():
        key = (r["date"], float(r["log_moneyness"]), float(r["tau"]))
        assert key_to_pred_shuf[key] == pytest.approx(float(result_orig.pred[i]), rel=1e-5, abs=1e-6)


def test_predict_handles_date_with_no_observed_rows(tmp_path: Path):
    """A date with no observed context yields zero predictions for its rows
    but does not crash and other dates still work."""
    ckpt_path = _train_tiny(tmp_path)
    pred = ConditionalSurfacePredictor.from_checkpoint(ckpt_path, device=torch.device("cpu"))

    df = _synthetic_frame(3, 6, seed=11)
    # Set all rows for the first date to observed=False
    first_date = df["date"].min()
    df.loc[df["date"] == first_date, "observed"] = False

    result = pred.predict(df)
    first_mask = df["date"].values == first_date
    other_mask = ~first_mask
    assert np.all(result.pred[first_mask] == 0.0)
    assert np.all(result.pred[other_mask] > 0.0)
