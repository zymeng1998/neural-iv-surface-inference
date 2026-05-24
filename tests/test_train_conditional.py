"""Smoke test for the conditional training loop (2C.4)."""

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
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)


def _synthetic_frame(n_dates: int = 6, n_per_date: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        # Simple deterministic surface: iv = 0.2 + 0.5 * k^2 + 0.1 * tau
        k = rng.uniform(-0.3, 0.3, n_per_date).astype(np.float32)
        tau = rng.uniform(0.05, 1.0, n_per_date).astype(np.float32)
        iv_clean = 0.2 + 0.5 * k**2 + 0.1 * tau
        iv_noisy = iv_clean + rng.normal(0, 0.005, n_per_date).astype(np.float32)
        observed = np.ones(n_per_date, dtype=bool)
        observed[-2:] = False  # last two are unobserved
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "log_moneyness": k,
                    "tau": tau,
                    "implied_volatility": iv_noisy,
                    "iv_clean": iv_clean.astype(np.float32),
                    "observed": observed,
                    "split": "train",
                    "noise_sigma": 0.005,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _build_loader(df: pd.DataFrame, batch_size: int) -> DataLoader:
    ds = ConditionalIVSurfaceDataset(df)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_conditional
    )


@pytest.mark.unit
def test_smoke_loss_decreases_and_checkpoint_written(tmp_path: Path):
    torch.manual_seed(0)

    train_df = _synthetic_frame(n_dates=6, n_per_date=8, seed=0)
    val_df = _synthetic_frame(n_dates=3, n_per_date=8, seed=1)

    train_loader = _build_loader(train_df, batch_size=3)
    val_loader = _build_loader(val_df, batch_size=3)

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
        "epochs": 8,
        "patience": 20,
        "seed": 0,
    }

    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        log_every=100,  # silence
    )

    assert isinstance(model, ConditionalSurfaceModel)
    assert len(history) >= 1
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    ckpt = tmp_path / "best_conditional.pt"
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert "config" in payload


@pytest.mark.unit
def test_masked_loss_ignores_padded_query_positions(tmp_path: Path):
    """A padded query position with garbage target must not affect the loss."""
    from neural_iv_surface_inference.training.train_conditional import (
        masked_query_mse,
    )

    pred = torch.tensor([[0.2, 0.3, 0.0], [0.25, 0.0, 0.0]])
    target = torch.tensor([[0.2, 0.3, 999.0], [0.25, 999.0, 999.0]])  # garbage in pads
    mask = torch.tensor([[True, True, False], [True, False, False]])

    loss = masked_query_mse(pred, target, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


# ── W4 heads (2D.2) ────────────────────────────────────────────────


def _build_config(head_cfg: dict, epochs: int = 6) -> dict:
    return {
        "context_dim": 3,
        "coord_dim": 2,
        "hidden_dim": 16,
        "latent_dim": 8,
        "n_elem_layers": 2,
        "n_post_layers": 1,
        "n_decoder_layers": 2,
        "learning_rate": 5e-3,
        "weight_decay": 1e-5,
        "epochs": epochs,
        "patience": 50,
        "seed": 0,
        "head": head_cfg,
    }


def _train_with_head(head_cfg: dict, tmp_path: Path, epochs: int = 6):
    torch.manual_seed(0)
    train_df = _synthetic_frame(n_dates=6, n_per_date=8, seed=0)
    val_df = _synthetic_frame(n_dates=3, n_per_date=8, seed=1)
    train_loader = _build_loader(train_df, batch_size=3)
    val_loader = _build_loader(val_df, batch_size=3)
    cfg = _build_config(head_cfg, epochs=epochs)
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        log_every=100,
    )
    return model, history


@pytest.mark.unit
def test_point_head_regression_matches_baseline(tmp_path: Path):
    """With head.kind=point the trained losses must match the legacy 2C run."""
    # Baseline path: no head key at all (legacy code path).
    torch.manual_seed(0)
    train_df = _synthetic_frame(n_dates=6, n_per_date=8, seed=0)
    val_df = _synthetic_frame(n_dates=3, n_per_date=8, seed=1)
    train_loader = _build_loader(train_df, batch_size=3)
    val_loader = _build_loader(val_df, batch_size=3)
    cfg_a = _build_config({"kind": "point"}, epochs=4)
    cfg_b = dict(cfg_a)
    cfg_b.pop("head")
    _, hist_a = train_conditional(
        train_loader=train_loader, val_loader=val_loader, config=cfg_a,
        device=torch.device("cpu"), checkpoint_dir=tmp_path / "a", log_every=100,
    )
    _, hist_b = train_conditional(
        train_loader=train_loader, val_loader=val_loader, config=cfg_b,
        device=torch.device("cpu"), checkpoint_dir=tmp_path / "b", log_every=100,
    )
    assert len(hist_a) == len(hist_b)
    for ra, rb in zip(hist_a, hist_b):
        assert ra["train_loss"] == pytest.approx(rb["train_loss"], rel=1e-6, abs=1e-8)
        assert ra["val_loss"] == pytest.approx(rb["val_loss"], rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_gaussian_head_nll_finite_and_decreasing(tmp_path: Path):
    _, history = _train_with_head({"kind": "gaussian"}, tmp_path, epochs=8)
    assert len(history) >= 2
    losses = [r["train_loss"] for r in history]
    for v in losses:
        assert torch.isfinite(torch.tensor(v))
    assert losses[-1] < losses[0]


@pytest.mark.unit
def test_quantile_head_pinball_finite_and_decreasing(tmp_path: Path):
    head_cfg = {"kind": "quantile", "quantiles": [0.1, 0.5, 0.9]}
    model, history = _train_with_head(head_cfg, tmp_path, epochs=8)
    assert len(history) >= 2
    losses = [r["train_loss"] for r in history]
    for v in losses:
        assert v == v  # NaN check
        assert v > 0.0
    assert losses[-1] < losses[0]

    # Monotone quantiles at inference
    model.eval()
    val_df = _synthetic_frame(n_dates=2, n_per_date=6, seed=42)
    val_loader = _build_loader(val_df, batch_size=2)
    raw_batch = next(iter(val_loader))
    out = model(raw_batch["context"], raw_batch["context_mask"], raw_batch["query"])
    diffs = out["quantiles"].diff(dim=-1)
    assert torch.all(diffs >= -1e-6)
