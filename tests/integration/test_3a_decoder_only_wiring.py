"""Integration tests: 3A.3 freeze_encoder + encoder_init_from + coord_encoding kinds."""

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


def _synthetic_df(seed: int = 0, n_dates: int = 4, per_date: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-02")
    for d in range(n_dates):
        date = base + pd.Timedelta(days=d)
        k = rng.uniform(-0.2, 0.2, size=per_date).astype(np.float32)
        tau = rng.uniform(0.05, 1.0, size=per_date).astype(np.float32)
        iv = (0.2 + 0.1 * np.abs(k) + 0.05 * tau).astype(np.float32)
        observed = np.ones(per_date, dtype=bool)
        observed[-1] = False  # keep at least one unobserved query point
        for i in range(per_date):
            rows.append(
                {
                    "date": date,
                    "log_moneyness": float(k[i]),
                    "tau": float(tau[i]),
                    "implied_volatility": float(iv[i]),
                    "iv_clean": float(iv[i]),
                    "observed": bool(observed[i]),
                }
            )
    return pd.DataFrame(rows)


def _tiny_cond_cfg(extra: dict | None = None) -> dict:
    cfg = dict(
        context_dim=3,
        coord_dim=2,
        hidden_dim=16,
        latent_dim=8,
        n_elem_layers=1,
        n_post_layers=1,
        n_decoder_layers=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        epochs=1,
        patience=5,
        seed=42,
        head={"kind": "gaussian"},
    )
    if extra:
        cfg.update(extra)
    return cfg


def _build_loaders(df: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
    ds = ConditionalIVSurfaceDataset(df)
    return (
        DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_conditional),
        DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_conditional),
    )


def _save_source_checkpoint(tmp_path: Path) -> Path:
    """Build a fresh model and save it as a 2D.7-shaped checkpoint."""
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(
        context_dim=3,
        coord_dim=2,
        hidden_dim=16,
        latent_dim=8,
        n_elem_layers=1,
        n_post_layers=1,
        n_decoder_layers=2,
        head={"kind": "gaussian"},
    )
    ckpt_path = tmp_path / "source_2d7.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.0,
            "config": {},
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.mark.integration
def test_freeze_encoder_actually_freezes(tmp_path: Path):
    df = _synthetic_df()
    src = _save_source_checkpoint(tmp_path)
    src_payload = torch.load(str(src), map_location="cpu", weights_only=False)
    src_encoder = {
        k[len("encoder.") :]: v.detach().clone()
        for k, v in src_payload["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    cfg = _tiny_cond_cfg(
        {
            "encoder_init_from": str(src),
            "freeze_encoder": True,
            "coord_encoding": {"kind": "raw"},
        }
    )
    train_loader, val_loader = _build_loaders(df)
    ckpt_dir = tmp_path / "out"
    model, history = train_conditional(
        train_loader, val_loader, cfg, device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir, log_every=1000,
    )

    # Encoder params must be frozen.
    for p in model.encoder.parameters():
        assert p.requires_grad is False

    # Encoder weights unchanged.
    trained_enc = {k: v.detach() for k, v in model.encoder.state_dict().items()}
    assert set(trained_enc.keys()) == set(src_encoder.keys())
    for k in trained_enc:
        assert torch.equal(trained_enc[k], src_encoder[k]), f"encoder.{k} drifted"


@pytest.mark.integration
def test_decoder_updates_when_encoder_frozen(tmp_path: Path):
    df = _synthetic_df()
    src = _save_source_checkpoint(tmp_path)

    # Snapshot the decoder weights of a freshly built model seeded the same way.
    torch.manual_seed(42)
    pre_model = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=16, latent_dim=8,
        n_elem_layers=1, n_post_layers=1, n_decoder_layers=2,
        head={"kind": "gaussian"}, coord_encoding={"kind": "raw"},
    )
    pre_decoder = {k: v.detach().clone() for k, v in pre_model.decoder.state_dict().items()}

    cfg = _tiny_cond_cfg(
        {
            "encoder_init_from": str(src),
            "freeze_encoder": True,
            "coord_encoding": {"kind": "raw"},
            "epochs": 2,
        }
    )
    train_loader, val_loader = _build_loaders(df)
    model, _ = train_conditional(
        train_loader, val_loader, cfg, device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "out", log_every=1000,
    )

    post_decoder = {k: v.detach() for k, v in model.decoder.state_dict().items()}
    any_changed = any(
        not torch.equal(pre_decoder[k], post_decoder[k]) for k in post_decoder
    )
    assert any_changed, "decoder weights did not update during decoder-only retrain"


@pytest.mark.integration
def test_encoder_init_from_loads_weights(tmp_path: Path):
    df = _synthetic_df()
    src = _save_source_checkpoint(tmp_path)
    src_payload = torch.load(str(src), map_location="cpu", weights_only=False)
    src_encoder = {
        k[len("encoder.") :]: v.detach().clone()
        for k, v in src_payload["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    cfg = _tiny_cond_cfg(
        {
            "encoder_init_from": str(src),
            "freeze_encoder": False,
            "coord_encoding": {"kind": "raw"},
            "epochs": 0,  # treat as a load-only path; we'll just instantiate
        }
    )
    # Setting epochs=0 means the for-loop body doesn't run, history stays empty.
    # train_conditional still constructs the model and seeds the encoder.
    train_loader, val_loader = _build_loaders(df)
    model, _ = train_conditional(
        train_loader, val_loader, cfg, device=torch.device("cpu"),
        checkpoint_dir=None, log_every=1000,
    )
    loaded_enc = {k: v.detach() for k, v in model.encoder.state_dict().items()}
    for k in src_encoder:
        assert torch.equal(loaded_enc[k], src_encoder[k]), f"encoder.{k} not loaded"


@pytest.mark.integration
@pytest.mark.parametrize(
    "coord_encoding",
    [
        {"kind": "raw"},
        {"kind": "fourier", "num_bands": 4, "max_freq": 10.0, "include_input": True},
    ],
)
def test_both_coord_encoding_kinds_train(tmp_path: Path, coord_encoding: dict):
    df = _synthetic_df()
    src = _save_source_checkpoint(tmp_path)
    cfg = _tiny_cond_cfg(
        {
            "encoder_init_from": str(src),
            "freeze_encoder": True,
            "coord_encoding": coord_encoding,
            "epochs": 1,
        }
    )
    train_loader, val_loader = _build_loaders(df)
    _, history = train_conditional(
        train_loader, val_loader, cfg, device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "out", log_every=1000,
    )
    assert len(history) == 1
    assert not np.isnan(history[0]["val_loss"])
