"""Integration test: micro_v1 end-to-end wiring (3C.2 / ADR 0008).

Loads the real smoke config `configs/conditional_3C2_micro_smoke.yaml`,
synthesises an in-memory micro_v1 frame (raw data lives only on the Pod),
trains `train_conditional` for 1 epoch, reloads the checkpoint through
`ConditionalSurfacePredictor.from_checkpoint`, and asserts the 9-dim feature
path round-trips with finite outputs and a finite scalar loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.eval.adapters import ConditionalSurfacePredictor
from neural_iv_surface_inference.training.train_conditional import (
    masked_query_mse,
    train_conditional,
)
from neural_iv_surface_inference.utils.io import load_config

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "conditional_3C2_micro_smoke.yaml"
)


def _make_micro_frame(n_dates: int = 6, rows_per_date: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    parts = []
    base = pd.Timestamp("2024-01-02")
    for d in range(n_dates):
        n = rows_per_date
        observed = np.zeros(n, dtype=bool)
        observed[: max(2, n // 2)] = True  # guarantee >=2 observed
        rng.shuffle(observed)
        bid = rng.uniform(0.5, 5.0, n)
        ask = bid + rng.uniform(0.05, 0.5, n)
        parts.append(
            pd.DataFrame(
                {
                    "date": base + pd.Timedelta(days=d),
                    "log_moneyness": rng.uniform(-0.3, 0.3, n).astype(np.float32),
                    "tau": rng.uniform(0.05, 1.5, n).astype(np.float32),
                    "implied_volatility": rng.uniform(0.1, 0.5, n).astype(np.float32),
                    "iv_clean": rng.uniform(0.1, 0.5, n).astype(np.float32),
                    "observed": observed,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "volume": rng.integers(1, 500, n),
                    "open_interest": rng.integers(1, 5000, n),
                    "type": rng.choice(["call", "put"], n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _build_loader(df: pd.DataFrame, feature_set: str, batch_size: int) -> DataLoader:
    return DataLoader(
        ConditionalIVSurfaceDataset(df, feature_set=feature_set),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_conditional,
        num_workers=0,
    )


def test_micro_v1_end_to_end(tmp_path):
    config = load_config(_CONFIG_PATH)
    cond_cfg = dict(config["conditional"])
    feature_set = str(cond_cfg["feature_set"])
    assert feature_set == "micro_v1"
    batch_size = int(config["data"]["batch_size"])

    df = _make_micro_frame()
    train_loader = _build_loader(df, feature_set, batch_size)
    val_loader = _build_loader(df, feature_set, batch_size)

    device = torch.device("cpu")
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cond_cfg,
        device=device,
        checkpoint_dir=tmp_path,
        log_every=1,
    )
    assert len(history) == 1  # epochs: 1
    assert model.feature_set == "micro_v1"
    assert model.encoder.elem_mlp[0][0].in_features == 9

    ckpt_path = tmp_path / "best_conditional.pt"
    assert ckpt_path.exists()

    # The persisted config block must record feature_set (acceptance crit).
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt["config"]["feature_set"] == "micro_v1"

    # Round-trip through the predictor adapter with no operator override.
    predictor = ConditionalSurfacePredictor.from_checkpoint(ckpt_path, device=device)
    assert predictor.feature_set == "micro_v1"
    assert predictor.model.feature_set == "micro_v1"
    assert predictor.model.encoder.elem_mlp[0][0].in_features == 9

    result = predictor.predict(df)
    assert result.pred.shape == (len(df),)
    # Observed-row predictions must be finite (unobserved-context dates are
    # not present here — every date has >=2 observed quotes).
    assert np.isfinite(result.pred).all()

    # A forward pass on a fixed input yields a finite scalar loss.
    batch = next(iter(train_loader))
    out = predictor.model(batch["context"], batch["context_mask"], batch["query"])
    loss = masked_query_mse(out["mu"], batch["target"], batch["query_mask"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_legacy_minimal_checkpoint_still_loads(tmp_path):
    """A minimal (legacy-default) checkpoint round-trips as a 3-dim path."""
    df = _make_micro_frame()
    cfg = {
        "feature_set": "minimal",
        "context_dim": 3,
        "hidden_dim": 16,
        "latent_dim": 8,
        "n_elem_layers": 1,
        "n_post_layers": 1,
        "n_decoder_layers": 2,
        "epochs": 1,
        "patience": 1,
        "head": {"kind": "point"},
    }
    train_loader = _build_loader(df, "minimal", 4)
    val_loader = _build_loader(df, "minimal", 4)
    model, _ = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        log_every=1,
    )
    assert model.feature_set == "minimal"
    predictor = ConditionalSurfacePredictor.from_checkpoint(
        tmp_path / "best_conditional.pt", device=torch.device("cpu")
    )
    assert predictor.feature_set == "minimal"
    assert predictor.model.encoder.elem_mlp[0][0].in_features == 3
    res = predictor.predict(df)
    assert np.isfinite(res.pred).all()
