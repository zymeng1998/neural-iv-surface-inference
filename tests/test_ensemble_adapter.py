"""Tests for the deep-ensemble Predictor adapter (2D.3).

Covers the algebraic invariants of ``EnsembleConditionalPredictor`` against a
tiny dummy ensemble of mock members with deterministic outputs, then exercises
the manifest-loading path end-to-end by saving two real (untrained) member
checkpoints and round-tripping them through ``from_manifest``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
    EnsembleConditionalPredictor,
)
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)


class _MockMember:
    """Member double that returns a pre-baked, length-N prediction array."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float32)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        assert len(df) == len(self._values), (
            "mock member length must match df length"
        )
        return PredictionResult(
            pred=self._values.copy(),
            uncertainty=None,
            meta={"model": "mock"},
        )


def _dummy_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2024-01-01"),
            "log_moneyness": np.linspace(-0.2, 0.2, n).astype(np.float32),
            "tau": np.linspace(0.1, 1.0, n).astype(np.float32),
            "implied_volatility": np.full(n, 0.2, dtype=np.float32),
            "observed": np.array([True] * n),
        }
    )


def test_aggregates_mean_and_population_std():
    df = _dummy_frame(n=4)
    m1 = _MockMember(np.array([0.10, 0.20, 0.30, 0.40]))
    m2 = _MockMember(np.array([0.12, 0.18, 0.34, 0.36]))
    m3 = _MockMember(np.array([0.14, 0.22, 0.28, 0.44]))

    ens = EnsembleConditionalPredictor(members=[m1, m2, m3])
    res = ens.predict(df)

    expected_mean = np.mean(
        [[0.10, 0.20, 0.30, 0.40],
         [0.12, 0.18, 0.34, 0.36],
         [0.14, 0.22, 0.28, 0.44]], axis=0,
    ).astype(np.float32)
    expected_std = np.std(
        [[0.10, 0.20, 0.30, 0.40],
         [0.12, 0.18, 0.34, 0.36],
         [0.14, 0.22, 0.28, 0.44]], axis=0, ddof=0,
    ).astype(np.float32)

    assert res.pred.shape == (4,)
    assert res.uncertainty is not None
    assert res.uncertainty.shape == (4,)
    np.testing.assert_allclose(res.pred, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.uncertainty, expected_std, rtol=1e-6, atol=1e-6)
    assert res.lower is None and res.upper is None
    assert res.meta["model"] == "ensemble_conditional"
    assert res.meta["ensemble_size"] == 3
    assert res.meta["disagreement_ddof"] == 0


def test_alignment_to_input_df_order():
    """Result length must match the input frame length."""
    df = _dummy_frame(n=7)
    members = [
        _MockMember(np.full(7, 0.10 + i * 0.01, dtype=np.float32))
        for i in range(2)
    ]
    res = EnsembleConditionalPredictor(members=members).predict(df)
    assert len(res) == len(df)


def test_empty_members_rejected():
    with pytest.raises(ValueError):
        EnsembleConditionalPredictor(members=[])


def test_missing_member_checkpoint_raises_clear_error(tmp_path: Path):
    manifest_path = tmp_path / "members.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "ensemble_size": 1,
                "config_hash": "deadbeef",
                "members": [
                    {"seed": 1, "val_loss": 0.1,
                     "checkpoint": "seed_1/best_conditional.pt"},
                ],
            },
            f,
        )
    with pytest.raises(FileNotFoundError):
        EnsembleConditionalPredictor.from_manifest(manifest_path)


def test_manifest_roundtrip_with_real_checkpoints(tmp_path: Path):
    """End-to-end: write two untrained checkpoints + manifest, load + predict."""
    cfg = dict(
        context_dim=3, coord_dim=2, hidden_dim=16, latent_dim=8,
        n_elem_layers=2, n_post_layers=1, n_decoder_layers=2,
    )

    members = []
    for seed in (101, 202):
        torch.manual_seed(seed)
        model = ConditionalSurfaceModel(**cfg)
        member_dir = tmp_path / f"seed_{seed}"
        member_dir.mkdir(parents=True)
        ckpt_path = member_dir / "best_conditional.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "val_loss": 0.5,
                "config": cfg,
            },
            ckpt_path,
        )
        members.append(
            {"seed": seed, "val_loss": 0.5,
             "checkpoint": f"seed_{seed}/best_conditional.pt"}
        )

    manifest_path = tmp_path / "members.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {"version": 1, "ensemble_size": 2,
             "config_hash": "test", "members": members},
            f,
        )

    ens = EnsembleConditionalPredictor.from_manifest(
        manifest_path, device=torch.device("cpu")
    )
    assert len(ens.members) == 2
    df = _dummy_frame(n=5)
    res = ens.predict(df)
    assert res.pred.shape == (5,)
    assert res.uncertainty is not None
    # Distinct seeds → non-degenerate disagreement on at least one point.
    assert float(res.uncertainty.max()) > 0.0
