"""Parity test for ConditionalSurfacePredictor across decoder_kind (3B.3).

Asserts that the existing adapter loads either ``decoder_kind="deepsets"``
(Phase 2) or ``decoder_kind="anp"`` (Phase 3B.2) from a checkpoint and
returns a ``PredictionResult`` whose schema is identical modulo the
predicted values: same row order (== ``df`` input row order), same
column population pattern, same dtypes.

Local, CPU-only, no Pod time, no AV data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neural_iv_surface_inference.eval.adapters import ConditionalSurfacePredictor
from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)


_BASE_CFG = dict(
    context_dim=3,
    coord_dim=2,
    hidden_dim=16,
    latent_dim=8,
    n_elem_layers=1,
    n_post_layers=1,
    n_decoder_layers=2,
)


def _synthetic_frame(n_dates: int = 3, n_per_date: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    base = pd.Timestamp("2024-01-02")
    for d in range(n_dates):
        date = base + pd.Timedelta(days=d)
        k = rng.uniform(-0.2, 0.2, n_per_date).astype(np.float32)
        tau = rng.uniform(0.05, 1.0, n_per_date).astype(np.float32)
        iv = (0.2 + 0.1 * np.abs(k) + 0.05 * tau).astype(np.float32)
        observed = np.ones(n_per_date, dtype=bool)
        observed[-2:] = False
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "log_moneyness": k,
                    "tau": tau,
                    "implied_volatility": iv,
                    "iv_clean": iv,
                    "observed": observed,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _save_checkpoint(
    tmp_path: Path,
    *,
    decoder_kind: str,
    head_kind: str,
    tag: str,
    seed: int = 0,
) -> Path:
    torch.manual_seed(seed)
    head_cfg = {"kind": head_kind}
    if head_kind == "quantile":
        head_cfg["quantiles"] = [0.05, 0.5, 0.95]
    model = ConditionalSurfaceModel(
        **_BASE_CFG,
        head=head_cfg,
        decoder_kind=decoder_kind,  # type: ignore[arg-type]
        anp={"n_heads": 2, "mlp_hidden": 16} if decoder_kind == "anp" else None,
    )

    # Build the config blob the adapter will read back.
    config_blob = dict(_BASE_CFG)
    config_blob["head"] = dict(head_cfg)
    config_blob["decoder_kind"] = decoder_kind
    if decoder_kind == "anp":
        config_blob["anp"] = {"n_heads": 2, "mlp_hidden": 16}

    ckpt_path = tmp_path / f"{tag}.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.0,
            "config": config_blob,
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.mark.parametrize("head_kind", ["gaussian", "quantile", "point"])
def test_parity_across_decoder_kinds(tmp_path: Path, head_kind: str):
    df = _synthetic_frame()
    ds_ckpt = _save_checkpoint(
        tmp_path, decoder_kind="deepsets", head_kind=head_kind, tag="ds"
    )
    anp_ckpt = _save_checkpoint(
        tmp_path, decoder_kind="anp", head_kind=head_kind, tag="anp"
    )

    ds_pred = ConditionalSurfacePredictor.from_checkpoint(ds_ckpt).predict(df)
    anp_pred = ConditionalSurfacePredictor.from_checkpoint(anp_ckpt).predict(df)

    assert isinstance(ds_pred, PredictionResult)
    assert isinstance(anp_pred, PredictionResult)

    # Row count matches input frame for both.
    assert len(ds_pred) == len(df) == len(anp_pred)

    # Population pattern of the four fields is identical (both populated /
    # both None) for each head kind.
    for field in ("uncertainty", "lower", "upper"):
        ds_val = getattr(ds_pred, field)
        anp_val = getattr(anp_pred, field)
        assert (ds_val is None) == (anp_val is None), (
            f"field={field} populated mismatch: deepsets={ds_val is None} "
            f"anp={anp_val is None}"
        )
        if ds_val is not None:
            assert ds_val.shape == anp_val.shape == (len(df),)
            assert ds_val.dtype == anp_val.dtype

    # pred dtypes / shapes match.
    assert ds_pred.pred.shape == anp_pred.pred.shape == (len(df),)
    assert ds_pred.pred.dtype == anp_pred.pred.dtype

    # meta shape: same keys for the head-kind-conditional contents.
    assert ds_pred.meta.get("head_kind") == anp_pred.meta.get("head_kind") == head_kind

    # Head-kind-specific population checks.
    if head_kind == "gaussian":
        assert anp_pred.uncertainty is not None
        assert (anp_pred.uncertainty > 0).all()
        assert anp_pred.lower is None and anp_pred.upper is None
    elif head_kind == "quantile":
        assert anp_pred.lower is not None and anp_pred.upper is not None
        assert (anp_pred.lower <= anp_pred.upper).all()
        assert anp_pred.uncertainty is None
        assert "quantile_levels" in anp_pred.meta
    else:  # point
        assert anp_pred.uncertainty is None
        assert anp_pred.lower is None and anp_pred.upper is None


def test_anp_gaussian_uncertainty_positive_and_finite(tmp_path: Path):
    df = _synthetic_frame(n_dates=4)
    ckpt = _save_checkpoint(
        tmp_path, decoder_kind="anp", head_kind="gaussian", tag="anp_g"
    )
    res = ConditionalSurfacePredictor.from_checkpoint(ckpt).predict(df)
    assert res.uncertainty is not None
    assert (res.uncertainty > 0).all()
    assert np.isfinite(res.pred).all()
    assert np.isfinite(res.uncertainty).all()


def test_row_order_preserved_for_anp(tmp_path: Path):
    """PredictionResult.pred must align with df row order (2C.5 contract)."""
    df = _synthetic_frame(n_dates=2, n_per_date=8)
    ckpt = _save_checkpoint(
        tmp_path, decoder_kind="anp", head_kind="gaussian", tag="anp_order"
    )
    pred = ConditionalSurfacePredictor.from_checkpoint(ckpt).predict(df)

    # Shuffle the input — predictions must follow.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    df_shuf = df.iloc[perm].reset_index(drop=True)
    pred_shuf = ConditionalSurfacePredictor.from_checkpoint(ckpt).predict(df_shuf)

    np.testing.assert_allclose(pred_shuf.pred, pred.pred[perm], atol=1e-6)


def test_legacy_checkpoint_without_decoder_kind_defaults_to_deepsets(tmp_path: Path):
    """A Phase 2 checkpoint with no decoder_kind in its config still loads."""
    torch.manual_seed(11)
    model = ConditionalSurfaceModel(**_BASE_CFG, head={"kind": "gaussian"})
    legacy_cfg = dict(_BASE_CFG)
    legacy_cfg["head"] = {"kind": "gaussian"}
    # NOTE: no decoder_kind / coord_encoding / anp keys — mimics 2D.7 era ckpts.
    ckpt_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.0,
            "config": legacy_cfg,
        },
        ckpt_path,
    )
    loaded = ConditionalSurfacePredictor.from_checkpoint(ckpt_path)
    assert loaded.model.decoder_kind == "deepsets"
    df = _synthetic_frame()
    res = loaded.predict(df)
    assert res.uncertainty is not None
    assert (res.uncertainty > 0).all()
