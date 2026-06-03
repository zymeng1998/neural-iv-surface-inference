"""Unit tests for SetEncoder / ConditionalSurfaceModel under micro_v1 (3C.2).

Covers ADR 0008 acceptance:
  - SetEncoder(in_dim=9) round-trips shapes and `return_elements=True`, and the
    padding mask still zeroes padded rows.
  - ConditionalSurfaceModel(feature_set="micro_v1") builds with a 9-dim input
    projection and forwards finite outputs for both decoder kinds.
  - feature_set is authoritative for the context dim; a conflicting explicit
    context_dim raises.
"""

from __future__ import annotations

import pytest
import torch

from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
    SetEncoder,
)


def test_set_encoder_in_dim_9_shapes_and_elements():
    torch.manual_seed(0)
    enc = SetEncoder(in_dim=9, hidden_dim=16, latent_dim=8, n_elem_layers=2)
    assert enc.elem_mlp[0][0].in_features == 9

    b, n = 2, 5
    context = torch.randn(b, n, 9)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 3:] = False  # pad last two rows of sample 0

    z = enc(context, mask)
    assert z.shape == (b, 8)
    assert torch.isfinite(z).all()

    z2, h = enc(context, mask, return_elements=True)
    assert h.shape == (b, n, 16)
    # Padded element embeddings must be exactly zero.
    assert torch.equal(h[0, 3:], torch.zeros(2, 16))


def test_set_encoder_mask_changes_pool():
    """Masking a row must change z (padded rows cannot leak into the pool)."""
    torch.manual_seed(1)
    enc = SetEncoder(in_dim=9, hidden_dim=16, latent_dim=8)
    context = torch.randn(1, 4, 9)
    full = torch.ones(1, 4, dtype=torch.bool)
    partial = full.clone()
    partial[0, 3] = False
    z_full = enc(context, full)
    z_partial = enc(context, partial)
    assert not torch.allclose(z_full, z_partial)


def _tiny_model_kwargs() -> dict:
    return dict(
        coord_dim=2,
        hidden_dim=16,
        latent_dim=8,
        n_elem_layers=1,
        n_post_layers=1,
        n_decoder_layers=2,
        feature_set="micro_v1",
    )


@pytest.mark.parametrize("decoder_kind", ["deepsets", "anp"])
def test_model_micro_v1_builds_and_forwards(decoder_kind):
    torch.manual_seed(2)
    extra = {}
    if decoder_kind == "anp":
        extra["anp"] = {"n_heads": 2, "mlp_hidden": 16, "include_z_in_decoder": True}
    model = ConditionalSurfaceModel(
        **_tiny_model_kwargs(), decoder_kind=decoder_kind, **extra
    )
    assert model.feature_set == "micro_v1"
    assert model.encoder.elem_mlp[0][0].in_features == 9

    b, n_ctx, n_q = 2, 6, 4
    context = torch.randn(b, n_ctx, 9)
    context_mask = torch.ones(b, n_ctx, dtype=torch.bool)
    context_mask[0, 4:] = False
    query = torch.randn(b, n_q, 2)

    out = model(context, context_mask, query)
    assert "mu" in out
    assert out["mu"].shape == (b, n_q)
    assert torch.isfinite(out["mu"]).all()


def test_model_context_dim_conflict_raises():
    with pytest.raises(ValueError, match="conflicts with feature_set"):
        ConditionalSurfaceModel(context_dim=3, feature_set="micro_v1")


def test_model_minimal_default_in_dim_3():
    model = ConditionalSurfaceModel()
    assert model.feature_set == "minimal"
    assert model.encoder.elem_mlp[0][0].in_features == 3
