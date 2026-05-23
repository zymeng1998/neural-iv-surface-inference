"""Tests for the conditional surface architecture (2C.3)."""

from __future__ import annotations

import torch

from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
    CoordinateDecoder,
    SetEncoder,
)


def test_set_encoder_output_shape():
    torch.manual_seed(0)
    enc = SetEncoder(in_dim=3, hidden_dim=32, latent_dim=16)
    batch, max_ctx = 4, 7
    x = torch.randn(batch, max_ctx, 3)
    mask = torch.ones(batch, max_ctx, dtype=torch.bool)
    z = enc(x, mask)
    assert z.shape == (batch, 16)


def test_set_encoder_permutation_invariance():
    torch.manual_seed(0)
    enc = SetEncoder(in_dim=3, hidden_dim=32, latent_dim=16)
    enc.eval()
    batch, max_ctx = 2, 6
    x = torch.randn(batch, max_ctx, 3)
    mask = torch.ones(batch, max_ctx, dtype=torch.bool)

    z_ref = enc(x, mask)

    perm = torch.randperm(max_ctx)
    x_shuf = x[:, perm, :]
    mask_shuf = mask[:, perm]
    z_shuf = enc(x_shuf, mask_shuf)

    assert torch.allclose(z_ref, z_shuf, atol=1e-6)


def test_set_encoder_mask_invariance_to_padding():
    """Appending masked-out rows must leave z_t unchanged."""
    torch.manual_seed(0)
    enc = SetEncoder(in_dim=3, hidden_dim=32, latent_dim=16)
    enc.eval()

    batch, n_real = 3, 5
    x_real = torch.randn(batch, n_real, 3)
    mask_real = torch.ones(batch, n_real, dtype=torch.bool)
    z_real = enc(x_real, mask_real)

    # Append two padding rows with garbage features and mask=False
    pad = torch.randn(batch, 2, 3) * 100.0
    x_padded = torch.cat([x_real, pad], dim=1)
    mask_padded = torch.cat(
        [mask_real, torch.zeros(batch, 2, dtype=torch.bool)], dim=1
    )
    z_padded = enc(x_padded, mask_padded)

    assert torch.allclose(z_real, z_padded, atol=1e-6)


def test_coordinate_decoder_shape_and_positive_output():
    torch.manual_seed(0)
    dec = CoordinateDecoder(latent_dim=16, coord_dim=2, hidden_dim=32)
    batch, n_query = 4, 11
    z = torch.randn(batch, 16)
    q = torch.randn(batch, n_query, 2)
    out = dec(z, q)
    assert out.shape == (batch, n_query)
    assert torch.all(out > 0.0)


def test_coordinate_decoder_variable_query_counts():
    """Decoder must handle arbitrary query counts per batch (same n across batch is fine)."""
    torch.manual_seed(0)
    dec = CoordinateDecoder(latent_dim=8, coord_dim=2, hidden_dim=16)
    z = torch.randn(2, 8)
    for n in (1, 5, 20):
        q = torch.randn(2, n, 2)
        out = dec(z, q)
        assert out.shape == (2, n)
        assert torch.all(out > 0.0)


def test_conditional_surface_model_end_to_end():
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=32, latent_dim=16
    )
    model.eval()
    batch, max_ctx, n_q = 3, 6, 9
    ctx = torch.randn(batch, max_ctx, 3)
    ctx_mask = torch.ones(batch, max_ctx, dtype=torch.bool)
    # Mask out the last 2 rows of the first sample
    ctx_mask[0, -2:] = False
    q = torch.randn(batch, n_q, 2)

    pred = model(ctx, ctx_mask, q)
    assert pred.shape == (batch, n_q)
    assert torch.all(pred > 0.0)
