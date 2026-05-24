"""Tests for the conditional surface architecture (2C.3, extended by 2D.2)."""

from __future__ import annotations

import math

import pytest
import torch

from neural_iv_surface_inference.models.conditional_surface import (
    DEFAULT_QUANTILES,
    ConditionalSurfaceModel,
    CoordinateDecoder,
    MultiOutputDecoder,
    SetEncoder,
)
from neural_iv_surface_inference.models.losses import (
    gaussian_nll_loss,
    pinball_loss,
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
    assert isinstance(pred, dict)
    assert pred["mu"].shape == (batch, n_q)
    assert torch.all(pred["mu"] > 0.0)


# ── W4 heads (2D.2) ────────────────────────────────────────────────


def test_gaussian_head_shapes_and_positivity():
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=32, latent_dim=16,
        head={"kind": "gaussian"},
    )
    model.eval()
    B, N, Q = 3, 5, 7
    ctx = torch.randn(B, N, 3)
    ctx_mask = torch.ones(B, N, dtype=torch.bool)
    q = torch.randn(B, Q, 2)
    out = model(ctx, ctx_mask, q)
    assert set(out.keys()) >= {"mu", "sigma", "log_sigma2"}
    assert out["mu"].shape == (B, Q)
    assert out["sigma"].shape == (B, Q)
    assert out["log_sigma2"].shape == (B, Q)
    assert torch.all(out["mu"] > 0.0)
    assert torch.all(out["sigma"] > 0.0)


def test_quantile_head_shapes_and_sorting_in_eval():
    torch.manual_seed(0)
    qs = (0.1, 0.5, 0.9)
    model = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=32, latent_dim=16,
        head={"kind": "quantile", "quantiles": list(qs)},
    )
    B, N, Q = 4, 5, 6
    ctx = torch.randn(B, N, 3)
    ctx_mask = torch.ones(B, N, dtype=torch.bool)
    q = torch.randn(B, Q, 2)

    model.eval()
    out = model(ctx, ctx_mask, q)
    assert out["quantiles"].shape == (B, Q, len(qs))
    diffs = out["quantiles"].diff(dim=-1)
    assert torch.all(diffs >= -1e-6)  # monotone non-decreasing in eval mode
    # mu is the middle quantile
    assert torch.allclose(out["mu"], out["quantiles"][..., len(qs) // 2])


def test_quantile_head_invalid_levels_rejected():
    with pytest.raises(ValueError):
        MultiOutputDecoder(
            latent_dim=8, coord_dim=2, hidden_dim=16, n_layers=2,
            head_kind="quantile", quantiles=(0.9, 0.5, 0.1),
        )
    with pytest.raises(ValueError):
        MultiOutputDecoder(
            latent_dim=8, coord_dim=2, hidden_dim=16, n_layers=2,
            head_kind="quantile", quantiles=(0.0, 0.5, 0.9),
        )


def test_pinball_loss_zero_on_perfect_quantiles():
    # Perfectly predicting the target at every quantile level → loss == 0.
    target = torch.tensor([[0.2, 0.25]])
    quantiles = (0.1, 0.5, 0.9)
    preds = target.unsqueeze(-1).expand(-1, -1, len(quantiles)).clone()
    mask = torch.tensor([[True, True]])
    loss = pinball_loss(preds, target, mask, quantiles)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_pinball_loss_known_value():
    # K=1 with q=0.5: pinball = 0.5 * |target - pred|.
    target = torch.tensor([[1.0]])
    preds = torch.tensor([[[0.0]]])
    mask = torch.tensor([[True]])
    loss = pinball_loss(preds, target, mask, (0.5,))
    assert float(loss) == pytest.approx(0.5, abs=1e-6)


def test_gaussian_nll_finite_and_matches_closed_form():
    # Closed form: with sigma=1, target=mu, NLL = 0.5 * log(2π * 1) per point.
    mu = torch.tensor([[0.2, 0.3]])
    sigma = torch.ones_like(mu)
    target = mu.clone()
    mask = torch.tensor([[True, True]])
    loss = gaussian_nll_loss(mu, sigma, target, mask)
    expected = 0.5 * math.log(1.0)  # log(sigma^2) term + (mu-target)^2 = 0
    # gaussian_nll_loss includes only 0.5*(log_sigma2 + sq_err/sigma2) — drops the
    # 0.5*log(2π) additive constant (it cancels in optimization). Verify the
    # written formula:
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_default_quantiles_constant_matches_spec():
    assert DEFAULT_QUANTILES == (0.05, 0.5, 0.95)
