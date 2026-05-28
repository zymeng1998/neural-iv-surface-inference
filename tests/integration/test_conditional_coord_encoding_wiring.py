"""Integration tests: `coord_encoding` flag wiring through ConditionalSurfaceModel (3A.2)."""

from __future__ import annotations

import pytest
import torch

from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)


def _tiny_kwargs() -> dict:
    return dict(
        context_dim=3,
        coord_dim=2,
        hidden_dim=16,
        latent_dim=8,
        n_elem_layers=1,
        n_post_layers=1,
        n_decoder_layers=2,
        head={"kind": "gaussian"},
    )


def _synthetic_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    context = torch.randn(2, 5, 3)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    query = torch.randn(2, 4, 2)
    return context, context_mask, query


@pytest.mark.integration
def test_forward_raw_gaussian_keys_and_shapes():
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(**_tiny_kwargs(), coord_encoding={"kind": "raw"})
    context, mask, query = _synthetic_batch()
    out = model(context, mask, query)
    assert set(out.keys()) == {"mu", "sigma", "log_sigma2"}
    assert out["mu"].shape == (2, 4)
    assert out["sigma"].shape == (2, 4)
    assert out["log_sigma2"].shape == (2, 4)
    assert torch.all(out["sigma"] > 0)


@pytest.mark.integration
def test_forward_fourier_gaussian_keys_and_shapes():
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(
        **_tiny_kwargs(),
        coord_encoding={
            "kind": "fourier",
            "num_bands": 8,
            "max_freq": 10.0,
            "include_input": True,
        },
    )
    context, mask, query = _synthetic_batch()
    out = model(context, mask, query)
    assert set(out.keys()) == {"mu", "sigma", "log_sigma2"}
    assert out["mu"].shape == (2, 4)
    assert torch.all(out["sigma"] > 0)


@pytest.mark.integration
def test_raw_default_matches_explicit_raw():
    """Default (None) coord_encoding must be bit-for-bit equal to {"kind": "raw"}."""
    torch.manual_seed(123)
    model_default = ConditionalSurfaceModel(**_tiny_kwargs())
    torch.manual_seed(123)
    model_raw = ConditionalSurfaceModel(
        **_tiny_kwargs(), coord_encoding={"kind": "raw"}
    )

    # Parameter parity.
    for (n1, p1), (n2, p2) in zip(
        model_default.named_parameters(), model_raw.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param mismatch on {n1}"

    context, mask, query = _synthetic_batch(seed=7)
    out_default = model_default(context, mask, query)
    out_raw = model_raw(context, mask, query)
    for key in out_default:
        assert torch.equal(out_default[key], out_raw[key]), f"output mismatch on {key}"


@pytest.mark.integration
def test_param_count_delta_matches_closed_form():
    """Switching raw→fourier expands ONLY the decoder trunk input layer."""
    torch.manual_seed(0)
    raw = ConditionalSurfaceModel(**_tiny_kwargs(), coord_encoding={"kind": "raw"})
    torch.manual_seed(0)
    fou = ConditionalSurfaceModel(
        **_tiny_kwargs(),
        coord_encoding={
            "kind": "fourier",
            "num_bands": 8,
            "max_freq": 10.0,
            "include_input": True,
        },
    )

    n_raw = sum(p.numel() for p in raw.parameters())
    n_fou = sum(p.numel() for p in fou.parameters())

    # Encoded-dim grows from 2 to 34, so the trunk's first Linear gains
    # (34 - 2) * hidden_dim extra weights (bias unchanged).
    kwargs = _tiny_kwargs()
    expected_delta = (34 - 2) * kwargs["hidden_dim"]
    assert n_fou - n_raw == expected_delta


@pytest.mark.integration
def test_coord_encoding_cfg_persisted():
    """coord_encoding_cfg attribute reflects the resolved config (for checkpoint reload)."""
    model = ConditionalSurfaceModel(
        **_tiny_kwargs(),
        coord_encoding={"kind": "fourier", "num_bands": 4, "max_freq": 8.0},
    )
    assert model.coord_encoding_cfg["kind"] == "fourier"
    assert model.coord_encoding_cfg["num_bands"] == 4
    assert model.coord_encoding.encoded_dim == 2 + 2 * 2 * 4  # = 18


@pytest.mark.integration
def test_forward_point_head_with_fourier():
    """Point-head decoder path also routes through the encoded query."""
    kwargs = _tiny_kwargs()
    kwargs["head"] = {"kind": "point"}
    torch.manual_seed(0)
    model = ConditionalSurfaceModel(
        **kwargs,
        coord_encoding={"kind": "fourier", "num_bands": 4, "max_freq": 8.0},
    )
    context, mask, query = _synthetic_batch()
    out = model(context, mask, query)
    assert set(out.keys()) == {"mu"}
    assert out["mu"].shape == (2, 4)
    assert torch.all(out["mu"] > 0)
