"""Unit tests for the coordinate-encoding module (3A.2)."""

from __future__ import annotations

import pytest
import torch

from neural_iv_surface_inference.features.coord_encoding import (
    FourierCoordEncoding,
    RawCoordEncoding,
    build_coord_encoding,
)


@pytest.mark.unit
def test_raw_encoding_passthrough_and_shape():
    enc = RawCoordEncoding(coord_dim=2)
    assert enc.encoded_dim == 2
    x = torch.randn(3, 5, 2)
    out = enc(x)
    assert out.shape == (3, 5, 2)
    assert torch.equal(out, x)


@pytest.mark.unit
def test_fourier_encoded_dim_matches_closed_form():
    enc = FourierCoordEncoding(
        coord_dim=2, num_bands=8, max_freq=10.0, include_input=True
    )
    # encoded_dim = coord_dim + 2 * coord_dim * num_bands = 2 + 2*2*8 = 34
    assert enc.encoded_dim == 34
    x = torch.randn(4, 6, 2)
    out = enc(x)
    assert out.shape == (4, 6, 34)


@pytest.mark.unit
def test_fourier_encoded_dim_without_input():
    enc = FourierCoordEncoding(
        coord_dim=2, num_bands=8, include_input=False
    )
    assert enc.encoded_dim == 32
    out = enc(torch.randn(2, 3, 2))
    assert out.shape == (2, 3, 32)


@pytest.mark.unit
def test_fourier_determinism():
    enc = FourierCoordEncoding(coord_dim=2, num_bands=4, max_freq=8.0)
    x = torch.randn(2, 7, 2)
    a = enc(x)
    b = enc(x)
    assert torch.equal(a, b)


@pytest.mark.unit
def test_fourier_gradient_flow():
    enc = FourierCoordEncoding(coord_dim=2, num_bands=4)
    x = torch.randn(2, 5, 2, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_raw_gradient_flow():
    enc = RawCoordEncoding(coord_dim=3)
    x = torch.randn(2, 4, 3, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_builder_roundtrip_raw():
    enc = build_coord_encoding({"kind": "raw"}, coord_dim=2)
    assert isinstance(enc, RawCoordEncoding)
    assert enc.encoded_dim == 2


@pytest.mark.unit
def test_builder_roundtrip_fourier():
    cfg = {
        "kind": "fourier",
        "num_bands": 8,
        "max_freq": 10.0,
        "include_input": True,
    }
    enc = build_coord_encoding(cfg, coord_dim=2)
    assert isinstance(enc, FourierCoordEncoding)
    assert enc.num_bands == 8
    assert enc.max_freq == 10.0
    assert enc.include_input is True
    assert enc.encoded_dim == 34


@pytest.mark.unit
def test_builder_default_none_is_raw():
    enc = build_coord_encoding(None, coord_dim=2)
    assert isinstance(enc, RawCoordEncoding)


@pytest.mark.unit
def test_builder_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown coord_encoding kind"):
        build_coord_encoding({"kind": "siren"}, coord_dim=2)


@pytest.mark.unit
def test_fourier_invalid_args_raise():
    with pytest.raises(ValueError):
        FourierCoordEncoding(coord_dim=0)
    with pytest.raises(ValueError):
        FourierCoordEncoding(coord_dim=2, num_bands=0)
    with pytest.raises(ValueError):
        FourierCoordEncoding(coord_dim=2, max_freq=0.0)


@pytest.mark.unit
def test_fourier_value_check_sin_cos_at_zero():
    """At x=0, sin component is 0 and cos component is 1 for every band."""
    enc = FourierCoordEncoding(
        coord_dim=2, num_bands=4, max_freq=4.0, include_input=False
    )
    x = torch.zeros(1, 1, 2)
    out = enc(x)  # (1, 1, 16) — 4 bands * 2 coords * (sin, cos)
    # By construction in the module, the last axis interleaves (sin, cos)
    # pairs per (coord, band). Sins should all be 0; cosines should be 1.
    assert torch.allclose(out[..., 0::2], torch.zeros_like(out[..., 0::2]))
    assert torch.allclose(out[..., 1::2], torch.ones_like(out[..., 1::2]))
