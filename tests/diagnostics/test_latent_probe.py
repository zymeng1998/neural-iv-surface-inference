"""Smoke test for the latent-extraction hook (Story 2E.2).

Builds a tiny synthetic loader + a real ``ConditionalSurfaceModel`` and
verifies the hook captures latents with the expected shape and row count.
No data files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neural_iv_surface_inference.diagnostics.latent_probe import (
    LatentCache,
    extract_latents,
)
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)


pytestmark = pytest.mark.unit


def _make_loader(
    n_batches: int = 3,
    batch_size: int = 4,
    n_ctx: int = 5,
    n_query: int = 6,
) -> list[dict]:
    rng = torch.Generator().manual_seed(20260526)
    batches = []
    for _ in range(n_batches):
        batches.append(
            {
                "context": torch.randn(
                    batch_size, n_ctx, 3, generator=rng
                ).double(),
                "context_mask": torch.ones(
                    batch_size, n_ctx, dtype=torch.bool
                ),
                "query": torch.randn(
                    batch_size, n_query, 2, generator=rng
                ).double(),
                "target": torch.rand(
                    batch_size, n_query, generator=rng
                ).double(),
                "query_mask": torch.ones(
                    batch_size, n_query, dtype=torch.bool
                ),
            }
        )
    return batches


def test_extract_latents_returns_aligned_cache() -> None:
    latent_dim = 8
    model = ConditionalSurfaceModel(
        context_dim=3,
        coord_dim=2,
        hidden_dim=16,
        latent_dim=latent_dim,
        n_elem_layers=2,
        n_post_layers=1,
        n_decoder_layers=2,
        head={"kind": "gaussian"},
    ).double()
    loader = _make_loader(n_batches=3, batch_size=4)

    cache = extract_latents(model, loader, device=torch.device("cpu"))

    assert isinstance(cache, LatentCache)
    n_total = 3 * 4
    assert cache.Z.shape == (n_total, latent_dim)
    assert cache.query.shape == (n_total, 6, 2)
    assert cache.target.shape == (n_total, 6)
    assert cache.query_mask.shape == (n_total, 6)
    assert cache.Z.device.type == "cpu"
    assert torch.isfinite(cache.Z).all()


def test_extract_latents_matches_direct_encoder_call() -> None:
    model = ConditionalSurfaceModel(
        context_dim=3,
        coord_dim=2,
        hidden_dim=8,
        latent_dim=4,
        head={"kind": "point"},
    ).double()
    model.eval()
    loader = _make_loader(n_batches=2, batch_size=3)

    cache = extract_latents(model, loader, device=torch.device("cpu"))

    expected = []
    with torch.no_grad():
        for b in loader:
            expected.append(model.encoder(b["context"], b["context_mask"]))
    Z_direct = torch.cat(expected, dim=0)

    assert torch.allclose(cache.Z, Z_direct, atol=1e-10)


def test_extract_latents_raises_on_empty_loader() -> None:
    model = ConditionalSurfaceModel(latent_dim=4).double()
    with pytest.raises(RuntimeError, match="no batches"):
        extract_latents(model, iter([]), device=torch.device("cpu"))


def test_extract_latents_pads_variable_query_widths_across_batches() -> None:
    # Regression: collate_conditional pads each batch to its own max Q, so the
    # captured per-batch tensors can disagree on dim=1. The hook must pad to
    # the global max before concatenating.
    model = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2,
        hidden_dim=8, latent_dim=4,
        head={"kind": "point"},
    ).double()
    loader = [
        _make_loader(n_batches=1, batch_size=2, n_query=5)[0],
        _make_loader(n_batches=1, batch_size=2, n_query=9)[0],
        _make_loader(n_batches=1, batch_size=2, n_query=7)[0],
    ]

    cache = extract_latents(model, loader, device=torch.device("cpu"))

    assert cache.Z.shape == (6, 4)
    assert cache.query.shape == (6, 9, 2)
    assert cache.target.shape == (6, 9)
    assert cache.query_mask.shape == (6, 9)
    # Real rows from the n_query=5 batch keep their original mask True; the
    # extra 4 padded positions must be False so per-row loss ignores them.
    first_batch_real_mask = cache.query_mask[:2, :5]
    first_batch_pad_mask = cache.query_mask[:2, 5:]
    assert first_batch_real_mask.all()
    assert not first_batch_pad_mask.any()


def test_extract_latents_requires_encoder_attribute() -> None:
    class Bare(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return None

    with pytest.raises(AttributeError, match="encoder"):
        extract_latents(Bare(), iter([]), device=torch.device("cpu"))
