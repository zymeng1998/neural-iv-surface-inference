"""Integration tests for ConditionalSurfaceModel(decoder_kind=...) (3B.2).

Synthetic, CPU-only, fast. Asserts:
  - `decoder_kind="deepsets"` produces bit-for-bit identical outputs to the
    pre-3B.2 baseline path on a fixed input batch (regression guard).
  - `decoder_kind="anp"` reduces a masked-MSE loss over 50 steps on a tiny
    synthetic batch (smoke).
  - Per-row mask shuffling of the context set leaves outputs invariant
    (set-equivariance).
  - Masking out all-but-one context element returns finite output
    (degenerate-mask robustness).
  - All three head.kind values compose with decoder_kind="anp".
"""

from __future__ import annotations

import pytest
import torch

from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)


def _make_inputs(
    batch: int = 4,
    n_ctx: int = 32,
    n_q: int = 16,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    context = torch.randn(batch, n_ctx, 3, generator=g)
    query = torch.randn(batch, n_q, 2, generator=g)
    mask = torch.ones(batch, n_ctx, dtype=torch.bool)
    # Pad the tail of the first row to exercise the mask.
    mask[0, -4:] = False
    target = 0.2 + 0.05 * torch.randn(batch, n_q, generator=g).abs()
    target_mask = torch.ones(batch, n_q, dtype=torch.bool)
    return context, mask, query, target, target_mask


def _build_model(decoder_kind: str, head_kind: str = "gaussian", seed: int = 7):
    torch.manual_seed(seed)
    return ConditionalSurfaceModel(
        context_dim=3,
        coord_dim=2,
        hidden_dim=32,
        latent_dim=16,
        n_elem_layers=1,
        n_post_layers=1,
        n_decoder_layers=2,
        head={"kind": head_kind},
        decoder_kind=decoder_kind,
        anp={"n_heads": 4, "mlp_hidden": 32},
    )


@pytest.mark.integration
def test_deepsets_path_bitforbit_baseline():
    """Default decoder_kind='deepsets' is the Phase 2 path; verify constructor
    leaves head dispatch untouched and produces a finite forward pass."""
    context, mask, query, _, _ = _make_inputs()

    torch.manual_seed(7)
    ref = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=32, latent_dim=16,
        n_elem_layers=1, n_post_layers=1, n_decoder_layers=2,
        head={"kind": "gaussian"},
    )
    ref.eval()
    out_ref = ref(context, mask, query)

    torch.manual_seed(7)
    new = ConditionalSurfaceModel(
        context_dim=3, coord_dim=2, hidden_dim=32, latent_dim=16,
        n_elem_layers=1, n_post_layers=1, n_decoder_layers=2,
        head={"kind": "gaussian"},
        decoder_kind="deepsets",
    )
    new.eval()
    out_new = new(context, mask, query)

    assert torch.equal(out_ref["mu"], out_new["mu"])
    assert torch.equal(out_ref["sigma"], out_new["sigma"])


@pytest.mark.integration
@pytest.mark.parametrize("head_kind", ["gaussian", "quantile", "point"])
def test_anp_forward_all_heads(head_kind: str):
    context, mask, query, _, _ = _make_inputs()
    model = _build_model("anp", head_kind=head_kind)
    model.eval()
    out = model(context, mask, query)
    assert "mu" in out
    assert out["mu"].shape == (4, 16)
    assert torch.isfinite(out["mu"]).all()
    if head_kind == "gaussian":
        assert out["sigma"].shape == (4, 16)
        assert (out["sigma"] > 0).all()
    if head_kind == "quantile":
        assert out["quantiles"].shape[-1] == 3


@pytest.mark.integration
def test_anp_synthetic_smoke_loss_decreases():
    context, mask, query, target, tmask = _make_inputs()
    model = _build_model("anp", head_kind="point", seed=11)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)

    def _step():
        optim.zero_grad()
        out = model(context, mask, query)
        # masked MSE on positive target
        diff = (out["mu"] - target) ** 2
        loss = (diff * tmask).sum() / tmask.sum().clamp_min(1)
        loss.backward()
        optim.step()
        return float(loss.detach())

    initial = _step()
    losses = [_step() for _ in range(50)]
    assert losses[-1] < initial, (
        f"smoke loss did not decrease over 50 steps: initial={initial} "
        f"final={losses[-1]}"
    )


@pytest.mark.integration
def test_anp_context_permutation_invariance():
    context, mask, query, _, _ = _make_inputs()
    model = _build_model("anp", head_kind="gaussian", seed=13)
    model.eval()
    out_ref = model(context, mask, query)["mu"]

    perm = torch.randperm(context.shape[1])
    out_perm = model(context[:, perm, :], mask[:, perm], query)["mu"]
    assert torch.allclose(out_ref, out_perm, atol=1e-5)


@pytest.mark.integration
def test_anp_degenerate_mask_single_real_element():
    context, mask, query, _, _ = _make_inputs()
    mask = torch.zeros_like(mask)
    mask[:, 0] = True
    model = _build_model("anp", head_kind="gaussian", seed=17)
    model.eval()
    out = model(context, mask, query)
    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["sigma"]).all()


@pytest.mark.integration
def test_anp_decoder_kind_routes_through_anp_module():
    """Sanity: with decoder_kind='anp' the decoder must be an ANPDecoder."""
    from neural_iv_surface_inference.models.anp_decoder import ANPDecoder

    model = _build_model("anp", head_kind="gaussian")
    assert isinstance(model.decoder, ANPDecoder)


@pytest.mark.integration
def test_invalid_decoder_kind_raises():
    with pytest.raises(ValueError):
        ConditionalSurfaceModel(decoder_kind="bogus")  # type: ignore[arg-type]


@pytest.mark.integration
def test_train_conditional_forwards_decoder_kind_and_anp_cfg():
    """train_conditional must thread decoder_kind / anp from config to the model.

    Regression guard: the 3B.2 plumbing originally stopped at
    ConditionalSurfaceModel; train_conditional silently used the default
    DeepSets decoder regardless of YAML. 3B.4 relies on this forwarding.
    """
    from torch.utils.data import DataLoader

    import pandas as pd

    from neural_iv_surface_inference.data.conditional_loaders import (
        ConditionalIVSurfaceDataset,
        collate_conditional,
    )
    from neural_iv_surface_inference.models.anp_decoder import ANPDecoder
    from neural_iv_surface_inference.training.train_conditional import (
        train_conditional,
    )

    g = torch.Generator().manual_seed(0)
    n_days = 4
    rows = []
    for d in range(n_days):
        for i in range(8):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d),
                    "log_moneyness": float(torch.randn(1, generator=g).item()),
                    "tau": float(0.1 + 0.5 * torch.rand(1, generator=g).item()),
                    "implied_volatility": float(
                        0.2 + 0.05 * torch.rand(1, generator=g).item()
                    ),
                    "iv_clean": float(
                        0.2 + 0.05 * torch.rand(1, generator=g).item()
                    ),
                    "observed": True,
                    "split": "train",
                }
            )
    df = pd.DataFrame(rows)
    loader = DataLoader(
        ConditionalIVSurfaceDataset(df),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_conditional,
        num_workers=0,
    )

    cfg = {
        "seed": 42,
        "context_dim": 3,
        "coord_dim": 2,
        "hidden_dim": 32,
        "latent_dim": 16,
        "n_elem_layers": 1,
        "n_post_layers": 1,
        "n_decoder_layers": 2,
        "epochs": 1,
        "patience": 1,
        "head": {"kind": "gaussian"},
        "decoder_kind": "anp",
        "anp": {"n_heads": 2, "mlp_hidden": 32},
    }
    model, history = train_conditional(
        train_loader=loader,
        val_loader=loader,
        config=cfg,
        device=torch.device("cpu"),
        checkpoint_dir=None,
    )
    assert model.decoder_kind == "anp"
    assert isinstance(model.decoder, ANPDecoder)
    assert len(history) >= 1
