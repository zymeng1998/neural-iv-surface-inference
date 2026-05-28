"""Unit tests for the ANP cross-attention decoder (3B.2)."""

from __future__ import annotations

import pytest
import torch

from neural_iv_surface_inference.models.anp_decoder import ANPDecoder


def _make_batch(
    batch: int = 3,
    n_ctx: int = 7,
    n_q: int = 5,
    d_h: int = 16,
    d_z: int = 8,
    d_query: int = 2,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(batch, n_ctx, d_h, generator=g)
    z = torch.randn(batch, d_z, generator=g)
    q = torch.randn(batch, n_q, d_query, generator=g)
    mask = torch.ones(batch, n_ctx, dtype=torch.bool)
    # Drop the last two positions in row 0 to exercise the mask.
    mask[0, -2:] = False
    return H, mask, z, q


@pytest.mark.parametrize(
    "head_kind,expected_keys",
    [
        ("gaussian", {"mu", "sigma", "log_sigma2"}),
        ("quantile", {"mu", "quantiles", "quantile_levels"}),
        ("point", {"mu"}),
    ],
)
def test_forward_shapes_and_keys(head_kind: str, expected_keys: set[str]):
    torch.manual_seed(0)
    dec = ANPDecoder(
        d_h=16, d_z=8, d_query=2, n_heads=4, mlp_hidden=32, head_kind=head_kind
    )
    H, mask, z, q = _make_batch()
    out = dec(H=H, mask=mask, z=z, q=q)
    assert set(out.keys()) == expected_keys
    assert out["mu"].shape == (3, 5)
    if head_kind == "gaussian":
        assert out["sigma"].shape == (3, 5)
        assert (out["sigma"] > 0).all()
    if head_kind == "quantile":
        assert out["quantiles"].shape == (3, 5, 3)
        assert out["quantile_levels"].shape == (3,)


def test_mu_is_strictly_positive():
    torch.manual_seed(1)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    H, mask, z, q = _make_batch()
    out = dec(H=H, mask=mask, z=z, q=q)
    assert (out["mu"] >= 0).all()
    assert (out["sigma"] > 0).all()


def test_quantile_sorted_at_eval_unsorted_at_train():
    torch.manual_seed(2)
    dec = ANPDecoder(
        d_h=16, d_z=8, d_query=2, head_kind="quantile",
        quantiles=(0.1, 0.5, 0.9),
    )
    H, mask, z, q = _make_batch()

    dec.eval()
    out_eval = dec(H=H, mask=mask, z=z, q=q)
    q_eval = out_eval["quantiles"]
    assert torch.all(q_eval[..., :-1] <= q_eval[..., 1:])

    dec.train()
    # train-mode output need not be sorted; just verify it runs and shapes match.
    out_train = dec(H=H, mask=mask, z=z, q=q)
    assert out_train["quantiles"].shape == q_eval.shape


def test_mask_blocks_padded_rows():
    """Replacing the *content* of masked rows must leave outputs unchanged."""
    torch.manual_seed(3)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    dec.eval()
    H, mask, z, q = _make_batch()
    out_ref = dec(H=H, mask=mask, z=z, q=q)["mu"]

    H_alt = H.clone()
    # Row 0 last two positions are masked; jam them with junk values.
    H_alt[0, -2:, :] = 1e3 * torch.randn_like(H_alt[0, -2:, :])
    out_alt = dec(H=H_alt, mask=mask, z=z, q=q)["mu"]
    assert torch.allclose(out_ref, out_alt, atol=1e-5)


def test_context_permutation_invariance():
    """ANP output for fixed queries is invariant under context permutation."""
    torch.manual_seed(4)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    dec.eval()
    H, mask, z, q = _make_batch(batch=2, n_ctx=6)
    # Pick a permutation; permute both H rows and mask rows together.
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    H_p = H[:, perm, :]
    mask_p = mask[:, perm]

    out_ref = dec(H=H, mask=mask, z=z, q=q)["mu"]
    out_perm = dec(H=H_p, mask=mask_p, z=z, q=q)["mu"]
    assert torch.allclose(out_ref, out_perm, atol=1e-5)


def test_gradient_flow_to_inputs_and_params():
    torch.manual_seed(5)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    H, mask, z, q = _make_batch()
    H = H.requires_grad_(True)
    z = z.requires_grad_(True)
    q = q.requires_grad_(True)
    out = dec(H=H, mask=mask, z=z, q=q)
    loss = out["mu"].sum() + out["sigma"].sum()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    any_param_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and (p.grad.abs() > 0).any()
        for p in dec.parameters()
    )
    assert any_param_grad


def test_determinism_under_fixed_seed():
    H, mask, z, q = _make_batch(seed=7)

    def _run():
        torch.manual_seed(123)
        dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
        dec.eval()
        return dec(H=H, mask=mask, z=z, q=q)["mu"]

    a = _run()
    b = _run()
    assert torch.equal(a, b)


def test_degenerate_mask_one_real_element():
    """All but one masked → output must be finite (no NaN)."""
    torch.manual_seed(8)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    dec.eval()
    H, mask, z, q = _make_batch()
    mask = torch.zeros_like(mask)
    mask[:, 0] = True
    out = dec(H=H, mask=mask, z=z, q=q)
    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["sigma"]).all()


def test_degenerate_mask_all_padded_finite():
    """Even a fully-padded row must not produce NaN (boundary robustness)."""
    torch.manual_seed(9)
    dec = ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="gaussian")
    dec.eval()
    H, mask, z, q = _make_batch()
    mask = torch.zeros_like(mask)  # nothing real
    out = dec(H=H, mask=mask, z=z, q=q)
    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["sigma"]).all()


def test_rejects_invalid_head_kind():
    with pytest.raises(ValueError):
        ANPDecoder(d_h=16, d_z=8, d_query=2, head_kind="bogus")  # type: ignore[arg-type]


def test_rejects_unsorted_quantiles():
    with pytest.raises(ValueError):
        ANPDecoder(
            d_h=16, d_z=8, d_query=2,
            head_kind="quantile", quantiles=(0.9, 0.5, 0.1),
        )
