"""Tests for the date-grouped conditional dataset + collation (2C.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)


def _make_frame(rows_per_date: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Build a tiny synthetic benchmark frame.

    rows_per_date maps a date string -> (n_observed, n_unobserved).
    """
    rng = np.random.default_rng(0)
    parts = []
    for date_str, (n_obs, n_unobs) in rows_per_date.items():
        n = n_obs + n_unobs
        observed = np.array([True] * n_obs + [False] * n_unobs)
        rng.shuffle(observed)
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(date_str),
                "log_moneyness": rng.uniform(-0.3, 0.3, n).astype(np.float32),
                "tau": rng.uniform(0.05, 1.5, n).astype(np.float32),
                "implied_volatility": rng.uniform(0.1, 0.5, n).astype(np.float32),
                "iv_clean": rng.uniform(0.1, 0.5, n).astype(np.float32),
                "observed": observed,
                "split": "train",
                "noise_sigma": 0.01,
            }
        )
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_dataset_returns_one_date_per_index_with_observed_context():
    df = _make_frame({"2024-01-02": (5, 3), "2024-01-03": (7, 2)})
    ds = ConditionalIVSurfaceDataset(df)

    assert len(ds) == 2  # two unique dates

    sample = ds[0]
    # Two observed counts depending on date order (sorted)
    obs_counts = [5, 7]
    n_obs = obs_counts[0]
    n_total = 8

    assert sample["context"].shape == (n_obs, 3)  # [k, tau, iv_input]
    assert sample["query"].shape == (n_total, 2)
    assert sample["target"].shape == (n_total,)
    assert sample["query_observed"].shape == (n_total,)
    assert int(sample["query_observed"].sum()) == n_obs

    # Context rows are only the observed rows (verify by IV values being in
    # observed-row implied_volatility set)
    day_df = df[df["date"] == pd.to_datetime("2024-01-02")].reset_index(drop=True)
    obs_iv = set(np.round(day_df.loc[day_df["observed"], "implied_volatility"].values, 6))
    ctx_iv = set(np.round(sample["context"][:, 2].numpy(), 6))
    assert ctx_iv == obs_iv


def test_collate_pads_and_builds_mask_correctly():
    df = _make_frame({"2024-01-02": (5, 3), "2024-01-03": (7, 2)})
    ds = ConditionalIVSurfaceDataset(df)
    batch = collate_conditional([ds[0], ds[1]])

    assert batch["context"].shape[0] == 2  # batch size
    max_ctx = max(5, 7)
    max_q = max(8, 9)
    assert batch["context"].shape == (2, max_ctx, 3)
    assert batch["context_mask"].shape == (2, max_ctx)
    assert batch["query"].shape == (2, max_q, 2)
    assert batch["query_mask"].shape == (2, max_q)
    assert batch["target"].shape == (2, max_q)

    # Per-row context_mask.sum() equals each date's observed count
    assert int(batch["context_mask"][0].sum()) == 5
    assert int(batch["context_mask"][1].sum()) == 7

    # Padded positions are zero-filled
    row0_pad = batch["context"][0, 5:max_ctx]
    assert torch.all(row0_pad == 0.0)

    # query_mask sums to the per-date total query count
    assert int(batch["query_mask"][0].sum()) == 8
    assert int(batch["query_mask"][1].sum()) == 9


def test_dataset_compatible_with_torch_dataloader():
    df = _make_frame({"2024-01-02": (3, 1), "2024-01-03": (4, 1), "2024-01-04": (2, 2)})
    ds = ConditionalIVSurfaceDataset(df)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_conditional)
    batches = list(loader)
    assert len(batches) == 2  # 3 dates with batch_size=2 -> 2 batches

    # All tensors are torch.Tensor with float32/bool
    b0 = batches[0]
    assert b0["context"].dtype == torch.float32
    assert b0["context_mask"].dtype == torch.bool
    assert b0["query_mask"].dtype == torch.bool


def test_dataset_preserves_chronological_date_ordering():
    df = _make_frame({"2024-01-04": (2, 1), "2024-01-02": (3, 1), "2024-01-03": (2, 2)})
    ds = ConditionalIVSurfaceDataset(df)
    expected = [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert ds.dates == expected


def test_dataset_rejects_dates_with_zero_observed():
    df = _make_frame({"2024-01-02": (0, 3)})
    with pytest.raises(ValueError, match="zero observed"):
        ConditionalIVSurfaceDataset(df)
