"""Unit tests for the `feature_set` flag on ConditionalIVSurfaceDataset (3C.2).

Covers ADR 0008:
  - `micro_v1` produces the 9-dim context in the frozen column order.
  - derived `bid_ask_spread_rel` and `put_call_indicator` are correct.
  - `put_call_indicator ∈ {-1, +1}`.
  - `minimal` (default) is bit-for-bit identical to the pre-3C.2 loader on a
    fixed input (a frozen reference array asserted byte-for-byte).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    build_context_matrix,
    feature_set_context_dim,
    resolve_context_features,
)

# ADR 0008 frozen column order for micro_v1.
_MICRO_V1_ORDER = (
    "log_moneyness",
    "tau",
    "implied_volatility",
    "bid",
    "ask",
    "bid_ask_spread_rel",
    "volume",
    "open_interest",
    "put_call_indicator",
)


def _make_micro_frame() -> pd.DataFrame:
    """One date, 4 rows; observed at indices 0, 2, 3 (index 1 unobserved).

    All values are fixed literals so the derived features and column order
    can be asserted exactly.
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime("2024-01-02"),
            "log_moneyness": np.array([-0.10, 0.00, 0.05, 0.20], dtype=np.float32),
            "tau": np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
            "implied_volatility": np.array([0.20, 0.25, 0.30, 0.35], dtype=np.float32),
            "iv_clean": np.array([0.21, 0.26, 0.31, 0.36], dtype=np.float32),
            "observed": np.array([True, False, True, True]),
            "bid": np.array([1.00, 2.00, 3.00, 4.00], dtype=np.float64),
            "ask": np.array([1.20, 2.20, 3.30, 4.40], dtype=np.float64),
            "mid": np.array([1.10, 2.10, 3.15, 4.20], dtype=np.float64),
            "volume": np.array([10, 20, 30, 40], dtype=np.int64),
            "open_interest": np.array([100, 200, 300, 400], dtype=np.int64),
            "type": ["call", "put", "call", "put"],
        }
    )


def test_resolver_dims():
    assert resolve_context_features("minimal") == (
        "log_moneyness",
        "tau",
        "implied_volatility",
    )
    assert resolve_context_features("micro_v1") == _MICRO_V1_ORDER
    assert feature_set_context_dim("minimal") == 3
    assert feature_set_context_dim("micro_v1") == 9
    with pytest.raises(ValueError):
        resolve_context_features("nope")


def test_micro_v1_shape_and_column_order():
    df = _make_micro_frame()
    ds = ConditionalIVSurfaceDataset(df, feature_set="micro_v1")

    assert ds.feature_set == "micro_v1"
    assert ds.context_features == _MICRO_V1_ORDER
    assert ds.context_in_features == 9

    ctx = ds[0]["context"].numpy()
    assert ctx.shape == (3, 9)  # 3 observed rows, 9 features

    # Expected from the observed rows (indices 0, 2, 3) per ADR 0008 order.
    # bid_ask_spread_rel = (ask - bid) / max(mid, 1e-4)
    expected = np.array(
        [
            [-0.10, 0.10, 0.20, 1.00, 1.20, 0.20 / 1.10, 10, 100, +1.0],
            [0.05, 0.30, 0.30, 3.00, 3.30, 0.30 / 3.15, 30, 300, +1.0],
            [0.20, 0.40, 0.35, 4.00, 4.40, 0.40 / 4.20, 40, 400, -1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ctx, expected, rtol=0, atol=1e-6)


def test_put_call_indicator_domain():
    df = _make_micro_frame()
    ds = ConditionalIVSurfaceDataset(df, feature_set="micro_v1")
    pc = ds[0]["context"].numpy()[:, _MICRO_V1_ORDER.index("put_call_indicator")]
    assert set(np.unique(pc).tolist()) <= {-1.0, 1.0}


def test_micro_v1_rejects_bad_type():
    df = _make_micro_frame()
    df.loc[0, "type"] = "future"
    with pytest.raises(ValueError, match="type"):
        ConditionalIVSurfaceDataset(df, feature_set="micro_v1")


def test_micro_v1_mid_fallback_matches_bid_ask_mean():
    df = _make_micro_frame().drop(columns=["mid"])
    ds = ConditionalIVSurfaceDataset(df, feature_set="micro_v1")
    ctx = ds[0]["context"].numpy()
    col = _MICRO_V1_ORDER.index("bid_ask_spread_rel")
    # Fallback mid = (bid + ask) / 2 for observed rows 0, 2, 3.
    expected_spread = np.array(
        [
            0.20 / ((1.00 + 1.20) / 2),
            0.30 / ((3.00 + 3.30) / 2),
            0.40 / ((4.00 + 4.40) / 2),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ctx[:, col], expected_spread, rtol=0, atol=1e-6)


def test_micro_v1_missing_raw_columns_raises():
    df = _make_micro_frame().drop(columns=["open_interest"])
    with pytest.raises(ValueError, match="open_interest"):
        ConditionalIVSurfaceDataset(df, feature_set="micro_v1")


def test_minimal_is_default():
    df = _make_micro_frame()
    ds_default = ConditionalIVSurfaceDataset(df)
    assert ds_default.feature_set == "minimal"
    assert ds_default.context_in_features == 3
    assert ds_default[0]["context"].shape == (3, 3)


def test_minimal_bit_for_bit_frozen_reference():
    """`minimal` must reproduce the pre-3C.2 loader byte-for-byte.

    The frozen reference is the exact pre-change computation:
    `group.loc[observed, [log_moneyness, tau, implied_volatility]]` as float32
    over the fixed input — observed rows are indices 0, 2, 3.
    """
    df = _make_micro_frame()
    ds = ConditionalIVSurfaceDataset(df, feature_set="minimal")
    ctx = ds[0]["context"].numpy()

    frozen_reference = np.array(
        [
            [-0.10, 0.10, 0.20],
            [0.05, 0.30, 0.30],
            [0.20, 0.40, 0.35],
        ],
        dtype=np.float32,
    )
    assert ctx.dtype == np.float32
    assert ctx.shape == frozen_reference.shape
    # Byte-for-byte identical (not just numerically close).
    assert ctx.tobytes() == frozen_reference.tobytes()

    # And equivalently via the shared helper used by the predictor adapter.
    via_helper = build_context_matrix(
        df.loc[df["observed"]], "minimal"
    )
    assert via_helper.tobytes() == frozen_reference.tobytes()
