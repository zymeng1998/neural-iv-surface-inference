"""Date-grouped conditional dataset + collation for the W3 conditional model.

The Phase 1 ``IVSurfaceDataset`` yields one *point* per sample, which is right
for a coordinate-regression MLP but wrong for a model conditioned on the
observed chain ``O_t``. The conditional model needs each sample to be a
**date**: a variable-cardinality context set of observed points plus the query
points to predict.

``ConditionalIVSurfaceDataset`` groups a benchmark split frame by ``date`` and
emits one ``(context, query)`` pair per date. ``collate_conditional`` pads
ragged context and query sets into batch tensors with boolean masks so the
2C.3 set encoder can pool over real (non-padded) elements only.

The point-wise ``IVSurfaceDataset`` is left untouched.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── Feature-set contract (ADR 0008) ───────────────────────────────────────
#
# ``minimal`` is the legacy three-dim per-quote tuple and the **default** —
# every committed Phase 2 / Phase 3 / 3X checkpoint stays reproducible
# byte-for-byte. ``micro_v1`` appends the six AV-native microstructure
# features frozen in ADR 0008, in the exact order below.
_CONTEXT_FEATURES_MINIMAL = ("log_moneyness", "tau", "implied_volatility")
_MICRO_V1_EXTRA = (
    "bid",
    "ask",
    "bid_ask_spread_rel",
    "volume",
    "open_interest",
    "put_call_indicator",
)
_CONTEXT_FEATURES_MICRO_V1 = _CONTEXT_FEATURES_MINIMAL + _MICRO_V1_EXTRA

# Back-compat alias: the legacy module constant is the ``minimal`` tuple.
_CONTEXT_FEATURES = _CONTEXT_FEATURES_MINIMAL
_QUERY_FEATURES = ("log_moneyness", "tau")

# Map of supported feature sets -> ordered per-quote context columns.
FEATURE_SET_FEATURES: dict[str, tuple[str, ...]] = {
    "minimal": _CONTEXT_FEATURES_MINIMAL,
    "micro_v1": _CONTEXT_FEATURES_MICRO_V1,
}

# ε for the relative bid/ask spread (ADR 0008 §Decision, feature #3).
_SPREAD_EPS = 1e-4

# Columns every feature set needs.
_BASE_REQUIRED = {
    "date",
    "log_moneyness",
    "tau",
    "implied_volatility",
    "iv_clean",
    "observed",
}
# Raw columns ``micro_v1`` needs in order to *materialise* its six extra
# features. ``bid_ask_spread_rel`` and ``put_call_indicator`` are derived
# in-loader from these (they are not stored on the surface table); ``mid``
# is optional and falls back to ``(bid + ask) / 2``.
_MICRO_RAW_REQUIRED = {"bid", "ask", "volume", "open_interest", "type"}

DEFAULT_FEATURE_SET = "minimal"


def resolve_context_features(feature_set: str = DEFAULT_FEATURE_SET) -> tuple[str, ...]:
    """Return the ordered context columns for ``feature_set`` (ADR 0008)."""
    try:
        return FEATURE_SET_FEATURES[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature_set {feature_set!r}; "
            f"expected one of {sorted(FEATURE_SET_FEATURES)}"
        ) from exc


def feature_set_context_dim(feature_set: str = DEFAULT_FEATURE_SET) -> int:
    """Per-quote context dimensionality implied by ``feature_set``."""
    return len(resolve_context_features(feature_set))


def _materialize_micro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the two derived ``micro_v1`` columns added.

    Derives, per ADR 0008:
      - ``bid_ask_spread_rel = (ask - bid) / max(mid, ε)``, ε = 1e-4
        (``mid`` column if present, else ``(bid + ask) / 2``).
      - ``put_call_indicator = +1`` for ``type == "call"``, ``-1`` for
        ``"put"``. Any other ``type`` value is a hard error.

    The remaining four ``micro_v1`` features (``bid``, ``ask``, ``volume``,
    ``open_interest``) are AV-native columns used verbatim.
    """
    out = df.copy()
    bid = pd.to_numeric(out["bid"], errors="coerce").to_numpy(dtype=np.float64)
    ask = pd.to_numeric(out["ask"], errors="coerce").to_numpy(dtype=np.float64)
    if "mid" in out.columns:
        mid = pd.to_numeric(out["mid"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        mid = (bid + ask) / 2.0
    out["bid_ask_spread_rel"] = (ask - bid) / np.maximum(mid, _SPREAD_EPS)

    type_norm = out["type"].astype(str).str.lower()
    valid = type_norm.isin(["call", "put"])
    if not bool(valid.all()):
        bad = sorted(set(type_norm[~valid].tolist()))
        raise ValueError(
            f"micro_v1: 'type' column has values outside {{call, put}}: {bad}"
        )
    out["put_call_indicator"] = np.where(type_norm.to_numpy() == "call", 1.0, -1.0)
    return out


def build_context_matrix(
    df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET
) -> np.ndarray:
    """Build the float32 ``(n_rows, context_dim)`` matrix for ``feature_set``.

    Single source of truth for the per-quote context tuple, shared by the
    dataset (training) and the predictor adapter (inference) so both order
    columns identically per ADR 0008. For ``minimal`` this is a verbatim
    column select (bit-for-bit identical to the pre-3C.2 loader); for
    ``micro_v1`` the two derived columns are materialised first.
    """
    features = resolve_context_features(feature_set)
    if feature_set == "micro_v1":
        df = _materialize_micro_columns(df)
    return df.loc[:, list(features)].to_numpy(dtype=np.float32)


class ConditionalIVSurfaceDataset(Dataset):
    """Date-grouped dataset for the conditional surface model.

    Each sample corresponds to one date and carries:
      - ``context``  : float32 tensor of shape ``(n_obs, 3)`` —
                       ``(log_moneyness, tau, implied_volatility)`` for that
                       date's observed rows only.
      - ``query``    : float32 tensor of shape ``(n_query, 2)`` —
                       ``(log_moneyness, tau)`` for the query points.
      - ``target``   : float32 tensor of shape ``(n_query,)`` —
                       ``iv_clean`` for the query points.
      - ``query_observed`` : bool tensor of shape ``(n_query,)`` —
                       per-query-point observed flag (so the eval/loss can
                       split observed vs unobserved query performance).
      - ``date``     : ``pandas.Timestamp`` — the date this sample represents.

    By default, the query set is all rows for the date (observed + unobserved).
    This mirrors the Phase 1 evaluation contract where the model predicts
    everywhere and is scored on the full surface.

    Parameters
    ----------
    df : pd.DataFrame
        A benchmark dataset (output of Step 4) filtered to a single split.
        Must contain at least the columns ``date``, ``log_moneyness``, ``tau``,
        ``implied_volatility``, ``iv_clean``, ``observed``. When
        ``feature_set="micro_v1"`` it must also contain the raw AV columns
        ``bid``, ``ask``, ``volume``, ``open_interest``, ``type`` (and
        optionally ``mid``) so the loader can materialise the ADR 0008
        microstructure tuple.
    feature_set : {"minimal", "micro_v1"}
        ADR 0008 feature contract. ``minimal`` (default) is the legacy
        3-dim ``(log_moneyness, tau, implied_volatility)`` context;
        ``micro_v1`` appends the six frozen microstructure features for a
        9-dim context. The default preserves byte-for-byte behaviour for
        every committed checkpoint.
    """

    def __init__(self, df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET):
        self.feature_set = feature_set
        self.context_features = resolve_context_features(feature_set)
        self.context_in_features = len(self.context_features)

        required = set(_BASE_REQUIRED)
        if feature_set == "micro_v1":
            required |= _MICRO_RAW_REQUIRED
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ConditionalIVSurfaceDataset: missing columns {sorted(missing)}")

        # Sort chronologically; preserves time-based split integrity.
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
        # Materialise the two derived micro_v1 columns once (vectorised) so
        # the per-date select below is a plain column lookup.
        if feature_set == "micro_v1":
            df = _materialize_micro_columns(df)

        self.dates: list[pd.Timestamp] = []
        self._contexts: list[torch.Tensor] = []
        self._queries: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._query_observed: list[torch.Tensor] = []

        for date, group in df.groupby("date", sort=True):
            obs_mask = group["observed"].values.astype(bool)
            if obs_mask.sum() == 0:
                raise ValueError(
                    f"ConditionalIVSurfaceDataset: date {date} has zero observed rows; "
                    "the conditional model cannot form a context for it."
                )

            ctx_rows = group.loc[obs_mask, list(self.context_features)].to_numpy(dtype=np.float32)
            query_rows = group[list(_QUERY_FEATURES)].to_numpy(dtype=np.float32)
            target_rows = group["iv_clean"].to_numpy(dtype=np.float32)
            query_obs = obs_mask.astype(np.bool_)

            self.dates.append(pd.Timestamp(date))
            self._contexts.append(torch.from_numpy(ctx_rows))
            self._queries.append(torch.from_numpy(query_rows))
            self._targets.append(torch.from_numpy(target_rows))
            self._query_observed.append(torch.from_numpy(query_obs))

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        return {
            "context": self._contexts[idx],
            "query": self._queries[idx],
            "target": self._targets[idx],
            "query_observed": self._query_observed[idx],
            "date": self.dates[idx],
        }


def collate_conditional(samples: Sequence[dict]) -> dict:
    """Pad ragged context/query sets and build boolean masks.

    Parameters
    ----------
    samples : sequence of dicts produced by ``ConditionalIVSurfaceDataset``.

    Returns
    -------
    dict with keys:
      - ``context``      : float32 ``(B, max_ctx, 3)``, zero-padded
      - ``context_mask`` : bool    ``(B, max_ctx)``, True for real rows
      - ``query``        : float32 ``(B, max_q, 2)``, zero-padded
      - ``query_mask``   : bool    ``(B, max_q)``, True for real query points
      - ``target``       : float32 ``(B, max_q)``, zero-padded
      - ``query_observed`` : bool  ``(B, max_q)``, True for observed query rows
                             (masked-out rows are False)
      - ``dates``        : list[pd.Timestamp]
    """
    if not samples:
        raise ValueError("collate_conditional: empty batch")

    batch_size = len(samples)
    max_ctx = max(s["context"].shape[0] for s in samples)
    max_q = max(s["query"].shape[0] for s in samples)

    ctx_dim = samples[0]["context"].shape[1]
    q_dim = samples[0]["query"].shape[1]

    context = torch.zeros(batch_size, max_ctx, ctx_dim, dtype=torch.float32)
    context_mask = torch.zeros(batch_size, max_ctx, dtype=torch.bool)
    query = torch.zeros(batch_size, max_q, q_dim, dtype=torch.float32)
    query_mask = torch.zeros(batch_size, max_q, dtype=torch.bool)
    target = torch.zeros(batch_size, max_q, dtype=torch.float32)
    query_observed = torch.zeros(batch_size, max_q, dtype=torch.bool)
    dates: list[pd.Timestamp] = []

    for i, s in enumerate(samples):
        n_ctx = s["context"].shape[0]
        n_q = s["query"].shape[0]
        context[i, :n_ctx] = s["context"]
        context_mask[i, :n_ctx] = True
        query[i, :n_q] = s["query"]
        query_mask[i, :n_q] = True
        target[i, :n_q] = s["target"]
        query_observed[i, :n_q] = s["query_observed"]
        dates.append(s["date"])

    return {
        "context": context,
        "context_mask": context_mask,
        "query": query,
        "query_mask": query_mask,
        "target": target,
        "query_observed": query_observed,
        "dates": dates,
    }
