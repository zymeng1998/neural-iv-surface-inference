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


_CONTEXT_FEATURES = ("log_moneyness", "tau", "implied_volatility")
_QUERY_FEATURES = ("log_moneyness", "tau")


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
        ``implied_volatility``, ``iv_clean``, ``observed``.
    """

    def __init__(self, df: pd.DataFrame):
        required = {"date", "log_moneyness", "tau", "implied_volatility",
                    "iv_clean", "observed"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ConditionalIVSurfaceDataset: missing columns {sorted(missing)}")

        # Sort chronologically; preserves time-based split integrity.
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

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

            ctx_rows = group.loc[obs_mask, list(_CONTEXT_FEATURES)].to_numpy(dtype=np.float32)
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
