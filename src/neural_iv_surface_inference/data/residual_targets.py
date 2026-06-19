"""Phase 4A residual-target construction (ADR 0010).

The Phase 4 hybrid predicts ``sigma_hat(k, tau) = RBF_t(k, tau) + f_theta(...)``.
The neural model ``f_theta`` is trained on the **additive residual**

    r_t(k, tau) = iv_clean - rbf_pred_t(k, tau)

at query coordinates, where ``rbf_pred_t`` is the per-date RBF baseline
prediction. The RBF here is reused **verbatim** from
``models.interpolation`` — the same baseline scored in story 3X.6 — so no
new interpolation math is introduced (ADR 0010 §Decision 2; 4A.2 non-goals).

This module is the only genuinely new code surface in Phase 4: it computes
the residual target and exposes the column names the loader / builder use.
The full-dataset materialisation is story 4A.3; training is 4A.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Column written by :func:`add_rbf_pred_column` holding the per-date RBF
#: prediction at every row's query coordinate.
RBF_PRED_COLUMN = "rbf_pred"
#: Column holding the additive residual target ``iv_clean - rbf_pred``.
RESIDUAL_COLUMN = "residual_target"


def rbf_predict_at_queries(df_date: pd.DataFrame) -> np.ndarray:
    """Per-date RBF prediction at every row's ``(log_moneyness, tau)``.

    Thin reuse of
    :func:`models.interpolation.interpolate_surface_date` with
    ``method="rbf"`` — fits on the date's observed quotes and predicts at all
    rows (the query coordinates). Returns a ``float64`` array of shape
    ``(len(df_date),)``.
    """
    # Lazy import: importing ``models`` at module load creates a circular
    # import via ``models.conditional_surface`` → ``data.conditional_loaders``
    # (which imports this module). Deferring it to call time breaks the cycle.
    from ..models.interpolation import interpolate_surface_date

    return interpolate_surface_date(df_date, method="rbf")


def residual_targets_for_date(df_date: pd.DataFrame) -> np.ndarray:
    """Additive residual ``r = iv_clean - rbf_pred`` for a single date."""
    rbf_pred = rbf_predict_at_queries(df_date)
    return df_date["iv_clean"].to_numpy(dtype=float) - rbf_pred


def add_rbf_pred_column(df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    """Return a copy of ``df`` with ``rbf_pred`` + ``residual_target`` columns.

    Computes the per-date RBF prediction at every row (mirroring the loop in
    ``models.interpolation.run_interpolation_baseline``) and the additive
    residual. Index is reset so the per-date masks line up positionally.
    """
    out = df.reset_index(drop=True).copy()
    rbf_pred = np.full(len(out), np.nan, dtype=float)

    dates = out["date"].unique()
    for i, date in enumerate(dates):
        mask = (out["date"] == date).to_numpy()
        rbf_pred[mask] = rbf_predict_at_queries(out.loc[mask])
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(dates)}] dates RBF-predicted")

    out[RBF_PRED_COLUMN] = rbf_pred
    out[RESIDUAL_COLUMN] = out["iv_clean"].to_numpy(dtype=float) - rbf_pred
    if verbose:
        print(f"  residual targets built: {len(dates)} dates, "
              f"{int(np.isfinite(rbf_pred).sum())}/{len(out)} finite rbf_pred")
    return out
