#!/usr/bin/env python3
"""Paired (date-clustered) bootstrap CI on the per-query MAE difference
between two prediction columns (story 4A.7 — the Phase 4 accuracy bar).

Given one prediction CSV that carries a model column and a baseline column
plus a target column, compute per-row absolute errors for each arm and a
**date-clustered** bootstrap distribution of the mean MAE difference
``mean(|model − target|) − mean(|baseline − target|)``. Resampling whole
dates (not rows) respects the within-date correlation of a surface.

If the 95% CI is entirely below 0, the model is significantly more accurate
than the baseline. This is the formal Phase 4 bar: model = the calibrated
RBF-prior hybrid (`mu`), baseline = RBF (`rbf_pred`), target = `iv_clean`
(the same target the RBF floor 0.00613 is measured against).

Usage:
    python scripts/eval_mae_bootstrap_ci.py \
        --pred-csv artifacts/runs/4A4/gaussian/test_predictions.csv \
        --pred-col mu --baseline-col rbf_pred --target-col iv_clean \
        --group-col date --n-boot 2000 --seed 0 \
        --output results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/mae_delta_ci.json

Stdlib + numpy + pandas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mae_delta_ci(
    abs_err_model: np.ndarray,
    abs_err_base: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Date-clustered bootstrap CI on mean(abs_err_model − abs_err_base).

    Resamples unique groups (dates) with replacement; the statistic is the
    overall mean per-row delta on the resampled rows.
    """
    delta = abs_err_model - abs_err_base
    uniq, inv = np.unique(groups, return_inverse=True)
    # Precompute per-group row indices once.
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(uniq) + 1))
    group_rows = [order[boundaries[g]:boundaries[g + 1]] for g in range(len(uniq))]

    rng = np.random.default_rng(seed)
    n_groups = len(uniq)
    point = float(delta.mean())
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_rows[g] for g in pick])
        boot[b] = float(delta[idx].mean())
    lo = float(np.quantile(boot, (1 - ci) / 2))
    hi = float(np.quantile(boot, 1 - (1 - ci) / 2))
    return {
        "mean_delta": point,                       # < 0 ⇒ model better
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "n_boot": n_boot,
        "n_groups": int(n_groups),
        "n_rows": int(len(delta)),
        "significant": bool(hi < 0.0),             # model significantly better
        "mae_model": float(abs_err_model.mean()),
        "mae_baseline": float(abs_err_base.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--pred-col", default="mu")
    ap.add_argument("--baseline-col", default="rbf_pred")
    ap.add_argument("--target-col", default="iv_clean")
    ap.add_argument("--group-col", default="date")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    cols = [args.pred_col, args.baseline_col, args.target_col, args.group_col]
    df = pd.read_csv(args.pred_csv, usecols=cols)
    y = df[args.target_col].to_numpy(dtype=np.float64)
    err_model = np.abs(df[args.pred_col].to_numpy(dtype=np.float64) - y)
    err_base = np.abs(df[args.baseline_col].to_numpy(dtype=np.float64) - y)
    res = bootstrap_mae_delta_ci(
        err_model, err_base, df[args.group_col].to_numpy(),
        n_boot=args.n_boot, seed=args.seed,
    )
    res["pred_csv"] = args.pred_csv
    res["pred_col"] = args.pred_col
    res["baseline_col"] = args.baseline_col
    res["target_col"] = args.target_col
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    verdict = ("SIGNIFICANTLY BETTER" if res["significant"]
               else "not significant (CI includes 0)")
    print(f"[4A.7-CI] {args.pred_col} vs {args.baseline_col} on {args.target_col}: "
          f"mae_model={res['mae_model']:.6f} mae_base={res['mae_baseline']:.6f} "
          f"mean_delta={res['mean_delta']:+.6f} 95%CI=[{res['ci_low']:+.6f}, "
          f"{res['ci_high']:+.6f}] → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
