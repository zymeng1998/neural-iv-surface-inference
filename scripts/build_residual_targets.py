#!/usr/bin/env python3
"""Materialise residual targets for a benchmark dataset (Phase 4A / ADR 0010).

Adds two columns to a Step-4 benchmark parquet:
  - ``rbf_pred``        : per-date RBF baseline prediction at each query coord
  - ``residual_target`` : ``iv_clean - rbf_pred`` (the hybrid `f_theta` target)

The RBF is reused verbatim from ``models.interpolation`` (the 3X.6 baseline);
no new interpolation math. Story 4A.2 ships this driver + a smoke run on a
small slice; story 4A.3 runs it on the full OTM fold (remote CPU).

Usage
-----
    python3 scripts/build_residual_targets.py \
        --input  data_benchmarks/spy_phase1_random40_noiselow_otm.parquet \
        --output data_benchmarks/spy_phase1_random40_noiselow_otm_residual.parquet
    # smoke on a few dates:
    python3 scripts/build_residual_targets.py --input IN --output OUT --max-dates 5

Stdlib + pandas + the project package only. No training, no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_iv_surface_inference.data.residual_targets import (  # noqa: E402
    RBF_PRED_COLUMN,
    RESIDUAL_COLUMN,
    add_rbf_pred_column,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build residual targets (rbf_pred + residual_target) for a benchmark."
    )
    parser.add_argument("--input", required=True, help="Input benchmark parquet.")
    parser.add_argument("--output", required=True, help="Output parquet with residual columns.")
    parser.add_argument(
        "--max-dates", type=int, default=None,
        help="Smoke mode: only process the first N dates (chronological).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input)
    if args.max_dates is not None:
        keep = sorted(df["date"].unique())[: args.max_dates]
        df = df[df["date"].isin(keep)].copy()
        if args.verbose:
            print(f"[residual-build] smoke: {len(keep)} dates, {len(df)} rows")

    out = add_rbf_pred_column(df, verbose=args.verbose)

    n_nonfinite = int((~np.isfinite(out[RBF_PRED_COLUMN])).sum())
    if n_nonfinite:
        print(
            f"[residual-build] BLOCKED — {n_nonfinite} non-finite rbf_pred rows",
            file=sys.stderr,
        )
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    mae_rbf = float(np.abs(out[RESIDUAL_COLUMN]).mean())
    print(
        f"[residual-build] wrote {args.output}  rows={len(out)}  "
        f"mean|residual|={mae_rbf:.5f}  (≈ RBF MAE; sanity-check vs 3X.6)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
