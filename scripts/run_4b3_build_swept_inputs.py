#!/usr/bin/env python3
"""Materialise the Phase 4B swept evaluation inputs (story 4B.3, remote CPU).

For every regime x rung of the 4B.2 sparsity sweep, re-fit the per-date RBF
baseline on the *sparser observed context* and predict at the full fixed query
grid, on the OTM benchmark. Emits a finiteness + MAE-sanity audit:

  - RBF MAE rises (monotone non-decreasing) as the observed context thins;
  - the dense rung (intensity 0 == full observed context) reproduces the 4A RBF
    floor (overall test MAE 0.006132) — the provenance anchor.

The RBF is reused **verbatim** from ``models.interpolation`` (the 3X.6 / 4A.3
baseline) — only the ``observed`` context mask changes per rung, exactly the
construction the 4B.2 harness pins. No model, no GPU.

The committed deliverable is the small audit bundle
``artifacts/runs/4B3/{manifest.json, rbf_sweep_stats.csv}``. The (large) per-rung
RBF / context parquet stays on the Pod volume (``--parquet-out``) as the handoff
to 4B.4 (the hybrid forward sweep); it is regenerable and gitignored.

Usage
-----
    python3 scripts/run_4b3_build_swept_inputs.py \
        --input data_processed/spy/benchmarks/spy_phase1_random40_noiselow_otm_residual.parquet \
        --split test \
        --parquet-out artifacts/runs/4B3/swept_rbf_test.parquet
    # smoke on a few dates (no floor assertion):
    python3 scripts/run_4b3_build_swept_inputs.py --input IN --max-dates 5 --no-floor-check

Stdlib + pandas/numpy + scipy + the project package only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_iv_surface_inference.diagnostics.sparsity_sweep import (  # noqa: E402
    REGIMES,
    build_all_regimes,
)
from neural_iv_surface_inference.models.interpolation import (  # noqa: E402
    interpolate_surface_date,
)

# 3X.6 / 4A.3 committed RBF floor: overall test MAE vs iv_clean (the value the
# dense rung must reproduce). See results/3/.../rbf/metrics_summary.csv.
DEFAULT_RBF_FLOOR = 0.006131847035635011
DEFAULT_INTENSITIES = (0.0, 0.2, 0.4, 0.6, 0.8)
# Per-date seed fold so different dates draw different (but reproducible) masks.
_DATE_SEED_MULT = 1_000_003


def _parse_intensities(text: str) -> tuple[float, ...]:
    vals = tuple(float(x) for x in text.split(",") if x.strip())
    if not vals or vals[0] != 0.0:
        raise ValueError("intensities must start at 0.0 (the dense rung)")
    if list(vals) != sorted(vals):
        raise ValueError("intensities must be ascending (nested rungs)")
    return vals


class _Acc:
    """Running MAE / finiteness accumulator for one (regime, rung)."""

    __slots__ = (
        "sum_abs_overall", "n_overall", "sse_overall",
        "sum_abs_query", "n_query",
        "n_retained", "n_nonfinite",
    )

    def __init__(self) -> None:
        self.sum_abs_overall = 0.0
        self.sse_overall = 0.0
        self.n_overall = 0
        self.sum_abs_query = 0.0
        self.n_query = 0
        self.n_retained = 0
        self.n_nonfinite = 0

    def update(self, pred: np.ndarray, truth: np.ndarray,
               query_mask: np.ndarray, n_retained: int) -> None:
        finite = np.isfinite(pred)
        self.n_nonfinite += int((~finite).sum())
        err = np.abs(pred - truth)
        self.sum_abs_overall += float(np.nansum(err))
        self.sse_overall += float(np.nansum((pred - truth) ** 2))
        self.n_overall += int(pred.shape[0])
        self.sum_abs_query += float(np.nansum(err[query_mask]))
        self.n_query += int(query_mask.sum())
        self.n_retained += int(n_retained)

    @property
    def mae_overall(self) -> float:
        return self.sum_abs_overall / self.n_overall if self.n_overall else float("nan")

    @property
    def rmse_overall(self) -> float:
        return (self.sse_overall / self.n_overall) ** 0.5 if self.n_overall else float("nan")

    @property
    def mae_query(self) -> float:
        return self.sum_abs_query / self.n_query if self.n_query else float("nan")


def build_sweep_inputs(
    df: pd.DataFrame,
    *,
    regimes: tuple[str, ...],
    intensities: tuple[float, ...],
    seed: int,
    parquet_out: str | None,
    maturity_col: str = "tau",
    verbose: bool = False,
) -> dict[tuple[str, int], _Acc]:
    """Re-fit RBF per date x regime x rung; accumulate MAE + finiteness.

    Returns accumulators keyed by ``(regime, rung_index)``. When ``parquet_out``
    is set, the long-form per-rung RBF / context table (the 4B.4 handoff) is
    **streamed to disk one date at a time** via a pyarrow ``ParquetWriter`` so
    memory stays flat regardless of split size (the container is cgroup-capped,
    so holding ~55M rows in a list OOM-kills).
    """
    accs: dict[tuple[str, int], _Acc] = {
        (r, i): _Acc() for r in regimes for i in range(len(intensities))
    }
    ladders = {r: intensities for r in regimes}
    dates = sorted(df["date"].unique())

    writer = None
    pq = pa = None
    if parquet_out is not None:
        import pyarrow as pa  # noqa: F811
        import pyarrow.parquet as pq  # noqa: F811
        Path(parquet_out).parent.mkdir(parents=True, exist_ok=True)

    try:
        for di, date in enumerate(dates):
            df_date = df[df["date"] == date]
            truth = df_date["iv_clean"].to_numpy(dtype=float)
            base_observed = df_date["observed"].to_numpy(dtype=bool)
            query_mask = ~base_observed  # fixed query set (= original unobserved)
            date_seed = (int(seed) * _DATE_SEED_MULT + di) % (2 ** 63)

            sweeps = build_all_regimes(
                df_date, ladders, seed=date_seed, maturity_col=maturity_col
            )

            # The dense rung (intensity 0) is the full observed context for every
            # regime -> identical RBF prediction. Fit it once, reuse per regime.
            work = df_date.copy()
            dense_pred = interpolate_surface_date(work, method="rbf")

            date_chunks: list[pd.DataFrame] = []
            for regime in regimes:
                for rung in sweeps[regime].rungs:
                    if rung.rung_index == 0:
                        pred = dense_pred
                    else:
                        work = work.assign(observed=rung.retained_observed)
                        pred = interpolate_surface_date(work, method="rbf")
                    pred = np.asarray(pred, dtype=float)
                    accs[(regime, rung.rung_index)].update(
                        pred, truth, query_mask, int(rung.retained_observed.sum())
                    )
                    if parquet_out is not None:
                        date_chunks.append(pd.DataFrame({
                            "date": df_date["date"].to_numpy(),
                            "regime": regime,
                            "rung_index": np.int16(rung.rung_index),
                            "intensity": np.float32(rung.intensity),
                            "log_moneyness": df_date["log_moneyness"].to_numpy(),
                            "tau": df_date["tau"].to_numpy(),
                            "iv_clean": truth,
                            "observed_base": base_observed,
                            "observed_rung": rung.retained_observed,
                            "is_query_fixed": query_mask,
                            "rbf_pred": pred,
                        }))

            if parquet_out is not None:
                table = pa.Table.from_pandas(
                    pd.concat(date_chunks, ignore_index=True), preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(parquet_out, table.schema, compression="zstd")
                writer.write_table(table)

            if verbose and (di + 1) % 200 == 0:
                print(f"  [{di + 1}/{len(dates)}] dates swept", flush=True)
    finally:
        if writer is not None:
            writer.close()

    return accs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build Phase 4B swept eval inputs (RBF re-fit per rung).")
    ap.add_argument("--input", required=True, help="OTM benchmark parquet (with iv_clean, observed, split).")
    ap.add_argument("--split", default="test", help="Split to build (default test).")
    ap.add_argument("--regimes", default=",".join(REGIMES), help="Comma-separated regimes.")
    ap.add_argument("--intensities", default=",".join(str(x) for x in DEFAULT_INTENSITIES),
                    help="Ascending intensity ladder starting at 0.0.")
    ap.add_argument("--seed", type=int, default=4023)
    ap.add_argument("--maturity-col", default="tau",
                    help="Column identifying expirations for missing_maturities (benchmark uses tau).")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "artifacts/runs/4B3"))
    ap.add_argument("--parquet-out", default=None, help="Optional Pod path for the long-form swept parquet (handoff to 4B.4).")
    ap.add_argument("--rbf-floor", type=float, default=DEFAULT_RBF_FLOOR)
    ap.add_argument("--floor-tol", type=float, default=5e-5)
    ap.add_argument("--no-floor-check", action="store_true", help="Skip the dense-rung floor assertion (smoke mode).")
    ap.add_argument("--max-dates", type=int, default=None, help="Smoke: only first N dates.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    regimes = tuple(r.strip() for r in args.regimes.split(",") if r.strip())
    bad = set(regimes) - set(REGIMES)
    if bad:
        print(f"[4b3] unknown regimes {sorted(bad)}; valid {REGIMES}", file=sys.stderr)
        return 2
    intensities = _parse_intensities(args.intensities)

    cols = ["date", "log_moneyness", "tau", "implied_volatility", "iv_clean", "observed", "split"]
    df = pd.read_parquet(args.input, columns=cols)
    df = df[df["split"] == args.split].copy()
    if args.max_dates is not None:
        keep = sorted(df["date"].unique())[: args.max_dates]
        df = df[df["date"].isin(keep)].copy()
    n_dates = int(df["date"].nunique())
    n_rows = int(len(df))
    print(f"[4b3] split={args.split} dates={n_dates} rows={n_rows} "
          f"regimes={list(regimes)} rungs={intensities}", flush=True)

    t0 = time.time()
    accs = build_sweep_inputs(
        df, regimes=regimes, intensities=intensities, seed=args.seed,
        parquet_out=args.parquet_out, maturity_col=args.maturity_col,
        verbose=args.verbose,
    )
    wall = time.time() - t0

    # --- assemble per-(regime, rung) stats ---------------------------------
    rows = []
    for regime in regimes:
        for i, intensity in enumerate(intensities):
            a = accs[(regime, i)]
            rows.append({
                "regime": regime,
                "rung_index": i,
                "intensity": intensity,
                "mae_overall": a.mae_overall,
                "rmse_overall": a.rmse_overall,
                "mae_query_fixed": a.mae_query,
                "n_overall": a.n_overall,
                "n_query": a.n_query,
                "n_retained_context": a.n_retained,
                "retained_frac_of_base": (
                    a.n_retained / accs[(regime, 0)].n_retained
                    if accs[(regime, 0)].n_retained else float("nan")
                ),
                "n_nonfinite": a.n_nonfinite,
            })
    stats = pd.DataFrame(rows)

    # --- audit -------------------------------------------------------------
    total_nonfinite = int(stats["n_nonfinite"].sum())
    monotone = {}
    for regime in regimes:
        seq = stats[stats["regime"] == regime].sort_values("rung_index")["mae_overall"].to_numpy()
        diffs = np.diff(seq)
        monotone[regime] = {
            "non_decreasing": bool(np.all(diffs >= -1e-6)),
            "max_decrease": float(max(0.0, -diffs.min())) if diffs.size else 0.0,
        }
    dense_overall = float(stats[(stats["rung_index"] == 0)]["mae_overall"].iloc[0])
    floor_abs_err = abs(dense_overall - args.rbf_floor)
    floor_ok = floor_abs_err <= args.floor_tol

    # --- write committed bundle -------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(outdir / "rbf_sweep_stats.csv", index=False)

    manifest = {
        "story": "4B.3",
        "input": str(args.input),
        "split": args.split,
        "regimes": list(regimes),
        "intensities": list(intensities),
        "seed": args.seed,
        "n_dates": n_dates,
        "n_rows": n_rows,
        "rbf_floor_ref": args.rbf_floor,
        "dense_rung_mae_overall": dense_overall,
        "dense_rung_floor_abs_err": floor_abs_err,
        "dense_rung_reproduces_floor": floor_ok,
        "floor_tol": args.floor_tol,
        "total_nonfinite": total_nonfinite,
        "monotone_overall": monotone,
        "wall_seconds": round(wall, 1),
        "parquet_out": args.parquet_out,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.parquet_out is not None:
        print(f"[4b3] streamed swept parquet -> {args.parquet_out}", flush=True)

    # --- report ------------------------------------------------------------
    print("\n[4b3] per-(regime,rung) RBF MAE (overall / query-fixed):", flush=True)
    for regime in regimes:
        sub = stats[stats["regime"] == regime].sort_values("rung_index")
        ov = ", ".join(f"{m:.6f}" for m in sub["mae_overall"])
        print(f"  {regime:<22} overall=[{ov}]  monotone={monotone[regime]['non_decreasing']}")
    print(f"\n[4b3] dense rung overall MAE = {dense_overall:.6f}  "
          f"(floor {args.rbf_floor:.6f}, |err|={floor_abs_err:.2e}, ok={floor_ok})")
    print(f"[4b3] total non-finite = {total_nonfinite}")
    print(f"[4b3] wall = {wall:.1f}s")
    print(f"[4b3] wrote {outdir/'manifest.json'} + {outdir/'rbf_sweep_stats.csv'}")

    if total_nonfinite:
        print("[4b3] BLOCKED — non-finite RBF predictions present", file=sys.stderr)
        return 1
    if not args.no_floor_check and not floor_ok:
        print(f"[4b3] BLOCKED — dense rung MAE {dense_overall:.6f} != floor "
              f"{args.rbf_floor:.6f} (tol {args.floor_tol})", file=sys.stderr)
        return 1
    nonmono = [r for r, m in monotone.items() if not m["non_decreasing"]]
    if nonmono:
        print(f"[4b3] WARNING — non-monotone overall MAE in: {nonmono} "
              f"(sanity flag; recorded in manifest)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
