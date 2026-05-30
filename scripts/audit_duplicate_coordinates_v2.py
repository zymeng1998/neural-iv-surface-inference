#!/usr/bin/env python3
"""Vectorised audit v2 of duplicate model-coordinates (Phase 3X.3).

Functional contract is identical to ``audit_duplicate_coordinates.py``
(v1):

- Same CLI flags.
- Same output files in ``--output-dir``:
  ``duplicate_summary.csv``, ``duplicate_iv_dispersion.csv``,
  ``observed_hidden_leakage.csv``, ``sparse_density_sensitivity.csv``,
  ``headline.json``.
- Same markdown report path/format.

Only the **inner loops** that walked one duplicate group at a time are
replaced with vectorised ``groupby.agg`` + numpy boolean ops. v1 spent
~2h39m on the strict file + a single benchmark; v2 needs to make the
12-artifact OTM audit (strict + 11 benchmarks) practical.

v1 is preserved unchanged as historical evidence of the discovery
method (ADR 0006 / D6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import audit_duplicate_coordinates as v1  # noqa: E402

# Re-export constants so reports/tests can use either module identically.
ROUND_DECIMALS = v1.ROUND_DECIMALS
MONEYNESS_BUCKETS = v1.MONEYNESS_BUCKETS
MATURITY_BUCKETS = v1.MATURITY_BUCKETS
DEFAULT_CHUNK_ROWS = v1.DEFAULT_CHUNK_ROWS
StrictAcc = v1.StrictAcc
LeakageRow = v1.LeakageRow
DensityAcc = v1.DensityAcc
quantiles = v1.quantiles
bucketise = v1.bucketise
iter_dates = v1.iter_dates


# ----------------------------------------------------------------------
# Vectorised strict accumulator update
# ----------------------------------------------------------------------

def update_strict_acc_vectorised(
    acc: StrictAcc,
    group_df: pd.DataFrame,
    key_cols: list[str],
    has_type: bool,
) -> None:
    """Vectorised replacement for ``v1.update_strict_acc``.

    Replaces the ``for key, idx in grouped.indices.items()`` per-group
    Python loop with ``groupby.agg([...])`` over the whole date.
    """
    acc.total_rows += len(group_df)
    if group_df.empty:
        return

    iv_col = "implied_volatility"
    work = group_df[key_cols + [iv_col] + (["type"] if has_type else [])].copy()

    if has_type:
        types_lower = work["type"].astype(str).str.lower()
        work["is_call"] = types_lower.isin(["call", "c"]).astype(np.int64)
        work["is_put"] = types_lower.isin(["put", "p"]).astype(np.int64)

    agg_dict: dict[str, tuple[str, str]] = {
        "n": (iv_col, "size"),
        "iv_min": (iv_col, "min"),
        "iv_max": (iv_col, "max"),
    }
    if has_type:
        agg_dict["n_call"] = ("is_call", "sum")
        agg_dict["n_put"] = ("is_put", "sum")

    grouped = work.groupby(key_cols, sort=False).agg(**agg_dict)

    sizes = grouped["n"].to_numpy()
    dup_mask = sizes >= 2
    if not dup_mask.any():
        return

    dup_sizes = sizes[dup_mask]
    acc.rows_in_dup_groups += int(dup_sizes.sum())
    acc.n_dup_groups += int(dup_mask.sum())

    # Group-size histogram, vectorised via np.unique counts.
    uniq_sizes, uniq_counts = np.unique(dup_sizes, return_counts=True)
    for s, c in zip(uniq_sizes.tolist(), uniq_counts.tolist()):
        acc.group_size_hist[int(s)] += int(c)

    iv_ranges = (grouped["iv_max"].to_numpy() - grouped["iv_min"].to_numpy())[dup_mask]

    if has_type:
        n_call = grouped["n_call"].to_numpy()[dup_mask]
        n_put = grouped["n_put"].to_numpy()[dup_mask]
        mix = (n_call >= 1) & (n_put >= 1)
    else:
        mix = np.zeros(dup_mask.sum(), dtype=bool)

    acc.n_groups_call_put_mix += int(mix.sum())
    acc.n_groups_same_type += int((~mix).sum())
    acc.iv_ranges_mixed.extend(iv_ranges[mix].tolist())
    acc.iv_ranges_same.extend(iv_ranges[~mix].tolist())


# ----------------------------------------------------------------------
# Vectorised density accumulator update
# ----------------------------------------------------------------------

def update_density_vectorised(
    accs: dict[tuple, DensityAcc],
    group_df: pd.DataFrame,
    decimals: int,
    benchmark_name: str,
    max_samples: int = 200_000,
) -> None:
    """Vectorised replacement for ``v1.update_density``.

    Same three modes (``naive``, ``dedup_obs``, ``exclude_self_dup``).
    The expensive ``exclude_self_dup`` per-twin recomputation loop in v1
    is replaced by a one-shot masked-d2 reduction.
    """
    if group_df.empty:
        return

    df = group_df
    lm_round = df["log_moneyness"].round(decimals).to_numpy()
    tau_round = df["tau"].round(decimals).to_numpy()
    observed = df["observed"].to_numpy(dtype=bool)

    if not (~observed).any():
        return

    obs_coords = np.column_stack(
        [df.loc[observed, "log_moneyness"].to_numpy(dtype=float),
         df.loc[observed, "tau"].to_numpy(dtype=float)]
    )
    hid_coords = np.column_stack(
        [df.loc[~observed, "log_moneyness"].to_numpy(dtype=float),
         df.loc[~observed, "tau"].to_numpy(dtype=float)]
    )
    obs_lm_r = lm_round[observed]
    obs_tau_r = tau_round[observed]
    hid_lm_r = lm_round[~observed]
    hid_tau_r = tau_round[~observed]

    if obs_coords.size == 0:
        d_naive = np.full(len(hid_coords), np.inf)
        d_dedup = d_naive.copy()
        d_excl = d_naive.copy()
        has_twin = np.zeros(len(hid_coords), dtype=bool)
    else:
        d2 = (
            (hid_coords[:, None, 0] - obs_coords[None, :, 0]) ** 2
            + (hid_coords[:, None, 1] - obs_coords[None, :, 1]) ** 2
        )
        d_naive = np.sqrt(d2.min(axis=1))

        # Dedup mode: keep first occurrence of each rounded obs coord.
        obs_key_df = pd.DataFrame({"lm": obs_lm_r, "tau": obs_tau_r})
        dedup_mask = ~obs_key_df.duplicated(keep="first").to_numpy()
        d2_dedup = d2[:, dedup_mask]
        d_dedup = np.sqrt(d2_dedup.min(axis=1))

        # Exclude self-dup: mask obs entries sharing rounded key with hidden row.
        match = (
            (hid_lm_r[:, None] == obs_lm_r[None, :])
            & (hid_tau_r[:, None] == obs_tau_r[None, :])
        )
        d2_excl = np.where(match, np.inf, d2)
        with np.errstate(invalid="ignore"):
            d_excl = np.sqrt(d2_excl.min(axis=1))
        has_twin = match.any(axis=1)

    for mode, dvec in (
        ("naive", d_naive),
        ("dedup_obs", d_dedup),
        ("exclude_self_dup", d_excl),
    ):
        key = (benchmark_name, decimals, mode)
        acc = accs.get(key)
        if acc is None:
            acc = DensityAcc(benchmark=benchmark_name, decimals=decimals, mode=mode)
            accs[key] = acc
        acc.n_total += len(dvec)
        acc.n_zero_distance += int(np.sum(dvec == 0))
        if mode == "naive":
            acc.n_held_out_with_exact_obs_dup += int(has_twin.sum())
        if len(acc.distances) < max_samples:
            take = min(max_samples - len(acc.distances), len(dvec))
            acc.distances.extend(dvec[:take].tolist())


# ----------------------------------------------------------------------
# Top-level audit drivers (mirror v1.audit_strict / v1.audit_benchmark)
# ----------------------------------------------------------------------

def audit_strict(strict_path: Path, chunk_rows: int) -> list[StrictAcc]:
    cols_needed = ["date", "expiration", "strike", "type",
                   "log_moneyness", "tau", "implied_volatility"]

    accs: dict[str, StrictAcc] = {
        "contract": StrictAcc(key_name="date+expiration+strike"),
    }
    for d in ROUND_DECIMALS:
        accs[f"coord_{d}dp"] = StrictAcc(
            key_name=f"date+round(lm,{d})+round(tau,{d})"
        )

    t0 = time.time()
    n_dates = 0
    for date_df in iter_dates(strict_path, cols_needed, chunk_rows=chunk_rows):
        update_strict_acc_vectorised(
            accs["contract"], date_df,
            key_cols=["expiration", "strike"], has_type=True,
        )
        for d in ROUND_DECIMALS:
            tmp = date_df.copy()
            tmp["lm_round"] = tmp["log_moneyness"].round(d)
            tmp["tau_round"] = tmp["tau"].round(d)
            update_strict_acc_vectorised(
                accs[f"coord_{d}dp"], tmp,
                key_cols=["lm_round", "tau_round"], has_type=True,
            )
        n_dates += 1
        if n_dates % 200 == 0:
            print(f"  [strict] processed {n_dates} dates in {time.time()-t0:.1f}s",
                  flush=True)

    print(f"  [strict] DONE: {n_dates} dates in {time.time()-t0:.1f}s",
          flush=True)
    return list(accs.values())


def audit_benchmark(
    bench_path: Path,
    chunk_rows: int,
) -> tuple[dict[tuple, LeakageRow], dict[tuple, DensityAcc]]:
    cols_needed = ["date", "log_moneyness", "tau", "implied_volatility",
                   "iv_clean", "observed"]
    pf = pq.ParquetFile(str(bench_path))
    if "split" in pf.schema_arrow.names:
        cols_needed.append("split")

    leak_rows: dict[tuple, LeakageRow] = {}
    density: dict[tuple, DensityAcc] = {}
    benchmark_name = bench_path.stem

    t0 = time.time()
    n_dates = 0
    for date_df in iter_dates(bench_path, cols_needed, chunk_rows=chunk_rows):
        for d in ROUND_DECIMALS:
            # update_benchmark is already vectorised in v1; reuse it.
            v1.update_benchmark(leak_rows, leak_rows, date_df,
                                decimals=d, benchmark_name=benchmark_name)
        update_density_vectorised(
            density, date_df,
            decimals=ROUND_DECIMALS[1],
            benchmark_name=benchmark_name,
        )
        n_dates += 1
        if n_dates % 200 == 0:
            print(f"  [{benchmark_name}] processed {n_dates} dates in "
                  f"{time.time()-t0:.1f}s", flush=True)

    print(f"  [{benchmark_name}] DONE: {n_dates} dates in "
          f"{time.time()-t0:.1f}s", flush=True)
    return leak_rows, density


# ----------------------------------------------------------------------
# CLI (mirrors v1; only differs in --help blurb)
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit duplicate model-coordinates (v2, vectorised).",
    )
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/audits/duplicate_coordinates"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("docs/research/duplicate_coordinate_audit.md"),
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args(argv)

    if not args.strict.exists():
        print(f"ERROR: strict file not found at {args.strict}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Auditing strict surface table (v2): {args.strict}")
    strict_accs = audit_strict(args.strict, chunk_rows=args.chunk_rows)

    print(f"[2/3] Auditing benchmark splits ({len(args.benchmark)} file(s))")
    leak_rows: dict[tuple, LeakageRow] = {}
    density: dict[tuple, DensityAcc] = {}
    for bench in args.benchmark:
        if not bench.exists():
            print(f"  WARN: benchmark not found: {bench}", file=sys.stderr)
            continue
        bench_leak, bench_density = audit_benchmark(
            bench, chunk_rows=args.chunk_rows
        )
        leak_rows.update(bench_leak)
        density.update(bench_density)

    print(f"[3/3] Writing artifacts to {args.output_dir} and {args.report}")
    v1.write_duplicate_summary(
        strict_accs, args.output_dir / "duplicate_summary.csv"
    )
    v1.write_dispersion(
        strict_accs, args.output_dir / "duplicate_iv_dispersion.csv"
    )
    v1.write_leakage(leak_rows, args.output_dir / "observed_hidden_leakage.csv")
    v1.write_density(density, args.output_dir / "sparse_density_sensitivity.csv")
    v1.render_report(
        strict_accs, leak_rows, density,
        out_path=args.report,
        strict_path=args.strict,
        benchmark_paths=args.benchmark,
    )

    headline = {
        "strict_path": str(args.strict),
        "benchmarks": [str(b) for b in args.benchmark],
        "contract_dup_share_pct": (
            100.0 * strict_accs[0].rows_in_dup_groups
            / max(strict_accs[0].total_rows, 1)
        ),
        "n_call_put_groups": strict_accs[0].n_groups_call_put_mix,
        "n_same_type_groups": strict_accs[0].n_groups_same_type,
        "audit_version": "v2",
    }
    (args.output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2)
    )
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
