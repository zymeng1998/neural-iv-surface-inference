#!/usr/bin/env python3
"""Audit duplicate model-coordinates in the strict surface table and benchmark splits.

Background
----------
The conditional surface model and the RBF baseline both treat IV as a
single-valued function of ``(log_moneyness, tau)``. The raw option chain,
however, carries ``type`` (call vs put). The current pipeline
(``src/data/03_build_spy_surface_table.py``) does **not** deduplicate by
``(date, expiration, strike)``, and the strict modelling subset retains both
the call and the put leg whenever each individually passes the IV / tau /
log-moneyness gates. Downstream code (``conditional_loaders.py``,
``interpolation.py``) drops ``type`` entirely, so a single date can carry two
distinct rows at the same model-coordinate with two distinct ``iv_clean``
labels.

This script quantifies the problem, end-to-end, with three outputs:

1. ``artifacts/audits/duplicate_coordinates/duplicate_summary.csv``
   Per-dataset row counts, dup-group counts, by key definition.
2. ``artifacts/audits/duplicate_coordinates/duplicate_iv_dispersion.csv``
   Quantiles of ``iv_range = max - min`` inside each duplicate group,
   split by call-put-mix vs same-type, and by key-rounding tolerance.
3. ``artifacts/audits/duplicate_coordinates/observed_hidden_leakage.csv``
   For each benchmark split, the share of held-out (``observed=False``)
   rows that have an exact-coordinate observed twin on the same date,
   sliced by split / moneyness region / maturity bucket, plus the MAE of
   the trivial "predict the observed twin's mean IV" baseline.

The script is **read-only**. It does not mutate the strict surface table
or the benchmark parquets.

Usage
-----
::

    python scripts/audit_duplicate_coordinates.py \\
        --strict data_processed/spy/spy_surface_points_strict.parquet \\
        --benchmark data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet \\
        --output-dir artifacts/audits/duplicate_coordinates \\
        --report docs/research/duplicate_coordinate_audit.md

Multiple ``--benchmark`` flags are supported; each is audited separately
and the report aggregates them.

Memory safety
-------------
The strict file is ~22 M rows and benchmark files are similar. We stream
them in row-group sized chunks via PyArrow and process **one date at a
time**, buffering only the trailing partial date between chunks. Per-date
peak memory is bounded by the largest date's row count (~10k for SPY).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Decimals to round (log_moneyness, tau) at, for the model-coordinate key.
# 8 dp is loose enough that nearly any float32 round-trip collapses; 12 dp
# is tight enough to keep genuinely distinct points apart. We report all
# three to bracket FP noise.
ROUND_DECIMALS = (8, 10, 12)

# Moneyness regions, mirrors src/neural_iv_surface_inference/training/eval.py.
MONEYNESS_BUCKETS = [
    ("deep_put_wing", -np.inf, -0.20),
    ("put_wing", -0.20, -0.05),
    ("atm", -0.05, 0.05),
    ("call_wing", 0.05, 0.20),
    ("deep_call_wing", 0.20, np.inf),
]

MATURITY_BUCKETS = [
    ("short", 0.0, 30.0 / 365.0),
    ("medium", 30.0 / 365.0, 180.0 / 365.0),
    ("long", 180.0 / 365.0, np.inf),
]

DEFAULT_CHUNK_ROWS = 500_000


# ----------------------------------------------------------------------
# Bucket helpers
# ----------------------------------------------------------------------

def bucketise(values: np.ndarray, edges: list[tuple[str, float, float]]) -> np.ndarray:
    """Return a string array assigning each value to the named edge bucket."""
    out = np.empty(values.shape, dtype=object)
    for name, lo, hi in edges:
        sel = (values >= lo) & (values < hi)
        out[sel] = name
    # last bucket caps inf-inclusive
    return out


# ----------------------------------------------------------------------
# Chunked per-date iterator
# ----------------------------------------------------------------------

def iter_dates(
    parquet_path: Path,
    columns: list[str],
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per distinct ``date`` from a parquet file.

    The file is assumed to be (approximately) sorted by date. We stream
    by row groups, buffer the trailing partial date between chunks, and
    flush whenever the date changes. If the file is not sorted, this is
    still correct but yields more (and smaller) fragments per date.
    """
    pf = pq.ParquetFile(str(parquet_path))
    available = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"{parquet_path.name}: missing required columns {missing}. "
            f"Available: {sorted(available)}"
        )

    buf: pd.DataFrame | None = None
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=columns):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        if buf is not None:
            df = pd.concat([buf, df], ignore_index=True)
            buf = None

        # Group consecutive runs of identical date; flush all but the last.
        if df.empty:
            continue

        # Compute run-length boundaries.
        date_vals = df["date"].values
        change_mask = np.empty(len(df), dtype=bool)
        change_mask[0] = True
        change_mask[1:] = date_vals[1:] != date_vals[:-1]
        change_idx = np.flatnonzero(change_mask)
        run_starts = list(change_idx) + [len(df)]

        # The last run may continue into the next chunk; buffer it.
        for i in range(len(run_starts) - 2):
            lo, hi = run_starts[i], run_starts[i + 1]
            yield df.iloc[lo:hi].reset_index(drop=True)
        buf = df.iloc[run_starts[-2]:].reset_index(drop=True)

    if buf is not None and not buf.empty:
        yield buf


# ----------------------------------------------------------------------
# Strict-surface accumulator
# ----------------------------------------------------------------------

@dataclass
class StrictAcc:
    """Running aggregates for one key definition over the strict file."""

    key_name: str
    total_rows: int = 0
    rows_in_dup_groups: int = 0
    n_dup_groups: int = 0
    group_size_hist: Counter = field(default_factory=Counter)
    # Call-put mix
    n_groups_call_put_mix: int = 0
    n_groups_same_type: int = 0
    # Streaming IV dispersion per group (range = max - min).
    iv_ranges_mixed: list[float] = field(default_factory=list)
    iv_ranges_same: list[float] = field(default_factory=list)


def update_strict_acc(
    acc: StrictAcc,
    group_df: pd.DataFrame,
    key_cols: list[str],
    has_type: bool,
) -> None:
    """Update ``acc`` with the duplicate groups in one date's DataFrame.

    ``group_df`` carries exactly one date worth of rows. The grouping is
    on ``key_cols`` (which never include ``date`` — the date is implicit
    in the per-date iteration).
    """
    acc.total_rows += len(group_df)
    grouped = group_df.groupby(key_cols, sort=False)
    sizes = grouped.size()
    dup_sizes = sizes[sizes >= 2]
    acc.rows_in_dup_groups += int(dup_sizes.sum())
    acc.n_dup_groups += int(len(dup_sizes))
    acc.group_size_hist.update(dup_sizes.tolist())

    if len(dup_sizes) == 0:
        return

    iv_col = "implied_volatility"
    for key, idx in grouped.indices.items():
        if len(idx) < 2:
            continue
        sub = group_df.iloc[idx]
        iv_vals = sub[iv_col].to_numpy(dtype=float)
        iv_range = float(iv_vals.max() - iv_vals.min())

        if has_type:
            types = set(sub["type"].astype(str).str.lower())
            mix = ("call" in types and "put" in types) or (
                "c" in types and "p" in types
            )
        else:
            mix = False

        if mix:
            acc.n_groups_call_put_mix += 1
            acc.iv_ranges_mixed.append(iv_range)
        else:
            acc.n_groups_same_type += 1
            acc.iv_ranges_same.append(iv_range)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {q: float("nan") for q in ("p50", "p90", "p95", "p99", "max", "mean", "n")}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


# ----------------------------------------------------------------------
# Benchmark accumulator
# ----------------------------------------------------------------------

@dataclass
class LeakageRow:
    benchmark: str
    key_decimals: int
    split: str
    moneyness_bucket: str
    maturity_bucket: str
    n_hidden_rows: int = 0
    n_hidden_with_obs_twin: int = 0
    abs_err_sum: float = 0.0
    abs_err_n: int = 0
    iv_range_sum: float = 0.0
    iv_range_n: int = 0


def make_leakage_key(
    benchmark: str, decimals: int, split: str, money: str, mat: str
) -> tuple:
    return (benchmark, decimals, split, money, mat)


def update_benchmark(
    leak_rows: dict[tuple, LeakageRow],
    nbd_obs: dict[tuple, LeakageRow],  # alias kept for clarity
    group_df: pd.DataFrame,
    decimals: int,
    benchmark_name: str,
) -> None:
    """Update leakage rows for one date of one benchmark at one rounding."""
    if group_df.empty:
        return

    df = group_df.copy()
    df["lm_round"] = df["log_moneyness"].round(decimals)
    df["tau_round"] = df["tau"].round(decimals)
    df["money_bucket"] = bucketise(df["log_moneyness"].to_numpy(), MONEYNESS_BUCKETS)
    df["mat_bucket"] = bucketise(df["tau"].to_numpy(), MATURITY_BUCKETS)
    # Fallback for any None (shouldn't happen if input is in strict bounds).
    df["money_bucket"] = df["money_bucket"].astype(object).where(df["money_bucket"].notnull(), "unknown")
    df["mat_bucket"] = df["mat_bucket"].astype(object).where(df["mat_bucket"].notnull(), "unknown")

    obs_mean = (
        df[df["observed"]]
        .groupby(["lm_round", "tau_round"])["implied_volatility"]
        .mean()
    )
    obs_count = (
        df[df["observed"]]
        .groupby(["lm_round", "tau_round"])["implied_volatility"]
        .size()
    )

    hidden = df[~df["observed"]]
    if hidden.empty and df.empty:
        return

    # Identify hidden rows with an observed twin at the exact coordinate.
    twin_iv = obs_mean.reindex(
        list(zip(hidden["lm_round"].to_numpy(), hidden["tau_round"].to_numpy()))
    )
    twin_count = obs_count.reindex(twin_iv.index)
    twin_iv_vals = twin_iv.to_numpy()
    has_twin = ~np.isnan(twin_iv_vals)

    # Per-row leakage stats keyed by split/money/maturity.
    splits = hidden.get("split")
    if splits is None:
        splits = pd.Series(["all"] * len(hidden), index=hidden.index)
    splits = splits.astype(str).to_numpy()
    moneys = hidden["money_bucket"].to_numpy()
    mats = hidden["mat_bucket"].to_numpy()
    iv_clean = hidden["iv_clean"].to_numpy(dtype=float) if "iv_clean" in hidden.columns else hidden["implied_volatility"].to_numpy(dtype=float)

    abs_err = np.abs(twin_iv_vals - iv_clean)

    # We also need iv-range across BOTH observed-twin and hidden labels at
    # the same coordinate, to characterise label spread.
    full_range = (
        df.groupby(["lm_round", "tau_round"])["iv_clean"]
        .agg(lambda s: float(s.max() - s.min()))
        if "iv_clean" in df.columns
        else df.groupby(["lm_round", "tau_round"])["implied_volatility"]
        .agg(lambda s: float(s.max() - s.min()))
    )
    iv_range_per_hidden = full_range.reindex(
        list(zip(hidden["lm_round"].to_numpy(), hidden["tau_round"].to_numpy()))
    ).to_numpy()

    for s_val, m_val, t_val, twin, err, rng in zip(
        splits, moneys, mats, has_twin, abs_err, iv_range_per_hidden
    ):
        key = make_leakage_key(benchmark_name, decimals, s_val, m_val, t_val)
        row = leak_rows.get(key)
        if row is None:
            row = LeakageRow(
                benchmark=benchmark_name,
                key_decimals=decimals,
                split=s_val,
                moneyness_bucket=m_val,
                maturity_bucket=t_val,
            )
            leak_rows[key] = row
        row.n_hidden_rows += 1
        if twin:
            row.n_hidden_with_obs_twin += 1
            if not np.isnan(err):
                row.abs_err_sum += float(err)
                row.abs_err_n += 1
            if not np.isnan(rng):
                row.iv_range_sum += float(rng)
                row.iv_range_n += 1


# ----------------------------------------------------------------------
# Sparse-region density audit
# ----------------------------------------------------------------------

@dataclass
class DensityAcc:
    """Q1/Q2/Q3/Q4 density bucket counts for one (rounding, dedup-mode)."""

    benchmark: str
    decimals: int
    mode: str  # "naive", "dedup_obs", "exclude_self_dup"
    # Streaming nearest-neighbour distance samples; we keep a reservoir.
    distances: list[float] = field(default_factory=list)
    n_zero_distance: int = 0
    n_total: int = 0
    n_held_out_with_exact_obs_dup: int = 0


def nearest_observed_distance(
    obs_coords: np.ndarray,
    query_coords: np.ndarray,
) -> np.ndarray:
    """Return the L2 nearest-neighbour distance for each query → observed set."""
    if obs_coords.size == 0 or query_coords.size == 0:
        return np.full(len(query_coords), np.inf)
    # All-pairs is fine because per-date sizes are small (≤ ~10k).
    d2 = (
        (query_coords[:, None, 0] - obs_coords[None, :, 0]) ** 2
        + (query_coords[:, None, 1] - obs_coords[None, :, 1]) ** 2
    )
    return np.sqrt(d2.min(axis=1))


def update_density(
    accs: dict[tuple, DensityAcc],
    group_df: pd.DataFrame,
    decimals: int,
    benchmark_name: str,
    max_samples: int = 200_000,
) -> None:
    if group_df.empty:
        return

    df = group_df.copy()
    df["lm_round"] = df["log_moneyness"].round(decimals)
    df["tau_round"] = df["tau"].round(decimals)

    obs = df[df["observed"]]
    hid = df[~df["observed"]]
    if hid.empty:
        return

    obs_coords = obs[["log_moneyness", "tau"]].to_numpy(dtype=float)
    obs_keys = set(zip(obs["lm_round"].to_numpy(), obs["tau_round"].to_numpy()))
    hid_coords = hid[["log_moneyness", "tau"]].to_numpy(dtype=float)
    hid_keys = list(zip(hid["lm_round"].to_numpy(), hid["tau_round"].to_numpy()))

    # Mode A: naive.
    d_naive = nearest_observed_distance(obs_coords, hid_coords)

    # Mode B: deduplicated observed (collapse by rounded key, keep one
    # representative per coordinate; we just sub-select first occurrence).
    obs_dedup_idx = ~obs.duplicated(subset=["lm_round", "tau_round"], keep="first")
    obs_dedup_coords = obs_coords[obs_dedup_idx.to_numpy()]
    d_dedup = nearest_observed_distance(obs_dedup_coords, hid_coords)

    # Mode C: exclude exact-coordinate observed twins of each hidden row.
    has_twin = np.array([k in obs_keys for k in hid_keys])
    d_excl = d_naive.copy()
    if has_twin.any():
        # For rows with twin, drop the twin coordinate from obs and re-nn.
        # We do this by mask-and-recompute per twin coordinate. For
        # efficiency, we vectorise by setting the distance to those exact
        # twin coords to inf inside the d2 matrix.
        # Simpler: drop ALL obs that share rounded key with the hidden row.
        for i, (k, h_coord) in enumerate(zip(hid_keys, hid_coords)):
            if not has_twin[i]:
                continue
            mask = ~(
                (obs["lm_round"].to_numpy() == k[0])
                & (obs["tau_round"].to_numpy() == k[1])
            )
            sub = obs_coords[mask]
            if sub.size == 0:
                d_excl[i] = np.inf
            else:
                d_excl[i] = float(
                    np.sqrt(
                        ((sub - h_coord) ** 2).sum(axis=1).min()
                    )
                )

    for mode, dvec in (
        ("naive", d_naive),
        ("dedup_obs", d_dedup),
        ("exclude_self_dup", d_excl),
    ):
        key = (benchmark_name, decimals, mode)
        acc = accs.get(key)
        if acc is None:
            acc = DensityAcc(
                benchmark=benchmark_name, decimals=decimals, mode=mode
            )
            accs[key] = acc
        acc.n_total += len(dvec)
        acc.n_zero_distance += int(np.sum(dvec == 0))
        if mode == "naive":
            acc.n_held_out_with_exact_obs_dup += int(has_twin.sum())
        if len(acc.distances) < max_samples:
            take = min(max_samples - len(acc.distances), len(dvec))
            acc.distances.extend(dvec[:take].tolist())


# ----------------------------------------------------------------------
# Main pass
# ----------------------------------------------------------------------

def audit_strict(strict_path: Path, chunk_rows: int) -> list[StrictAcc]:
    """Audit the strict surface file for duplicate coordinates."""
    cols_needed = ["date", "expiration", "strike", "type",
                   "log_moneyness", "tau", "implied_volatility"]

    accs: dict[str, StrictAcc] = {}
    accs["contract"] = StrictAcc(key_name="date+expiration+strike")
    for d in ROUND_DECIMALS:
        accs[f"coord_{d}dp"] = StrictAcc(key_name=f"date+round(lm,{d})+round(tau,{d})")

    t0 = time.time()
    n_dates = 0
    for date_df in iter_dates(strict_path, cols_needed, chunk_rows=chunk_rows):
        # Contract-level key
        update_strict_acc(
            accs["contract"], date_df,
            key_cols=["expiration", "strike"], has_type=True,
        )
        # Model-coordinate keys at multiple rounding tolerances
        for d in ROUND_DECIMALS:
            tmp = date_df.copy()
            tmp["lm_round"] = tmp["log_moneyness"].round(d)
            tmp["tau_round"] = tmp["tau"].round(d)
            update_strict_acc(
                accs[f"coord_{d}dp"], tmp,
                key_cols=["lm_round", "tau_round"], has_type=True,
            )
        n_dates += 1
        if n_dates % 200 == 0:
            print(f"  [strict] processed {n_dates} dates in {time.time()-t0:.1f}s",
                  flush=True)

    print(f"  [strict] DONE: {n_dates} dates in {time.time()-t0:.1f}s", flush=True)
    return list(accs.values())


def audit_benchmark(
    bench_path: Path,
    chunk_rows: int,
) -> tuple[dict[tuple, LeakageRow], dict[tuple, DensityAcc]]:
    """Audit one benchmark file for observed-hidden leakage + sparse density."""
    cols_needed = ["date", "log_moneyness", "tau", "implied_volatility",
                   "iv_clean", "observed"]
    # split is optional but expected.
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
            update_benchmark(leak_rows, leak_rows, date_df,
                             decimals=d, benchmark_name=benchmark_name)
        # Density audit at the tightest decimals only (cheapest signal).
        update_density(density, date_df,
                       decimals=ROUND_DECIMALS[1],
                       benchmark_name=benchmark_name)
        n_dates += 1
        if n_dates % 200 == 0:
            print(f"  [{benchmark_name}] processed {n_dates} dates in "
                  f"{time.time()-t0:.1f}s", flush=True)

    print(f"  [{benchmark_name}] DONE: {n_dates} dates in "
          f"{time.time()-t0:.1f}s", flush=True)
    return leak_rows, density


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_duplicate_summary(strict_accs: list[StrictAcc], out_path: Path) -> None:
    rows = []
    for acc in strict_accs:
        rows.append({
            "key_name": acc.key_name,
            "total_rows": acc.total_rows,
            "rows_in_dup_groups": acc.rows_in_dup_groups,
            "pct_rows_in_dup_groups": (
                100.0 * acc.rows_in_dup_groups / max(acc.total_rows, 1)
            ),
            "n_dup_groups": acc.n_dup_groups,
            "n_groups_call_put_mix": acc.n_groups_call_put_mix,
            "n_groups_same_type": acc.n_groups_same_type,
            "group_size_2": acc.group_size_hist.get(2, 0),
            "group_size_3": acc.group_size_hist.get(3, 0),
            "group_size_4_plus": sum(
                c for k, c in acc.group_size_hist.items() if k >= 4
            ),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def write_dispersion(strict_accs: list[StrictAcc], out_path: Path) -> None:
    rows = []
    for acc in strict_accs:
        for label, vals in (("call_put_mix", acc.iv_ranges_mixed),
                            ("same_type", acc.iv_ranges_same),
                            ("all_dup_groups", acc.iv_ranges_mixed + acc.iv_ranges_same)):
            q = quantiles(vals)
            rows.append({
                "key_name": acc.key_name,
                "group_kind": label,
                **q,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def write_leakage(
    leak_rows: dict[tuple, LeakageRow], out_path: Path
) -> None:
    rows = []
    for r in leak_rows.values():
        rows.append({
            "benchmark": r.benchmark,
            "key_decimals": r.key_decimals,
            "split": r.split,
            "moneyness_bucket": r.moneyness_bucket,
            "maturity_bucket": r.maturity_bucket,
            "n_hidden_rows": r.n_hidden_rows,
            "n_hidden_with_obs_twin": r.n_hidden_with_obs_twin,
            "pct_hidden_with_twin": (
                100.0 * r.n_hidden_with_obs_twin / max(r.n_hidden_rows, 1)
            ),
            "twin_mae": (
                r.abs_err_sum / r.abs_err_n if r.abs_err_n > 0 else float("nan")
            ),
            "twin_iv_range_mean": (
                r.iv_range_sum / r.iv_range_n if r.iv_range_n > 0 else float("nan")
            ),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def write_density(
    density: dict[tuple, DensityAcc], out_path: Path
) -> None:
    rows = []
    for acc in density.values():
        arr = np.asarray(acc.distances, dtype=float)
        if arr.size > 0:
            qs = np.quantile(arr, [0.25, 0.5, 0.75, 0.95])
        else:
            qs = [float("nan")] * 4
        rows.append({
            "benchmark": acc.benchmark,
            "key_decimals": acc.decimals,
            "mode": acc.mode,
            "n_total_hidden": acc.n_total,
            "n_zero_distance": acc.n_zero_distance,
            "pct_zero_distance": (
                100.0 * acc.n_zero_distance / max(acc.n_total, 1)
            ),
            "n_hidden_with_exact_obs_dup": acc.n_held_out_with_exact_obs_dup,
            "q25_distance": float(qs[0]),
            "q50_distance": float(qs[1]),
            "q75_distance": float(qs[2]),
            "q95_distance": float(qs[3]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------

def render_report(
    strict_accs: list[StrictAcc],
    leak_rows: dict[tuple, LeakageRow],
    density: dict[tuple, DensityAcc],
    out_path: Path,
    strict_path: Path,
    benchmark_paths: list[Path],
) -> None:
    contract = next(a for a in strict_accs if a.key_name.startswith("date+expiration"))
    coord_8 = next(a for a in strict_accs if a.key_name.endswith("(lm,8)+round(tau,8)"))
    coord_10 = next(a for a in strict_accs if a.key_name.endswith("(lm,10)+round(tau,10)"))
    coord_12 = next(a for a in strict_accs if a.key_name.endswith("(lm,12)+round(tau,12)"))

    def pct(numer, denom):
        return 100.0 * numer / max(denom, 1)

    # Headline rule for the verdict.
    contract_share = pct(contract.rows_in_dup_groups, contract.total_rows)
    coord10_share = pct(coord_10.rows_in_dup_groups, coord_10.total_rows)
    if contract_share < 0.5 and coord10_share < 0.5:
        verdict = "negligible"
    elif contract_share < 10 and coord10_share < 15:
        verdict = "moderate"
    else:
        verdict = "severe"

    lines = []
    lines.append("# Duplicate-coordinate audit\n")
    lines.append("---")
    lines.append("status: executed")
    lines.append("type: data_audit")
    lines.append(f"input_strict: {strict_path}")
    lines.append(f"input_benchmarks: {[p.name for p in benchmark_paths]}")
    lines.append("---\n")

    lines.append("## 1. Verdict\n")
    lines.append(f"**Severity: {verdict.upper()}**.\n")
    lines.append(
        f"- {contract.rows_in_dup_groups:,} / {contract.total_rows:,} rows "
        f"({contract_share:.2f}%) live inside a duplicate "
        f"`(date, expiration, strike)` group in the strict surface table.\n"
    )
    lines.append(
        f"- {coord_10.rows_in_dup_groups:,} / {coord_10.total_rows:,} rows "
        f"({coord10_share:.2f}%) live inside a duplicate "
        f"`(date, round(log_m,10), round(tau,10))` group — the key the "
        f"conditional model and RBF baseline both treat as the surface "
        f"coordinate.\n"
    )
    lines.append(
        f"- Of the contract-level dup groups, "
        f"{contract.n_groups_call_put_mix:,} are call-put pairs "
        f"({pct(contract.n_groups_call_put_mix, contract.n_dup_groups):.1f}%) "
        f"and {contract.n_groups_same_type:,} are same-type duplicates.\n"
    )

    lines.append("## 2. Exact numbers\n")
    lines.append("### 2.1 Duplicate counts by key definition\n")
    lines.append("| Key | Total rows | Rows in dup groups | % | Dup groups | size=2 | size=3 | size≥4 | Call-put mix | Same-type |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for acc in strict_accs:
        lines.append(
            f"| `{acc.key_name}` | {acc.total_rows:,} | "
            f"{acc.rows_in_dup_groups:,} | "
            f"{pct(acc.rows_in_dup_groups, acc.total_rows):.2f}% | "
            f"{acc.n_dup_groups:,} | "
            f"{acc.group_size_hist.get(2,0):,} | "
            f"{acc.group_size_hist.get(3,0):,} | "
            f"{sum(c for k,c in acc.group_size_hist.items() if k>=4):,} | "
            f"{acc.n_groups_call_put_mix:,} | "
            f"{acc.n_groups_same_type:,} |"
        )

    lines.append("\n### 2.2 IV-label dispersion within duplicate groups\n")
    lines.append("`iv_range = max(IV) − min(IV)` inside each group.\n")
    lines.append("| Key | Group kind | n | mean | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for acc in strict_accs:
        for label, vals in (("call_put_mix", acc.iv_ranges_mixed),
                            ("same_type", acc.iv_ranges_same),
                            ("all", acc.iv_ranges_mixed + acc.iv_ranges_same)):
            q = quantiles(vals)
            lines.append(
                f"| `{acc.key_name}` | {label} | {q['n']:,} | {q['mean']:.4f} | "
                f"{q['p50']:.4f} | {q['p90']:.4f} | {q['p95']:.4f} | "
                f"{q['p99']:.4f} | {q['max']:.4f} |"
            )

    lines.append("\n### 2.3 Benchmark observed-hidden leakage\n")
    lines.append(
        "Held-out rows whose coordinate (rounded to 10 d.p.) coincides with "
        "at least one observed row on the same date. The 'twin MAE' column "
        "is the MAE of the trivial baseline that predicts the mean observed "
        "IV at the exact coordinate; it tells us whether nearest-distance=0 "
        "is genuine local density or a duplicate-leg artifact.\n"
    )
    lines.append("| Benchmark | dp | Split | Money | Maturity | Hidden | With twin | % | Twin MAE | iv_range mean |")
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    keys_sorted = sorted(leak_rows.keys())
    for k in keys_sorted:
        r = leak_rows[k]
        lines.append(
            f"| {r.benchmark} | {r.key_decimals} | {r.split} | "
            f"{r.moneyness_bucket} | {r.maturity_bucket} | "
            f"{r.n_hidden_rows:,} | {r.n_hidden_with_obs_twin:,} | "
            f"{pct(r.n_hidden_with_obs_twin, r.n_hidden_rows):.2f}% | "
            f"{(r.abs_err_sum / r.abs_err_n) if r.abs_err_n else float('nan'):.4f} | "
            f"{(r.iv_range_sum / r.iv_range_n) if r.iv_range_n else float('nan'):.4f} |"
        )

    lines.append("\n### 2.4 Sparse-region density sensitivity\n")
    lines.append(
        "Nearest-observed L2 distance from each held-out point to the "
        "observed set on the same date, computed three ways: **naive** "
        "(raw observed set), **dedup_obs** (collapse observed by rounded "
        "coordinate, keep one representative), **exclude_self_dup** (drop "
        "observed rows sharing the rounded coordinate of the hidden row).\n"
    )
    lines.append("| Benchmark | dp | Mode | n hidden | n zero-dist | % zero | q25 | q50 | q75 | q95 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(density.keys()):
        acc = density[k]
        arr = np.asarray(acc.distances, dtype=float)
        if arr.size > 0:
            qs = np.quantile(arr, [0.25, 0.5, 0.75, 0.95])
        else:
            qs = [float("nan")] * 4
        lines.append(
            f"| {acc.benchmark} | {acc.decimals} | {acc.mode} | "
            f"{acc.n_total:,} | {acc.n_zero_distance:,} | "
            f"{pct(acc.n_zero_distance, acc.n_total):.2f}% | "
            f"{qs[0]:.4f} | {qs[1]:.4f} | {qs[2]:.4f} | {qs[3]:.4f} |"
        )

    lines.append("\n## 3. Interpretation\n")
    lines.append(
        "The IV surface is defined as `IV(log_moneyness, tau)`. Including "
        "both calls and puts at the same `(date, expiration, strike)` "
        "violates the single-valued-function assumption: at the same "
        "model coordinate the loss sees two distinct labels, and the RBF "
        "kernel must average over them. Put-call parity says the *price* "
        "structure is consistent, but the *implied volatilities* reported "
        "by the data source diverge whenever bid/ask sits asymmetrically "
        "around the parity-implied mid (which is the usual case for "
        "non-ATM strikes). Section 2.2 quantifies that divergence.\n"
    )
    lines.append(
        "Section 2.3 then quantifies how often held-out targets in the "
        "benchmark have an observed exact twin on the same date. Whenever "
        "they do, the sparse-region analysis's nearest-observed distance "
        "is *forced* to zero — not because the local neighbourhood is "
        "dense, but because the held-out point is literally one leg of a "
        "call-put pair whose other leg leaked into context.\n"
    )

    lines.append("\n## 4. Recommendation\n")
    if verdict == "negligible":
        rec = (
            "Duplicate coordinates exist but affect a negligible fraction of "
            "rows. Run the sparse-region ANP-vs-RBF experiment as-is, but "
            "include the dedup-aware density definition as a sanity check "
            "in the report appendix."
        )
    elif verdict == "moderate":
        rec = (
            "Duplicate coordinates are material. Before running sparse-region "
            "ANP-vs-RBF, add an Option-D pre-filter: when computing per-date "
            "nearest-observed density, deduplicate the observed set by "
            "`(date, round(lm,10), round(tau,10))` and exclude observed "
            "rows that share a hidden row's exact coordinate. Re-run the "
            "experiment with and without the filter and report sensitivity. "
            "Do NOT rebuild the strict surface yet."
        )
    else:
        rec = (
            "Duplicate coordinates are severe. The current strict surface "
            "table is not a single-valued IV surface; it is a quote table. "
            "**Recommended action before any sparse-region run:** ship "
            "Option A (true OTM surface, puts for K<S / calls for K>S, "
            "tie-break at ATM) as a new derived parquet, e.g. "
            "`spy_surface_points_strict_otm.parquet`, leave the existing "
            "file untouched, regenerate the benchmark from the OTM file, "
            "and re-run the sparse-region experiment on it. Option D "
            "(dedup-aware density) is a stopgap if Option A is too costly "
            "to ship before the experiment, but the experiment then needs "
            "to be reported with the duplicate-affected slice excluded."
        )
    lines.append(rec + "\n")

    lines.append("## 5. Sparse-region experiment status\n")
    if verdict == "negligible":
        lines.append(
            "Sparse-region ANP-vs-RBF can proceed as-is, with a one-line "
            "sensitivity check.\n"
        )
    elif verdict == "moderate":
        lines.append(
            "Sparse-region ANP-vs-RBF needs Option-D dedup-aware density "
            "before headline numbers are reported.\n"
        )
    else:
        lines.append(
            "Sparse-region ANP-vs-RBF needs data reconstruction (Option A) "
            "before it can be claimed as a clean test of the hypothesis.\n"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit duplicate model-coordinates in IV surface data."
    )
    parser.add_argument(
        "--strict", type=Path, required=True,
        help="Path to strict surface parquet (output of step 03).",
    )
    parser.add_argument(
        "--benchmark", type=Path, action="append", default=[],
        help="Path to a benchmark parquet. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/audits/duplicate_coordinates"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("docs/research/duplicate_coordinate_audit.md"),
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
    )
    args = parser.parse_args(argv)

    if not args.strict.exists():
        print(f"ERROR: strict file not found at {args.strict}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Auditing strict surface table: {args.strict}")
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
    write_duplicate_summary(
        strict_accs, args.output_dir / "duplicate_summary.csv"
    )
    write_dispersion(
        strict_accs, args.output_dir / "duplicate_iv_dispersion.csv"
    )
    write_leakage(leak_rows, args.output_dir / "observed_hidden_leakage.csv")
    write_density(density, args.output_dir / "sparse_density_sensitivity.csv")
    render_report(
        strict_accs, leak_rows, density,
        out_path=args.report,
        strict_path=args.strict,
        benchmark_paths=args.benchmark,
    )

    # JSON sidecar for machine-readable headline.
    headline = {
        "strict_path": str(args.strict),
        "benchmarks": [str(b) for b in args.benchmark],
        "contract_dup_share_pct": (
            100.0 * strict_accs[0].rows_in_dup_groups
            / max(strict_accs[0].total_rows, 1)
        ),
        "n_call_put_groups": strict_accs[0].n_groups_call_put_mix,
        "n_same_type_groups": strict_accs[0].n_groups_same_type,
    }
    (args.output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2)
    )
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
