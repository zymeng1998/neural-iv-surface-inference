#!/usr/bin/env python3
"""Step 5: Build the OTM-restricted strict surface table.

Implements Phase 3X / ADR 0006 (Correction A) + decision points D5 + D7.

Reads the dirty strict surface table
``data_processed/spy/spy_surface_points_strict.parquet`` (multi-valued
at ``(date, log_moneyness, tau)`` because both call and put legs are
kept per ``(date, expiration, strike)``) and writes a single-valued
counterpart ``spy_surface_points_strict_otm.parquet``.

Selection rule (per row, with ``log_m = log_moneyness`` and a
configurable ``atm_band`` half-width on ``|log_m|``):

* ``log_m <  -atm_band``  →  keep the ``put``  leg (OTM put side)
* ``log_m >  +atm_band``  →  keep the ``call`` leg (OTM call side)
* ``|log_m| <= atm_band`` (near-ATM): group by ``(date, expiration,
  strike)`` and pick the side with the **tighter** relative spread
  ``(ask - bid) / max(mid, eps)``; ties or missing-quote fields fall
  through to a deterministic fallback (``put``, declared order).

After the OTM rule, same-type residual duplicates at the rounded
``(date, log_m, tau)`` key (D7) are resolved deterministically: drop on
economic equivalence, else quality-select on (relative spread, then
``open_interest``, then ``volume``); ``keep='first'`` after a stable
sort is the documented final fallback. Residual groups are emitted to
``artifacts/audits/otm_residual_same_type.csv``.

The builder is **streaming** (PyArrow ``iter_batches``) — it buffers at
most a single calendar date in RAM, so the same code that runs on
local synthetic fixtures will run on the 22 M-row real strict file on
a CPU pod within ~16 GB RAM (story 3X.4).

No synthetic IV quotes are ever constructed; every output row is an
observed market leg. See ADR 0006 §"Rejected alternatives" / Option B.

Usage
-----
::

    # Real run (writes parquet + manifest)
    python src/data/05_build_otm_surface.py

    # Custom paths / dry-run (no output parquet, just manifest to stdout)
    python src/data/05_build_otm_surface.py \\
        --input  path/to/fixture_strict.parquet \\
        --output path/to/fixture_strict_otm.parquet \\
        --atm-band 0.0025 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Allow importing ``config`` whether invoked as a script from the repo
# root, from ``src/data/``, or via ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    PROJECT_ROOT,
    SURFACE_POINTS_STRICT_FILE,
    SURFACE_POINTS_STRICT_OTM_FILE,
    ATM_BAND_ABS_LOG_MONEYNESS,
    SMALL_EPS,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Round-decimals for the residual D7 key. Mirrors the audit script's
# 10-dp setting, which collapses float32 round-trip noise without
# merging genuinely distinct points.
RESIDUAL_KEY_DECIMALS = 10

# Sensitivity bands reported in the manifest (D5).
ATM_BAND_SENSITIVITY = (1e-12, 0.001, 0.0025, 0.005)

# Default streaming chunk size for the input reader.
DEFAULT_CHUNK_ROWS = 500_000

# Deterministic tie-break declaration: when ATM tie-break cannot pick a
# side (tie on relative spread or missing quote fields), pick this leg.
FALLBACK_TYPE = "put"

# Default residual-CSV location, relative to repo root.
DEFAULT_RESIDUAL_CSV = (
    PROJECT_ROOT / "artifacts" / "audits" / "otm_residual_same_type.csv"
)


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------


@dataclass
class BuildManifest:
    """Aggregated build statistics + provenance."""

    input_path: str
    output_path: str
    atm_band: float

    rows_in: int = 0
    rows_out: int = 0

    dropped_wrong_type: int = 0           # non-ATM rows whose type didn't match
    atm_groups: int = 0                   # near-ATM (date,exp,strike) groups
    atm_rows_seen: int = 0
    atm_rows_kept: int = 0
    atm_fallback_count: int = 0           # ATM groups resolved by declared fallback

    residual_groups: int = 0              # D7 groups with >1 row at the (date,log_m,tau) key
    residual_rows_in: int = 0
    residual_rows_dropped_equivalent: int = 0
    residual_rows_quality_picked: int = 0
    residual_keep_first_fallback: int = 0  # D7 final-fallback count

    atm_sensitivity: dict = field(default_factory=dict)

    input_sha256: str = ""
    output_sha256: str = ""

    dates_processed: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "atm_band": self.atm_band,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_wrong_type": self.dropped_wrong_type,
            "atm_groups": self.atm_groups,
            "atm_rows_seen": self.atm_rows_seen,
            "atm_rows_kept": self.atm_rows_kept,
            "atm_fallback_count": self.atm_fallback_count,
            "residual_groups": self.residual_groups,
            "residual_rows_in": self.residual_rows_in,
            "residual_rows_dropped_equivalent": self.residual_rows_dropped_equivalent,
            "residual_rows_quality_picked": self.residual_rows_quality_picked,
            "residual_keep_first_fallback": self.residual_keep_first_fallback,
            "atm_sensitivity": self.atm_sensitivity,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "dates_processed": self.dates_processed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def sha256_of_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path`` read in ``chunk_bytes`` chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blob = f.read(chunk_bytes)
            if not blob:
                break
            h.update(blob)
    return h.hexdigest()


def iter_by_date(
    parquet_path: Path,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per distinct ``date`` from a parquet file.

    The file is assumed to be (approximately) sorted by date — exactly
    the assumption used by ``scripts/audit_duplicate_coordinates.py``.
    The trailing partial date is buffered across batches so a date
    split across two batches is yielded as a single chunk.
    """
    pf = pq.ParquetFile(str(parquet_path))

    buf: pd.DataFrame | None = None
    for batch in pf.iter_batches(batch_size=chunk_rows):
        df = batch.to_pandas()
        if "date" not in df.columns:
            raise ValueError(
                f"{parquet_path.name}: missing required column 'date'"
            )
        df["date"] = pd.to_datetime(df["date"])
        if buf is not None:
            df = pd.concat([buf, df], ignore_index=True)
            buf = None
        if df.empty:
            continue

        date_vals = df["date"].values
        change_mask = np.empty(len(df), dtype=bool)
        change_mask[0] = True
        change_mask[1:] = date_vals[1:] != date_vals[:-1]
        change_idx = np.flatnonzero(change_mask)
        run_starts = list(change_idx) + [len(df)]

        # All but the last run are complete (within this batch). The
        # last run may continue into the next batch, so buffer it.
        for i in range(len(run_starts) - 2):
            lo, hi = run_starts[i], run_starts[i + 1]
            yield df.iloc[lo:hi].reset_index(drop=True)
        buf = df.iloc[run_starts[-2]:].reset_index(drop=True)

    if buf is not None and not buf.empty:
        yield buf


def _relative_spread(df: pd.DataFrame) -> np.ndarray:
    """Vectorised ``(ask - bid) / max(mid, eps)``; NaN where quotes missing."""
    ask = pd.to_numeric(df.get("ask"), errors="coerce").to_numpy(dtype=float)
    bid = pd.to_numeric(df.get("bid"), errors="coerce").to_numpy(dtype=float)
    if "mid" in df.columns:
        mid = pd.to_numeric(df["mid"], errors="coerce").to_numpy(dtype=float)
    else:
        mid = (ask + bid) / 2.0
    denom = np.maximum(mid, SMALL_EPS)
    spread = ask - bid
    out = spread / denom
    out[np.isnan(spread) | np.isnan(denom)] = np.nan
    return out


def select_otm_for_date(
    df: pd.DataFrame,
    atm_band: float,
    manifest: BuildManifest,
) -> pd.DataFrame:
    """Apply the OTM rule to one date's rows.

    Mutates ``manifest`` for per-rule counters.
    """
    log_m = pd.to_numeric(df["log_moneyness"], errors="coerce").to_numpy(dtype=float)

    # --- ATM-band sensitivity (D5) ------------------------------------
    for band in ATM_BAND_SENSITIVITY:
        key = f"atm_rows_at_band_{band:g}"
        manifest.atm_sensitivity[key] = (
            manifest.atm_sensitivity.get(key, 0) + int((np.abs(log_m) <= band).sum())
        )

    # --- Split rows into OTM-side strict and ATM tie-break ------------
    atm_mask = np.abs(log_m) <= atm_band
    non_atm = df.loc[~atm_mask].copy()
    atm = df.loc[atm_mask].copy()

    # Non-ATM: keep only the row whose type matches the OTM convention.
    type_vals = non_atm["type"].astype(str).str.lower()
    nlog_m = log_m[~atm_mask]
    want_put = nlog_m < -atm_band
    want_call = nlog_m > +atm_band
    keep_mask = (want_put & (type_vals == "put").to_numpy()) | (
        want_call & (type_vals == "call").to_numpy()
    )
    dropped_non_atm = int((~keep_mask).sum())
    manifest.dropped_wrong_type += dropped_non_atm
    non_atm_kept = non_atm.loc[keep_mask].copy()

    # ATM tie-break: group by (date,expiration,strike), pick tighter
    # spread; ties / NaN spreads fall to FALLBACK_TYPE.
    manifest.atm_rows_seen += len(atm)
    if not atm.empty:
        atm["_rel_spread"] = _relative_spread(atm)
        # Higher type-priority wins ties — fallback-type rows sort last
        # ascending, so we sort with priority **descending** to break
        # the tie in their favour.
        atm["_type_priority"] = (
            atm["type"].astype(str).str.lower() == FALLBACK_TYPE
        ).astype(int)
        # Spread NaN must lose to a finite spread → push NaNs to the bottom.
        atm_sorted = atm.sort_values(
            ["date", "expiration", "strike", "_rel_spread", "_type_priority"],
            ascending=[True, True, True, True, False],
            kind="mergesort",   # stable
            na_position="last",
        )
        group_keys = ["date", "expiration", "strike"]
        atm_kept = atm_sorted.drop_duplicates(subset=group_keys, keep="first")

        # Count groups + fallback usage: a group used the declared
        # fallback if its top row's spread is NaN OR if there are ≥ 2
        # rows whose spread ties the winner.
        group_sizes = atm_sorted.groupby(group_keys, sort=False).size()
        manifest.atm_groups += int(len(group_sizes))
        # Detect fallback per group: compare top row spread vs others.
        # Cheap heuristic — count groups where the winning row's spread
        # is NaN OR tied with at least one other row's spread.
        fallback = 0
        for _, grp in atm_sorted.groupby(group_keys, sort=False):
            if len(grp) <= 1:
                continue
            winner_spread = grp["_rel_spread"].iloc[0]
            if pd.isna(winner_spread):
                fallback += 1
                continue
            others = grp["_rel_spread"].iloc[1:].to_numpy()
            if np.any(others == winner_spread):
                fallback += 1
        manifest.atm_fallback_count += fallback

        atm_kept = atm_kept.drop(columns=["_rel_spread", "_type_priority"])
        manifest.atm_rows_kept += len(atm_kept)
    else:
        atm_kept = atm

    return pd.concat([non_atm_kept, atm_kept], ignore_index=True)


# Columns checked for D7 economic equivalence (within float tolerance).
_EQUIV_COLS = ("bid", "ask", "mid", "implied_volatility", "volume", "open_interest")
_EQUIV_TOL = 1e-8


def _is_economically_equivalent(group: pd.DataFrame) -> bool:
    """All rows agree on the economic columns (within tolerance / equal NaN)."""
    for col in _EQUIV_COLS:
        if col not in group.columns:
            continue
        vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        finite = ~np.isnan(vals)
        if finite.sum() == 0:
            continue
        if finite.sum() != len(vals):
            return False   # some NaN, some finite → not equivalent
        if not np.all(np.abs(vals - vals[0]) <= _EQUIV_TOL):
            return False
    return True


def _quality_rank(group: pd.DataFrame) -> pd.DataFrame:
    """Stable sort: tighter spread first, then higher OI, then volume."""
    g = group.copy()
    g["_rel_spread"] = _relative_spread(g)
    oi = pd.to_numeric(g.get("open_interest"), errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(g.get("volume"), errors="coerce").to_numpy(dtype=float)
    g["_neg_oi"] = -np.nan_to_num(oi, nan=-np.inf)
    g["_neg_vol"] = -np.nan_to_num(vol, nan=-np.inf)
    out = g.sort_values(
        ["_rel_spread", "_neg_oi", "_neg_vol"],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    )
    return out.drop(columns=["_rel_spread", "_neg_oi", "_neg_vol"])


def resolve_residuals(
    df: pd.DataFrame,
    manifest: BuildManifest,
    residual_writer: "ResidualCsvWriter | None" = None,
) -> pd.DataFrame:
    """Apply D7 to one date's OTM-kept rows. Returns the deduped frame."""
    if df.empty:
        return df

    keyed = df.copy()
    keyed["_lm_key"] = pd.to_numeric(keyed["log_moneyness"], errors="coerce").round(
        RESIDUAL_KEY_DECIMALS
    )
    keyed["_tau_key"] = pd.to_numeric(keyed["tau"], errors="coerce").round(
        RESIDUAL_KEY_DECIMALS
    )
    counts = keyed.groupby(["date", "_lm_key", "_tau_key"], sort=False).size()
    dup_keys = counts[counts > 1].index
    if len(dup_keys) == 0:
        return keyed.drop(columns=["_lm_key", "_tau_key"])

    idx_obj = keyed.set_index(["date", "_lm_key", "_tau_key"]).index
    dup_mask = idx_obj.isin(dup_keys)
    dup_rows = keyed.loc[dup_mask]
    clean_rows = keyed.loc[~dup_mask].drop(columns=["_lm_key", "_tau_key"])

    manifest.residual_groups += int(len(dup_keys))
    manifest.residual_rows_in += int(len(dup_rows))

    if residual_writer is not None and not dup_rows.empty:
        residual_writer.write(dup_rows.drop(columns=["_lm_key", "_tau_key"]))

    picked_chunks: list[pd.DataFrame] = [clean_rows]
    for _, group in dup_rows.groupby(["date", "_lm_key", "_tau_key"], sort=False):
        gnk = group.drop(columns=["_lm_key", "_tau_key"])
        if _is_economically_equivalent(gnk):
            sorted_g = gnk.sort_values(["type"], kind="mergesort")
            picked_chunks.append(sorted_g.iloc[[0]])
            manifest.residual_rows_dropped_equivalent += len(gnk) - 1
            continue
        ranked = _quality_rank(gnk)
        # If the winner ties on relative spread with at least one other
        # row AND has equal OI / volume, that's the documented final
        # fallback (``keep='first'`` after stable sort).
        winner = ranked.iloc[0]
        others = ranked.iloc[1:]
        if not others.empty:
            w_spread = _relative_spread(ranked.iloc[[0]])[0]
            o_spreads = _relative_spread(others)
            spread_tie = np.any(
                (o_spreads == w_spread)
                | (np.isnan(o_spreads) & np.isnan(w_spread))
            )
            if spread_tie:
                manifest.residual_keep_first_fallback += 1
        picked_chunks.append(ranked.iloc[[0]])
        manifest.residual_rows_quality_picked += 1

    return pd.concat(picked_chunks, ignore_index=True)


class ResidualCsvWriter:
    """Append-only CSV writer for D7 same-type residuals."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialised = False

    def write(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            self.path,
            mode="w" if not self._initialised else "a",
            header=not self._initialised,
            index=False,
        )
        self._initialised = True


# ----------------------------------------------------------------------
# Main build loop
# ----------------------------------------------------------------------


def build_otm(
    input_path: Path,
    output_path: Path,
    atm_band: float = ATM_BAND_ABS_LOG_MONEYNESS,
    dry_run: bool = False,
    residual_csv: Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    verbose: bool = True,
) -> BuildManifest:
    """Stream-build the OTM strict surface; return the populated manifest."""
    if not input_path.exists():
        raise FileNotFoundError(f"input parquet not found: {input_path}")

    t0 = time.time()
    manifest = BuildManifest(
        input_path=str(input_path),
        output_path=str(output_path) if not dry_run else "(dry-run)",
        atm_band=float(atm_band),
    )

    residual_writer: ResidualCsvWriter | None = None
    if not dry_run and residual_csv is not None:
        # Truncate any prior file by deleting it; will be re-created on first write.
        if residual_csv.exists():
            residual_csv.unlink()
        residual_writer = ResidualCsvWriter(residual_csv)

    writer: pq.ParquetWriter | None = None

    for date_df in iter_by_date(input_path, chunk_rows=chunk_rows):
        manifest.rows_in += len(date_df)
        manifest.dates_processed += 1

        otm = select_otm_for_date(date_df, atm_band=atm_band, manifest=manifest)
        deduped = resolve_residuals(
            otm, manifest=manifest, residual_writer=residual_writer
        )
        if deduped.empty:
            continue

        manifest.rows_out += len(deduped)

        if dry_run:
            continue

        # Stable column order: preserve input schema.
        deduped = deduped[list(date_df.columns)]
        table = pa.Table.from_pandas(deduped, preserve_index=False)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(output_path), table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    manifest.input_sha256 = sha256_of_file(input_path)
    if not dry_run and output_path.exists():
        manifest.output_sha256 = sha256_of_file(output_path)
    manifest.elapsed_seconds = time.time() - t0

    if verbose:
        print(json.dumps(manifest.to_dict(), indent=2, default=str))

    return manifest


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=SURFACE_POINTS_STRICT_FILE,
        help=f"input strict parquet (default: {SURFACE_POINTS_STRICT_FILE})",
    )
    p.add_argument(
        "--output", type=Path, default=SURFACE_POINTS_STRICT_OTM_FILE,
        help=f"output OTM parquet (default: {SURFACE_POINTS_STRICT_OTM_FILE})",
    )
    p.add_argument(
        "--atm-band", type=float, default=ATM_BAND_ABS_LOG_MONEYNESS,
        help=(
            "half-width of the near-ATM band on |log_moneyness| "
            f"(default: {ATM_BAND_ABS_LOG_MONEYNESS})"
        ),
    )
    p.add_argument(
        "--residual-csv", type=Path, default=DEFAULT_RESIDUAL_CSV,
        help="path for D7 same-type residual CSV "
             f"(default: {DEFAULT_RESIDUAL_CSV})",
    )
    p.add_argument(
        "--manifest", type=Path, default=None,
        help="optional path to write the manifest JSON sidecar",
    )
    p.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
        help="streaming batch size in rows",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="process the input + print the manifest, but do not write parquet",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="do not print the manifest to stdout",
    )
    args = p.parse_args()

    manifest = build_otm(
        input_path=args.input,
        output_path=args.output,
        atm_band=args.atm_band,
        dry_run=args.dry_run,
        residual_csv=args.residual_csv if not args.dry_run else None,
        chunk_rows=args.chunk_rows,
        verbose=not args.quiet,
    )

    if args.manifest is not None and not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ OTM build failed: {e}", file=sys.stderr)
        raise
