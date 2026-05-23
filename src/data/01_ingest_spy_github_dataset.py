#!/usr/bin/env python3
"""
Step 1: Download SPY options + underlying data.

***************************************************************************
*  DEFUNCT (2026-05-22) — DO NOT RUN.                                     *
*                                                                         *
*  The upstream sources this script depends on are unreachable:           *
*    - https://static.philippdubach.com/data/options/spy/options.parquet  *
*      → HTTP 404 (verified with `curl --fail`, exit 56)                  *
*    - https://static.philippdubach.com/data/options/spy/underlying.parquet
*      → HTTP 404                                                         *
*    - https://github.com/philippdubach/options-dataset-hist              *
*      → repo absent from the maintainer's GitHub account                 *
*                                                                         *
*  The active SPY options data source is Alpha Vantage HISTORICAL_OPTIONS *
*  (paid Standard tier). See:                                             *
*    - docs/decisions/0003_spy_options_data_source_migration.md           *
*    - docs/tasks/specs/2C.6_remote_sync_data_refresh.md                  *
*                                                                         *
*  The replacement implementation is                                      *
*  `src/data/01_ingest_spy_alpha_vantage.py` (story 2C.6).                *
*  Story 2C.7 will git-remove this file once the new ingest has been      *
*  validated end-to-end on the remote.                                    *
***************************************************************************

Options source:     static.philippdubach.com (DEFUNCT)
Underlying source:  static.philippdubach.com first (DEFUNCT), then Yahoo
                    Finance fallback (still works for the underlying only;
                    the option chain cannot be recovered from yfinance)

Usage:
    (do not run — see the DEFUNCT notice above)
"""

import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import urllib.request

from config import (
    RAW_DIR, RAW_OPTIONS_FILE, RAW_UNDERLYING_FILE,
    SPY_OPTIONS_URL, SPY_UNDERLYING_URL,
    REPORTS_DIR,
)


def download_with_progress(url: str, dest, label: str) -> None:
    """Download a file with progress feedback."""
    print(f"[download] {label}")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    start = time.time()

    # Use urllib for progress tracking
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "neural-iv-surface-inference/0.1")

    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        last_report = 0

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                # Report every 50 MB
                if downloaded - last_report >= 50 * 1024 * 1024:
                    elapsed = time.time() - start
                    if total:
                        pct = downloaded / total * 100
                        print(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB "
                              f"({pct:.0f}%) — {elapsed:.0f}s elapsed")
                    else:
                        print(f"  {downloaded / 1e6:.0f} MB — {elapsed:.0f}s elapsed")
                    last_report = downloaded

    elapsed = time.time() - start
    size_mb = dest.stat().st_size / 1e6
    print(f"  Done: {size_mb:.1f} MB in {elapsed:.1f}s\n")


def summarize_parquet(path, label: str) -> dict:
    """Read Parquet metadata and return summary dict."""
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    schema = pf.schema_arrow

    # Read just enough to get date range
    tbl = pf.read(columns=["date"] if "date" in schema.names else [])
    dates = tbl.column("date").to_pylist() if "date" in schema.names else []

    summary = {
        "label": label,
        "file": str(path),
        "num_rows": meta.num_rows,
        "num_columns": meta.num_columns,
        "columns": schema.names,
        "column_types": {name: str(schema.field(name).type) for name in schema.names},
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }

    if dates:
        date_strs = [str(d) for d in dates if d is not None]
        if date_strs:
            summary["min_date"] = min(date_strs)
            summary["max_date"] = max(date_strs)

    return summary


def write_ingest_report(summaries: list[dict]) -> None:
    """Write a markdown + JSON summary of ingested data."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON metadata
    meta_path = RAW_DIR / "ingest_metadata.json"
    meta = {
        "source": "philippdubach/options-data (static.philippdubach.com)",
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
        "files": summaries,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[metadata] Written to {meta_path}")

    # Markdown report
    report_path = REPORTS_DIR / "spy_ingest_summary.md"
    lines = [
        "# SPY Data Ingestion Summary\n",
        f"**Downloaded**: {meta['date_downloaded']}\n",
        f"**Source**: {meta['source']}\n",
    ]
    for s in summaries:
        lines.append(f"\n## {s['label']}\n")
        lines.append(f"- Rows: {s['num_rows']:,}")
        lines.append(f"- Columns: {s['num_columns']}")
        lines.append(f"- Size: {s['size_mb']} MB")
        if "min_date" in s:
            lines.append(f"- Date range: {s['min_date']} → {s['max_date']}")
        lines.append(f"- Column list: `{'`, `'.join(s['columns'])}`\n")

    report_path.write_text("\n".join(lines))
    print(f"[report]   Written to {report_path}")


def download_underlying_with_fallback() -> None:
    """Download underlying prices; fall back to Yahoo Finance if Dubach file is empty."""
    if RAW_UNDERLYING_FILE.exists():
        # Check if existing file has rows
        pf = pq.ParquetFile(RAW_UNDERLYING_FILE)
        if pf.metadata.num_rows > 0:
            print(f"[skip] {RAW_UNDERLYING_FILE} already exists "
                  f"({pf.metadata.num_rows:,} rows)")
            return
        else:
            print("[warn] Existing underlying file has 0 rows, re-downloading...")

    # Try Dubach source first
    try:
        download_with_progress(
            SPY_UNDERLYING_URL, RAW_UNDERLYING_FILE, "SPY underlying.parquet (Dubach)"
        )
        pf = pq.ParquetFile(RAW_UNDERLYING_FILE)
        if pf.metadata.num_rows > 0:
            return
        print("[warn] Dubach underlying file is empty. Falling back to Yahoo Finance...")
    except Exception as e:
        print(f"[warn] Dubach underlying download failed: {e}")
        print("[fallback] Trying Yahoo Finance...")

    # Fallback: yfinance (handles Yahoo auth properly)
    print("[download] SPY underlying from Yahoo Finance via yfinance...")
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError(
            "yfinance not installed. Run: pip install yfinance"
        )

    ticker = yf.Ticker("SPY")
    # Dynamic end-date (was hardcoded "2026-01-01" — a known bug that would
    # silently cap the underlying series in the past after that date).
    # Kept for completeness even though this whole script is DEFUNCT.
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = ticker.history(start="2008-01-01", end=today_str, auto_adjust=False)
    df = df.reset_index()

    # Normalize column names to match expected schema
    col_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
        "Dividends": "dividend_amount",
        "Stock Splits": "split_coefficient",
    }
    df = df.rename(columns=col_map)

    # Ensure date is timezone-naive
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df["symbol"] = "SPY"

    # Add any missing expected columns as null
    for col in ["adjusted_close", "dividend_amount", "split_coefficient"]:
        if col not in df.columns:
            df[col] = None

    df.to_parquet(RAW_UNDERLYING_FILE, index=False)
    print(f"  Saved {len(df):,} rows to {RAW_UNDERLYING_FILE}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}\n")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download options ─────────────────────────────────────────────
    if RAW_OPTIONS_FILE.exists():
        print(f"[skip] {RAW_OPTIONS_FILE} already exists "
              f"({RAW_OPTIONS_FILE.stat().st_size / 1e6:.0f} MB)")
    else:
        download_with_progress(
            SPY_OPTIONS_URL, RAW_OPTIONS_FILE, "SPY options.parquet"
        )

    # ── Download underlying (with Yahoo fallback) ────────────────────
    download_underlying_with_fallback()

    # ── Summarize ────────────────────────────────────────────────────
    print("=" * 60)
    print("Generating ingestion summary...")
    summaries = [
        summarize_parquet(RAW_OPTIONS_FILE, "SPY Options"),
        summarize_parquet(RAW_UNDERLYING_FILE, "SPY Underlying"),
    ]

    for s in summaries:
        print(f"\n  {s['label']}: {s['num_rows']:,} rows, "
              f"{s['num_columns']} cols, {s['size_mb']} MB")
        if "min_date" in s:
            print(f"    Date range: {s['min_date']} → {s['max_date']}")

    write_ingest_report(summaries)
    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}", file=sys.stderr)
        raise
