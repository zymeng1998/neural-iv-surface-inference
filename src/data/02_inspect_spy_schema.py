#!/usr/bin/env python3
"""
Step 2: Inspect SPY raw data — schema, dtypes, quality, suspicious rows.

Produces: reports/spy_schema_report.md

Usage:
    python src/data/02_inspect_spy_schema.py
"""

import sys

import pyarrow.parquet as pq
import pandas as pd

from config import (
    RAW_OPTIONS_FILE, RAW_UNDERLYING_FILE,
    REQUIRED_OPTIONS_COLS, REQUIRED_UNDERLYING_COLS,
    REPORTS_DIR,
)


def inspect_file(path, label: str, required_cols: list[str]) -> list[str]:
    """Inspect a single Parquet file, return report lines."""
    lines = [f"## {label}\n", f"**File**: `{path}`\n"]

    if not path.exists():
        lines.append(f"**ERROR**: file not found — run step 01 first.\n")
        return lines

    df = pd.read_parquet(path)
    lines.append(f"**Rows**: {len(df):,}")
    lines.append(f"**Columns**: {len(df.columns)}\n")

    # ── Schema ───────────────────────────────────────────────────────
    lines.append("### Column Schema\n")
    lines.append("| Column | Dtype | Non-null | Null % |")
    lines.append("|--------|-------|----------|--------|")
    for col in df.columns:
        n_nonnull = df[col].notna().sum()
        null_pct = (1 - n_nonnull / len(df)) * 100 if len(df) > 0 else 0
        lines.append(f"| `{col}` | {df[col].dtype} | {n_nonnull:,} | {null_pct:.2f}% |")
    lines.append("")

    # ── Required columns check ───────────────────────────────────────
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        lines.append(f"**⚠ MISSING REQUIRED COLUMNS**: {missing_required}\n")
    else:
        lines.append("**✓ All required columns present.**\n")

    # ── Key ranges ───────────────────────────────────────────────────
    lines.append("### Key Ranges\n")
    range_cols = {
        "date": True, "expiration": True,
        "strike": False, "implied_volatility": False,
        "bid": False, "ask": False,
        "close": False,
    }
    for col, is_date in range_cols.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            lines.append(f"- `{col}`: all null")
            continue
        lines.append(f"- `{col}`: min={s.min()}, max={s.max()}")
    lines.append("")

    # ── Distinct option types ────────────────────────────────────────
    if "type" in df.columns:
        types = df["type"].dropna().unique().tolist()
        lines.append(f"### Option Types\n")
        lines.append(f"Distinct values: `{types}`\n")

    # ── Suspicious rows ──────────────────────────────────────────────
    lines.append("### Suspicious Row Counts\n")
    lines.append("| Check | Count | % of total |")
    lines.append("|-------|------:|----------:|")

    checks = {}
    if "bid" in df.columns:
        checks["bid < 0"] = (df["bid"] < 0).sum()
    if "ask" in df.columns:
        checks["ask < 0"] = (df["ask"] < 0).sum()
    if "bid" in df.columns and "ask" in df.columns:
        checks["ask < bid"] = (df["ask"] < df["bid"]).sum()
    if "implied_volatility" in df.columns:
        checks["IV <= 0"] = (df["implied_volatility"] <= 0).sum()
        checks["IV > 5.0"] = (df["implied_volatility"] > 5.0).sum()
    for col in ["date", "expiration", "strike", "type"]:
        if col in df.columns:
            checks[f"{col} is null"] = df[col].isna().sum()

    for desc, count in checks.items():
        pct = count / len(df) * 100 if len(df) > 0 else 0
        lines.append(f"| {desc} | {count:,} | {pct:.3f}% |")
    lines.append("")

    return lines


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = ["# SPY Schema & Quality Report\n"]

    print("Inspecting SPY options data...")
    report_lines += inspect_file(
        RAW_OPTIONS_FILE, "SPY Options", REQUIRED_OPTIONS_COLS
    )

    print("Inspecting SPY underlying data...")
    report_lines += inspect_file(
        RAW_UNDERLYING_FILE, "SPY Underlying", REQUIRED_UNDERLYING_COLS
    )

    # ── Write report ─────────────────────────────────────────────────
    report_path = REPORTS_DIR / "spy_schema_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n✓ Report written to {report_path}")

    # Also print to stdout
    print("\n" + "=" * 60)
    print("\n".join(report_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Inspection failed: {e}", file=sys.stderr)
        raise
