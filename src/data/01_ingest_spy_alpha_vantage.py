#!/usr/bin/env python3
"""Alpha Vantage HISTORICAL_OPTIONS ingest (story 2C.6).

Replaces the defunct Dubach Parquet ingest (`01_ingest_spy_github_dataset.py`).
Pulls SPY option chains one date at a time from Alpha Vantage's paid Standard
tier (75 req/min) and writes a single Parquet matching the pipeline's
`REQUIRED_OPTIONS_COLS` schema. Underlying prices still come from `yfinance`
with a dynamic end-date.

Local (2C.6) usage: small-sample validation, e.g.

    python src/data/01_ingest_spy_alpha_vantage.py \\
        --dates 2024-01-05 2026-05-15 \\
        --out-options /tmp/spy_options_sample.parquet

Full pull (2C.7, RunPod only):

    python src/data/01_ingest_spy_alpha_vantage.py --full

API key: provided via the environment variable `ALPHAVANTAGE_API_KEY`. The
key is never written to a tracked file and is redacted in any logged URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests  # type: ignore

# When run via `python src/data/01_ingest_spy_alpha_vantage.py`, Python adds
# the script's parent directory to sys.path so `from config import ...` works
# without packaging.
from config import (
    ALPHAVANTAGE_API_KEY_ENV,
    ALPHAVANTAGE_BASE_URL,
    ALPHAVANTAGE_FUNCTION,
    ALPHAVANTAGE_RATE_LIMIT_PER_MIN,
    INGEST_START_DATE,
    RAW_DIR,
    RAW_OPTIONS_FILE,
    RAW_UNDERLYING_FILE,
    REPORTS_DIR,
    REQUIRED_OPTIONS_COLS,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AuthError(RuntimeError):
    """Raised on HTTP 401 or any premium/auth-related payload from Alpha Vantage.

    These are not retriable — fail hard and surface the error to the operator.
    """


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter: at most ``max_per_min`` acquires per 60s."""

    def __init__(self, max_per_min: int = ALPHAVANTAGE_RATE_LIMIT_PER_MIN):
        if max_per_min <= 0:
            raise ValueError("max_per_min must be > 0")
        self.max_per_min = max_per_min
        self._window: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        window = 60.0
        # Drop timestamps older than the window
        while self._window and (now - self._window[0]) > window:
            self._window.popleft()
        if len(self._window) >= self.max_per_min:
            sleep_for = window - (now - self._window[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
            # After sleeping, prune again
            now = time.monotonic()
            while self._window and (now - self._window[0]) > window:
                self._window.popleft()
        self._window.append(time.monotonic())


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


_KEY_RE = re.compile(r"(apikey=)([^&\s]+)", flags=re.IGNORECASE)


def redact_key(text: str, key: str | None = None) -> str:
    """Redact an API key from a URL or log message.

    Replaces any literal occurrence of ``key`` and any ``apikey=...`` query
    parameter value with ``***``.
    """
    out = text
    if key:
        out = out.replace(key, "***")
    out = _KEY_RE.sub(r"\1***", out)
    return out


def require_api_key() -> str:
    """Read the AV API key from the env. Raises if absent or empty."""
    key = os.environ.get(ALPHAVANTAGE_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{ALPHAVANTAGE_API_KEY_ENV} env var is required; "
            "the API key must never be written to a tracked file."
        )
    return key


def _is_auth_payload(body: dict) -> bool:
    """Return True if the JSON body indicates an auth/premium failure."""
    for k in ("Information", "Note", "Error Message"):
        if k in body:
            msg = str(body[k]).lower()
            if any(s in msg for s in ("premium", "api key", "invalid", "unauthor")):
                return True
    return False


def fetch_one_date(
    date: str,
    api_key: str,
    base_url: str = ALPHAVANTAGE_BASE_URL,
    max_retries: int = 5,
    backoff_base: float = 1.5,
    rate_limiter: RateLimiter | None = None,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Fetch one date's HISTORICAL_OPTIONS and return a parsed DataFrame.

    Retries on transient HTTP 5xx / network errors with exponential backoff;
    hard-fails on HTTP 401 / 403 or any premium-endpoint payload.
    """
    params = {
        "function": ALPHAVANTAGE_FUNCTION,
        "symbol": "SPY",
        "date": date,
        "apikey": api_key,
    }
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            resp = requests.get(base_url, params=params, timeout=timeout_s)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(backoff_base ** attempt)
            continue

        status = resp.status_code
        if status in (401, 403):
            raise AuthError(
                f"Alpha Vantage rejected the request for {date}: HTTP {status}. "
                f"Body: {redact_key(resp.text, api_key)[:200]}"
            )
        if status >= 500:
            last_exc = RuntimeError(f"HTTP {status}")
            time.sleep(backoff_base ** attempt)
            continue
        if status != 200:
            raise RuntimeError(
                f"Alpha Vantage HTTP {status} for {date}: "
                f"{redact_key(resp.text, api_key)[:200]}"
            )

        try:
            body = resp.json()
        except Exception:
            last_exc = RuntimeError("invalid JSON")
            time.sleep(backoff_base ** attempt)
            continue

        if _is_auth_payload(body):
            raise AuthError(
                f"Alpha Vantage refused the request for {date}: {body}"
            )

        return parse_response_to_df(body)

    raise RuntimeError(
        f"Alpha Vantage fetch failed for {date} after {max_retries} retries: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_FLOAT_COLS = (
    "strike", "bid", "ask", "last", "mark",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
)
_INT_COLS = ("bid_size", "ask_size", "volume", "open_interest")


def _to_float(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(x: object) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def parse_response_to_df(body: dict) -> pd.DataFrame:
    """Convert an AV HISTORICAL_OPTIONS payload to the pipeline schema.

    All numeric fields are coerced from AV's string values to typed Python
    primitives. Date columns are parsed to pandas datetimes. Optional fields
    (greeks, sizes, contractID, etc.) are kept when present.
    """
    rows = body.get("data") or []
    if not rows:
        # Build an empty frame with the required columns so downstream
        # concatenation does not crash on no-data days.
        empty: dict[str, list] = {c: [] for c in REQUIRED_OPTIONS_COLS}
        return pd.DataFrame(empty)

    out: dict[str, list] = {}
    for col in (
        "date", "expiration", "strike", "type",
        "bid", "ask", "volume", "open_interest", "implied_volatility",
        "contractID", "symbol", "last", "mark",
        "bid_size", "ask_size",
        "delta", "gamma", "theta", "vega", "rho",
    ):
        out[col] = []

    for r in rows:
        out["date"].append(r.get("date"))
        out["expiration"].append(r.get("expiration"))
        out["strike"].append(_to_float(r.get("strike")))
        out["type"].append(str(r.get("type", "")).lower())
        out["bid"].append(_to_float(r.get("bid")))
        out["ask"].append(_to_float(r.get("ask")))
        out["volume"].append(_to_int(r.get("volume")))
        out["open_interest"].append(_to_int(r.get("open_interest")))
        out["implied_volatility"].append(_to_float(r.get("implied_volatility")))

        out["contractID"].append(r.get("contractID"))
        out["symbol"].append(r.get("symbol"))
        out["last"].append(_to_float(r.get("last")))
        out["mark"].append(_to_float(r.get("mark")))
        out["bid_size"].append(_to_int(r.get("bid_size")))
        out["ask_size"].append(_to_int(r.get("ask_size")))
        for g in ("delta", "gamma", "theta", "vega", "rho"):
            out[g].append(_to_float(r.get(g)))

    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


# ---------------------------------------------------------------------------
# Underlying via yfinance (dynamic end)
# ---------------------------------------------------------------------------


def download_underlying(out_path: Path) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance not installed; pip install yfinance") from e

    today = datetime.now().date().isoformat()
    df = yf.Ticker("SPY").history(start="2008-01-01", end=today, auto_adjust=False)
    df = df.reset_index()
    df = df.rename(
        columns={
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
    )
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["symbol"] = "SPY"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestSummary:
    dates_requested: int
    dates_returned: int
    rows: int
    date_min: str | None
    date_max: str | None
    out_path: str


def pull_dates(
    dates: Iterable[str],
    api_key: str,
    out_path: Path,
    base_url: str = ALPHAVANTAGE_BASE_URL,
) -> IngestSummary:
    """Fetch a list of trading dates and write a single options Parquet."""
    limiter = RateLimiter()
    frames: list[pd.DataFrame] = []
    n_dates = 0
    for date in dates:
        n_dates += 1
        df = fetch_one_date(date, api_key=api_key, base_url=base_url, rate_limiter=limiter)
        if len(df):
            frames.append(df)

    if not frames:
        out_df = pd.DataFrame({c: [] for c in REQUIRED_OPTIONS_COLS})
    else:
        out_df = pd.concat(frames, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    date_min = str(out_df["date"].min().date()) if len(out_df) else None
    date_max = str(out_df["date"].max().date()) if len(out_df) else None
    return IngestSummary(
        dates_requested=n_dates,
        dates_returned=int(out_df["date"].nunique()) if len(out_df) else 0,
        rows=len(out_df),
        date_min=date_min,
        date_max=date_max,
        out_path=str(out_path),
    )


def write_metadata(summary: IngestSummary) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / "ingest_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "alphavantage HISTORICAL_OPTIONS",
        "date_downloaded": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "rows": summary.rows,
            "dates_requested": summary.dates_requested,
            "dates_returned": summary.dates_returned,
            "date_min": summary.date_min,
            "date_max": summary.date_max,
            "out_path": summary.out_path,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    report_path = REPORTS_DIR / "spy_ingest_summary.md"
    lines = [
        "# SPY Data Ingestion Summary",
        "",
        f"**Downloaded**: {meta['date_downloaded']}",
        f"**Source**: {meta['source']}",
        "",
        f"- Rows: {summary.rows:,}",
        f"- Dates requested: {summary.dates_requested}",
        f"- Dates returned: {summary.dates_returned}",
        f"- Date range: {summary.date_min} → {summary.date_max}",
        f"- Output: `{summary.out_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Explicit list of YYYY-MM-DD trading dates to pull (sample mode).",
    )
    parser.add_argument(
        "--start", default=INGEST_START_DATE,
        help="Range mode start date (only used with --full or --range).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Range mode end date (defaults to today).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Pull every trading day in [start, end]. Reserved for 2C.7 (remote).",
    )
    parser.add_argument(
        "--out-options", default=str(RAW_OPTIONS_FILE),
        help="Output Parquet path for the options chain.",
    )
    parser.add_argument(
        "--with-underlying", action="store_true",
        help="Also refresh the SPY underlying via yfinance (dynamic end).",
    )
    args = parser.parse_args(argv)

    api_key = require_api_key()

    if args.dates:
        dates = list(args.dates)
    elif args.full or args.end is not None:
        # Trading-day enumeration via pandas business-day range (good enough
        # for the local sample; the remote run will use an exchange calendar).
        end = args.end or datetime.now().date().isoformat()
        dates = [
            d.date().isoformat()
            for d in pd.bdate_range(start=args.start, end=end)
        ]
    else:
        parser.error("provide --dates ... or --full or --end")

    print(
        f"[ingest] fetching {len(dates)} date(s) "
        f"({dates[0]} → {dates[-1]}) → {args.out_options}"
    )
    summary = pull_dates(
        dates=dates,
        api_key=api_key,
        out_path=Path(args.out_options),
    )
    print(
        f"[ingest] {summary.rows:,} rows from {summary.dates_returned} "
        f"date(s); span {summary.date_min} → {summary.date_max}"
    )

    if args.with_underlying:
        print("[underlying] fetching SPY from yfinance ...")
        u_df = download_underlying(Path(RAW_UNDERLYING_FILE))
        print(f"[underlying] {len(u_df):,} rows -> {RAW_UNDERLYING_FILE}")

    write_metadata(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
