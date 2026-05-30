"""Centralized configuration for the SPY data pipeline."""

from pathlib import Path

# ── Project root (two levels up from this file) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Directory paths ──────────────────────────────────────────────────
RAW_DIR       = PROJECT_ROOT / "data_raw" / "spy"
PROCESSED_DIR = PROJECT_ROOT / "data_processed" / "spy"
REPORTS_DIR   = PROJECT_ROOT / "reports"
PLOTS_DIR     = PROJECT_ROOT / "plots"
LOGS_DIR      = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ── Data source URLs ─────────────────────────────────────────────────
#
# DEFUNCT (2026-05-22) — the Philipp Dubach static-Parquet source and the
# fallback GitHub LFS repo (`options-dataset-hist`) are both unreachable:
# the parquet URLs and the host's `/data/options/spy/` path return HTTP 404
# (verified with `curl --fail`, exit 56), and the GitHub repo is absent
# from the maintainer's account. The maintainer's current vol project uses
# live APIs instead. See:
#   - docs/decisions/0003_spy_options_data_source_migration.md
#   - docs/data/data_lineage.md §9 ("Upstream data source unreachable")
#
# These constants are retained ONLY as historical evidence; do NOT use them.
# Story 2C.7 will remove `01_ingest_spy_github_dataset.py` entirely.
SPY_OPTIONS_URL    = "https://static.philippdubach.com/data/options/spy/options.parquet"  # DEFUNCT
SPY_UNDERLYING_URL = "https://static.philippdubach.com/data/options/spy/underlying.parquet"  # DEFUNCT
ETF_REPO_BASE      = "https://github.com/philippdubach/options-dataset-hist"  # DEFUNCT
ETF_PARQUET_YEARS  = range(2008, 2026)  # historical Dubach coverage

# ── Active data source: Alpha Vantage HISTORICAL_OPTIONS ─────────────
#
# Chosen replacement source per ADR 0003. Verified 2026-05-22 against the
# paid Standard tier: returns SPY option chains 2008 → present with all
# required pipeline columns (date, expiration, strike, type, bid, ask,
# volume, open_interest, implied_volatility) plus greeks. Response is JSON;
# values arrive as strings and must be parsed.
#
# The implementation script is `src/data/01_ingest_spy_alpha_vantage.py`
# (story 2C.6). The API key MUST come from the environment variable below
# and must never be written to any tracked file.
ALPHAVANTAGE_BASE_URL: str             = "https://www.alphavantage.co/query"
ALPHAVANTAGE_FUNCTION: str             = "HISTORICAL_OPTIONS"
ALPHAVANTAGE_API_KEY_ENV: str          = "ALPHAVANTAGE_API_KEY"
ALPHAVANTAGE_RATE_LIMIT_PER_MIN: int   = 75   # Standard tier ($49.99/mo)
INGEST_START_DATE: str                 = "2008-01-02"  # first SPY trading day in scope
# INGEST_END_DATE: resolved dynamically at runtime to today's date — do NOT
# hardcode a calendar date here (the `yfinance` fallback's hardcoded
# end="2026-01-01" in the old ingest script is the kind of bug we are
# avoiding).

# ── Raw file names ───────────────────────────────────────────────────
RAW_OPTIONS_FILE    = RAW_DIR / "spy_options.parquet"
RAW_UNDERLYING_FILE = RAW_DIR / "spy_underlying.parquet"

# ── Processed file names ─────────────────────────────────────────────
SURFACE_POINTS_FILE             = PROCESSED_DIR / "spy_surface_points.parquet"
SURFACE_POINTS_STRICT_FILE      = PROCESSED_DIR / "spy_surface_points_strict.parquet"
SURFACE_POINTS_STRICT_OTM_FILE  = PROCESSED_DIR / "spy_surface_points_strict_otm.parquet"

# ── Schema: required columns (pipeline will fail if missing) ─────────
REQUIRED_OPTIONS_COLS = [
    "date", "expiration", "strike", "type",
    "bid", "ask", "volume", "open_interest",
    "implied_volatility",
]

REQUIRED_UNDERLYING_COLS = [
    "date", "close",
]

# ── Schema: optional columns (kept if present, no failure) ───────────
OPTIONAL_OPTIONS_COLS = [
    "contract_id", "symbol", "last", "mark",
    "bid_size", "ask_size",
    "delta", "gamma", "theta", "vega", "rho",
    "in_the_money",
]

OPTIONAL_UNDERLYING_COLS = [
    "symbol", "open", "high", "low",
    "adjusted_close", "volume",
    "dividend_amount", "split_coefficient",
]

# ── Cleaning thresholds ─────────────────────────────────────────────
# Hard drops
MIN_IV = 0.0       # drop IV <= 0
MAX_IV = 5.0       # drop IV > 500% (likely data error)
MIN_TAU = 0.0      # drop tau <= 0 (expired)
MAX_TAU = 3.0      # drop tau > 3 years (illiquid far-dated)

# Strict subset thresholds
STRICT_MIN_IV  = 0.01   # 1%
STRICT_MAX_IV  = 3.0    # 300%
STRICT_MIN_TAU = 1 / 365.0   # at least 1 day
STRICT_MAX_TAU = 2.0    # 2 years
STRICT_MIN_LOG_MONEYNESS = -1.0   # deep in-the-money puts
STRICT_MAX_LOG_MONEYNESS =  1.0   # deep OTM

# Wide spread flag threshold
WIDE_SPREAD_THRESHOLD = 0.5   # (ask - bid) / mid > 50%
SMALL_EPS = 1e-8              # avoid division by zero

# ── OTM-restricted surface (Phase 3X, ADR 0006 + D5) ────────────────
# Half-width of the near-ATM band on |log_moneyness|. Rows with
# |log_m| <= ATM_BAND_ABS_LOG_MONEYNESS fall into the tie-break path
# (pick the side with tighter relative spread; deterministic fallback
# = "put"). Builder reports sensitivity counts at
# {1e-12, 0.001, 0.0025, 0.005} in its manifest.
ATM_BAND_ABS_LOG_MONEYNESS = 0.0025
