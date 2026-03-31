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

# ── Data source URLs (Philipp Dubach, broader repo) ─────────────────
SPY_OPTIONS_URL    = "https://static.philippdubach.com/data/options/spy/options.parquet"
SPY_UNDERLYING_URL = "https://static.philippdubach.com/data/options/spy/underlying.parquet"

# Fallback: ETF-focused repo (year-partitioned)
ETF_REPO_BASE = "https://github.com/philippdubach/options-dataset-hist"
ETF_PARQUET_YEARS = range(2008, 2026)  # 2008–2025

# ── Raw file names ───────────────────────────────────────────────────
RAW_OPTIONS_FILE    = RAW_DIR / "spy_options.parquet"
RAW_UNDERLYING_FILE = RAW_DIR / "spy_underlying.parquet"

# ── Processed file names ─────────────────────────────────────────────
SURFACE_POINTS_FILE        = PROCESSED_DIR / "spy_surface_points.parquet"
SURFACE_POINTS_STRICT_FILE = PROCESSED_DIR / "spy_surface_points_strict.parquet"

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
