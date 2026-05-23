# SPY Data Pipeline — Assumptions & Cleaning Rules

## Phase 1 Scope

- Underlying: SPY only
- Granularity: EOD only
- Source (Phase 1): Philipp Dubach historical options dataset *(now defunct
  — see below; replaced by Alpha Vantage on 2026-05-22 per ADR 0003)*

## Data Source

**Active (2026-05-22 onward — per [ADR 0003](decisions/0003_spy_options_data_source_migration.md), used by Phase 2C and all
subsequent training/eval runs):**

- **Primary**: Alpha Vantage `HISTORICAL_OPTIONS` — paid Standard tier
  ($49.99/month, 75 req/min, cancel-anytime). One HTTP request per trading
  date returns the full SPY option chain with `date, expiration, strike,
  type, bid, ask, bid_size, ask_size, volume, open_interest,
  implied_volatility, delta, gamma, theta, vega, rho, last, mark,
  contractID, symbol`. Values arrive as JSON strings and are parsed in
  ingest. API key from `os.environ["ALPHAVANTAGE_API_KEY"]`.
- **Underlying**: `yfinance` SPY daily close, end-date resolved dynamically
  to today at runtime.
- **Coverage**: 2008-01-02 → present (verified against the live endpoint on
  2026-05-22; 2026-05-15 returned 13,796 contracts).
- **Ingest script**: `src/data/01_ingest_spy_alpha_vantage.py` (story 2C.6).
  Full pull on the remote happens in story 2C.7.

**Defunct (Phase 1 evidence only — do not use):**

- ~~`https://static.philippdubach.com/data/options/spy/options.parquet`~~ — HTTP 404
- ~~`https://static.philippdubach.com/data/options/spy/underlying.parquet`~~ — HTTP 404
- Phase 1 coverage from this source: 2008-01-02 through 2025-12-16 (~24.7M rows).
  Replaced wholesale on the remote in story 2C.7 — the Dubach snapshot is
  deleted, not kept alongside the new data.

## Spot Price Convention

- `spot = close` (same-day closing price of SPY)
- Do NOT use `adjusted_close` as the spot for moneyness computation
- `adjusted_close`, `dividend_amount`, `split_coefficient` are kept as audit fields only
- Rationale: option strikes are quoted against unadjusted prices; using adjusted close would distort moneyness

## Derived Columns

| Column | Formula | Notes |
|--------|---------|-------|
| `mid` | `(bid + ask) / 2` | |
| `days_to_expiry` | `(expiration - date).days` | integer |
| `tau` | `days_to_expiry / 365.0` | annualized |
| `spot` | `close` | unadjusted |
| `log_moneyness` | `ln(strike / spot)` | positive = OTM call / ITM put |

## Hard Cleaning Rules (rows dropped)

| Rule | Threshold | Rationale |
|------|-----------|-----------|
| Null critical field | date, expiration, strike, type, IV, close | Cannot compute surface coordinates |
| Negative bid | `bid < 0` | Invalid quote |
| Negative ask | `ask < 0` | Invalid quote |
| Crossed quote | `ask < bid` | Market microstructure error |
| Non-positive IV | `IV <= 0` | Undefined or clearly wrong |
| Extreme IV | `IV > 5.0` (500%) | Likely data error |
| Expired option | `tau <= 0` | Already expired |
| Far-dated option | `tau > 3.0` (3 years) | Illiquid, unreliable IV |

## Quality Flags (kept, not dropped)

| Flag | Condition | Notes |
|------|-----------|-------|
| `flag_zero_bid` | `bid == 0` | Common for deep OTM; may want to exclude in strict subset |
| `flag_zero_volume` | `volume == 0` | No trades that day |
| `flag_zero_oi` | `open_interest == 0` | No existing positions |
| `flag_wide_spread` | `(ask - bid) / max(mid, ε) > 0.5` | Spread > 50% of mid |

## Strict Modeling Subset Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_iv` | 0.01 (1%) | Floor for reasonable volatility |
| `max_iv` | 3.0 (300%) | Ceiling for reasonable volatility |
| `min_tau` | 1/365 (~1 day) | At least 1 calendar day |
| `max_tau` | 2.0 (2 years) | Reasonable liquidity horizon |
| `min_log_moneyness` | -1.0 | Excludes extreme deep ITM puts |
| `max_log_moneyness` | 1.0 | Excludes extreme deep OTM |

These thresholds are first-pass conservative choices. They will be revisited after EDA.

## What Is NOT Done Yet

- No arbitrage-free smoothing or constraints
- No smile/skew interpolation
- No cross-sectional consistency checks
- No intraday data
- No multi-asset
