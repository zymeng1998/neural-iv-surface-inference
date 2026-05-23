# ADR 0003: SPY Options Data Source Migration

## Status

Accepted

## Date

2026-05-22

## Context

The Phase 1 pipeline ingests SPY EOD option-chain data from a static Parquet
hosted by Philipp Dubach, configured in `src/data/config.py`:

- `https://static.philippdubach.com/data/options/spy/options.parquet`
- `https://static.philippdubach.com/data/options/spy/underlying.parquet`

with a documented fallback to `github.com/philippdubach/options-dataset-hist`
and a `yfinance` fallback for the underlying only.

On 2026-05-22, while preparing the Phase 2C remote data refresh (story 2C.6),
this source was found to be **discontinued**:

- Both Parquet URLs and the host path `/data/options/spy/` return HTTP 404
  (verified with `curl --fail`, exit 56, zero bytes).
- The dataset repositories (`options-data`, `options-dataset-hist`,
  `historic-options-dataset`) are absent from
  `api.github.com/users/philippdubach/repos`.
- The maintainer's current volatility project
  (`philippdubach/vol-regime-prediction`) pulls from **live APIs**
  (Alpha Vantage, CBOE, FRED, yfinance) instead of a hosted Parquet.

The original ingest covered 2008–2025; it is now 2026-05. The data is therefore
both stale and unrecoverable from its documented location. Full evidence is in
`docs/data/data_lineage.md` §9.

## Decision

Migrate the SPY options data source to **Alpha Vantage `HISTORICAL_OPTIONS`** as
the primary source, with **OptionsDX** as a no-cost fallback.

- **Primary — Alpha Vantage `HISTORICAL_OPTIONS`:** ~15+ years of EOD option
  chains (2008+) with implied volatility and Greeks; one request per trading
  date; requires an API key (free tier 25 req/day is impractical for full
  history; the ~$49.99/mo Standard tier at 75 req/min fetches the full history in
  ≈1 hour). This is the closest match to the original dataset's depth and is the
  provider the original maintainer himself migrated to.
- **Fallback — OptionsDX:** free EOD SPY option chains (registration), coverage
  from ~2010, distributed as downloadable monthly files. Used if a paid API key
  is unavailable; accepts the loss of 2008–2009 and slower update cadence.

> **OptionsDX downgraded to documentation only (2026-05-22):** verified that
> OptionsDX's downloadable SPY EOD data stops at **2023** (no 2024–2026), so it
> cannot satisfy "all data up to today." The QuantConnect AlgoSeek path (~$8–60
> seat + per-file QCC) was also considered but starts only in **2012**, with a
> two-dataset merge and LEAN→parquet ETL. Alpha Vantage Standard is the only
> path that delivers a **single consistent source from 2008 → present**, and is
> therefore the chosen source — no fallback is active.

Rejected for now: OptionsDX (2010–2023, fails recency), QuantConnect AlgoSeek
US Equity Options (2012+, two-dataset merge), DoltHub `dolthub/options`
(2019–2024 only — too short for Phase 1 parity), CBOE DataShop / Databento OPRA
(paid, intraday-grade, heavier than EOD needs), WRDS OptionMetrics IvyDB (gold
standard but requires institutional access — revisit if access is confirmed).

## Verification (2026-05-22)

The paid Alpha Vantage Standard key was provisioned and tested against two
dates (one mid-history, one recent). Results:

- `HISTORICAL_OPTIONS` returns real data on both dates (the free tier returns a
  premium-gated error). Standard tier confirmed active at **75 req/min**, full
  ~4,500-trading-day SPY history pulls in ≈1 hour.
- **2024-01-05:** 7,618 contracts; **2026-05-15** (most recent trading week):
  13,796 contracts. Coverage reaches the present trading week; density is
  roughly 2× the prior Dubach dataset.
- **Response shape:** `{endpoint, message, data: [...]}` where each contract in
  `data` has 20 string fields:
  `ask, ask_size, bid, bid_size, contractID, date, delta, expiration, gamma,
  implied_volatility, last, mark, open_interest, rho, strike, symbol, theta,
  type, vega, volume`.
- **Field mapping to our pipeline schema** (`REQUIRED_OPTIONS_COLS` in
  `src/data/config.py`): all required fields are present directly, named
  identically (`date, expiration, strike, type, bid, ask, volume,
  open_interest, implied_volatility`); greeks are provided but not consumed by
  the pipeline. All values are JSON strings — ingest must parse to
  float/int. The underlying `close` is fetched separately via the existing
  `yfinance` path (with `end` made dynamic).
- **Cost & cancellation:** $49.99/month, "Cancel anytime — no questions asked"
  per the AV pricing page. Pull-and-cancel pattern → one-time ≈ $50.
  Reminder created on the operator's calendar for 2026-06-18 (4 days before
  the ~2026-06-22 renewal).

## Rationale

- Preserves the longest feasible history (2008+) for comparability with the
  Phase 1 baselines and benchmark naming.
- Keeps re-fetch **scriptable and reproducible** (API key + loop over trading
  days) rather than dependent on a single third-party static file that can vanish
  again.
- Provides a free degradation path (OptionsDX) so the project is not hard-blocked
  on a paid subscription.

## Consequences

Positive:
- The pipeline no longer depends on a single fragile static URL.
- Re-fetch can be re-run on demand on the remote, satisfying the
  "re-runnable from scratch" data principle.

Trade-offs / follow-ups:
- Schema remap required: the Alpha Vantage / OptionsDX field names must be mapped
  into the pipeline's required schema (`date, expiration, strike, type, bid, ask,
  volume, open_interest, implied_volatility` + underlying `date, close`). Captured
  as the first implementation step of story 2C.6.
- Per-date paging (Alpha Vantage) makes a full historical pull a long-running,
  rate-limited job; output must be redirected to logs and cached incrementally.
- A human action is required before ingest code can be wired: provide an Alpha
  Vantage API key, or complete the OptionsDX registration/download.
- The `yfinance` underlying fallback hardcodes `end="2026-01-01"` and must be
  fixed to a current/dynamic date.
- Benchmark datasets (`spy_phase1_*`) must be regenerated against the new source;
  any row-count / date-range differences vs the Phase 1 snapshot are documented in
  the experiment journal and data lineage.
