# Data Lineage — Phase 1 (SPY EOD)

---
created_at: 2026-04-02T02:00:00-04:00
last_updated_at: 2026-05-22T21:30:00-04:00
---

> Repo-specific data lineage for the Neural IV Surface Inference project.
> Covers the raw → processed → modeling-ready data flow for Phase 1 (SPY-only, EOD-only).

---

## 1. Purpose

This document traces how data flows through the repository:

- Where raw data comes from
- How it is validated
- How it is transformed into modeling-ready surfaces
- What scripts, configs, and docs govern each step
- What decisions and constraints apply

It is grounded in actual repository evidence. Sections where the pipeline has not yet been fully executed on the current machine are marked explicitly.

---

## 2. Data layers

| Layer | Path | Contents | Persistence |
|---|---|---|---|
| **Raw** | `data_raw/spy/` | Downloaded Parquet files from external sources | Gitignored; downloaded on-demand by ingestion script |
| **Processed (partitions)** | `data_processed/spy/partitions/` | Year-by-year cleaned surface point Parquet files | Gitignored; produced by build script |
| **Processed (consolidated)** | `data_processed/spy/` | `spy_surface_points.parquet` (conservative) and `spy_surface_points_strict.parquet` (strict) | Gitignored; produced by build script |
| **Reports** | `reports/` | Markdown summaries from each pipeline step | Gitignored; produced by pipeline scripts |
| **Benchmarks** | `data_processed/spy/benchmarks/` | Per-variant masked/noised/split Parquet files with provenance metadata | Gitignored; produced by benchmark task script |
| **Metadata** | `data_raw/ingest_metadata.json` | Download timestamp and source info | Gitignored; produced by ingestion script |

All data files are gitignored. The pipeline is designed to be re-run from scratch on any machine with network access.

---

## 3. Source-of-truth paths

### External sources

**Active (2026-05-22 onward):**

| Source | Endpoint | What it provides |
|---|---|---|
| Alpha Vantage `HISTORICAL_OPTIONS` (paid Standard, 75 req/min) | `https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=SPY&date=YYYY-MM-DD&apikey=…` | Full SPY EOD option chain per date (2008 → present); 20 fields per contract incl. IV, bid/ask, volume, open_interest, greeks |
| Yahoo Finance (`yfinance`) | Python library | SPY underlying close (`date`, `close`), dynamic end-date — used until/unless a primary AV underlying ingest is added |

API key handling: read from `os.environ["ALPHAVANTAGE_API_KEY"]`; never written
to a tracked file. Provenance + cost rationale: see
[ADR 0003](../decisions/0003_spy_options_data_source_migration.md).

**Defunct (kept as historical evidence only):**

| Source | URL | Status |
|---|---|---|
| Philipp Dubach options dataset | `https://static.philippdubach.com/data/options/spy/options.parquet` | HTTP 404; host path removed; covered 2008–2025 (~24.7M rows) |
| Philipp Dubach underlying dataset | `https://static.philippdubach.com/data/options/spy/underlying.parquet` | HTTP 404 |
| Philipp Dubach ETF fallback repo | `https://github.com/philippdubach/options-dataset-hist` | Repo absent from maintainer's account |

See §9 "Upstream data source unreachable" for the discovery + replacement
decision trail.

### Internal file paths

| File | Role | Produced by |
|---|---|---|
| `data_raw/spy/spy_options.parquet` | Raw options chain | `src/data/01_ingest_spy_alpha_vantage.py` *(story 2C.6 — new active script; the old `01_ingest_spy_github_dataset.py` is DEFUNCT and slated for removal in 2C.7)* |
| `data_raw/spy/spy_underlying.parquet` | Raw underlying prices | `src/data/01_ingest_spy_alpha_vantage.py` (yfinance, dynamic end-date) |
| `data_processed/spy/partitions/spy_surface_YYYY.parquet` | Per-year cleaned surface points | `src/data/03_build_spy_surface_table.py` |
| `data_processed/spy/spy_surface_points.parquet` | Conservative cleaned surface (hard drops + quality flags) | `src/data/03_build_spy_surface_table.py` |
| `data_processed/spy/spy_surface_points_strict.parquet` | Strict modeling subset (tighter thresholds) | `src/data/03_build_spy_surface_table.py` |
| `reports/spy_ingest_summary.md` | Ingestion report | `src/data/01_ingest_spy_alpha_vantage.py` |
| `reports/spy_schema_report.md` | Schema validation report | `src/data/02_inspect_spy_schema.py` |
| `reports/spy_build_report.md` | Build processing report | `src/data/03_build_spy_surface_table.py` |
| `data_processed/spy/benchmarks/spy_phase1_*.parquet` | Benchmark task datasets (masked, noised, split) | `src/data/04_build_benchmark_tasks.py` |
| `artifacts/checkpoints/best_mlp.pt` | Best MLP model checkpoint | `scripts/run_baseline.py` via `training/train.py` (refit on new data in 2C.8) |
| `artifacts/results/baseline_results.csv` | Baseline comparison metrics (interp vs MLP) | `scripts/run_baseline.py` (refreshed on new data in 2C.8) |
| (loader) `src/neural_iv_surface_inference/data/loaders.py::IVSurfaceDataset` | Point-wise loader used by the Phase 1 MLP baseline | Phase 1 |
| (loader) `src/neural_iv_surface_inference/data/conditional_loaders.py::ConditionalIVSurfaceDataset` + `collate_conditional` | **Date-grouped** loader for the W3 conditional surface model (2C.2): one date per sample, ragged context = observed rows, padded with boolean masks for the set encoder | 2C.2 |

---

## 4. Pipeline flow

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — Ingest  (ACTIVE: Alpha Vantage)               │
│  Script: src/data/01_ingest_spy_alpha_vantage.py        │
│          (the old 01_ingest_spy_github_dataset.py is    │
│           DEFUNCT — Dubach static Parquet is dead;      │
│           ADR 0003 / story 2C.7 removes it)             │
│  Config: src/data/config.py (AV constants + paths)      │
│                                                         │
│  Input:  Alpha Vantage HISTORICAL_OPTIONS (paid),       │
│          one HTTP GET per trading date, 75 req/min;     │
│          underlying via yfinance with dynamic end-date  │
│  Output: data_raw/spy/spy_options.parquet                │
│          data_raw/spy/spy_underlying.parquet             │
│          reports/spy_ingest_summary.md                   │
│          data_raw/ingest_metadata.json                   │
│                                                         │
│  Notes:  Streaming-to-Parquet (no raw JSON cache).      │
│          API key from ALPHAVANTAGE_API_KEY env var only.│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — Inspect / Validate Schema                     │
│  Script: src/data/02_inspect_spy_schema.py              │
│  Config: src/data/config.py (required/optional columns) │
│                                                         │
│  Input:  data_raw/spy/spy_options.parquet                │
│          data_raw/spy/spy_underlying.parquet             │
│  Output: reports/spy_schema_report.md                   │
│                                                         │
│  Checks: Required columns present, null counts,         │
│          data types, date ranges, suspicious rows        │
│          (negative bids, crossed quotes, extreme IV)     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 — Build Surface Table                           │
│  Script: src/data/03_build_spy_surface_table.py         │
│  Config: src/data/config.py (thresholds, paths)         │
│  Docs:   docs/data_assumptions_and_cleaning.md          │
│                                                         │
│  Input:  data_raw/spy/spy_options.parquet                │
│          data_raw/spy/spy_underlying.parquet             │
│                                                         │
│  Processing (per year, 2008–2025):                      │
│    1. Load year's options via chunked scan (500k rows)  │
│    2. Prune to required + optional columns              │
│    3. Join with underlying on date                      │
│    4. Derive: mid, days_to_expiry, tau, spot,           │
│       log_moneyness                                     │
│    5. Apply hard cleaning drops                         │
│    6. Add quality flags                                 │
│    7. Write partition to partitions/ dir                 │
│                                                         │
│  After all years:                                       │
│    8. Concatenate partitions → conservative file        │
│    9. Apply strict thresholds → strict subset file      │
│   10. Generate build report                             │
│                                                         │
│  Output: data_processed/spy/partitions/spy_surface_*    │
│          data_processed/spy/spy_surface_points.parquet   │
│          data_processed/spy/spy_surface_points_strict.p… │
│          reports/spy_build_report.md                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — EDA / First Look (manual)                    │
│  Notebook: notebooks/01_spy_data_firstlook.ipynb        │
│                                                         │
│  Input:  data_processed/spy/spy_surface_points.parquet   │
│          data_processed/spy/spy_surface_points_strict.p… │
│                                                         │
│  Analyses: Row counts by year, call/put split,          │
│            IV distribution, tau distribution,            │
│            log-moneyness distribution,                   │
│            IV surface sample dates, spread diagnostics,  │
│            missingness diagnostics, strict subset stats  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — Build Benchmark Tasks (memory-safe streaming) │
│  Script: src/data/04_build_benchmark_tasks.py           │
│  Config: configs/benchmark_tasks.yaml                   │
│  Modules:                                               │
│    src/neural_iv_surface_inference/data/masking.py      │
│    src/neural_iv_surface_inference/data/noise.py        │
│    src/neural_iv_surface_inference/data/splits.py       │
│                                                         │
│  Input:  data_processed/spy/spy_surface_points_strict.  │
│          parquet (from Step 3)                          │
│          data_processed/spy/partitions/ (for date scan) │
│                                                         │
│  Processing:                                            │
│    0. Precompute global date→split map by scanning      │
│       partition files (reads only date column, ~4.5K    │
│       dates, trivial memory)                            │
│    Per benchmark variant:                               │
│    1. Stream strict file via PyArrow iter_batches       │
│       (500K rows per batch)                             │
│    2. Per batch: apply mask → inject noise → assign     │
│       split from precomputed map → write via            │
│       ParquetWriter                                     │
│    3. gc.collect() after each batch                     │
│    4. Close writer with provenance metadata             │
│                                                         │
│  Output: data_processed/spy/benchmarks/                 │
│          spy_phase1_<strategy><pct>_noise<level>.parquet│
│          (11 variants configured)                       │
│                                                         │
│  Status: Implemented and tested. Rewritten from eager   │
│          full-load to streaming after OOM on RunPod.    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5 — Baselines + Training + Evaluation             │
│  Entry point: scripts/run_baseline.py                   │
│  Config: configs/baseline.yaml                          │
│                                                         │
│  Models:                                                │
│    models/interpolation.py — S3.1 per-date RBF/griddata│
│    models/baseline_mlp.py  — S3.2 global masked MLP    │
│    models/losses.py        — masked MSE/MAE/combined   │
│                                                         │
│  Data:                                                  │
│    data/loaders.py — IVSurfaceDataset + DataLoader      │
│      (column-pruned load + gc.collect after split)      │
│                                                         │
│  Training:                                              │
│    training/train.py — training loop, early stopping,   │
│                        checkpointing, LR scheduling     │
│    training/eval.py  — MAE/RMSE/MAPE, regional         │
│                        diagnostics (maturity/moneyness) │
│                                                         │
│  Input:  data_processed/spy/benchmarks/*.parquet        │
│          (from Step 4)                                  │
│                                                         │
│  Output: artifacts/checkpoints/best_mlp.pt              │
│          artifacts/results/baseline_results.csv          │
│                                                         │
│  Status: Implemented and tested (25 unit tests).        │
│          Awaiting benchmark data (Steps 3+4 on RunPod)  │
│          before full-scale execution.                   │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Derived columns

| Column | Formula | Defined in |
|---|---|---|
| `mid` | `(bid + ask) / 2` | `03_build_spy_surface_table.py`, `data_assumptions_and_cleaning.md` |
| `days_to_expiry` | `(expiration - date).days` | `03_build_spy_surface_table.py` |
| `tau` | `days_to_expiry / 365.0` | `03_build_spy_surface_table.py` |
| `spot` | `close` (unadjusted) | `03_build_spy_surface_table.py`, `data_assumptions_and_cleaning.md` |
| `log_moneyness` | `ln(strike / spot)` | `03_build_spy_surface_table.py`, `data_assumptions_and_cleaning.md` |

**Critical convention:** `spot = close` (unadjusted closing price), NOT `adjusted_close`. This is because option strikes are quoted against unadjusted prices; using adjusted close would distort moneyness computations. Documented in `docs/data_assumptions_and_cleaning.md`.

---

## 6. Cleaning thresholds

### Hard drops (conservative — rows removed entirely)

| Rule | Threshold | Source |
|---|---|---|
| Null critical field | date, expiration, strike, type, IV, close | `config.py`, `data_assumptions_and_cleaning.md` |
| Negative bid | `bid < 0` | `config.py` |
| Negative ask | `ask < 0` | `config.py` |
| Crossed quote | `ask < bid` | `config.py` |
| Non-positive IV | `IV <= 0` | `config.py` |
| Extreme IV | `IV > 5.0` (500%) | `config.py` |
| Expired option | `tau <= 0` | `config.py` |
| Far-dated option | `tau > 3.0` (3 years) | `config.py` |

### Strict subset additional thresholds

| Parameter | Range | Source |
|---|---|---|
| IV | 0.01 – 3.0 (1% – 300%) | `config.py`, `data_assumptions_and_cleaning.md` |
| Tau | 1/365 – 2.0 years | `config.py`, `data_assumptions_and_cleaning.md` |
| Log-moneyness | -1.0 – 1.0 | `config.py`, `data_assumptions_and_cleaning.md` |
| Bid | > 0 (positive) | `config.py` |
| Ask | > 0 (positive) | `config.py` |
| Mid | > 0 (nonzero) | `config.py` |

### Quality flags (attached, not used for exclusion in conservative set)

| Flag | Condition | Source |
|---|---|---|
| `flag_zero_bid` | `bid == 0` | `config.py` |
| `flag_zero_volume` | `volume == 0` | `config.py` |
| `flag_zero_oi` | `open_interest == 0` | `config.py` |
| `flag_wide_spread` | `(ask - bid) / max(mid, ε) > 0.5` | `config.py` |

---

## 7. Known decisions and constraints

| Decision | Rationale | ADR / Reference |
|---|---|---|
| SPY-only for Phase 1 | Most liquid equity option; lowest data quality risk | `docs/decisions/0002_phase1_scope_freeze.md` |
| EOD-only for Phase 1 | Simplifies pipeline; avoids real-time infrastructure | `docs/decisions/0002_phase1_scope_freeze.md` |
| Spot = unadjusted close | Option strikes quoted against unadjusted prices | `docs/data_assumptions_and_cleaning.md` |
| Year-by-year partitioned processing | Avoids OOM on 24.7M-row dataset | `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` |
| Two-tier output (conservative + strict) | Conservative preserves market structure for auditing; strict enables cleaner early experiments | `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` |
| Yahoo Finance fallback for underlying | Direct URL download was blocked | Commits `b0cfde5`, `e6b31de` |
| Thresholds are first-pass conservative | Will be revisited after EDA | `docs/data_assumptions_and_cleaning.md` |

---

## 8. Governing references

| Document | What it governs |
|---|---|
| `src/data/config.py` | Centralized paths, URLs, thresholds, column lists |
| `docs/data_assumptions_and_cleaning.md` | Cleaning rules, spot price convention, threshold rationale |
| `docs/decisions/0002_phase1_scope_freeze.md` | Scope: SPY-only, EOD-only |
| `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` | Why partitioned processing; two-output strategy |
| `configs/data.yaml` | Pipeline config (partially stubbed) |
| `configs/baseline.yaml` | Baseline model config (partially stubbed) |
| `configs/benchmark_tasks.yaml` | Benchmark variant definitions (masking, noise, split params) |

---

## 9. Open questions and unresolved lineage details

### Pipeline execution status on current machine

The pipeline scripts exist and the code is committed. However, the `data_raw/spy/` and `data_processed/spy/` directories on the current local clone are empty. The full pipeline execution (Steps 1–3) has been run on RunPod based on progress log evidence, but the resulting data files are not present in this clone. Data files are gitignored by design.

**Resolved (Step 4 output validation):** Benchmark parquet outputs were verified directly on RunPod. Evidence:
- strict source file (`data_processed/spy/spy_surface_points_strict.parquet`) row count = `21,386,789`
- benchmark outputs present = `11/11` configured files
- every benchmark file row count = `21,386,789` (matches strict source exactly)
- per-file split counts are internally consistent: train `10,458,763`, val `5,617,807`, test `5,310,219` (sum = total)

### Feature engineering pipeline

Task construction modules (S2.1–S2.3) are now implemented and tested:
- **S2.1 Masking**: `src/neural_iv_surface_inference/data/masking.py` — 7 strategies (random, structured maturity/moneyness, realistic liquidity)
- **S2.2 Noise**: `src/neural_iv_surface_inference/data/noise.py` — 4 regimes (none/low/med/high), i.i.d. and heteroscedastic modes
- **S2.3 Splits**: `src/neural_iv_surface_inference/data/splits.py` — chronological train/val/test (70/15/15), benchmark naming and Parquet save/load with provenance

Orchestration: `src/data/04_build_benchmark_tasks.py` reads `configs/benchmark_tasks.yaml` (11 variants) and produces one Parquet per variant in `data_processed/spy/benchmarks/`. The script uses memory-safe streaming (PyArrow `iter_batches`, 500K rows/batch) with a precomputed date→split map via `compute_date_split_map()` in `splits.py`.

`run_interpolation_baseline()` returns `np.ndarray` (not a DataFrame copy) to avoid duplicating the full dataset in memory. `loaders.py` prunes to only needed columns on load and calls `gc.collect()` after splitting.

**Remaining stubs**: `scripts/prepare_data.py` and `src/neural_iv_surface_inference/data/cleaning.py` still contain TODO placeholders. All other modules (loaders, models, training, evaluation) are now implemented.

### Config alignment

`configs/data.yaml` references paths (`data/raw`, `data/interim`, `data/processed`) that differ from the actual paths used by the pipeline scripts (`data_raw/spy/`, `data_processed/spy/`). The config file has TODO markers and appears to be a template that was not updated to match the actual pipeline. The operative config is `src/data/config.py`, not `configs/data.yaml`.

### Vendor-style reference data

Subtask S3.3 (vendor-style reference) is currently **blocked**. No vendor reference data files or integration scripts are present in the repository yet.

Concrete next dependency to unblock S3.3:
- approve one vendor-style reference source and access path
- freeze minimal schema mapping into current surface coordinates (`date`, `tau`, `log_moneyness`, reference IV field)
- define ingestion location under `data_raw/` and alignment output format under `data_processed/`

### Upstream data source unreachable (discovered 2026-05-22)

During Phase 2C planning, the documented external sources in §3 were re-probed
and **all return HTTP 404**:

- `https://static.philippdubach.com/data/options/spy/options.parquet` → 404
- `https://static.philippdubach.com/data/options/spy/underlying.parquet` → 404
- `https://static.philippdubach.com/` (host root) → 404
- `https://github.com/philippdubach/options-dataset-hist` (ETF fallback) → 404

The original ingest covered **2008–2025**; it is now 2026-05, so the dataset is
at minimum ~5 months stale and the source may be **gone entirely**. The 404s may
be transient or reflect a moved/renamed path, but the host-root and fallback-repo
404s suggest a genuine relocation/removal.

Follow-up confirmation (2026-05-22): the source is **discontinued**, not merely
moved. The maintainer's GitHub no longer hosts `options-data` /
`options-dataset-hist` / `historic-options-dataset` (absent from
`api.github.com/users/philippdubach/repos`), and his current volatility project
(`philippdubach/vol-regime-prediction`) pulls from **live APIs**
(Alpha Vantage, CBOE, FRED, yfinance) rather than a static SPY options parquet.
The `static.philippdubach.com` host is alive (serves other assets) but the
`/data/options/spy/` path is gone. Treat the documented parquet source as dead.

Implications:
- A data refresh (story **2C.6**) must first **re-establish a working options
  source** — not just re-download. The `yfinance` fallback covers only the
  **underlying** price series; it cannot reconstruct the option chain.
- The `yfinance` fallback in `01_ingest_spy_github_dataset.py` also hardcodes
  `end="2026-01-01"`, which would cap the underlying in the past and must be
  fixed during refresh.
- If no compatible options source is found, Phase 2C full train/eval must proceed
  from the last-known-good snapshot on the remote (if preserved), with that
  decision logged (candidate ADR in `docs/decisions/`).
