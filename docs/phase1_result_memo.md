# Phase 1 Result Memo

> **Data-source note (2026-05-22):** The numbers in this memo were produced
> against the Philipp Dubach SPY static-Parquet dataset (2008–2025, ~24.7M
> rows). That source is now defunct (HTTP 404 + repos removed) and the
> project has migrated to **Alpha Vantage `HISTORICAL_OPTIONS`** per
> [ADR 0003](decisions/0003_spy_options_data_source_migration.md). The
> benchmark Parquet files those baselines were trained/evaluated against
> will be **replaced** (story **2C.7**) and the Phase 1 baselines
> (interpolation + MLP) will be **re-derived on the new dataset** in story
> **2C.8**. Until 2C.8 lands, the figures below are *historical Phase-1
> numbers* — preserve them as context, but do not cite them as the current
> baseline for Phase 2C comparisons.

## Scope

Phase 1 remained SPY-only and EOD-only with a fixed task:
reconstruct dense IV surfaces from sparse/noisy observations.

## What Worked

- Real data pipeline is running end-to-end (`ingest -> inspect -> build`).
- Benchmark generation now creates reproducible sparse/noisy train/val/test datasets.
- A simple interpolation baseline and a neural baseline (MLP) both run through one evaluation path.
- Required output artifacts are generated in a consistent structure:
  figures, metrics tables, and run metadata.

## Baseline Results (benchmark `spy_phase1_random40_noiselow`)

Evaluated against `iv_clean` on chronological train/val/test splits. Source:
`artifacts/results/baseline_results.csv`.

| Model | Split | Overall MAE | Overall RMSE | Observed MAE | Unobserved MAE |
|---|---|---|---|---|---|
| interp_rbf | test | **0.0687** | 0.1137 | 0.0562 | 0.0769 |
| mlp        | test | 0.0967 | 0.1662 | 0.0966 | 0.0967 |
| interp_rbf | val  | 0.0452 | 0.0886 | 0.0378 | 0.0501 |
| mlp        | val  | 0.1006 | 0.1658 | 0.1006 | 0.1006 |
| interp_rbf | train| 0.0507 | 0.0897 | 0.0391 | 0.0584 |
| mlp        | train| 0.0976 | 0.1632 | 0.0975 | 0.0976 |

The masked MLP uses only `(log_moneyness, tau)` as input. The `mlp` rows were
regenerated during Phase 1 closeout by re-evaluating the saved checkpoint
`artifacts/checkpoints/best_mlp.pt` (no retraining); see the 2026-05-20 entry in
`docs/experiments/experiment_journal.md`.

## What Failed or Is Weak

- The current neural baseline is intentionally small and **underperforms the
  interpolation floor on every split** (test MAE 0.0967 vs 0.0687). Its error is
  essentially flat between observed and unobserved points (≈0.0966 vs ≈0.0967),
  indicating it learned only a smooth global fit rather than genuine surface
  structure. It is weakest exactly where reconstruction matters most — short
  maturities (test MAE ≈ 0.13) and the deep-ITM / deep-OTM wings (≈0.19 / ≈0.15).
- Vendor-style reference is represented as a placeholder alignment row and still needs a true external surface integration.
- Region-level metrics are still coarse and should be expanded with richer bucket analysis.
- A process gap surfaced at closeout: result CSVs were not committed as evidence, so
  the documented `mlp` metrics had to be regenerated from the checkpoint. Result
  artifacts should be committed alongside checkpoints going forward.

## Hardest Part of the Task

The hardest part is robust reconstruction in high-sparsity regions with little observed information,
especially for tail moneyness and short/long maturity edges.

## Is Simple Interpolation Enough?

No. It provides a transparent and useful floor, but it cannot capture richer cross-region structure
once masks become sparse and noisy.

## Why Phase 2 Is Needed

Phase 2 is needed to improve model capacity, add stronger inductive structure,
and incorporate better reference targets and uncertainty-aware evaluation.

Concretely, the baseline numbers above show that a naive coordinate-regression MLP
does not beat interpolation and does not exploit observed points. This motivates
**structured / latent / mask-aware surface-level models** (e.g. conditional latent
surface inference that ingests the observed set as context) rather than pointwise
coordinate regression, plus arbitrage-aware constraints and calibrated uncertainty.
