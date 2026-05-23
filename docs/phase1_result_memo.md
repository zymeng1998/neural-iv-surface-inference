# Phase 1 Result Memo

> **2026-05-23 update (Alpha Vantage rerun):** The Phase 1 baselines have
> been re-derived on the new AV-sourced benchmarks (stories **2C.7** + **2C.8**).
> See the new **"AV-era refresh"** section below for the current numbers; the
> historical Dubach-era numbers immediately under "Baseline Results" are
> preserved as comparison context. The AV-era numbers are the active baseline
> floor for all Phase 2C+ comparisons.
>
> Both datasets produce nearly identical interp/MLP test MAE (interp 0.0662 vs
> historical 0.0687; MLP 0.0951 vs 0.0967), confirming the data-source migration
> didn't shift the comparison floor.

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

## AV-era Refresh (Alpha Vantage, 2026-05-23) — current baseline

Story **2C.8** rerun on the AV-sourced `spy_phase1_random40_noiselow` benchmark
(22.5M rows, conservative cleaning; chronological train/val/test = 11.1M/5.6M/5.8M).
Source: `artifacts/results/baseline_results.csv` (refreshed),
`uncertainty_eval_av_rerun_20260523_075616_*.csv`.

| Model | Split | Overall MAE | Overall RMSE | Observed MAE | Unobserved MAE |
|---|---|---|---|---|---|
| interp_rbf         | test  | **0.0662** | 0.1124 | 0.0542 | 0.0742 |
| **conditional (W3)** | test  | **0.0753** | 0.1193 | 0.0753 | 0.0754 |
| mlp                | test  | 0.0951 | 0.1665 | 0.0951 | 0.0951 |
| interp_rbf         | val   | 0.0463 | 0.0905 | 0.0379 | 0.0519 |
| **conditional (W3)** | val   | 0.0580 | 0.0987 | 0.0580 | 0.0581 |
| mlp                | val   | 0.0949 | 0.1617 | 0.0949 | 0.0949 |
| interp_rbf         | train | 0.0485 | 0.0888 | 0.0367 | 0.0563 |
| **conditional (W3)** | train | 0.0573 | 0.0943 | 0.0572 | 0.0574 |
| mlp                | train | 0.1400 | 0.2177 | 0.1399 | 0.1400 |

Headline shifts vs the historical Dubach-era table above:

- AV-era interp/MLP test MAE: 0.0662 / 0.0951 vs Dubach 0.0687 / 0.0967 —
  within ~3%. Data-source migration did **not** shift the comparison floor.
- The **conditional surface model** (story 2C.4-R, 85,057 params,
  ConditionalSurfaceModel = SetEncoder + CoordinateDecoder, trained 50 epochs
  in 3.7 min on RTX A4500) **beats the Phase 1 MLP by ~21% on test MAE
  (0.0753 vs 0.0951)** but **loses to RBF interpolation by ~14% (0.0753 vs 0.0662)**.
- Conditional model's observed/unobserved MAE are nearly identical
  (0.0753 / 0.0754) — characteristic of the parametric inductive bias; the
  RBF interp shows the expected observed (0.0542) vs unobserved (0.0742) gap.
- Phase 2 acceptance criterion #4 ("conditional model evaluated on the same
  benchmarks as the Phase 1 baselines") is met on real SPY data.

W3 caveat: this is a point predictor; uncertainty signals arrive in epic 2D.

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
