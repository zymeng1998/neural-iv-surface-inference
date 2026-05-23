# Neural IV Surface Inference

## Overview

Neural IV Surface Inference is an ML × Finance project focused on recovering implied-volatility surfaces from sparse, noisy, and irregular option observations. The current direction is **reliability-aware surface inference**: beyond producing a smooth surface, the system aims to know where its predictions are trustworthy and to abstain where they are not. Future focus areas include uncertainty estimation, abstention, no-arbitrage diagnostics, and conditional neural models.

## Current Phase

**Phase 1 baseline foundation complete; in Phase 2 — Reliability-Aware Implied Volatility Surface Inference (Phase 2A & 2B done; Phase 2C in progress).**

> **Data-source migration (2026-05-22):** the upstream Philipp Dubach SPY
> Parquet that powered Phase 1 is now defunct (HTTP 404, repos removed). The
> project has migrated to **Alpha Vantage `HISTORICAL_OPTIONS`** (paid
> Standard tier, ~$50 one-month pull-and-cancel). The historical Phase 1
> baseline numbers will be re-derived on the new dataset under stories 2C.7
> (full pull on the remote) and 2C.8 (baseline + W1/W2 rerun). See
> [ADR 0003](docs/decisions/0003_spy_options_data_source_migration.md).

## Current Status

Phase 1 baseline foundation completed:
- Real SPY EOD option-chain data pipeline (ingest → inspect → build surface table → build benchmark tasks), memory-safe streaming over ~21M rows
- Benchmark construction with configurable sparse masking (7 strategies) and noise regimes (none/low/med/high + heteroscedastic), chronological train/val/test splits
- Per-date interpolation baseline (RBF / griddata)
- Naive neural MLP baseline (global masked MLP) — runs end-to-end but is **intentionally limited**: it is a coordinate-regression model `(log_moneyness, tau) → implied_vol` and underperforms the interpolation floor, since it does not condition on the observed chain
- Evaluation metrics: MAE / RMSE / MAPE, observed vs. unobserved split, regional diagnostics by maturity and moneyness
- 59 passing unit tests; reproducible Phase 1 artifacts (figures, summary tables, result memo)
- Reproducible remote (RunPod) development workflow and project-memory documentation system

### Next Direction — Phase 2

The naive MLP result motivates moving beyond pointwise interpolation toward a
reliability-aware system. Phase 2 introduces uncertainty evaluation, masking
sensitivity and no-arbitrage diagnostics, a conditional neural surface model
(`observed chain O_t → latent z_t`, then `(k, tau, z_t) → sigma_hat`), and an
abstention / tradability decision layer. Target decision-grade outputs:
`sigma_hat`, `confidence_score`, `uncertainty_band`, `tradability_score`,
`no_arb_risk_flags`, and `abstain_flag`. See the Phase 2 roadmap below.

## Documentation Map

- `docs/roadmaps/phase1_structural_roadmap.md` — Phase 1 task decomposition and subtask matrix
- `docs/roadmaps/phase2_reliability_aware_surface_inference.md` — Phase 2 plan: workstreams, outputs, acceptance criteria
- `docs/phase1_result_memo.md` — Phase 1 baseline results and analysis
- `docs/tasks/BOARD.md` — Jira-style task board (epics, stories, statuses) for all implementation work
- `docs/workflows/ai_human_collaboration.md` — human-AI operating model: modes, task lifecycle, validation gate
- `docs/workflows/session_protocol.md` — start/end-of-session checklists and handoff template
- `docs/workflows/reusable_prompts.md` — copy-pasteable prompts for starting new sessions (epic decomposition, story implementation)
- `docs/setup/remote_dev.md` — sanitized remote development workflow and environment notes
- `docs/setup/private_runbook_template.md` — template for local-only private ops notes
- `docs/logs/progress_log.md` — chronological project progress log
- `docs/decisions/0001_remote_dev_stack.md` — architecture / workflow decision record

## Repository Structure

```text
src/neural_iv_surface_inference/   Python package (data, features, models, training, utils)
scripts/                           Entry-point scripts (smoke test, data prep, training)
configs/                           YAML configuration files
notebooks/                         Jupyter notebooks
tests/                             Test suite
data/                              Data directories (raw, interim, processed, samples)
artifacts/                         Output artifacts (figures, tables, checkpoints)
docs/                              Project documentation
```

## Immediate Next Steps

- Build the model-agnostic uncertainty evaluation layer (Phase 2A)
- Add masking-sensitivity and no-arbitrage structure diagnostics (Phase 2B)
- Implement the first conditional neural surface model — Set Encoder + Coordinate Decoder (Phase 2C)

## Security Note

Tracked documentation intentionally excludes sensitive operational details such as IPs, ports, usernames, SSH config details, private key paths, and tokens. Those details belong only in a local-only private runbook that must not be committed.
