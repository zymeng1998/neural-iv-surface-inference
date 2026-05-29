# Neural IV Surface Inference

## Overview

Neural IV Surface Inference is an ML × Finance project focused on recovering
implied-volatility surfaces from sparse, noisy, and irregular option
observations. The current direction is **reliability-aware surface
inference**: beyond producing a smooth surface, the system knows where its
predictions are trustworthy and abstains where they are not.

## Current Phase

**Phase 3 — Accuracy Push: Beat RBF Without Losing Reliability — in
progress (3A + 3B closed, 3C next).** Phase 2 (Reliability-Aware Surface
Inference) closed on 2026-05-25 with both mandatory acceptance numbers
green; Phase 3 attacks the remaining accuracy gap versus the per-date RBF
interpolation baseline.

### Phase 3 (current)

| Epic | Workstream | Status |
|---|---|---|
| 3A | W10 — Coordinate-representation ablation (Fourier vs raw `(k, τ)`, decoder-only) | `done` — raw beats Fourier (0.0760 vs 0.0790 full-fold test MAE); gap-to-RBF unclosed |
| 3B | W11 — Cross-attention decoder (end-to-end DeepSets + ANP, raw `(k, τ)`) | `done` (in_review) — **bar NOT met**: ANP best-case +2.7 % vs RBF; reliability holds |
| 3C | W12 — Feature & inductive-bias expansion (microstructure features, optional SVI head) | `backlog` (next) |
| 3D | W13 — Phase 3 closing memo + re-evaluation versus RBF | `backlog` |

Acceptance bar: **test MAE ≤ 0.95 × RBF** on both the 2D.9 slice
(≤ 0.0693) and the full 2D.4 fold (≤ 0.0629), with no reliability
regression. The conditional neural model must beat RBF **on its own**
— RBF-as-prior / RBF-residual hybrids are explicitly out of scope for
Phase 3 and reserved as a Phase 4 production fallback. See
[`docs/roadmaps/phase3_accuracy_push.md`](docs/roadmaps/phase3_accuracy_push.md)
and [ADR 0004](docs/decisions/0004_phase3_accuracy_push_framing.md).

**3B closing result (epic 3B, 2026-05-28).** The end-to-end ANP
cross-attention decoder is the strongest conditional model produced so
far — it beats the 3A decoder-only raw variant (~10 %) and the Phase 2D
DeepSets-decoder family (~5 %), narrowing the gap-to-RBF from ~29 % to
**~2.7 %** (full-fold point head 0.0680 vs RBF 0.0662). But it does not
clear the ≥ 5 % bar on any view (slice 0.0813, full-fold gaussian
0.0722, full-fold point 0.0680). Reliability holds (coverage 0.9149
within ±2 pp; hi-conf MAE 0.0542 < no-abstention 0.0813). Evidence:
[`results/3/spy_phase1_random40_noiselow/3b_compare/comparison.csv`](results/3/spy_phase1_random40_noiselow/3b_compare/comparison.csv)
and the §W11 closing addendum in the roadmap. Implication: pure
decoder-architecture iteration has plateaued; 3C should prioritise
feature / inductive-bias expansion.

The single fresh-session entry point for Phase 3 is
[`docs/PHASE3_INDEX.md`](docs/PHASE3_INDEX.md) — read it first if you
are picking up Phase 3 work cold.

### Phase 2 (complete, 2026-05-25)

All four Phase 2 epics are `done`:

| Epic | Workstream | Status |
|---|---|---|
| 2A | W1 — Uncertainty evaluation layer | `done` (5/5 stories) |
| 2B | W2 — Masking sensitivity + no-arbitrage diagnostics | `done` (5/5 stories) |
| 2C | W3 — Conditional neural surface model (Set Encoder + Coordinate Decoder) | `done` (8/8 stories) |
| 2D | W4 + W5 — Uncertainty-aware inference + abstention / tradability decision layer | `done` (10/10 stories) |

Phase 2E (follow-ups) remains open for capacity-sweep diagnostics that
validate but do not gate Phase 3.

> **Data-source migration (2026-05-22, completed):** the upstream Philipp
> Dubach SPY Parquet that powered the original Phase 1 work is defunct (HTTP
> 404, repos removed). The project migrated to **Alpha Vantage
> `HISTORICAL_OPTIONS`** (paid Standard tier, ~$50 one-month pull-and-cancel).
> Phase 1 baselines and the W1 / W2 evaluation were re-run on the full AV
> dataset under stories 2C.7 (full pull, 26.06 M rows, 4,623 trading days,
> 2008-01-02 → 2026-05-22) and 2C.8 (baseline + W1/W2 rerun). See
> [ADR 0003](docs/decisions/0003_spy_options_data_source_migration.md).

## Current Status

### Phase 1 — baseline foundation (complete)

- Real SPY EOD option-chain data pipeline (ingest → inspect → build surface
  table → build benchmark tasks), memory-safe streaming over the full AV
  history.
- Benchmark construction with configurable sparse masking (7 strategies) and
  noise regimes (none / low / med / high / heteroscedastic), chronological
  train / val / test splits.
- Per-date interpolation baseline (RBF / griddata).
- Naive masked MLP baseline — intentionally limited: a coordinate-regression
  model `(log_moneyness, τ) → σ` that underperforms the interpolation floor
  because it does not condition on the observed chain.
- Reproducible Phase 1 artifacts (figures, summary tables, result memo).

### Phase 2 — reliability-aware inference (complete, 2026-05-25)

- **Model-agnostic uncertainty evaluator** (W1 / 2A): empirical coverage,
  interval width, error-vs-uncertainty correlation, abstention curves,
  high-confidence MAE. Interpolation, MLP, and the conditional model all run
  through the same evaluator.
- **Sensitivity + structure diagnostics** (W2 / 2B): masking-sensitivity
  harness producing per-(k, τ) stability; no-arbitrage checks
  (monotonicity, convexity, calendar) producing violation counts and per-row
  `no_arb_risk_flags`; (k, τ) region heatmaps.
- **Conditional neural surface model** (W3 / 2C): Set Encoder + Coordinate
  Decoder (85,057 params) ingests the observed chain `O_t → z_t` and decodes
  at any `(k, τ)`. Beats the Phase 1 MLP by ~21 % on test MAE; ~14 % above
  the RBF interpolation floor.
- **Calibrated reliability signals** (W4 / 2D.2–2D.4 + 2D.7–2D.8):
  heteroscedastic Gaussian / quantile head, K = 5 deep ensemble, fused
  calibrator combining Gaussian σ, ensemble disagreement, and
  masking-sensitivity into one `confidence_score` + calibrated `(lower,
  upper)` band.
- **Decision layer** (W5 / 2D.5 + 2D.9): config-driven thresholds emit
  `abstain_flag`, `tradability_score`, `decision_reason`, and consume
  `no_arb_risk_flags` from W2.
- **Six decision-grade outputs per query** are now produced end-to-end:
  `sigma_hat`, `confidence_score`, `uncertainty_band`, `tradability_score`,
  `no_arb_risk_flags`, `abstain_flag`.

**Headline acceptance numbers** (test split on
`spy_phase1_random40_noiselow`, AV benchmark):

| Acceptance criterion (roadmap §5 / §6) | Result | Status |
|---|---|---|
| Empirical coverage at α = 0.90 within ±2 pp | **0.9184** on the 2D.9 test slice (\|Δ\| = 1.84 pp); **0.8955** on the full 2D.4 test fold (\|Δ\| = 0.45 pp) | PASS |
| High-confidence MAE < no-abstention test MAE | **0.0606** at `keep_fraction = 0.8` vs **0.0855** no-abstention test MAE (~29 % reduction) | PASS |

Evidence:
[`docs/phase2_result_memo.md`](docs/phase2_result_memo.md),
[`notebooks/05_phase2_results.ipynb`](notebooks/05_phase2_results.ipynb),
[`results/2D/comparison_summary.csv`](results/2D/comparison_summary.csv),
[`docs/experiments/experiment_journal.md`](docs/experiments/experiment_journal.md)
(2026-05-25T11:00 closing entry).

## Documentation Map

### Roadmaps and results

- [`docs/roadmaps/phase1_structural_roadmap.md`](docs/roadmaps/phase1_structural_roadmap.md) — Phase 1 task decomposition and subtask matrix
- [`docs/roadmaps/phase2_reliability_aware_surface_inference.md`](docs/roadmaps/phase2_reliability_aware_surface_inference.md) — Phase 2 plan: workstreams, acceptance criteria, closing-status block
- [`docs/roadmaps/phase2_followups.md`](docs/roadmaps/phase2_followups.md) — Phase 2E follow-ups (capacity diagnostics, calibration drift placeholders)
- [`docs/roadmaps/phase3_accuracy_push.md`](docs/roadmaps/phase3_accuracy_push.md) — Phase 3 plan: workstreams W10–W13, acceptance bar versus RBF
- [`docs/PHASE3_INDEX.md`](docs/PHASE3_INDEX.md) — Phase 3 fresh-session entry point (per-story checkpoints, conflict matrix)
- [`docs/phase1_result_memo.md`](docs/phase1_result_memo.md) — Phase 1 baseline results and analysis
- [`docs/phase2_result_memo.md`](docs/phase2_result_memo.md) — Phase 2 closing memo (acceptance map, vs Phase 1 / 2C deltas, open questions)
- [`notebooks/04_phase2c_results.ipynb`](notebooks/04_phase2c_results.ipynb) — Phase 2C interactive results notebook
- [`notebooks/05_phase2_results.ipynb`](notebooks/05_phase2_results.ipynb) — Phase 2 closing interactive notebook
- [`docs/experiments/experiment_journal.md`](docs/experiments/experiment_journal.md) — chronological experiment journal (raw evidence per run)

### Operating model

- [`docs/tasks/BOARD.md`](docs/tasks/BOARD.md) — single canonical board (epics, stories, statuses)
- [`docs/workflows/ai_human_collaboration.md`](docs/workflows/ai_human_collaboration.md) — human-AI operating model: modes, task lifecycle, validation gate
- [`docs/workflows/session_protocol.md`](docs/workflows/session_protocol.md) — start / end-of-session checklists and handoff template
- [`docs/workflows/reusable_prompts.md`](docs/workflows/reusable_prompts.md) — copy-pasteable prompts for new sessions
- [`docs/logs/progress_log.md`](docs/logs/progress_log.md) — chronological project progress log

### Setup and decisions

- [`docs/setup/remote_dev.md`](docs/setup/remote_dev.md) — sanitized remote development workflow and environment notes
- [`docs/setup/private_runbook_template.md`](docs/setup/private_runbook_template.md) — template for local-only private ops notes
- [`docs/decisions/0001_remote_dev_stack.md`](docs/decisions/0001_remote_dev_stack.md) — remote dev stack ADR
- [`docs/decisions/0002_phase1_scope_freeze.md`](docs/decisions/0002_phase1_scope_freeze.md) — Phase 1 scope-freeze ADR
- [`docs/decisions/0003_spy_options_data_source_migration.md`](docs/decisions/0003_spy_options_data_source_migration.md) — Dubach → Alpha Vantage migration ADR
- [`docs/decisions/0004_phase3_accuracy_push_framing.md`](docs/decisions/0004_phase3_accuracy_push_framing.md) — Phase 3 framing ADR: locality + inductive bias, not capacity

## Repository Structure

```text
src/neural_iv_surface_inference/   Python package (data, features, models, training, eval, diagnostics)
scripts/                           Entry-point scripts (data prep, training, eval runners, notebook generation)
configs/                           YAML configuration files (training, calibration, decision layer, runner configs)
notebooks/                         Jupyter notebooks (Phase 1 surface gallery, Phase 2C results, Phase 2 closing notebook)
tests/                             Test suite
data/                              Data directories (raw, interim, processed, samples) — gitignored
artifacts/                         Output artifacts (figures, tables, checkpoints, run manifests)
results/                           Committed evidence artifacts (e.g. results/2D/)
docs/                              Project documentation (roadmaps, memos, tasks, decisions, experiments)
```

## Immediate Next Steps

Phase 3 is **opening**. The active next steps are the four
decomposition stories (one per epic). Each runs in **Plan mode**, writes
the breakdown of its epic into atomic stories, and waits for human
review before any implementation. See
[`docs/PHASE3_INDEX.md`](docs/PHASE3_INDEX.md) for the full live
status.

| Story | Mode | Trigger condition |
|---|---|---|
| `3A.1` Decompose Phase 3A | Plan | Human promotes from `backlog → todo`, then runs |
| `3B.1` Decompose Phase 3B | Plan | After 3A produces positional-feature decision |
| `3C.1` Decompose Phase 3C | Plan | After 3B picks a cross-attention architecture |
| `3D.1` Decompose Phase 3D | Plan | After 3C lands (or earlier, if 3D is purely synthesis) |

Carried Phase 2 / 2E follow-ups (lower priority, do not gate Phase 3):

- Re-tune `configs/decision_layer.yaml` operating point (current
  `max_relative_width = 0.5` is tighter than the calibrated Gaussian
  band; forces 100 %-abstention on test).
- Re-run the decision-layer eval across additional AV masking strategies
  and noise regimes (current evidence is `random40_noiselow` only).
- Conformal calibration under the chronological split under-covers by
  ≈ 4.3 pp due to exchangeability violation — document or address.
- Story 2E.3 `latent_dim` sweep (validates the 2E.2 finding; not on
  Phase 3 critical path).

## Security Note

Tracked documentation intentionally excludes sensitive operational details
such as IPs, ports, usernames, SSH config details, private key paths, and
tokens. Those details belong only in a local-only private runbook that must
not be committed.
