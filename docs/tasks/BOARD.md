# Task Board

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-25T00:00:00-04:00
---

The single canonical board for all work on this project. Every epic and story
ever created lives here, for the entire project lifecycle. **Rows are never
deleted** — completed work stays on the board marked `done`.

See `docs/workflows/ai_human_collaboration.md` for the operating model and
`docs/tasks/README.md` for board conventions.

## Status legend

| Status | Meaning |
|---|---|
| `backlog` | Captured but not yet pulled into active work. Epics start here; an epic stays `backlog` until it is entered and decomposed. Newly created (undefined) stories also start here. |
| `todo` | Reviewed, fully specified, and prioritized — safe for a session to pick up cold. |
| `in_progress` | Currently being worked in a session. |
| `in_review` | Implementation done; awaiting diff review + tests. |
| `blocked` | Cannot proceed; blocker noted in the story's spec. |
| `done` | Reviewed, tests pass, results committed. Never removed from the board. |

## Hierarchy

- **Epic** = a phase (`2A`, `2B`, `2C`, `2D`). Defined in `docs/roadmaps/`.
- **Story** = an atomic task under an epic (`2A.1`, `2A.2`, ...). Each story has a
  spec in `docs/tasks/specs/`.

Epics are decomposed into stories **one phase at a time** (progressive
decomposition). The first story of any epic is always its decomposition story.

## Board

| ID | Type | Title | Status | Spec / Definition | Updated |
|---|---|---|---|---|---|
| 2A | Epic | Reliability evaluation infrastructure | `done` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W1 / §5) | 2026-05-22 |
| 2A.1 | Story | Decompose Phase 2A | `done` | `docs/tasks/specs/2A.1_decompose_phase_2a.md` | 2026-05-22 |
| 2A.2 | Story | Model-agnostic predictor interface | `done` | `docs/tasks/specs/2A.2_predictor_interface.md` | 2026-05-22 |
| 2A.3 | Story | Core uncertainty-evaluation metrics | `done` | `docs/tasks/specs/2A.3_core_uncertainty_metrics.md` | 2026-05-22 |
| 2A.4 | Story | Abstention / selective-prediction curves | `done` | `docs/tasks/specs/2A.4_abstention_curves.md` | 2026-05-22 |
| 2A.5 | Story | Uncertainty-evaluation runner + artifacts | `done` | `docs/tasks/specs/2A.5_evaluation_runner_artifacts.md` | 2026-05-22 |
| 2B | Epic | Sensitivity & structure diagnostics | `done` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W2 / §5) | 2026-05-22 |
| 2B.1 | Story | Decompose Phase 2B | `done` | `docs/tasks/specs/2B.1_decompose_phase_2b.md` | 2026-05-22 |
| 2B.2 | Story | Masking-sensitivity harness | `done` | `docs/tasks/specs/2B.2_masking_sensitivity.md` | 2026-05-22 |
| 2B.3 | Story | No-arbitrage diagnostics | `done` | `docs/tasks/specs/2B.3_no_arb_diagnostics.md` | 2026-05-22 |
| 2B.4 | Story | Risk-flag synthesis + region heatmaps | `done` | `docs/tasks/specs/2B.4_risk_flags_heatmaps.md` | 2026-05-22 |
| 2B.5 | Story | Diagnostics runner + artifacts | `done` | `docs/tasks/specs/2B.5_diagnostics_runner_artifacts.md` | 2026-05-22 |
| 2C | Epic | Conditional neural surface model | `done` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W3 / §5) | 2026-05-23 |
| 2C.1 | Story | Decompose Phase 2C | `done` | `docs/tasks/specs/2C.1_decompose_phase_2c.md` | 2026-05-22 |
| 2C.2 | Story | Date-grouped conditional dataset + collation | `done` | `docs/tasks/specs/2C.2_conditional_dataset.md` | 2026-05-22 |
| 2C.3 | Story | Set-encoder + coordinate-decoder architecture | `done` | `docs/tasks/specs/2C.3_set_encoder_decoder.md` | 2026-05-22 |
| 2C.4 | Story | Conditional training loop + config | `done` | `docs/tasks/specs/2C.4_conditional_training.md` | 2026-05-22 |
| 2C.5 | Story | Predictor adapter + evaluation parity | `done` | `docs/tasks/specs/2C.5_predictor_adapter_eval.md` | 2026-05-22 |
| 2C.6 | Story | Alpha Vantage ingest implementation + sample validation (local) | `done` | `docs/tasks/specs/2C.6_remote_sync_data_refresh.md` | 2026-05-22 |
| 2C.7 | Story | Remote: full SPY AV pull, replace old dataset, rebuild pipeline | `done` | `docs/tasks/specs/2C.7_remote_full_pull_replace_dataset.md` | 2026-05-23 |
| 2C.8 | Story | Re-run Phase 1 baselines + 2A/2B eval on new AV data | `done` | `docs/tasks/specs/2C.8_rerun_baselines_eval_on_new_data.md` | 2026-05-23 |
| 2D | Epic | Uncertainty-aware inference & decision layer | `done` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W4+W5 / §5) | 2026-05-25 |
| 2D.1 | Story | Decompose Phase 2D | `done` | `docs/tasks/specs/2D.1_decompose_phase_2d.md` | 2026-05-24 |
| 2D.2 | Story | Local: heteroscedastic / quantile head — code + synthetic smoke | `done` | `docs/tasks/specs/2D.2_heteroscedastic_quantile_head.md` | 2026-05-24 |
| 2D.3 | Story | Local: deep ensemble adapter — manifest + dummy tests + synthetic smoke | `done` | `docs/tasks/specs/2D.3_deep_ensemble_disagreement.md` | 2026-05-24 |
| 2D.4 | Story | Local: calibrated confidence score + uncertainty band | `done` | `docs/tasks/specs/2D.4_calibrated_confidence_score.md` | 2026-05-25 |
| 2D.5 | Story | Local: abstention + tradability + risk-flag decision layer | `done` | `docs/tasks/specs/2D.5_abstention_tradability_decision_layer.md` | 2026-05-25 |
| 2D.6 | Story | Local: decision-layer runner skeleton + synthetic smoke | `done` | `docs/tasks/specs/2D.6_decision_layer_runner_artifacts.md` | 2026-05-25 |
| 2D.7 | Story | Remote: full AV training — Gaussian + quantile heads (+ point control) | `done` | `docs/tasks/specs/2D.7_remote_train_heteroscedastic_quantile.md` | 2026-05-25 |
| 2D.8 | Story | Remote: K-seed deep ensemble training on AV | `done` | `docs/tasks/specs/2D.8_remote_train_deep_ensemble.md` | 2026-05-25 |
| 2D.9 | Story | Remote: end-to-end decision-layer eval on AV + artifacts + journal | `done` | `docs/tasks/specs/2D.9_remote_decision_layer_e2e.md` | 2026-05-25 |

> When an epic is entered, set it to `in_progress`, add its decomposition story
> (e.g. `2A.1`), then add the resulting stories as new rows. Do not delete or
> renumber existing rows.
