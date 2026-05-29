# Task Board

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-28T13:00:00-04:00
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
| `cancelled` | Deliberately abandoned — superseded, no longer informative, or scope-killed by upstream findings. Row stays on the board with a one-line rationale in the linked spec. Never removed. |

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
| 2D.10 | Story | Local: Phase 2 results memo + notebook | `done` | `docs/tasks/specs/2D.10_phase2_results_memo_notebook.md` | 2026-05-25 |
| 2E | Epic | Phase 2 follow-ups | `in_progress` | `docs/roadmaps/phase2_followups.md` | 2026-05-26 |
| 2E.1 | Story | Decompose Phase 2E | `done` | `docs/tasks/specs/2E.1_decompose_phase_2e.md` | 2026-05-26 |
| 2E.2 | Story | Latent capacity diagnostic — effective rank + PCA + contribution analysis on the 2D.7 checkpoint | `done` | `docs/tasks/specs/2E.2_latent_capacity_diagnostic.md` | 2026-05-27 |
| 2E.3 | Story | `latent_dim` sweep (scope set by 2E.2 findings) | `cancelled` | `docs/tasks/specs/2E.3_latent_dim_sweep.md` | 2026-05-28 |
| 3A | Epic | Phase 3 — Coordinate-representation ablation (Fourier vs raw `(k, τ)`, decoder-only) — → raw beats Fourier on full-fold test MAE (0.0760 vs 0.0790, Δ +0.00300); gap-to-RBF unclosed; 3B default = raw | `done` | `docs/roadmaps/phase3_accuracy_push.md` (W10 / §4) | 2026-05-28 |
| 3A.1 | Story | Decompose Phase 3A | `done` | `docs/tasks/specs/3A.1_decompose_phase_3a.md` | 2026-05-28 |
| 3A.2 | Story | Local: Fourier-feature module + `coord_encoding` flag + unit tests + synthetic smoke | `done` | `docs/tasks/specs/3A.2_local_fourier_feature_module.md` | 2026-05-28 |
| 3A.3 | Story | Remote: decoder-only retrain on frozen 2D.7 encoder — Fourier vs raw variants | `done` | `docs/tasks/specs/3A.3_remote_decoder_only_retrain.md` | 2026-05-28 |
| 3A.4 | Story | Local: W1 evaluation of both variants vs 2D.9 baselines + journal + roadmap addendum | `done` | `docs/tasks/specs/3A.4_local_eval_and_addendum.md` | 2026-05-28 |
| 3B | Epic | Phase 3 — Cross-attention decoder (ANP picked per ADR 0005; end-to-end DeepSets+ANP, raw `(k, τ)`) | `in_progress` | `docs/roadmaps/phase3_accuracy_push.md` (W11 / §4) | 2026-05-28 |
| 3B.1 | Story | Decompose Phase 3B (ADR 0005 + 3B.2–3B.7 specs) | `in_review` | `docs/tasks/specs/3B.1_decompose_phase_3b.md` | 2026-05-28 |
| 3B.2 | Story | Local: ANP cross-attention decoder module + `decoder_kind` flag + unit / smoke / integration tests | `done` | `docs/tasks/specs/3B.2_local_anp_cross_attention_decoder.md` | 2026-05-28 |
| 3B.3 | Story | Local: ANP predictor-adapter wiring (evaluator parity test, ≤ minimal patch) | `done` | `docs/tasks/specs/3B.3_local_predictor_adapter.md` | 2026-05-28 |
| 3B.4 | Story | Remote: full AV training of ANP across `head.kind ∈ {gaussian, quantile, point}` | `done` | `docs/tasks/specs/3B.4_remote_full_av_training.md` | 2026-05-28 |
| 3B.5 | Story | Remote: K=5 deep ensemble of ANP point head on AV (mirrors 2D.8) | `done` | `docs/tasks/specs/3B.5_remote_deep_ensemble.md` | 2026-05-29 |
| 3B.6 | Story | Local: calibrator re-fit on ANP val predictions (mirrors 2D.4) | `in_review` | `docs/tasks/specs/3B.6_local_calibrator_refit.md` | 2026-05-28 |
| 3B.7 | Story | Local: end-to-end decision-layer eval of ANP vs Phase 2D baselines + journal + roadmap closing addendum (ANP +2.7% vs RBF best-case; bar NOT met) | `in_review` | `docs/tasks/specs/3B.7_local_decision_layer_eval.md` | 2026-05-28 |
| 3C | Epic | Phase 3 — Feature & inductive-bias expansion (microstructure, optional SVI head) | `backlog` | `docs/roadmaps/phase3_accuracy_push.md` (W12 / §4) | 2026-05-27 |
| 3C.1 | Story | Decompose Phase 3C | `backlog` | `docs/tasks/specs/3C.1_decompose_phase_3c.md` | 2026-05-27 |
| 3D | Epic | Phase 3 — Closing memo + re-evaluation versus RBF | `backlog` | `docs/roadmaps/phase3_accuracy_push.md` (W13 / §4) | 2026-05-27 |
| 3D.1 | Story | Decompose Phase 3D | `backlog` | `docs/tasks/specs/3D.1_decompose_phase_3d.md` | 2026-05-27 |

> When an epic is entered, set it to `in_progress`, add its decomposition story
> (e.g. `2A.1`), then add the resulting stories as new rows. Do not delete or
> renumber existing rows.

## Phase 3 entry point

For Phase 3 work specifically, read [`docs/PHASE3_INDEX.md`](../PHASE3_INDEX.md)
first. It mirrors the relevant subset of this board plus per-story
"last checkpoint" snippets, so a fresh session does not need to load
the entire repo to know the exact next action.
