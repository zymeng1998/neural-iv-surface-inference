# Task Board

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-06-18T00:00:00-04:00
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
| 3B | Epic | Phase 3 — Cross-attention decoder (ANP picked per ADR 0005; end-to-end DeepSets+ANP, raw `(k, τ)`) — → ANP best-case +2.7% vs RBF on dirty substrate; bar NOT met; **dirty-substrate verdict SUPERSEDED by 3X clean-OTM restatement (gap widened to +61% best head)** | `done` | `docs/roadmaps/phase3_accuracy_push.md` (W11 / §4) | 2026-06-14 |
| 3B.1 | Story | Decompose Phase 3B (ADR 0005 + 3B.2–3B.7 specs) | `done` | `docs/tasks/specs/3B.1_decompose_phase_3b.md` | 2026-06-14 |
| 3B.2 | Story | Local: ANP cross-attention decoder module + `decoder_kind` flag + unit / smoke / integration tests | `done` | `docs/tasks/specs/3B.2_local_anp_cross_attention_decoder.md` | 2026-05-28 |
| 3B.3 | Story | Local: ANP predictor-adapter wiring (evaluator parity test, ≤ minimal patch) | `done` | `docs/tasks/specs/3B.3_local_predictor_adapter.md` | 2026-05-28 |
| 3B.4 | Story | Remote: full AV training of ANP across `head.kind ∈ {gaussian, quantile, point}` | `done` | `docs/tasks/specs/3B.4_remote_full_av_training.md` | 2026-05-28 |
| 3B.5 | Story | Remote: K=5 deep ensemble of ANP point head on AV (mirrors 2D.8) | `done` | `docs/tasks/specs/3B.5_remote_deep_ensemble.md` | 2026-05-29 |
| 3B.6 | Story | Local: calibrator re-fit on ANP val predictions (mirrors 2D.4) | `done` | `docs/tasks/specs/3B.6_local_calibrator_refit.md` | 2026-06-14 |
| 3B.7 | Story | Local: end-to-end decision-layer eval of ANP vs Phase 2D baselines + journal + roadmap closing addendum (ANP +2.7% vs RBF best-case; bar NOT met — **dirty-substrate verdict superseded by 3X clean-OTM restatement**) | `done` | `docs/tasks/specs/3B.7_local_decision_layer_eval.md` | 2026-06-14 |
| 3X | Epic | Phase 3 — Data correction: OTM-restricted surface + paired-coordinate masking + full model-family restatement on `random40_noiselow_otm` (per ADR 0006) — **CLOSED**: RBF still wins, gap WIDENED on clean OTM (+2.7%→+61% best head); DeepSets→ANP architecture story survives; ADR 0006 Implemented | `done` | `docs/roadmaps/phase3_accuracy_push.md` (W11.5 / §4) | 2026-06-02 |
| 3X.1 | Story | Decompose Phase 3X (ADR 0006 addendum + 3X.2–3X.14 specs) | `done` | `docs/tasks/specs/3X.1_decompose_phase_3x.md` | 2026-06-02 |
| 3X.2 | Story | Local: OTM-surface builder `05_build_otm_surface.py` + step-04 `--source` flag + ATM-band (D5) + same-type residual handling (D7) + tests | `done` | `docs/tasks/specs/3X.2_local_otm_surface_builder.md` | 2026-05-30 |
| 3X.3 | Story | Local: vectorised audit v2 + paired-coordinate masking flag (default off) + parity tests | `done` | `docs/tasks/specs/3X.3_local_audit_v2_and_paired_masking.md` | 2026-05-30 |
| 3X.4 | Story | Remote (CPU): build OTM strict surface + rebuild all 11 OTM benchmarks — → single-valued PASS (10,531,499 rows, 0 dup groups), 11 `_otm` benchmarks, dirty files hash-unchanged | `done` | `docs/tasks/specs/3X.4_remote_build_otm_strict_and_benchmarks.md` | 2026-05-30 |
| 3X.5 | Story | Remote (CPU): audit OTM strict + all 11 OTM benchmarks — **HUMAN REVIEW GATE** → **PASS 12/12** (dup 0.0000%, twin leakage 0.0000% all splits; 93.61%→0%) | `done` | `docs/tasks/specs/3X.5_remote_audit_otm_gate.md` | 2026-05-30 |
| 3X.6 | Story | Remote (CPU): early RBF-on-OTM baseline (floor sanity check before GPU spend) — → test MAE **0.00613** (val 0.00615), 0 non-finite; **~10.8× below** RBF-on-dirty 0.0662 → OTM floor far lower, raises the neural bar for 3X.7+ | `done` | `docs/tasks/specs/3X.6_remote_rbf_on_otm_baseline.md` | 2026-05-31 |
| 3X.7 | Story | Remote (GPU): MLP baseline on OTM (Q1 — ladder anchor) — → test MAE **0.03006** (val 0.03391), early-stop ep12/best ep2, finite; dirty MLP 0.0951 → ~3.2× lower on OTM | `done` | `docs/tasks/specs/3X.7_remote_mlp_on_otm.md` | 2026-06-02 |
| 3X.8 | Story | Remote (GPU): Phase 2D DeepSets on OTM — single (2D.7-equiv) + K=5 ensemble (2D.8-equiv) (D2) — → test MAE quantile **0.01418** / gaussian 0.01530 / ensemble 0.01594 / point 0.01752; dirty 2D 0.072–0.079 → ~5× lower on OTM; qmono ok, disagreement>0 | `done` | `docs/tasks/specs/3X.8_remote_deepsets_on_otm.md` | 2026-06-02 |
| 3X.9 | Story | Remote (GPU): ANP on OTM — all three heads gaussian/quantile/point (D1) — → test MAE gaussian **0.01440** / quantile **0.01175** / point **0.00987**, qmono ok; dirty 3B.4 0.0726/0.0681/0.0684 → ~5–7× lower on OTM | `done` | `docs/tasks/specs/3X.9_remote_anp_on_otm.md` | 2026-06-02 |
| 3X.10 | Story | Remote (GPU): ANP K=5 deep ensemble on OTM (mirror 3B.5) — → ensemble test MAE **0.01220**, disagreement mean **0.00679** (dirty 3B.5 disagreement 0.0121 → ~56 % on OTM) | `done` | `docs/tasks/specs/3X.10_remote_anp_ensemble_on_otm.md` | 2026-06-02 |
| 3X.11 | Story | Local: calibrator re-fit on OTM val predictions (mirror 3B.6) — → val cov 0.9000, T=1.005, ens_scale=1.91; val hi-conf MAE 0.00849 < no-abst 0.01349; test cov 0.866 (val→test drift > 3B.6 dirty) | `done` | `docs/tasks/specs/3X.11_local_calibration_on_otm.md` | 2026-06-02 |
| 3X.12 | Story | Remote: decision-layer eval on OTM, thresholds held constant (Q2; mirror 3B.7) — → test MAE **0.01162** / hi-conf MAE **0.00835** / cov@0.90 **0.9295** / mean_width **0.0538** / abstain 1.0 / flag-viol **1814** (dirty 3B.7: 0.0813 / 0.0542 / 0.915 / 0.304 / 1.0 / 9007 — Q2 invariant held: thresholds unchanged) | `done` | `docs/tasks/specs/3X.12_remote_decision_layer_eval_on_otm.md` | 2026-06-02 |
| 3X.13 | Story | Local: dirty-vs-OTM side-by-side comparison tables (matched `random40_noiselow`) — → 11 family×head pairs assembled; OTM beats dirty by **3.0×–10.8×** on test MAE (RBF 10.80×, calibrated-fused 7.00×, MLP 3.01×); long+wide tables cite committed bundles only | `done` | `docs/tasks/specs/3X.13_local_dirty_vs_otm_comparison.md` | 2026-06-02 |
| 3X.14 | Story | Local: 3X closing addendum + methodology-progression narrative (NOT full Phase 3 memo — Q3) — → **verdict: RBF-vs-ANP unchanged in direction, gap WIDENED** (+2.7% dirty → +61% OTM best head, +90% calibrated; bar still NOT MET); DeepSets→ANP architecture story SURVIVES (ANP beats DeepSets all heads, 1.06–1.77×) → no Q5 reopen; ADR 0006 → Implemented | `done` | `docs/tasks/specs/3X.14_local_closing_addendum.md` | 2026-06-02 |
| 3C | Epic | Phase 3 — Feature & inductive-bias expansion (microstructure-only per 3C.1 / ADR 0008; SVI deferred) — on OTM clean substrate — → **CLOSED 2026-06-14 (3C.8): `micro_v1` WORSENED test MAE in all 3 heads (gauss +0.00194 / quant +0.00187 / point +0.00452 vs 3X.9); clean negative result, bar NOT met; ADR 0008 Implemented; downstream eval stories 3C.4–3C.7 cancelled as no longer informative** | `done` | `docs/roadmaps/phase3_accuracy_push.md` (W12 / §4) | 2026-06-14 |
| 3C.1 | Story | Decompose Phase 3C (ADR 0008 + 3C.2–3C.8 specs; microstructure-only / `micro_v1` / 3-head sweep / OTM) | `done` | `docs/tasks/specs/3C.1_decompose_phase_3c.md` | 2026-06-02 |
| 3C.2 | Story | Local: microstructure feature pipeline + `feature_set ∈ {minimal, micro_v1}` flag on loader/encoder/model/predictor + unit/integration tests (no training) | `done` | `docs/tasks/specs/3C.2_local_micro_feature_pipeline.md` | 2026-06-03 |
| 3C.3 | Story | Remote (GPU): full AV retrain of DeepSets+ANP with `feature_set: micro_v1` across `head.kind ∈ {gaussian, quantile, point}` on OTM (mirrors 3X.9). Ran; micro_v1 WORSE than 3X.9 on test MAE in all 3 heads (gauss 0.01634 / quant 0.01362 / point 0.01439 vs 0.01440 / 0.01175 / 0.00987) | `done` | `docs/tasks/specs/3C.3_remote_anp_micro_three_head.md` | 2026-06-03 |
| 3C.4 | Story | Remote (GPU): K=5 ANP+`micro_v1` point-head deep ensemble on OTM (mirrors 3X.10) | `cancelled` | `docs/tasks/specs/3C.4_remote_anp_micro_ensemble.md` | 2026-06-14 |
| 3C.5 | Story | Local: calibrator re-fit on ANP+`micro_v1` val predictions (mirrors 3X.11 / 3B.6) | `cancelled` | `docs/tasks/specs/3C.5_local_calibrator_refit_micro.md` | 2026-06-14 |
| 3C.6 | Story | Remote: decision-layer eval on OTM with thresholds held constant against 3X.12 (Q2 invariant; mirrors 3X.12) | `cancelled` | `docs/tasks/specs/3C.6_remote_decision_layer_eval_micro.md` | 2026-06-14 |
| 3C.7 | Story | Local: OTM-baseline vs OTM+`micro_v1` comparison tables (long + wide) on matched substrate (mirrors 3X.13) | `cancelled` | `docs/tasks/specs/3C.7_local_micro_vs_baseline_comparison.md` | 2026-06-14 |
| 3C.8 | Story | Local: 3C closing addendum on §W12 + ADR 0008 → Implemented + journal/README sync (NOT full Phase 3 memo — 3D) — closed on 3C.3 training evidence alone after 3C.4–3C.7 cancelled | `done` | `docs/tasks/specs/3C.8_local_3c_closing_addendum.md` | 2026-06-14 |
| 3D | Epic | Phase 3 — Closing memo + re-evaluation versus RBF — **CLOSED 2026-06-16: Phase 3 NEGATIVE on accuracy (best clean-OTM head +61% vs RBF; bar NOT met); memo + executed notebook + ADR 0009 (Accepted/Implemented) shipped; RBF stays production baseline; forward = Phase 4 RBF-prior hybrid** | `done` | `docs/roadmaps/phase3_accuracy_push.md` (W13 / §4) | 2026-06-16 |
| 3D.1 | Story | Decompose Phase 3D — executed: epic 3D entered, 3D.2/3D.3/3D.4 drafted, ADR 0009 skeleton created | `done` | `docs/tasks/specs/3D.1_decompose_phase_3d.md` | 2026-06-15 |
| 3D.2 | Story | Local: `scripts/generate_phase3_results_notebook.py` generator scaffold + smoke test (mirrors generate_phase2_results_notebook.py) — implemented: 19-cell build from committed bundles, 5 tests green | `done` | `docs/tasks/specs/3D.2_phase3_notebook_generator.md` | 2026-06-16 |
| 3D.3 | Story | Local: `docs/phase3_result_memo.md` — Phase 3 verdict vs RBF + vs 2D, dirty + clean-OTM restatement, §5 acceptance map — negative verdict (best clean-OTM head +61% vs RBF; bar NOT met) | `done` | `docs/tasks/specs/3D.3_phase3_result_memo.md` | 2026-06-16 |
| 3D.4 | Story | Local: emit `notebooks/06_phase3_results.ipynb` (executed, 0 cell errors), finalize ADR 0009 (Accepted/Implemented), journal close-out, flip epic 3D done + Phase 4 placeholder | `done` | `docs/tasks/specs/3D.4_phase3_notebook_adr_journal_close.md` | 2026-06-16 |
| 4A | Epic | **Phase 4 — RBF-prior hybrid / residual neural model** (per ADR 0004 / 0009 / 0010): σ̂ = RBF + f_θ(residual) + calibrated reliability layer. Entered + decomposed (4A.2–4A.8 + ADR 0010). Bar: any statistically meaningful gain over RBF (0.00613) + reliability preserved. Reopens GPU/Pod spend (4A.4/4A.5). | `in_progress` | `docs/roadmaps/phase4_hybrid_residual.md` | 2026-06-18 |
| 4A.1 | Story | Decompose Phase 4A + ADR 0010 + 4A.2–4A.8 specs | `done` | `docs/tasks/specs/4A.1_decompose_phase_4a.md` | 2026-06-18 |
| 4A.2 | Story | Local: residual-target builder + `target_mode ∈ {absolute, residual}` loader flag + unit/smoke tests (no full build/train) — `residual_targets.py` reuses 3X.6 RBF; absolute byte-identical; 20 tests green | `done` | `docs/tasks/specs/4A.2_residual_target_builder.md` | 2026-06-19 |
| 4A.3 | Story | Remote (CPU): build full residual-target dataset on OTM (per-date RBF at query coords → residuals) + finiteness/MAE-sanity audit — built 10.53M rows, 0 non-finite, val/test mean-abs-residual == 3X.6 RBF MAE; parquet on persistent /workspace | `done` | `docs/tasks/specs/4A.3_remote_build_residual_dataset.md` | 2026-06-20 |
| 4A.4 | Story | Remote (GPU): train residual hybrid across heads {gaussian,quantile,point} on OTM (ANP-residual default; report summed σ̂ MAE vs RBF + 3X.9) — **hybrid BEATS RBF: gaussian 0.006006 / quantile 0.005906 vs RBF 0.006132 (point ties); all ≪ 3X.9** (significance → 4A.7) | `done` | `docs/tasks/specs/4A.4_remote_train_residual_hybrid.md` | 2026-06-20 |
| 4A.5 | Story | Remote (GPU): K=5 deep ensemble of residual point head (mirror 3X.10); disagreement for the calibrator — ensemble test MAE 0.006141 (ties RBF, like 4A.4 single point); disagreement mean 0.000209 | `done` | `docs/tasks/specs/4A.5_remote_residual_ensemble.md` | 2026-06-20 |
| 4A.6 | Story | Local: calibrator re-fit on hybrid val predictions (mirror 3X.11) — fitted: T=1.147, ens_scale=438; test coverage 0.9181 (within ±2pp); 4 tests green | `in_review` | `docs/tasks/specs/4A.6_local_calibrator_refit_hybrid.md` | 2026-06-20 |
| 4A.7 | Story | Remote/local: decision-layer eval of calibrated hybrid on OTM (Q2 thresholds held) + paired bootstrap CI on MAE-delta vs RBF (bar adjudication) | `backlog` | `docs/tasks/specs/4A.7_decision_layer_eval_and_ci.md` | 2026-06-18 |
| 4A.8 | Story | Local: hybrid-vs-RBF-vs-neural comparison + Phase 4 closing memo + ADR 0010 Outcome + journal; flip epic 4A done | `backlog` | `docs/tasks/specs/4A.8_local_phase4_close.md` | 2026-06-18 |
| M1 | Epic | Meta — Multi-agent collaboration infrastructure (ADR 0007: AGENTS.md router + executable rule gates) — → 3 gates live (PMR + dep + scope) + commit-msg trailer; dep gate inspects push range; M1.6 routes waiver audit to an untracked log (no post-commit tracked-file mutation) | `done` | `docs/roadmaps/meta1_agent_collaboration.md` | 2026-06-17 |
| M1.1 | Story | Decompose M1 + ADR 0007 + 4 sub-specs | `done` | `docs/tasks/specs/M1.1_decompose_m1.md` | 2026-05-30 |
| M1.2 | Story | `AGENTS.md` router + `.cursor/rules/000-bootstrap.mdc` + CLAUDE.md re-point | `done` | `docs/tasks/specs/M1.2_agents_md_and_cursor_bootstrap.md` | 2026-05-30 |
| M1.3 | Story | `scripts/check_story_dependencies.py` + tests + `install_hooks.sh` (gate that would have caught 3X.4) | `done` | `docs/tasks/specs/M1.3_check_story_dependencies.md` | 2026-05-30 |
| M1.4 | Story | `scripts/check_file_scope.py` + tests + pre-push wiring | `done` | `docs/tasks/specs/M1.4_check_file_scope.md` | 2026-05-30 |
| M1.5 | Story | `commit-msg` hook — agent-trailer enforcement (Claude / Cursor / Codex / Aider) | `done` | `docs/tasks/specs/M1.5_commit_msg_trailer_hook.md` | 2026-05-30 |
| M1.6 | Story | Pre-push waivers must not mutate tracked files after the pushed commit — both gates record to untracked gitignored `docs/audit/waiver_log.md` via `record_waiver_audit`; 31 gate tests green; ADR 0007 addendum | `done` | `docs/tasks/specs/M1.6_waiver_audit_no_post_commit_mutation.md` | 2026-06-18 |

> When an epic is entered, set it to `in_progress`, add its decomposition story
> (e.g. `2A.1`), then add the resulting stories as new rows. Do not delete or
> renumber existing rows.

## Phase 3 entry point

For Phase 3 work specifically, read [`docs/PHASE3_INDEX.md`](../PHASE3_INDEX.md)
first. It mirrors the relevant subset of this board plus per-story
"last checkpoint" snippets, so a fresh session does not need to load
the entire repo to know the exact next action.
