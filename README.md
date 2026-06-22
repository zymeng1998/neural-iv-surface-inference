# Neural IV Surface Inference

## First-time setup (per clone)

```bash
bash scripts/install_hooks.sh
```

Installs project-managed git hooks (PMR + story-dep + file-scope + agent-
trailer gates). Required on every fresh clone — laptop, RunPod, CI. See
[`AGENTS.md`](AGENTS.md) for the multi-agent collaboration entry point
and [ADR 0007](docs/decisions/0007_multi_agent_handoff.md) for the why.

## Overview

Neural IV Surface Inference is an ML × Finance project focused on recovering
implied-volatility surfaces from sparse, noisy, and irregular option
observations. The current direction is **reliability-aware surface
inference**: beyond producing a smooth surface, the system knows where its
predictions are trustworthy and abstains where they are not.

## Current Phase

**Phase 4 — RBF-Prior Hybrid / Residual Neural Model — CLOSED POSITIVE
(epic `4A` done, 2026-06-20).** Phase 3 had closed **negative on accuracy**:
no pure conditional-neural variant beat RBF on its own (best ANP point head
+61 %; [ADR 0009](docs/decisions/0009_phase3_production_predictor_selection.md)),
so Phase 4 stopped fighting RBF and **stood on it**: predict
`σ̂ = RBF + f_θ(residual)`, RBF carrying the local interpolation it wins and
the neural model learning the residual plus the calibrated reliability/
abstention layer RBF lacks ([ADR 0010](docs/decisions/0010_rbf_prior_residual_hybrid.md),
now Implemented). **The bar was MET:** the calibrated gaussian hybrid posts
clean-OTM test MAE **0.006006** vs the RBF floor **0.006132** — mean Δ
**−0.000126**, date-clustered paired-bootstrap **95 % CI [−0.000144,
−0.000106]** (excludes 0) → the **first predictor in the project to
statistically significantly beat RBF** on the well-posed substrate.
Reliability preserved/tightened (coverage 0.9181 vs iv_true; hi-conf MAE
0.004710 < no-abstention; width 0.0328). **Production recommendation: adopt
the RBF-prior gaussian hybrid** (caveats: an iv_clean coverage-refit and a
deferred no-arb flag-count audit). Full synthesis:
[`docs/phase4_result_memo.md`](docs/phase4_result_memo.md); comparison bundle
[`results/4/spy_phase1_random40_noiselow_otm/4a_compare/`](results/4/spy_phase1_random40_noiselow_otm/4a_compare/comparison_wide.md).
See also
[`docs/roadmaps/phase4_hybrid_residual.md`](docs/roadmaps/phase4_hybrid_residual.md)
and [`docs/PHASE4_INDEX.md`](docs/PHASE4_INDEX.md).

### Next Direction (Phase 4B closed → accuracy survives; reliability-first in parallel)

Phase 4A is the adopted production point estimator. On the dense, clean
benchmark the win over RBF is small (≈ 2 % relative MAE) — but Phase 4B asked
the sharper question: does that edge **survive sparsity**? The forward direction
is staged ([ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md)):

1. **`4B` — sparsity-sweep diagnostic (the decision gate). ✅ Closed 2026-06-22 —
   verdict `ambiguous` → `true` (accuracy survives) after the fair retrain.**
   RBF vs the RBF-prior hybrid over four sparsity regimes × five rungs. The
   *eval-time* read (a full-context checkpoint scored on sparse context) showed
   the hybrid's relative edge **collapsing** under thinned wings / dropped
   maturities — but that was an out-of-distribution artifact. The **full fair
   retrain (4B.7)** — retraining the hybrid on each rung's sparse context —
   **reverses it**: the relative edge over RBF grows sharply with structured
   sparsity (thin wings **+16.6 %**, missing maturities **+37.1 %**, combined
   **+16.0 %** at the sparsest rung; all significant), while random unstructured
   thinning is where neither method has an edge. So raw accuracy **survives** and
   stays a core story. (Caveat: each cell trains a model for its specific
   sparsity; a single mixed-sparsity model is the production-realistic follow-up.)
   A parallel reliability finding — calibrated coverage collapses 0.96 → 0.35
   under sparsity — shows reliability degrades even faster, motivating Phase 5.
   ([`docs/roadmaps/phase4b_sparsity_sweep.md`](docs/roadmaps/phase4b_sparsity_sweep.md))
2. **`5A` — reliability-first surface inference / quote-risk layer (proceeds in
   parallel).** Formalize the durable Phase 2–4 asset — calibrated uncertainty,
   abstention, surface-level confidence, **per-regime recalibration**, no-arb /
   risk flags, quote/no-quote logic, and spread / model-risk-reserve suggestions
   — into a sell-side-desk-consumable system. The 4B coverage-collapse result
   makes this a high-value forward story alongside the (now alive) accuracy work.
   ([`docs/roadmaps/phase5_reliability_first_surface_inference.md`](docs/roadmaps/phase5_reliability_first_surface_inference.md))
3. **`6A` — structured-product pricing / FCN monetization demo.** A *sell-side*
   framing (bank desk using the model for pricing confidence, model-risk
   reserves, hedge-risk diagnostics — not retail, not the operator issuing
   notes): a **constrained** demo propagating surface uncertainty into
   fair-value intervals, sensitivities, and quote-risk diagnostics for an
   FCN-like payoff. It does **not** price FCNs today — structured notes are
   debt + embedded derivatives, so the IV surface is one critical pricing
   input, not the whole pricer (rates, dividends, funding/issuer spread,
   correlation, barriers, liquidity, and hedging stay out of scope).
   ([`docs/roadmaps/phase6_structured_product_pricing.md`](docs/roadmaps/phase6_structured_product_pricing.md))

The 4B decision gate has **resolved** (accuracy retired on this substrate);
**Phase 5A reliability-first is the active next phase**, with 6A (structured-
product pricing demo) downstream.

> **2026-05-29 — Data-integrity finding.** A duplicate-coordinate audit
> ([`docs/research/duplicate_coordinate_audit.md`](docs/research/duplicate_coordinate_audit.md))
> found that **93.61 %** of strict-table rows live inside a
> `(date, expiration, strike)` duplicate group, **100 %** of those
> duplicates are call-put leg pairs, and the median in-group IV range
> is **0.049** (≈ the current ANP test MAE bar). This invalidates the
> single-valued-function assumption the conditional model and the RBF
> baseline both rely on; cross-model MAE comparisons are biased in
> RBF's favour on dense regions, and the proposed sparse-region
> ANP-vs-RBF experiment is not interpretable as written. A new gating
> epic **3X — Data correction** (OTM-restricted surface +
> paired-coordinate masking + re-audit + ANP re-train + decision-layer
> re-eval, per
> [ADR 0006](docs/decisions/0006_duplicate_coordinate_data_correction.md))
> is inserted between 3B and 3C/3D. Phase 2 / Phase 3A / Phase 3B
> artifacts are preserved unchanged; the Phase 3D closing memo will
> append an OTM-clean re-statement of the 3B verdict alongside the
> original numbers. Full narrative:
> [retrospective 0002](docs/retrospectives/0002_call_put_duplicate_coordinate_discovery.md).
>
> **2026-06-02 — 3X closed.** The OTM-restricted surface was built,
> audited (PASS 12/12, 93.61 %→0 % duplication, 0 % twin leakage), and
> the full model-family ladder was restated on the matched clean
> benchmark `spy_phase1_random40_noiselow_otm`. **Verdict:** RBF still
> wins and the gap *widened* — the dirty call-put confound had been
> flattering the neural models by forcing RBF to average two-valued
> targets. Best ANP head goes from +2.7 % (dirty) to **+61 %** vs RBF on
> clean test MAE (calibrated production +90 %); the Phase 3 bar is still
> NOT met, by a wider margin. The DeepSets→ANP architecture story
> survives (ANP beats DeepSets at every matched head on OTM), so no
> Phase 2 reopen and ADR 0006 → **Implemented**. Narrative:
> [methodology progression](docs/research/duplicate_coordinate_methodology_progression.md).

### Phase 3 (complete, 2026-06-16 — negative on accuracy; superseded by Phase 4)

| Epic | Workstream | Status |
|---|---|---|
| 3A | W10 — Coordinate-representation ablation (Fourier vs raw `(k, τ)`, decoder-only) | `done` — raw beats Fourier (0.0760 vs 0.0790 full-fold test MAE); gap-to-RBF unclosed |
| 3B | W11 — Cross-attention decoder (end-to-end DeepSets + ANP, raw `(k, τ)`) | `done` — **bar NOT met**: ANP best-case +2.7 % vs RBF (dirty); reliability holds. Restated on clean OTM in 3X. |
| **3X** | **W11.5 — Data correction (OTM-restricted surface + paired-coordinate masking + re-audit + full model-family restatement + decision-layer re-eval)** | **`done` (2026-06-02)** — RBF still wins, gap WIDENED (+2.7 %→+61 % best head, +90 % calibrated; bar still NOT met); DeepSets→ANP survives; ADR 0006 Implemented |
| 3C | W12 — Feature & inductive-bias expansion (microstructure-only `micro_v1`; SVI deferred) | `done` (2026-06-14) — **`micro_v1` WORSENED test MAE in all 3 heads vs 3X.9; bar NOT met; ADR 0008 Implemented; downstream eval stories 3C.4–3C.7 cancelled** |
| 3D | W13 — Phase 3 closing memo + re-evaluation versus RBF (OTM-clean re-statement; frames Phase 4) | `done` (2026-06-16) — **Phase 3 CLOSED, negative on accuracy**: memo + executed notebook + ADR 0009 (Accepted/Implemented) shipped; RBF stays production baseline |

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

**3X closing result (epic 3X, 2026-06-02).** Restating the full
model-family ladder on the corrected single-valued OTM benchmark
sharpened the 3B verdict rather than overturning it. Every family
improves 3–11× on clean data (RBF floor 0.0662→0.00613, 10.8×; best ANP
point head 0.0684→0.00987, 6.9×), but because RBF was the model most
penalised by the dirty call-put confound, correcting it *widened* RBF's
lead: the best-ANP-head gap grew from +2.7 % to **+61 %** (calibrated
production +90 %), so the bar is missed by more, not less. The
DeepSets→ANP ranking holds on clean data (ANP beats DeepSets at every
matched head, 1.06–1.77×) — the architecture story survives and no
Phase 2 reopen is warranted. Scope is the matched
`random40_noiselow_otm` substrate only; all-11-variant robustness is
deferred future work. Evidence:
[`results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.md`](results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.md);
narrative:
[`docs/research/duplicate_coordinate_methodology_progression.md`](docs/research/duplicate_coordinate_methodology_progression.md).
This clean-OTM restatement is the operative verdict — it **supersedes**
the original 3B dirty-substrate interpretation (the §W11 addendum is
preserved for traceability only).

**3C closing result (epic 3C, 2026-06-14).** Feature expansion was the
last cheap lever. The `micro_v1` set (ADR 0008 — six AV-native per-quote
microstructure fields appended to the minimal `(log_moneyness, τ, iv)`
tuple → 9-dim context) was trained end-to-end on the matched clean OTM
substrate across all three heads (3C.3), a faithful restatement of 3X.9
with only the feature tuple changed. The result was a **clean
negative**: `micro_v1` *worsened* test MAE in every head (gaussian
0.01634 / quantile 0.01362 / point 0.01439 vs 3X.9's 0.01440 / 0.01175 /
0.00987), all still far above the RBF-on-OTM floor 0.00613. The
regression is head-uniform — the microstructure context did not give the
encoder a useful quote-reliability signal at this benchmark. Because
that settles the headline 3C question against `micro_v1`, the downstream
eval chain (ensemble / calibrator / decision-layer / comparison —
stories 3C.4–3C.7) was **cancelled** as no longer informative, and 3C.8
closed the epic on the 3C.3 evidence alone with **ADR 0008 →
Implemented**. The Phase 3 bar is still NOT met. Evidence:
`artifacts/runs/3C3/{gaussian,quantile,point}/manifest.json` and the
§W12 closing addendum in the roadmap.

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
- [`docs/roadmaps/phase4_hybrid_residual.md`](docs/roadmaps/phase4_hybrid_residual.md) — Phase 4 plan: RBF-prior residual hybrid, success bar, workstreams 4A.1–4A.8 (closed positive)
- [`docs/PHASE4_INDEX.md`](docs/PHASE4_INDEX.md) — Phase 4 fresh-session entry point (per-story checkpoints)
- [`docs/roadmaps/phase4b_sparsity_sweep.md`](docs/roadmaps/phase4b_sparsity_sweep.md) — Phase 4B plan: sparsity-sweep diagnostic (accuracy-survival gate; planned)
- [`docs/roadmaps/phase5_reliability_first_surface_inference.md`](docs/roadmaps/phase5_reliability_first_surface_inference.md) — Phase 5 plan: reliability-first surface inference / quote-risk layer (planned)
- [`docs/roadmaps/phase6_structured_product_pricing.md`](docs/roadmaps/phase6_structured_product_pricing.md) — Phase 6 plan: structured-product pricing / FCN monetization demo (planned)
- [`docs/phase1_result_memo.md`](docs/phase1_result_memo.md) — Phase 1 baseline results and analysis
- [`docs/phase2_result_memo.md`](docs/phase2_result_memo.md) — Phase 2 closing memo (acceptance map, vs Phase 1 / 2C deltas, open questions)
- [`docs/phase3_result_memo.md`](docs/phase3_result_memo.md) — Phase 3 closing memo (negative on accuracy; RBF floor vs neural ladder; frames Phase 4)
- [`docs/phase4_result_memo.md`](docs/phase4_result_memo.md) — Phase 4 closing memo (**positive**: RBF-prior hybrid beats RBF; production recommendation)
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
- [`docs/decisions/0005_cross_attention_architecture_choice.md`](docs/decisions/0005_cross_attention_architecture_choice.md) — Phase 3B architecture ADR: Attentive Neural Process (ANP)
- [`docs/decisions/0006_duplicate_coordinate_data_correction.md`](docs/decisions/0006_duplicate_coordinate_data_correction.md) — Phase 3X data-correction ADR: OTM-restricted surface + paired masking
- [`docs/decisions/0007_multi_agent_handoff.md`](docs/decisions/0007_multi_agent_handoff.md) — multi-agent handoff ADR: git-hook-enforced PMR / dep / scope / trailer gates
- [`docs/decisions/0008_microstructure_feature_set_freeze.md`](docs/decisions/0008_microstructure_feature_set_freeze.md) — Phase 3C feature-set ADR: `micro_v1` freeze (Implemented; negative result)
- [`docs/decisions/0009_phase3_production_predictor_selection.md`](docs/decisions/0009_phase3_production_predictor_selection.md) — Phase 3 close ADR: RBF stays the accuracy baseline; reliability layer retained
- [`docs/decisions/0010_rbf_prior_residual_hybrid.md`](docs/decisions/0010_rbf_prior_residual_hybrid.md) — Phase 4 ADR: RBF-prior residual hybrid (**Implemented** — hybrid adopted as production accuracy surface)
- [`docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md`](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md) — forward-strategy ADR (**Accepted**): adopt 4A as production estimator, stop benchmark grinding; 4B accuracy-survival gate → 5A reliability-first → 6A structured-product pricing demo
- [`docs/retrospectives/0002_call_put_duplicate_coordinate_discovery.md`](docs/retrospectives/0002_call_put_duplicate_coordinate_discovery.md) — discovery retrospective for the duplicate-coordinate finding

## Repository Structure

```text
src/neural_iv_surface_inference/   Python package (data, features, models, training, eval, diagnostics)
scripts/                           Entry-point scripts (data prep, training, eval runners, notebook generation)
configs/                           YAML configuration files (training, calibration, decision layer, runner configs)
notebooks/                         Jupyter notebooks (Phase 1 surface gallery, Phase 2C / Phase 2 closing, Phase 3 results)
tests/                             Test suite
data/                              Data directories (raw, interim, processed, samples) — gitignored
artifacts/                         Output artifacts (figures, tables, checkpoints, run manifests)
results/                           Committed evidence artifacts (results/2D/, results/3/, results/4/)
docs/                              Project documentation (roadmaps, memos, tasks, decisions, experiments)
```

## Immediate Next Steps

**Phase 4 (`4A`) closed POSITIVE (2026-06-20):** the RBF-prior gaussian
residual hybrid is adopted as the production accuracy surface (first predictor
to statistically significantly beat RBF; [ADR 0010](docs/decisions/0010_rbf_prior_residual_hybrid.md)
Implemented). All epics through Phase 4 are `done`; **no phase is currently
in flight.** The forward direction is staged as `4B → 5A → 6A`
([ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md);
see **Next Direction** above) — all `backlog`, with **4B the decision gate**
for whether accuracy stays a core story. The deferred follow-ups below are
backlog placeholders that fold into those phases (the all-11-variant sweep into
4B's robustness framing; the calibrator-refit and no-arb audit into 5A's
reliability layer):

| Follow-up | Mode | Notes |
|---|---|---|
| Calibrator re-fit on `iv_clean` | Implement | Tighten the conservative coverage side (hybrid over-covers iv_clean at 0.962; iv_true coverage 0.9181 is in-band). Local, CPU. |
| No-arb flag-count audit on the hybrid checkpoint | Implement | Confirm forbidden-flag behavior tracks RBF's; needs the released-pod checkpoint. Deferred per the 4A.7 spec. |
| All-11-OTM-variant robustness study | Plan → Implement | Every Phase 3/4 number is scoped to the matched `random40_noiselow_otm` substrate only; the broader sweep stays deferred. |
| Region-gated / multiplicative residual variants | Plan | ADR 0010 alternatives — only if a larger accuracy margin is wanted. |

Carried Phase 2 / 2E follow-ups (dormant, lower priority):

- Re-tune `configs/decision_layer.yaml` operating point (current
  `max_relative_width = 0.5` is tighter than the calibrated Gaussian
  band; forces 100 %-abstention on test).
- Re-run the decision-layer eval across additional AV masking strategies
  and noise regimes (current evidence is `random40_noiselow` only).
- Conformal calibration under the chronological split under-covers by
  ≈ 4.3 pp due to exchangeability violation — document or address.
- Story 2E.3 `latent_dim` sweep (validates the 2E.2 finding; dormant).

## Security Note

Tracked documentation intentionally excludes sensitive operational details
such as IPs, ports, usernames, SSH config details, private key paths, and
tokens. Those details belong only in a local-only private runbook that must
not be committed.
