# Phase 4 Roadmap — RBF-Prior Hybrid / Residual Neural Model

---
created_at: 2026-06-18T00:00:00-04:00
last_updated_at: 2026-06-18T00:00:00-04:00
status: in_progress
---

> Phase 4 is the deployment-engineering direction reserved by
> [ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md) and
> selected by [ADR 0009](../decisions/0009_phase3_production_predictor_selection.md)
> after Phase 3 closed **negative on accuracy** (no pure conditional-neural
> variant beat RBF on the clean OTM substrate; best head +61 %). The design
> is locked by [ADR 0010](../decisions/0010_rbf_prior_residual_hybrid.md).

## 1) Goal

Stop trying to beat RBF *from scratch*; instead **stand on it**. Let the
per-date RBF interpolation carry the local surface (which it already wins),
and train a neural model to predict the **residual** `r = iv − rbf_pred` —
the structured error RBF leaves behind (sparse wings, extreme maturities,
thin-quote regions) — plus the calibrated reliability / abstention layer RBF
lacks.

Prediction: `σ̂(k, τ) = RBF_t(k, τ) + f_θ(k, τ | context_t)`.

## 2) Success bar (operator-set, 2026-06-18)

Reframed from Phase 3's "beat RBF on its own" (moot for a hybrid). Phase 4
succeeds when **both** hold on `spy_phase1_random40_noiselow_otm` (test):

1. **Any statistically meaningful accuracy gain over RBF-alone.** The
   calibrated hybrid's test MAE is **significantly below** the RBF floor
   (0.00613, story 3X.6) — the margin need *not* reach 5 %, but the
   improvement must be real: a paired bootstrap 95 % CI on the per-query
   MAE difference (hybrid − RBF) excludes 0 (and/or a sign/Wilcoxon test).
2. **Reliability preserved.** Coverage within ±2 pp of 0.90; high-confidence
   MAE (`keep_fraction = 0.8`) strictly below no-abstention MAE;
   forbidden-flag violations not meaningfully worse than 3X.12.

**Negative branch (explicitly acceptable):** if the residual model cannot
significantly beat RBF *even with RBF as the prior*, that is a strong,
publishable finding — RBF is effectively optimal on this substrate — and the
Phase 4 deliverable becomes **the calibrated reliability/abstention layer on
top of RBF** (accuracy-neutral, reliability-positive). Either way Phase 4
ships a production recommendation.

No-overclaim guardrail (carried from 3X): all numbers are on the matched
`random40_noiselow_otm` substrate; the all-11-variant study stays deferred.

## 3) Workstreams (epic 4A)

| Story | Locale | Deliverable |
|---|---|---|
| 4A.1 | local | Decompose Phase 4A + ADR 0010 + these child specs |
| 4A.2 | local | Residual-target builder + `target_mode ∈ {absolute, residual}` loader flag + unit/smoke tests (no full build, no train) |
| 4A.3 | remote CPU | Build the full residual-target dataset on OTM (per-date RBF preds at query coords → residuals; finiteness audit) |
| 4A.4 | remote GPU | Train the residual hybrid across heads {gaussian, quantile, point} on OTM (reuse DeepSets+ANP backbone, residual target) |
| 4A.5 | remote GPU | K=5 deep ensemble of the residual point head (mirror 3X.10) |
| 4A.6 | local | Calibrator re-fit on hybrid val predictions (mirror 3X.11) |
| 4A.7 | remote/local | Decision-layer eval of the calibrated hybrid on OTM + paired bootstrap CI on MAE-delta vs RBF (the bar adjudication) |
| 4A.8 | local | Hybrid-vs-RBF-vs-ANP comparison + Phase 4 closing addendum + ADR 0010 Outcome + journal |

Dependency chain: `4A.1 → 4A.2 → 4A.3 → 4A.4 → 4A.5 → 4A.6 → 4A.7 → 4A.8`
(4A.5 mirrors the ensemble step; 4A.6 consumes 4A.4+4A.5 val predictions).
4A.4 + 4A.5 share one Pod-GPU window; 4A.3 is a CPU pre-step that can share a
CPU pod or run on the GPU pod before training. Each story is atomic — one
question, one artifact bundle, one acceptance check; no story spans
local + remote.

## 4) Substrate & baselines

- **Substrate:** `spy_phase1_random40_noiselow_otm` (the matched clean OTM
  benchmark; same chronological splits as 3X / 3C).
- **Floor to beat:** RBF-on-OTM test MAE **0.00613** (3X.6).
- **Reference points:** best pure-neural ANP point head 0.00987 (3X.9);
  calibrated production 0.01162 (3X.12). The hybrid should land **between
  RBF and these**, ideally below RBF.

## 5) Non-goals

- No new data source / ingest path (residuals are derived from committed OTM
  data + the existing RBF baseline). A new source would need a Phase-2-style
  data ADR first.
- No change to the OTM cleaning rules or benchmark construction.
- No GPU spend until 4A.2/4A.3 land and the operator green-lights the Pod
  window for 4A.4.
- Phase 4 does not reopen the Phase 3 verdict; it builds on it.

## 6) Decisions

- [ADR 0010](../decisions/0010_rbf_prior_residual_hybrid.md) — RBF-prior
  residual hybrid design (residual target, backbone, fusion, the success
  bar). Proposed at kickoff; Implemented on the 4A.8 close.
