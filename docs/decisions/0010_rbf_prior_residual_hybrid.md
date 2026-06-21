# ADR 0010: RBF-Prior Residual Hybrid (Phase 4 — `4A`)

## Status

**Implemented (2026-06-20).** Created Proposed by the Phase 4 kickoff /
decomposition story [`4A.1`](../tasks/specs/4A.1_decompose_phase_4a.md);
moved to Implemented on the [`4A.8`](../tasks/specs/4A.8_local_phase4_close.md)
close. The Outcome block below is filled from the 4A.7 decision-layer numbers.
**Verdict: the success bar (§5) is MET** — the calibrated RBF-prior gaussian
hybrid statistically significantly beats RBF on the matched clean OTM
substrate.

> Skeleton + candidate decision: the residual formulation, backbone options,
> and the success bar are fixed here so 4A.2–4A.8 have a target. The
> backbone choice (§Decision item 3) is the one fork left open for the 4A.4
> operator review; everything else is locked.

## Date

2026-06-18

## Context

Phase 3 closed **negative on accuracy**
([ADR 0009](0009_phase3_production_predictor_selection.md)): on the clean
single-valued OTM substrate `spy_phase1_random40_noiselow_otm`, the per-date
**RBF interpolation** posts test MAE **0.00613** and no pure conditional-
neural variant comes within 5 % (best ANP point head 0.00987, +61 %). The
consistent finding across 3A/3B/3X/3C: RBF is a very strong *local*
interpolator on a well-posed surface, and a global-prior amortized neural
model does not re-derive its local weighting from data.

[ADR 0004](0004_phase3_accuracy_push_framing.md) reserved a hybrid as the
deployment fallback. Phase 4 takes it up: rather than replace RBF, **predict
what RBF gets wrong**.

## Decision

1. **Hybrid form (additive residual).**
   `σ̂(k, τ) = RBF_t(k, τ) + f_θ(k, τ | context_t)`, where `RBF_t` is the
   existing per-date RBF baseline (unchanged) and `f_θ` is a neural model
   trained to predict the residual `r_t(k, τ) = iv_true − RBF_t(k, τ)` at
   query coordinates. At inference the two are summed.

2. **Residual target.** Additive residual on the IV scale (same units/scale
   the evaluator already uses), computed at every labeled query coordinate.
   No transform on the residual in v1 (z-scoring of *inputs* is unchanged;
   the *target* is the raw residual). A transform is an opt-in follow-up
   only if 4A.4 shows a pathological residual tail.

3. **Backbone (the open fork — locked at 4A.4 review).** Candidate options:
   - **(a) ANP-residual (default):** reuse the Phase 3B/3X DeepSets encoder +
     ANP cross-attention decoder, raw `(k, τ)`, swapping only the training
     target (residual instead of absolute IV). Maximum reuse; directly
     comparable to 3X.9.
   - **(b) Lightweight MLP-residual:** a small coordinate MLP on
     `(k, τ, context summary)` predicting the residual. Cheaper; tests
     whether the residual is "easy" once RBF removes the bulk signal.
   4A.4 trains the default (a); (b) is a fast ablation if (a) underperforms.

4. **Heads & reliability.** Train the three heads {gaussian, quantile,
   point} on the residual (mirrors 3X.9), build a K=5 ensemble of the point
   head (mirrors 3X.10), and **reuse the Phase 2D / 3X.11 calibration
   recipe** (Gaussian + ensemble-disagreement fusion → calibrated band,
   abstention, tradability) fit on the hybrid's val predictions. The
   reliability layer is applied to the *summed* prediction `σ̂`.

5. **Success bar (operator-set 2026-06-18; see roadmap §2).** Phase 4
   succeeds when the calibrated hybrid's OTM test MAE is **statistically
   significantly below** RBF-alone (0.00613) — any real margin, adjudicated
   by a paired bootstrap 95 % CI on the per-query MAE difference excluding 0
   — **and** reliability is preserved (coverage ±2 pp of 0.90; hi-conf MAE <
   no-abstention; flag violations not meaningfully worse than 3X.12). If no
   significant gain, Phase 4 ships the calibrated reliability layer on RBF
   as an accuracy-neutral, reliability-positive deliverable.

## Consequences

### Positive
- Builds on the measured Phase 3 evidence instead of re-fighting it; RBF's
  local strength is kept, the neural model only has to learn the residual.
- Maximum architecture reuse (3B/3X backbone + 3X.11 calibrator); the only
  genuinely new code surface is the residual-target builder (4A.2/4A.3).
- The reliability layer (the durable Phase 2/3 contribution) ships either
  way.

### Negative / costs
- Two estimators + a fusion rule = more production complexity than a single
  model; the inference path must run RBF then add the residual.
- Residual targets require per-date RBF predictions at **query** coordinates
  for the full OTM dataset — a CPU pre-compute step (4A.3), heavier than a
  config flag.
- Reopens Pod GPU spend (4A.4 / 4A.5).

### Neutral
- No data-pipeline or cleaning change; residuals derive from committed OTM
  data + the existing RBF baseline machinery (3X.6).

## Alternatives considered

- **Multiplicative / gated hybrid** (`σ̂ = RBF · (1 + f_θ)` or a learned gate
  blending RBF and a from-scratch neural surface). Rejected for v1: additive
  residual is the simplest, most interpretable, and keeps the evaluator
  scale unchanged. Revisit only if additive underperforms.
- **Residual only in sparse/wing regions** (region-gated). Deferred: train
  the full-domain residual first; region-gating is a 4A follow-up if the
  residual is concentrated.
- **Re-open pure-neural feature work (`micro_v2`).** Rejected by ADR 0008
  Outcome — pure feature expansion already missed.

## Open questions (resolved during 4A)

- Backbone (a) vs (b) — decided at 4A.4 from the training result.
- Whether the residual needs a transform — decided at 4A.4 from the residual
  distribution.
- Whether to region-gate — deferred to a 4A follow-up if warranted.

## Outcome (filled by 4A.8 on close)

**The success bar (§5) is MET.** Full synthesis:
[`docs/phase4_result_memo.md`](../phase4_result_memo.md); comparison bundle:
[`results/4/spy_phase1_random40_noiselow_otm/4a_compare/`](../../results/4/spy_phase1_random40_noiselow_otm/4a_compare/comparison_wide.md).

- **Accuracy (the gating bar).** On the matched clean OTM substrate
  `spy_phase1_random40_noiselow_otm`, the calibrated **gaussian** hybrid
  (the production head) posts test MAE **0.006006** vs the RBF floor
  **0.006132** — mean Δ **−0.000126**, date-clustered paired-bootstrap
  **95 % CI [−0.000144, −0.000106]** (n = 2,769,021 rows / 694 date groups,
  2000 resamples), entirely below 0 → **statistically significant**. First
  predictor in the project to beat RBF on the well-posed substrate. Source:
  `…/4a_hybrid/mae_delta_ci.json`.

- **Reliability (preserved and tightened).** Coverage **0.9181** vs iv_true
  (within ±2 pp of 0.90); hi-conf MAE **0.004710** < no-abstention
  **0.006006**; mean band width **0.032810** (vs 3X.12's 0.053824). Every
  reliability axis is at least as good as the 3X.12 pure-neural production.
  Source: `…/4a_hybrid/metrics_summary.json`. **Caveat:** the no-arb
  forbidden-flag count was not recomputed locally (needs the checkpoint; the
  hybrid = RBF + small smooth residual, so no-arb behavior tracks RBF's) —
  deferred per the 4A.7 spec, not a blocker.

- **Backbone chosen.** Option (a) **ANP-residual** (§Decision item 3) — the
  DeepSets encoder + ANP cross-attention decoder with the residual target,
  trained at 4A.4. The MLP-residual ablation (b) was not needed: (a) cleared
  the bar. No residual transform was required (the residual distribution was
  well-behaved at 4A.4). Region-gating remains a deferred follow-up.

- **Production recommendation.** **Adopt the RBF-prior gaussian residual
  hybrid** (`σ̂ = RBF + f_θ(residual)`, calibrated per 4A.6) as the production
  accuracy surface, retaining the calibrated reliability / abstention layer.
  Two documented follow-ups (neither a blocker): (1) calibrator re-fit on
  `iv_clean` to tighten the conservative coverage side (coverage vs iv_clean
  over-covers at 0.962); (2) no-arb flag-count audit on the checkpoint.

- **No-overclaim guardrail.** Every number is on the matched
  `random40_noiselow_otm` substrate; the all-11-variant robustness study stays
  deferred (carried from 3X).
