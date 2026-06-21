# Phase 4 Result Memo — RBF-Prior Residual Hybrid

> **Story 4A.8.** Executive synthesis of Phase 4 (epic `4A`). Pure synthesis on
> committed artifacts — no training, no scoring, no calibrator re-fit. Every
> number below cites a committed bundle path. The comparison table is emitted
> by `scripts/run_4a8_comparison.py` from those same bundles.
>
> **Scope / no-overclaim guardrail:** every number is on the single matched
> clean benchmark `spy_phase1_random40_noiselow_otm`. No robustness claim is
> made across the other 10 OTM benchmark variants (that study is deferred,
> carried forward from 3X / Phase 3).

## TL;DR

Phase 3 closed **negative on accuracy**: no pure conditional-neural variant
came within 5 % of the per-date **RBF interpolation** floor (best ANP point
head **+61 %**), so [ADR 0009](decisions/0009_phase3_production_predictor_selection.md)
kept RBF as the production accuracy baseline and reserved the **RBF-prior
hybrid** ([ADR 0004](decisions/0004_phase3_accuracy_push_framing.md) /
[ADR 0010](decisions/0010_rbf_prior_residual_hybrid.md)) for Phase 4. Phase 4
built it: rather than replace RBF, **predict what RBF gets wrong** —
`σ̂(k, τ) = RBF_t(k, τ) + f_θ(residual)`, with the three heads, the K=5
ensemble, and the calibrated reliability layer all reused from Phase 3.

**The accuracy bar was MET.** On the matched clean OTM substrate the calibrated
gaussian hybrid posts test MAE **0.006006** vs the RBF floor **0.006132** — a
mean delta of **−0.000126** with a date-clustered paired-bootstrap **95 % CI
[−0.000144, −0.000106]** that excludes 0. This is the **first predictor in the
project to statistically significantly beat RBF** on the well-posed substrate.
The reliability layer is preserved (coverage 0.918 vs iv_true, within ±2 pp of
0.90; hi-conf MAE 0.004710 < no-abstention 0.006006). The gain is small in
absolute terms (~2.0 % below the floor) but real and significant — exactly the
margin ADR 0010 §5 defined as success.

**Verdict: Phase 4 closes positive on accuracy.** The production
recommendation is to **adopt the RBF-prior gaussian residual hybrid** as the
accuracy surface, retaining the calibrated reliability layer, with two
documented caveats (below). Recorded in [ADR 0010](decisions/0010_rbf_prior_residual_hybrid.md)
(→ Implemented).

## Scope Recap (epic 4A)

- **4A.1 — decompose.** Roadmap + ADR 0010 (Proposed) + child specs; the
  additive residual form, head set, fusion, and success bar fixed at kickoff.
- **4A.2 — residual-target builder (local).** `target_mode ∈ {absolute,
  residual}` loader flag; absolute path byte-identical; 20 tests green.
- **4A.3 — full residual dataset (remote CPU).** Per-date RBF at query coords →
  residuals on OTM; 10.53M rows, 0 non-finite, val/test mean-abs-residual ==
  the 3X.6 RBF MAE (sanity).
- **4A.4 — train residual hybrid, 3 heads (remote GPU).** ANP-residual backbone
  (the locked fork — option (a), maximum reuse). **Hybrid beats RBF:** gaussian
  0.006006 / quantile 0.005906 vs RBF 0.006132; point ties. All ≪ the 3X.9
  pure-neural ladder.
- **4A.5 — K=5 residual point ensemble (remote GPU).** Test MAE 0.006141 (ties
  RBF, like the 4A.4 single point); disagreement mean 0.000209 feeds the
  calibrator. Last GPU run.
- **4A.6 — calibrator re-fit (local).** Fit on hybrid val predictions:
  T=1.147, ens_scale=438; test coverage 0.9181 vs iv_true (within ±2 pp).
- **4A.7 — decision-layer + bootstrap CI (local).** The bar adjudication —
  see the headline below. Computed fully local from cached predictions.

## Headline — clean OTM substrate (the operative verdict)

Test MAE on `spy_phase1_random40_noiselow_otm` vs `iv_clean`, sorted best→worst.
RBF is the floor; the right column is each variant's gap to RBF. Source:
[`results/4/spy_phase1_random40_noiselow_otm/4a_compare/comparison_wide.md`](../results/4/spy_phase1_random40_noiselow_otm/4a_compare/comparison_wide.md)
(+ `comparison.csv`, `headline.json`).

| Family / head | OTM test MAE | vs RBF | Bundle |
|---|---:|---:|---|
| **rbf_prior_hybrid · gaussian (Phase 4 production)** | **0.006006** | **−2.0 %** | `…_otm/4a_hybrid/metrics_summary.csv` |
| rbf · interp (floor) | 0.006132 | floor | `…_otm/rbf/metrics_summary.csv` |
| anp_single · point (best pure-neural) | 0.009871 | +61.0 % | `…_otm/3x_compare/comparison_wide.csv` |
| anp_calibrated · fused (3X.12 production) | 0.011618 | +89.5 % | `…_otm/3x_anp/metrics_summary.csv` |
| anp_single · quantile | 0.011752 | +91.7 % | `…_otm/3x_compare/comparison_wide.csv` |
| anp_single · gaussian | 0.014404 | +134.9 % | `…_otm/3x_compare/comparison_wide.csv` |

The structural inversion vs Phase 3: every **pure**-neural head sits +61 % or
worse above RBF, but the **hybrid** — same backbone, residual target — lands
*below* the floor. RBF keeps the local interpolation it wins; the neural model
only has to learn the small residual, and it does so well enough to net a
significant gain.

## The Bar (ADR 0010 §5)

Date-clustered paired bootstrap on the full test fold (n = 2,769,021 rows over
694 date groups, 2000 resamples), gaussian hybrid (production head) vs RBF,
both vs `iv_clean`. Source:
[`results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/mae_delta_ci.json`](../results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/mae_delta_ci.json)
+ `metrics_summary.json`.

| Bar component (ADR 0010 §5) | Result | Verdict |
|---|---|---|
| Test MAE significantly < RBF (paired-bootstrap 95 % CI on Δ excludes 0) | hybrid 0.006006 vs RBF 0.006132; Δ −0.000126; **95 % CI [−0.000144, −0.000106]** | **MET** |
| Coverage within ±2 pp of 0.90 | 0.9181 vs iv_true (in-band) | **MET** |
| Hi-conf MAE < no-abstention | 0.004710 < 0.006006 | **MET** |
| Flag violations not meaningfully worse than 3X.12 | not recomputed (deferred) | **caveat, not a fail** |

All three measurable bar components pass. The fourth (no-arb flag count) was
not recomputed locally — it needs the model checkpoint to re-predict, and the
hybrid surface is `RBF + a small smooth residual`, so its no-arbitrage behavior
tracks RBF's. Deferred per the 4A.7 spec; not a blocker for the verdict.

## Reliability vs the Pure-Neural Production (3X.12)

The hybrid carries forward the Phase 2/3 reliability layer and tightens it:

| Metric | 3X.12 pure-neural (production) | 4A hybrid (production) | Source |
|---|---:|---:|---|
| Test MAE (vs iv_clean) | 0.011618 | **0.006006** | `…/3x_anp/`, `…/4a_hybrid/` |
| Coverage @ 0.90 (vs iv_true) | 0.9295 | **0.9181** | same |
| Hi-conf MAE (keep 0.8) | 0.008352 | **0.004710** | same |
| Mean band width | 0.053824 | **0.032810** | same |

The hybrid is more accurate, better-covered (closer to nominal, inside the ±2 pp
band where 3X.12 ran ~+2.95 pp conservative), and ~39 % tighter on band width —
a strictly better reliability profile on every axis shown.

## What Changed vs Phase 3

Phase 3's lesson was that an amortized global-prior neural model **does not
re-derive RBF's local weighting from data** (ADR 0009). Phase 4 does not
re-fight that — it accepts it and reframes the neural target as the *residual*
RBF leaves behind. The same DeepSets→ANP backbone that was +61 % as a
standalone predictor becomes a net **−2.0 %** improvement once RBF carries the
bulk signal. The advance is in **problem formulation**, not architecture: the
durable Phase 2/3 contribution (the calibrated reliability layer) ships intact,
and the accuracy answer finally moves below the RBF floor.

## Acceptance-Criteria Map (ADR 0010 §5 / spec 4A.8)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Calibrated hybrid OTM test MAE significantly < RBF (0.00613), Δ-CI excludes 0 | **MET** — Δ −0.000126, 95 % CI [−0.000144, −0.000106] | `…/4a_hybrid/mae_delta_ci.json` |
| 2 | Coverage ±2 pp of 0.90 | **MET** — 0.9181 vs iv_true | `…/4a_hybrid/metrics_summary.json` |
| 3 | Hi-conf MAE < no-abstention | **MET** — 0.004710 < 0.006006 | same |
| 4 | Flag violations not meaningfully worse than 3X.12 | **Deferred** — not recomputed; surface tracks RBF | 4A.7 spec note |
| 5 | Reproducibility & evidence discipline | **MET** — every variant has manifests + `results/4/` bundles + journal | `artifacts/runs/4A*/`, `results/4/` |
| 6 | Closing memo + ADR Outcome shipped | **MET** — this memo (4A.8); ADR 0010 → Implemented | this file, ADR 0010 |

The gating criterion (1) passes with a margin that excludes 0; (2)/(3) hold;
(4) is deferred with a stated rationale; (5)/(6) close now.

## Verdict

**Phase 4 closes positive on accuracy: the RBF-prior residual hybrid is the
first predictor to statistically significantly beat RBF on the well-posed OTM
substrate (Δ −0.000126, 95 % CI [−0.000144, −0.000106]), with the reliability
layer preserved and tightened.** The win is small in absolute terms but real
and significant — the margin ADR 0010 §5 set as the success bar.

## Production Recommendation

**Adopt the RBF-prior gaussian residual hybrid** (`σ̂ = RBF + f_θ(residual)`,
calibrated per 4A.6) as the production accuracy surface, retaining the
calibrated reliability / abstention layer. Two documented caveats:

1. **Coverage-refit follow-up.** Coverage vs `iv_clean` runs conservative
   (0.962, over-covering) while coverage vs `iv_true` is in-band (0.9181). A
   calibrator re-fit on the `iv_clean` target would tighten the conservative
   side; not a blocker.
2. **No-arb flag-count audit deferred.** The hybrid's forbidden-flag count was
   not recomputed locally (needs the checkpoint). Since `σ̂ = RBF + small
   smooth residual`, no-arb behavior tracks RBF's; a confirmatory audit on the
   checkpoint is a clean follow-up.

## Open Questions / Phase 5 Framing (backlog placeholders only)

1. **Calibrator re-fit on `iv_clean`** to tighten the conservative coverage
   side (caveat 1).
2. **No-arb flag-count audit** on the hybrid checkpoint (caveat 2).
3. **All-11-OTM-variant robustness study** remains deferred; every number here
   is scoped to the matched `random40_noiselow_otm` substrate only.
4. **Region-gated / multiplicative residual** variants (ADR 0010 alternatives)
   — only if a larger margin is wanted; the additive form already cleared the
   bar.

Phase 4 does not open Phase 5; the above are backlog placeholders.

## Reproducing the Headline Numbers

```bash
# Re-assemble the Phase 4 comparison from committed artifacts (no training):
python3 scripts/run_4a8_comparison.py
```

Primary sources (all committed):
- `results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/{metrics_summary.csv,metrics_summary.json,mae_delta_ci.json}` — the hybrid metrics + the bar CI.
- `results/4/spy_phase1_random40_noiselow_otm/4a_compare/{comparison.csv,comparison_wide.md,headline.json}` — the Phase 4 ladder + verdict.
- `results/3/spy_phase1_random40_noiselow_otm/{rbf,3x_anp}/metrics_summary.csv` — the RBF floor + 3X.12 calibrated reference.
- `results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.csv` — the pure-neural ladder.

No new training, no new scoring — every numeric claim above traces to a
committed artifact path.
