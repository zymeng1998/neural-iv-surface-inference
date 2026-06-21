# ADR 0011: Forward Strategy After Phase 4 — Accuracy Survival, Reliability-First, Structured-Product Pricing

## Status

**Accepted (2026-06-21).** Forward-strategy decision taken at the Phase 4
close. Sets the staged direction for Phase 4B, Phase 5, and Phase 6 and the
**decision gate** (Phase 4B) that governs whether raw accuracy remains a core
project story. No implementation has run against this ADR yet; 4B/5A/6A are
**planned**, not executed. The Outcome block is a placeholder to be filled
when Phase 4B closes.

> This ADR does not reopen or revise the Phase 4 verdict
> ([ADR 0010](0010_rbf_prior_residual_hybrid.md), Implemented). It decides
> what the project does *next*, given that verdict.

## Date

2026-06-21

## Context

Phase 4 closed **positive on accuracy** (ADR 0010, Implemented): the
calibrated RBF-prior **gaussian** residual hybrid is the first predictor in
the project to *statistically significantly* beat per-date RBF on the matched
clean OTM substrate `spy_phase1_random40_noiselow_otm` — test MAE **0.006006**
vs RBF floor **0.006132**, mean Δ **−0.000126**, date-clustered paired-
bootstrap **95 % CI [−0.000144, −0.000106]** (excludes 0).

Two facts hold at once:

1. **The win is real.** The interval is entirely below zero under a
   date-clustered resample; reliability was preserved/tightened.
2. **The win is small.** The margin is ≈ **2 %** relative MAE reduction on a
   single dense, clean, well-posed benchmark. It is statistically meaningful
   but economically and model-strategically marginal — it does not, on its
   own, justify continuing to optimize the *same* dense clean SPY benchmark
   as the project's headline contribution.

The strategic question is therefore **not** "how do we squeeze more accuracy
out of this benchmark?" It is "does the accuracy story have enough life to
remain a core contribution, and if not, what is the project's durable value?"
The durable asset built across Phases 2–4 is the **calibrated reliability /
abstention / no-arb / decision layer**, which is independent of the small
point-estimate margin. A credible monetization framing exists on top of that
layer: sell-side / bank-style **quote-risk and structured-product pricing
support**, where a trustworthy surface plus an honest uncertainty band is
worth more than a 2 % MAE improvement.

## Decision

1. **Adopt Phase 4A as the production point estimator on current evidence.**
   The calibrated RBF-prior gaussian residual hybrid (`σ̂ = RBF + f_θ(residual)`)
   remains the recommended accuracy surface (ADR 0010). No change to that
   recommendation is made here.

2. **Stop optimizing the same dense, clean SPY benchmark as the headline
   pursuit.** Further accuracy work is justified only if it survives the
   Phase 4B gate below. The project will not keep grinding the
   `random40_noiselow_otm` substrate for fractional MAE.

3. **Phase 4B is the accuracy *survival* diagnostic (the decision gate).**
   Run RBF vs the RBF-prior hybrid over **increasing sparsity / thin wings /
   missing maturities** and measure how the hybrid's edge over RBF behaves as
   observations become sparse. See the gate below.

4. **Phase 5 (reliability-first surface inference / quote-risk layer) becomes
   the project's primary contribution if 4B is negative** — and is worth
   building regardless. It formalizes calibrated uncertainty, abstention,
   surface-level confidence, no-arbitrage / risk flags, quote/no-quote logic,
   and spread / model-risk-reserve suggestions into a coherent
   reliability-first system. This is the durable value of Phases 2–4.

5. **Phase 6 connects the system to sell-side structured-product pricing,
   beginning with FCN-style valuation support — not full issuer
   infrastructure.** The first Phase 6 deliverable is a *constrained demo*
   that propagates surface uncertainty into fair-value intervals,
   sensitivities, and quote-risk diagnostics for an option-embedded
   structured product (an FCN-like payoff first), framed as a bank / sell-side
   desk using the model for pricing confidence, model-risk reserves, and
   hedge-risk diagnostics. It does **not** claim to price FCNs today.

## Decision gate (Phase 4B)

The gate that determines whether raw accuracy remains a core story:

- **Experiment:** RBF vs RBF-prior hybrid, evaluated over a sweep of
  increasing context sparsity — fewer observed quotes, thinned wings (OTM
  tails removed), and dropped maturities — on the existing benchmark
  machinery (no new data source, no new architecture).
- **Survival criterion (accuracy story lives):** the hybrid's relative edge
  over RBF **grows materially** as observations become sparse (the hybrid
  degrades more gracefully than RBF where RBF's local interpolation has little
  to stand on). If so, sparse/illiquid regimes are a real accuracy frontier
  and accuracy work continues with that framing.
- **Stop criterion (pivot fully to reliability):** the hybrid's edge stays in
  the ~**0–2 %** band across the sweep (no material growth under sparsity).
  If so, the accuracy story is effectively settled on this asset; the project
  **stops accuracy chasing** and Phase 5 reliability-first inference becomes
  the primary contribution.

The exact sparsity ladder, whether the sweep is eval-time or requires
per-regime retraining, and the "material growth" threshold are fixed by the
Phase 4B decomposition story (`4B.1`), not here.

## Consequences

### Positive
- Resolves the "small but real win" tension with an explicit, falsifiable
  gate instead of open-ended benchmark grinding.
- Elevates the durable Phase 2–4 asset (the reliability/decision layer) to a
  first-class deliverable rather than a side effect of accuracy work.
- Gives monetization a concrete, scoped, sell-side framing that does not
  overclaim.

### Negative / costs
- Phase 4B reopens (modest) compute to run the sparsity sweep before the
  strategic direction fully commits.
- Phase 6 introduces a finance-modeling surface (payoff valuation) outside the
  current IV-surface codebase; scope discipline is required to keep the first
  deliverable a constrained demo, not an issuer pricer.

### Neutral
- No data-pipeline, cleaning, or benchmark-construction change is implied by
  this ADR. Phase 4B reuses committed benchmark machinery; Phase 5 reuses the
  Phase 2D / 4A reliability layer; Phase 6 consumes the surface + uncertainty
  outputs.

## Scope boundary — what an IV-surface model is and is not for pricing

Structured notes are typically **debt plus one or more embedded
derivatives**. The implied-volatility surface is **one critical pricing
input** to the embedded-derivative leg — but it is not the whole pricer. The
following remain **outside** the current IV-surface project and are explicitly
not claimed by Phase 6's first deliverable:

- interest rates / discounting curves and **dividends**;
- **funding / issuer credit spread** (the debt leg);
- **correlation** for multi-underlying baskets;
- **barriers**, autocall / knock-in features, path dependence;
- **liquidity** and bid/offer execution assumptions;
- **hedging** assumptions and hedge-cost / model-risk reserves beyond what the
  surface uncertainty layer informs.

Phase 6 contributes the **surface + uncertainty** input and its propagation
into fair-value intervals and quote-risk diagnostics, with the rest of the
pricing stack held fixed / stubbed and clearly labeled as such.

## Alternatives considered

- **Keep optimizing the dense clean benchmark for more accuracy.** Rejected
  as the headline pursuit: a 2 % significant-but-small margin does not warrant
  it. Permitted only if Phase 4B shows the edge grows under sparsity.
- **Declare accuracy dead now and skip 4B.** Rejected: that would overclaim a
  negative before the diagnostic runs. The decision is gated on evidence.
- **Jump straight to monetization (Phase 6) without the reliability
  formalization (Phase 5).** Rejected: the quote-risk / model-risk value
  proposition depends on a formalized, calibrated reliability layer; Phase 5
  is the foundation Phase 6 sells.

## Open questions (resolved during 4B / 5A / 6A)

- The exact sparsity ladder and the "material growth" threshold for the 4B
  gate — fixed by `4B.1`.
- Whether 4B requires per-regime retraining or is an eval-time context sweep —
  fixed by `4B.1`.
- The precise reliability-first deliverable set and operating points for
  Phase 5 — fixed by `5A.1`.
- The FCN payoff variant, fixed-input assumptions, and quote cross-check
  protocol for the Phase 6 demo — fixed by `6A.1`.

## Outcome (to be filled when Phase 4B closes)

_Pending._ Phase 4B will record whether the accuracy story survives (edge
grows under sparsity → accuracy remains core) or is retired (edge stays
~0–2 % → pivot fully to reliability). This block, ADR status, and the
downstream roadmap emphasis are updated on that close.
