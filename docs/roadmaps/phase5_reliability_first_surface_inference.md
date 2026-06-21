# Phase 5 Roadmap — Reliability-First Surface Inference / Quote-Risk Layer

---
created_at: 2026-06-21T00:00:00-04:00
last_updated_at: 2026-06-21T00:00:00-04:00
status: backlog
---

> **Planning document.** Phase 5 (epic `5A`) is not yet entered or decomposed.
> This roadmap scopes the direction; atomic child stories are authored by the
> decomposition story [`5A.1`](../tasks/specs/5A.1_decompose_phase_5a.md) when
> the operator green-lights the phase. Strategic framing:
> [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md).

## 1) Why Phase 5

Across Phases 2–4 the project's **durable asset** is not the point estimate —
it is the **calibrated reliability / abstention / no-arbitrage / decision
layer** that knows where the surface is trustworthy and abstains where it is
not. Phase 4 confirmed that the raw-accuracy advantage over RBF is real but
small (≈ 2 %; ADR 0010). Phase 5 makes reliability the **primary
contribution** rather than a side effect of accuracy work.

Phase 5 is worth building regardless of the Phase 4B gate, and becomes the
project's headline direction if 4B is negative (ADR 0011 Decision 4).

## 2) Goal

Formalize a **reliability-first surface inference** system: given sparse,
noisy, irregular option observations, produce not just a surface but a
defensible statement of *how much it can be trusted, where, and what to do
about it* — packaged as a quote-risk layer a desk could actually consume.

Target capabilities (consolidating and hardening Phase 2D / 4A):

1. **Calibrated uncertainty** — per-query bands with audited empirical
   coverage at the stated level, on the clean OTM substrate and under the
   sparsity regimes 4B exposes.
2. **Abstention / selective prediction** — principled quote/no-quote logic
   with high-confidence-MAE vs coverage trade-off curves.
3. **Surface-level confidence** — aggregate, region-aware confidence over the
   whole `(k, τ)` surface, not only per-point.
4. **No-arbitrage / risk flags** — monotonicity / convexity / calendar checks
   surfaced as actionable per-region risk flags.
5. **Quote/no-quote decision logic** — a configurable operating point mapping
   confidence + risk flags → a quote / widen / abstain decision.
6. **Spread / model-risk-reserve suggestions** — translate uncertainty and
   disagreement into a suggested spread or model-risk reserve, framed as a
   sell-side desk input (not a traded price).

## 3) Framing — pretend we are a sell-side desk

Phase 5 adopts a **bank / sell-side desk** consumer in mind: the model is used
for **pricing confidence, model-risk reserves, and hedge-risk diagnostics**,
not retail trading and not the operator issuing anything. Every output should
answer a desk question — "can I quote this? how wide? how much reserve?" — and
must be honest about abstaining when the surface is under-determined.

## 4) Substrate & inputs

- Reuse the Phase 2D / 4A calibrated hybrid and decision layer
  (`σ̂`, `confidence_score`, `uncertainty_band`, `tradability_score`,
  `no_arb_risk_flags`, `abstain_flag`).
- Reuse the committed OTM benchmark family; consume the sparsity regimes from
  Phase 4B to stress-test calibration and abstention where it matters most.
- No new data source; a new source would need a Phase-2-style data ADR first.

## 5) Non-goals

- No structured-product payoff valuation — that is Phase 6.
- No new market-data source or cleaning change.
- No claim that the reliability layer removes model risk; it *quantifies and
  communicates* it.
- No retail-trading framing.

## 6) Decisions

- [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
  — forward strategy (Phase 5 as primary contribution if 4B negative; worth
  building regardless). A Phase-5-specific design ADR is authored by `5A.1`
  if the formalization makes a constraining architectural choice.
