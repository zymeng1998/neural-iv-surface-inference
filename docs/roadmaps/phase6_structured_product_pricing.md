# Phase 6 Roadmap — Structured-Product Pricing / FCN Monetization Demo

---
created_at: 2026-06-21T00:00:00-04:00
last_updated_at: 2026-06-21T00:00:00-04:00
status: backlog
---

> **Planning document.** Phase 6 (epic `6A`) is not yet entered or decomposed.
> This roadmap scopes the monetization framing; atomic child stories are
> authored by the decomposition story
> [`6A.1`](../tasks/specs/6A.1_decompose_phase_6a.md) when the operator
> green-lights the phase. Strategic framing:
> [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md).

## 1) Why Phase 6

Phases 2–5 build a calibrated IV surface with an honest uncertainty / quote-
risk layer. Phase 6 asks the monetization question **without overclaiming**:
where is a trustworthy surface plus an uncertainty band actually worth money?
The answer is **sell-side / bank-style structured-product pricing support** —
where a 2 % point-estimate improvement is irrelevant but *quote confidence and
model-risk reserves* are valuable.

## 2) Framing — sell-side desk, not retail and not an issuer

Pretend we are a **bank / sell-side desk** using the model for **pricing,
quote confidence, model-risk reserves, and hedge-risk diagnostics** on
option-embedded structured products. This is explicitly **not**:

- retail trading;
- the operator personally issuing notes;
- a production issuer pricing/booking system.

The operator has access to **professional quote cross-checks through industry
contacts**, which makes a constrained pricing demo verifiable against real
desk quotes — the reason an FCN-like payoff is the chosen first target.

## 3) Goal — a constrained demo, not a full pricer

The first Phase 6 deliverable is a **constrained demo** that takes the Phase 4
/ Phase 5 surface + uncertainty layer and **propagates surface uncertainty
into**:

1. **Fair-value intervals** — a price range (not a single number) implied by
   the calibrated surface uncertainty for an FCN-like payoff.
2. **Sensitivities** — how the fair value moves with the surface inputs
   (vega-like surface sensitivity), surfaced per region.
3. **Quote-risk diagnostics** — where the surface is under-determined enough
   that the desk should widen, reserve, or abstain (consuming the Phase 5
   quote/no-quote layer).

It begins with an **FCN-like payoff** because the operator can cross-check it
against professional desk quotes. It does **not** promise a full production FCN
pricer.

## 4) Scope boundary — the surface is one input, not the whole pricer

Structured notes are typically **debt plus embedded derivatives**. The IV
surface is **one critical pricing input** to the embedded-derivative leg, but
the full pricing stack includes components **outside** this project. Phase 6's
first demo holds these fixed / stubbed and labels them as such:

- interest rates / discount curves and **dividends**;
- **funding / issuer credit spread** (the debt leg);
- **correlation** for multi-underlying baskets;
- **barriers**, autocall / knock-in features, path dependence;
- **liquidity** and bid/offer execution assumptions;
- **hedging** assumptions and hedge-cost reserves beyond what the surface
  uncertainty informs.

The project contributes the **surface + uncertainty** input and its
propagation into fair-value intervals and quote-risk diagnostics. The rest is
explicitly out of scope for the first deliverable.

## 5) Non-goals

- No claim that the model prices FCNs today, or replaces a desk pricer.
- No full rates / dividend / credit / correlation / barrier modeling stack.
- No booking, hedging, or execution system.
- No retail or issuer-facing product.

## 6) Decisions

- [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
  — forward strategy (Phase 6 connects the system to sell-side structured-
  product pricing, FCN-first, constrained demo). A Phase-6-specific ADR
  (payoff scope, fixed-input assumptions, cross-check protocol) is authored by
  `6A.1`.
