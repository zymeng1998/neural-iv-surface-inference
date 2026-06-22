# Phase 4B Roadmap — Sparsity-Sweep Diagnostic (Accuracy Survival Gate)

---
created_at: 2026-06-21T00:00:00-04:00
last_updated_at: 2026-06-21T12:00:00-04:00
status: in_progress
---

> **Entered + decomposed 2026-06-21** by [`4B.1`](../tasks/specs/4B.1_decompose_phase_4b.md).
> Operator chose the **staged (eval-first)** approach over **all four** sparsity
> regimes; the atomic child stories 4B.2–4B.7 are authored (backlog). The
> strategic framing and the decision gate live in
> [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md).

## 1) Why Phase 4B

Phase 4 closed **positive but small**: the RBF-prior gaussian hybrid beats
RBF on the clean OTM substrate by a *statistically significant* but ≈ **2 %**
relative MAE margin (0.006006 vs 0.006132; 95 % CI [−0.000144, −0.000106];
ADR 0010). A 2 % win on one dense, clean, well-posed benchmark is not, by
itself, a reason to keep optimizing that benchmark.

Phase 4B answers the only question that decides whether the accuracy story has
life: **does the hybrid's edge over RBF grow as observations become sparse?**
RBF is a strong *local* interpolator when the chain is dense; its advantage
should erode where there is little local data to interpolate from (thin wings,
missing maturities, few quotes). If the hybrid's residual model degrades more
gracefully there, sparse/illiquid regimes are a real accuracy frontier. If
not, accuracy is settled on this asset.

## 2) The gate (from ADR 0011)

- **Survival (accuracy story lives):** the hybrid's relative edge over RBF
  **grows materially** as sparsity increases → continue accuracy work with a
  sparse-regime framing.
- **Stop (pivot fully to reliability):** the edge stays in the ~**0–2 %** band
  across the sweep → stop accuracy chasing; Phase 5 reliability-first
  inference becomes the primary contribution.

This is a **decision gate**, not a bar to "pass". A negative result is a
clean, publishable finding (RBF is effectively optimal even under sparsity)
and is explicitly acceptable.

## 3) Diagnostic design (to be finalized by 4B.1)

Sweep RBF vs the RBF-prior hybrid across increasing context sparsity, reusing
the **committed benchmark machinery only** (the 7 masking strategies + noise
regimes already in the Phase 1 pipeline). Candidate sparsity axes:

- **Fewer observed quotes** — denser → sparser random masking (e.g.
  `random40 → random60 → random80`, or an explicit quote-count ladder).
- **Thin wings** — progressively remove OTM tail quotes so the deep-wing
  region must be inferred.
- **Missing maturities** — drop whole expirations from the observed context so
  calendar structure must be inferred.

Per regime, report RBF test MAE, hybrid test MAE, the absolute and relative
delta, and a date-clustered paired-bootstrap CI on the per-query delta (same
methodology as 4A.7). The headline is the **trajectory** of the relative edge
across the ladder, not any single point.

**Resolved at 4B.1 (operator, 2026-06-21):** the sweep is **eval-time first**
(staged) — RBF and the existing 4A checkpoint are compared at inference over
progressively sparser contexts (cheap; tests the trained model's robustness),
with **per-regime retraining held in reserve** as a *conditional* escalation
(4B.7) triggered only if the eval-time read is ambiguous. The fixed-query /
shrinking-context invariant keeps per-rung MAE deltas comparable. The "material
growth" threshold is **pre-registered inside 4B.5** before the trajectory is
computed.

## 3a) Workstreams (epic 4B)

| Story | Locale | Deliverable |
|---|---|---|
| 4B.1 | local | Decompose Phase 4B (this roadmap's child specs + resolved forks) |
| 4B.2 | local CPU | Sparsity-sweep harness (fixed-query / shrinking-context) + 4 regimes + unit/smoke tests (no full build, no checkpoint) |
| 4B.3 | remote CPU | Materialize swept eval inputs on full OTM: per-rung RBF at fixed query coords + finiteness/MAE-sanity audit |
| 4B.4 | remote | Run the 4A hybrid checkpoint **forward** across all rungs → σ̂ (eval-time; the only checkpoint-dependent story) |
| 4B.5 | local CPU | Edge-vs-sparsity trajectory + date-clustered bootstrap CIs (mirror 4A.7) + **ADR 0011 gate adjudication** |
| 4B.6 | local CPU | Phase 4B closing addendum + ADR 0011 Outcome + journal; flip epic 4B `done` |
| 4B.7 | remote GPU | **CONDITIONAL** per-regime retrain escalation — only if 4B.5 returns `ambiguous`; else `cancelled` |

Chain: `4B.1 → 4B.2 → 4B.3 → 4B.4 → 4B.5 → 4B.6`, with `4B.7` branching off
`4B.5` only on an `ambiguous` verdict. Each story is atomic; no story spans
local + remote.

**Sparsity regimes (parameters of the sweep, not separate stories):**
`fewer_quotes`, `thin_wings`, `missing_maturities`, and a `combined_quotes_wings`
joint stress (operator-requested) — all share the 4B.2–4B.5 chain.

## 4) Substrate & baselines

- **Substrate:** the existing `spy_phase1_*_otm` benchmark family (matched
  clean OTM splits; same chronological train/val/test as 3X/4A). No new data
  source, no cleaning change.
- **Reference floor:** RBF-on-OTM test MAE **0.006132** (3X.6 / 4A baseline)
  at the dense end of the ladder.
- **Reference hybrid:** the calibrated RBF-prior gaussian hybrid **0.006006**
  (4A) at the dense end.

## 5) Non-goals

- No new architecture, no new feature set, no new data source or cleaning
  change — this is a diagnostic over existing models and benchmark machinery.
- No reopening of the Phase 4 verdict (ADR 0010) — 4B builds on it.
- No production-recommendation change unless the gate flips the strategic
  direction (that decision lands in ADR 0011's Outcome, not here).
- No reliability-layer or pricing work — those are Phases 5 / 6.

## 6) Decisions

- [ADR 0011](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
  — forward strategy + the 4B accuracy-survival gate. The 4B Outcome feeds
  ADR 0011's Outcome block.

## 7) Close (2026-06-22) — verdict `ambiguous` → **`true` (accuracy survives)** after fair retrain

Phase 4B executed the staged eval-first diagnostic (4B.2–4B.5) **and** the full
fair retrain (4B.7). Provenance: `results/4/spy_phase1_random40_noiselow_otm/4b_sweep/`
(`trajectory.csv` eval-time; `fair_trajectory.csv` / `fair_trajectory_wide.md`
fair; `gate_verdict.json` resolved) + `artifacts/runs/4B7/`.

**Eval-time read (4B.5) was `ambiguous`** — a full-context checkpoint scored on
sparse context: relative edge `(RBF−hybrid)/RBF` **collapsed** for the wing
regimes (thin_wings 2.05 → 0.31 %, combined → 0.46 %), grew only for benign
random thinning (fewer_quotes → 4.32 %). Confounded by an eval-time OOD caveat.

**Fair retrain (4B.7) resolves it to `true`** — retraining the hybrid on each
rung's sparse context reverses the collapse. Relative edge (overall test MAE;
all wing/maturity gains significant, date-clustered 95 % CI < 0):

| regime | r0 dense | r1 | r2 | r3 | r4 sparse | eval-time r4 |
|---|---|---|---|---|---|---|
| missing_maturities | +2.05 % | +1.37 % | +9.53 % | +20.91 % | **+37.07 %** | (+1.56 %) |
| thin_wings | +2.05 % | +1.75 % | +5.25 % | +9.98 % | **+16.57 %** | (+0.31 %) |
| combined_quotes_wings | +2.05 % | +2.38 % | +5.39 % | +14.08 % | **+15.99 %** | (+0.46 %) |
| fewer_quotes | +2.05 % | +2.79 % | +1.94 % | +0.14 % | +0.07 % (n.s.) | (+4.32 %) |

- Both wing-sensitive regimes clear the pre-registered survive bar (grows,
  significant, ≥ 5 % at the sparse end) ⇒ **`accuracy_survives = true`**. The
  eval-time wing collapse was an OOD artifact; trained fairly, the neural prior
  fills **structured** gaps (wings/maturities) far better than RBF.
- Exception: `fewer_quotes` (random thinning) → ~0 — RBF interpolates scattered
  points fine; the hybrid is at worst tied.
- **Caveat:** each cell trains a separate model on its exact sparsity regime
  (proves the architecture *can* exploit structured sparsity); a single
  mixed-sparsity model is the production-realistic follow-up.
- **Reliability still stands (secondary):** dense-calibrated 90 % coverage falls
  0.962 → 0.35–0.39 (wing regimes) — so **Phase 5 reliability-first** remains
  highly valuable and proceeds **in parallel** with the (now alive) accuracy story.

Outcome recorded in [ADR 0011 §Outcome](../decisions/0011_forward_strategy_accuracy_reliability_pricing.md).
Epic 4B `done`; child stories 4B.1–4B.7 `done`.
