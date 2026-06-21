# STATUS — Phase 4 closed; 4B decomposed (4B.1 done); starting 4B.2 harness

**Updated:** 2026-06-21
**Branch:** main
**Mode:** local CPU. Docs/planning only — no GPU, no data/code/config/test edits.

## Headline

Phase 4 is **closed positive** (epic 4A `done`; RBF-prior gaussian hybrid is
the adopted production estimator; ADR 0010 Implemented). The win over RBF is
**statistically real but economically small** (≈ 2 % relative MAE on the dense
clean OTM benchmark), so the project will **not** keep grinding that benchmark.
The forward direction is staged and recorded in
[ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
(**Accepted**):

- **`4B` — sparsity-sweep diagnostic (the decision gate).** RBF vs hybrid over
  increasing sparsity / thin wings / missing maturities. Edge grows under
  sparsity ⇒ accuracy story lives; stays ~0–2 % ⇒ pivot fully to reliability.
- **`5A` — reliability-first surface inference / quote-risk layer.** Primary
  contribution if 4B is negative; worth building regardless.
- **`6A` — structured-product pricing / FCN monetization demo.** Sell-side
  framing; constrained demo, not a full FCN pricer.

## Where things stand

- Forward-strategy planning pass committed to the working tree: ADR 0011 +
  three roadmaps (`phase4b_sparsity_sweep`, `phase5_reliability_first_*`,
  `phase6_structured_product_pricing`) + decomposition specs `4B.1 / 5A.1 /
  6A.1` (all `backlog`) + BOARD rows `4B / 4B.1 / 5A / 5A.1 / 6A / 6A.1` +
  README Next Direction + progress_log entry.
- PMR gate dry-run: **PASS** (0 evidence-source files; docs-only).
- No experiment was run; the Phase 4A result is unchanged.

## Current task — 4B.1 DONE; starting 4B.2 (sparsity-sweep harness)

**Story 4B.1** decomposed Phase 4B into atomic child stories; promoted → `done`
(operator approved at push). Epic 4B is `in_progress`. **Next: 4B.2** —
sparsity-sweep harness (fixed-query / shrinking-context, 4 regimes) + unit/smoke
tests, local CPU, no checkpoint. Spec:
[`docs/tasks/specs/4B.2_local_sparsity_sweep_harness.md`](docs/tasks/specs/4B.2_local_sparsity_sweep_harness.md).

Spec: [`docs/tasks/specs/4B.1_decompose_phase_4b.md`](docs/tasks/specs/4B.1_decompose_phase_4b.md).
Roadmap: [`docs/roadmaps/phase4b_sparsity_sweep.md`](docs/roadmaps/phase4b_sparsity_sweep.md).

## Resolved design forks (operator, 2026-06-21)

1. **Approach → staged (eval-first).** Eval-time sweep on the existing 4A
   checkpoint first; per-regime retraining held as a *conditional* escalation
   (4B.7) only if the read is ambiguous.
2. **Axes → all four regimes:** `fewer_quotes`, `thin_wings`,
   `missing_maturities`, `combined_quotes_wings` (operator-added). Parameters of
   the sweep, not separate stories.
3. **Gate threshold** pre-registered inside 4B.5 before the trajectory is
   computed.

## Child stories authored (all `backlog`)

`4B.2` harness+tests (local CPU) → `4B.3` build swept inputs (remote CPU) →
`4B.4` hybrid forward sweep (remote; needs the 4A checkpoint volume) → `4B.5`
trajectory + CIs + **gate verdict** (local CPU) → `4B.6` close + ADR 0011
Outcome (local CPU). `4B.7` per-regime retrain is **conditional** off 4B.5
(`ambiguous` → run; else `cancelled`) — no Pod spend unless triggered.

## Next concrete action

- Operator promotes 4B.1 → `done`, then picks up **4B.2** (sparsity-sweep
  harness + tests, local CPU). Remote stories (4B.3 / 4B.4 / conditional 4B.7)
  gated on operator Pod go-ahead; 4B.4 requires the 4A checkpoint volume.
- All edits this session are docs/planning; PMR gate dry-run PASS (0
  evidence-source files).
