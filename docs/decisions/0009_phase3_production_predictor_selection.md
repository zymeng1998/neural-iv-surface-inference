# ADR 0009: Phase 3 Production Predictor Selection + Phase 4 Framing

## Status

**Proposed — SKELETON (2026-06-15).** Created by Phase 3D decomposition
story [`3D.1`](../tasks/specs/3D.1_decompose_phase_3d.md). To be moved to
**Accepted / Implemented** by [`3D.4`](../tasks/specs/3D.4_phase3_notebook_adr_journal_close.md)
once the Phase 3 closing memo ([`docs/phase3_result_memo.md`](../phase3_result_memo.md),
3D.3) is committed and the verdict is finalised against the
acceptance bar.

> This is a decision *skeleton*: the structure, the candidate decision,
> and the evidence pointers are laid down here so 3D.3 (memo) and 3D.4
> (close) have a fixed target. Every numeric value below is a
> **placeholder cite** to an already-committed bundle; 3D.4 confirms the
> final numbers from the 3D.3 memo and flips the status. Do not treat
> the Decision section as final until the status reads Accepted.

## Date

2026-06-15 (skeleton); decision date set by 3D.4 on close.

## Context

Phase 3 ("Accuracy Push — beat RBF without losing reliability", framed
by [ADR 0004](0004_phase3_accuracy_push_framing.md)) set a hard
acceptance bar: the conditional neural model must reach **test MAE
≤ 0.95 × RBF** on the benchmark, **on its own** (RBF-as-prior / residual
hybrids were explicitly reserved for a Phase 4 fallback, not Phase 3).

Three independent levers were tried, plus a data-correction interlude.
All are `done`; the consolidated ladder lives in the 3D.3 memo. Summary
(clean OTM substrate `spy_phase1_random40_noiselow_otm`, the operative
verdict per [ADR 0006](0006_duplicate_coordinate_data_correction.md)):

| Epic | Lever | Clean-OTM verdict vs RBF | Evidence bundle |
|---|---|---|---|
| 3A | coordinate representation (Fourier vs raw `(k,τ)`) | raw beats Fourier; gap-to-RBF unclosed | `results/3/spy_phase1_random40_noiselow/3a_compare/` |
| 3B | cross-attention decoder (DeepSets+ANP) | best head +2.7 % (dirty) → **+61 %** (clean OTM); bar NOT met | `results/3/spy_phase1_random40_noiselow_otm/3x_compare/` |
| 3X | data correction (single-valued OTM surface) | RBF lead *widened*, not closed (RBF floor 0.00613) | `results/3/spy_phase1_random40_noiselow_otm/3x_compare/` |
| 3C | feature expansion (`micro_v1` microstructure) | test MAE **worsened** in all 3 heads vs 3X.9 | `artifacts/runs/3C3/{gaussian,quantile,point}/manifest.json` |

(All figures above are carried from the committed 3B / 3X / 3C closing
addenda and journal entries; 3D.3 restates them in one table and 3D.4
locks the exact values cited here.)

The consistent finding: on a well-posed single-valued surface, the
per-date RBF interpolator is a very strong local baseline, and a
global-prior amortized conditional neural model did not re-derive its
local weighting from data within the levers Phase 3 budgeted.

## Decision (candidate — finalised by 3D.4)

1. **No pure conditional-neural predictor is promoted to production**
   for SPY IV-surface inference at the close of Phase 3. The Phase 3
   acceptance bar (≥ 5 % below RBF on test MAE) was **not met** by any
   variant on the clean OTM substrate.
2. **RBF interpolation remains the production accuracy baseline** for
   the surface itself, unchanged from Phase 1.
3. **The calibrated reliability layer is retained as a complementary
   signal**, not as the surface estimator: the ANP-based
   uncertainty / abstention / tradability outputs (Phase 2D + 3X.11/3X.12
   calibration recipe) remain the project's reliability contribution
   where a downstream consumer needs a trust signal, even though the
   point surface is served by RBF.
4. **Forward direction = Phase 4: RBF-prior hybrid / residual neural
   model.** Let RBF carry the local interpolation it already wins, and
   have a neural model learn (a) the residual where RBF is weak (sparse
   wings / extreme maturities) and (b) the calibrated reliability +
   abstention layer RBF lacks. This is a *deployment-engineering*
   answer, explicitly **not** a retraction of the Phase 3 research
   question. It is the fallback ADR 0004 reserved.

## Consequences

### Positive
- Honest close: the production recommendation matches the measured
  evidence rather than the original hypothesis.
- Phase 4 has a concrete, pre-justified starting point (residual hybrid)
  instead of an open-ended search.
- The reliability layer (the project's durable Phase 2/3 contribution)
  is preserved and given a defined role.

### Negative / costs
- The headline Phase 3 hypothesis ("a conditional neural model can beat
  RBF on its own") is **not** validated; the closing memo must state
  this plainly.
- Phase 4 introduces hybrid-system complexity (two estimators + a fusion
  rule) that a pure neural model would have avoided.

### Neutral
- No production code changes land in Phase 3D itself (synthesis only).
- The `feature_set` flag (ADR 0008) and the ANP architecture (ADR 0005)
  remain available and reproducible for Phase 4 reuse.

## Alternatives considered (to be expanded by 3D.4)

- **Promote the best clean-OTM ANP head anyway** (point head, the
  strongest neural variant). Rejected: it is +61 % above RBF on test
  MAE — promoting it would regress production accuracy.
- **Declare Phase 3 a failure and stop.** Rejected: the reliability
  layer is a real deliverable, and the residual-hybrid path is a
  well-motivated next step, not a dead end.
- **Re-open feature work (`micro_v2` with `vix` / realized-vol via a new
  data-source ADR).** Deferred (per ADR 0008 Outcome): lower expected
  value than the residual hybrid given pure feature expansion already
  regressed.

## Outcome (filled by 3D.4 on close)

_To be filled by 3D.4 with: the final consolidated ladder numbers from
`docs/phase3_result_memo.md`, the explicit acceptance-bar verdict, the
locked production recommendation, and (if warranted) a pointer to the
Phase 4 epic placeholder on the BOARD._
