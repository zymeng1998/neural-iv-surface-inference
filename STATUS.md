# STATUS — Phase 4B CLOSED POSITIVE (accuracy SURVIVES); next Phase 5A (parallel)

**Updated:** 2026-06-22
**Branch:** main
**Mode:** 4B fully closed (gate `true`). Next work (Phase 5A) is local; pod
terminable.

## Headline

Phase 4 is **closed positive** (epic 4A `done`; RBF-prior gaussian hybrid is
the adopted production estimator; ADR 0010 Implemented). The win over RBF is
**statistically real but economically small** (≈ 2 % relative MAE on the dense
clean OTM benchmark), so the project will **not** keep grinding that benchmark.
The forward direction is recorded in
[ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
(**Accepted; 4B gate resolved**):

- **`4B` — sparsity-sweep diagnostic (the decision gate). ✅ CLOSED POSITIVE
  2026-06-22.** Gate `ambiguous` (eval-time) → **`true` (accuracy survives)**
  after the full fair retrain (4B.7). Trained on sparse context, the hybrid's
  edge over RBF grows under structured sparsity (thin_wings +16.6%,
  missing_maturities +37.1%, combined +16.0% at sparsest; all sig); the
  eval-time wing collapse was an OOD artifact. Accuracy stays a core story.
- **`5A` — reliability-first surface inference / quote-risk layer. ← proceeds in
  parallel.** Coverage collapse 0.96→0.35 under sparsity + the per-regime
  accuracy result both motivate **per-regime recalibration**.
- **`6A` — structured-product pricing / FCN monetization demo.** Sell-side
  framing; constrained demo, not a full FCN pricer. Downstream of 5A.

## Where things stand

- **Phase 4B fully closed POSITIVE** (epic 4B `done`; **4B.1–4B.7 all `done`**).
  Gate resolved `true`; ADR 0011 §Outcome rewritten; roadmap §7 + README updated.
- Phase 4A result unchanged; RBF-prior hybrid remains the adopted estimator.
- Next phase: **5A** (reliability-first, in parallel with the now-alive accuracy
  story) — decomposition spec `5A.1` is the entry (`backlog`); roadmap
  `docs/roadmaps/phase5_reliability_first_surface_inference.md`.

## Current task — 4B closed positive; 4B.7 done

**Epic 4B `done` (closed positive 2026-06-22).** 4B.1–4B.7 all `done`. The full
16-cell fair retrain (4B.7, RTX 4000 Ada, 6.3 h GPU) resolved the gate
`ambiguous → true`: the eval-time wing collapse was an OOD artifact; trained
fairly the hybrid's relative edge over RBF grows under structured sparsity
(missing_maturities +37.1%, thin_wings +16.6%, combined +16.0% at the sparsest
rung — all significant; fewer_quotes → ~0, random thinning where RBF interpolates
fine). Caveat: per-cell models trained for each exact sparsity; a single
mixed-sparsity model is the production-realistic follow-up.

- Artifacts: `artifacts/runs/4B7/{manifest.json, fair_retrain_stats.csv,
  cells/*.json}` + `results/4/.../4b_sweep/{fair_trajectory.csv,
  fair_trajectory_wide.md, gate_verdict.json (resolved)}` — all local + committed.
- **Pod terminable** (4B.7 done, artifacts pulled). Pod-side checkpoints/preds
  gitignored + regenerable.

### Phase 4B verdict (final) — `accuracy_survives = true`

- The eval-time read (4B.5) was `ambiguous` (wing edge collapsed). The full fair
  retrain (4B.7) **reverses it**: trained on sparse context, the hybrid's
  relative edge over RBF **grows** under structured sparsity — sparsest rung
  thin_wings +16.6 %, missing_maturities +37.1 %, combined +16.0 % (all sig);
  fewer_quotes → ~0 (random thinning, RBF interpolates fine). Both wing-sensitive
  regimes clear the survive bar → **accuracy survives**.
- **ADR 0010 unchanged** — RBF-prior hybrid stays the adopted estimator.
- Reliability finding still holds: calibrated coverage collapses 0.962 → 0.35
  under sparsity → **Phase 5A reliability-first proceeds in parallel** (per-regime
  recalibration now doubly motivated).
- Recorded in ADR 0011 §Outcome (rewritten), roadmap §7, README, journal.

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

## Phase 4B chain (final)

`4B.1` decompose → `4B.2` harness → `4B.3` swept RBF → `4B.4` hybrid forward →
`4B.5` trajectory + gate (`ambiguous`) → `4B.7` full fair retrain (gate → `true`)
→ `4B.6` close (positive) — **all `done`. Epic 4B `done`.**

## Next concrete action — enter Phase 5A (reliability-first)

- Phase 4 + 4B closed (positive). Begin **Phase 5A**: reliability-first surface
  inference / quote-risk layer (calibrated uncertainty, abstention, surface
  confidence, **per-regime recalibration** — now doubly motivated by the 4B
  coverage collapse *and* the per-regime accuracy result — no-arb/risk flags,
  quote/no-quote logic). Entry is the `5A.1` decomposition spec
  (`docs/tasks/specs/5A.1_decompose_phase_5a.md`) + roadmap
  `docs/roadmaps/phase5_reliability_first_surface_inference.md`. Local; no pod.
- Possible accuracy follow-up (noted): a single model trained on *mixed*
  sparsity (production-realistic vs 4B.7's per-cell models).
