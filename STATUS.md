# STATUS — Phase 4B CLOSED (accuracy retired); next: Phase 5A reliability-first

**Updated:** 2026-06-22
**Branch:** main
**Mode:** Phase 4B closed. Next work (Phase 5A) is local CPU; no pod required.

## Headline

Phase 4 is **closed positive** (epic 4A `done`; RBF-prior gaussian hybrid is
the adopted production estimator; ADR 0010 Implemented). The win over RBF is
**statistically real but economically small** (≈ 2 % relative MAE on the dense
clean OTM benchmark), so the project will **not** keep grinding that benchmark.
The forward direction is recorded in
[ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
(**Accepted; 4B gate resolved**):

- **`4B` — sparsity-sweep diagnostic (the decision gate). ✅ CLOSED 2026-06-21.**
  Verdict `ambiguous` → accuracy **retired on this substrate** (relative edge
  collapses under wing/maturity sparsity; grows only mildly under benign
  thinning). 4B.7 fair-retrain **declined**. Hybrid stays adopted (ADR 0010).
- **`5A` — reliability-first surface inference / quote-risk layer. ← NOW PRIMARY.**
  The 4B coverage collapse (0.96→0.35 under sparsity) makes this the main story.
- **`6A` — structured-product pricing / FCN monetization demo.** Sell-side
  framing; constrained demo, not a full FCN pricer. Downstream of 5A.

## Where things stand

- **Phase 4B fully closed** (epic 4B `done`; 4B.1–4B.6 `done`; 4B.7 `cancelled`).
  ADR 0011 §Outcome filled; roadmap §7 close; README pivoted to Phase 5A.
- Phase 4A result unchanged; RBF-prior hybrid remains the adopted estimator.
- Next phase: **5A** (reliability-first) — decomposition spec `5A.1` is the entry
  (`backlog`); roadmap `docs/roadmaps/phase5_reliability_first_surface_inference.md`.

## Current task — Phase 4B CLOSED (4B.6 done); next Phase 5A

**Epic 4B `done`.** 4B.1–4B.6 `done`; **4B.7 `cancelled`** (operator declined the
GPU fair-retrain escalation on economic grounds). 4B.5 gate returned
`ambiguous`; 4B.6 closed the phase with **accuracy retired on this substrate**.

### Phase 4B verdict (final)

- The RBF-prior hybrid's **relative** edge over RBF does **not** survive realistic
  sparsity: collapses to 0.3–0.5 % under wing/maturity stress; grows only to
  ~4 % under benign random thinning (sub-5 % bar). All deltas significant; dense
  ties 4A exactly.
- **ADR 0010 unchanged** — the hybrid stays the adopted estimator. "Retired" =
  accuracy is no longer the primary forward *story*, not that the hybrid is
  discarded.
- Secondary: calibrated coverage collapses 0.962 → 0.35–0.39 under wing sparsity
  → reliability degrades faster than accuracy → **Phase 5A is now primary**.
- Recorded in ADR 0011 §Outcome, roadmap §7, README, journal. Artifacts:
  `results/4/spy_phase1_random40_noiselow_otm/4b_sweep/`.

### Pod status

**No pod needed.** The current CPU pod is fully terminable (all 4B remote work is
done and pulled local; 4B.7 cancelled). Phase 5A starts local.

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

## Phase 4B chain (final)

`4B.1` decompose → `4B.2` harness → `4B.3` swept RBF (remote) → `4B.4` hybrid
forward (remote) → `4B.5` trajectory + gate (`ambiguous`) → `4B.6` close — all
`done`. `4B.7` per-regime retrain **`cancelled`** (operator declined the GPU
escalation). Epic 4B `done`.

## Next concrete action — enter Phase 5A (reliability-first)

- Phase 4 and 4B are closed. Begin **Phase 5A**: the reliability-first surface
  inference / quote-risk layer (calibrated uncertainty, abstention, surface
  confidence, **per-regime recalibration** — directly motivated by the 4B
  coverage collapse — no-arb/risk flags, quote/no-quote logic). Entry is the
  `5A.1` decomposition spec (`docs/tasks/specs/5A.1_decompose_phase_5a.md`) +
  roadmap `docs/roadmaps/phase5_reliability_first_surface_inference.md`. Starts
  local; no pod required. Next step is to decompose 5A into atomic stories.
