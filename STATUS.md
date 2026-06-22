# STATUS — Phase 4B REOPENED; running full fair retrain (4B.7, GPU)

**Updated:** 2026-06-22
**Branch:** main
**Mode:** 4B reopened — full fair retrain (4B.7) on a fresh GPU pod (16 cells).
The 4B.6 close + ADR 0011 retired-verdict are provisional pending this.

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

## Current task — 4B REOPENED; full fair retrain (4B.7) in progress

**Epic 4B `in_progress` (reopened 2026-06-22).** 4B.1–4B.5 `done`; **4B.6
reopened to `in_progress`** (the close was provisional); **4B.7 `in_progress`**
— the operator reversed the earlier decline and chose the **full fair retrain**
(all 4 regimes × 4 non-dense rungs = 16 cells) to resolve the eval-time OOD
ambiguity from 4B.5.

### 4B.7 plan (full fair retrain)

Per (regime, rung): re-derive the rung's sparse mask on **all splits** (4B.2
harness, seed 4023 — test masking matches 4B.4 exactly), rebuild residual
targets with RBF re-fit on the **sparse** context, retrain the hybrid mirroring
4A.4 exactly (anp decoder, hidden 128 / latent 64, gaussian head, 50 epochs,
seed 42), score the sparse test context → σ̂, recompute the cell's edge + a
date-clustered CI. Dense rung (intensity 0) needs no retrain (= 4A.4 anchor).
Then refresh the contested trajectory rows and resolve `gate_verdict.json` to
`true`/`false`, hand back to 4B.6 for the (now real) close.

- Harness: `scripts/run_4b7_retrain_sweep.py` (+ `.sh`) + config
  `configs/conditional_4B7_retrain_gaussian_otm.yaml`.
- **Needs a fresh GPU pod** (operator spinning up; the prior pod was CPU-only).
  Reuses the residual parquet + OTM data on the volume (re-sync if new volume).

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

## Phase 4B chain

`4B.1` decompose → `4B.2` harness → `4B.3` swept RBF → `4B.4` hybrid forward →
`4B.5` trajectory + gate (`ambiguous`) all `done`. `4B.6` close **reopened**
(`in_progress`). `4B.7` full fair retrain **`in_progress`** (16 cells, GPU).

## Next concrete action — run 4B.7 on the GPU pod

1. Build/commit the retrain harness (`run_4b7_retrain_sweep.py` + `.sh` +
   config) — ready before the pod is up.
2. Operator provides the fresh GPU pod SSH string → scp harness + verify the
   residual parquet / OTM data are on the volume (re-sync if new).
3. Run the 16-cell fair retrain (~3–5 h GPU): per cell, sparse-context residual
   rebuild → retrain (mirror 4A.4) → score → edge + CI. Pull fair predictions
   local.
4. Recompute the fair trajectory vs the eval-time one; resolve
   `gate_verdict.json` to `true`/`false`; hand to 4B.6 for the real close.
