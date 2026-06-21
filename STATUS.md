# STATUS — Phase 4 CLOSED POSITIVE (epic 4A done); 4A.8 ready to push

**Updated:** 2026-06-20
**Branch:** main
**Mode:** local CPU. All GPU work for Phase 4 is complete.

## Headline

**Phase 4 closes POSITIVE on accuracy.** The RBF-prior residual hybrid is the
first predictor in the project to statistically significantly beat RBF on the
clean OTM substrate. Epic 4A (4A.1–4A.8) is `done`; ADR 0010 is Implemented.

## Where things stand

- origin/main is at **cf30a7c** (4A.7 pushed + promoted to `done`).
- **4A.8 (Phase 4 close) is done locally, NOT yet committed/pushed** — awaiting
  operator review, then commit + push.

## 4A.8 deliverables (this session, uncommitted)

- `scripts/run_4a8_comparison.py` — provenance-checked assembler; asserts every
  cited source exists and that the reported delta + RBF floor match the
  committed 4A.7 `mae_delta_ci.json` (no drift). Run → exit 0, `bar_met=True`.
- `results/4/spy_phase1_random40_noiselow_otm/4a_compare/` — `comparison.csv`,
  `comparison_wide.md`, `headline.json`.
- `docs/phase4_result_memo.md` — Phase 4 verdict memo (positive; production
  recommendation + caveats).
- ADR 0010 → **Implemented** (Outcome filled); roadmap `status: done` + close
  block; journal close-out; progress_log entry; BOARD + PHASE4_INDEX epic 4A +
  4A.8 → `done`; README Phase 4 callout → CLOSED POSITIVE.

## The verdict (cited)

- Accuracy: hybrid **0.006006** vs RBF **0.006132**; mean Δ **−0.000126**,
  **95 % CI [−0.000144, −0.000106]** (excludes 0) → significant.
  (`results/4/.../4a_hybrid/mae_delta_ci.json`)
- Reliability: coverage **0.9181** vs iv_true (±2 pp ✓); hi-conf MAE
  **0.004710** < no-abstention **0.006006**; width **0.0328**.
  (`results/4/.../4a_hybrid/metrics_summary.json`)
- No-arb flag count NOT recomputed (needs the checkpoint; hybrid = RBF + small
  smooth residual → no-arb tracks RBF's). Deferred caveat, not a blocker.

## Push readiness (4A.8 — designed zero-waiver)

- **PMR (live):** `scripts/` + `results/4/` + memo; journal + progress_log +
  ADR + roadmap + BOARD + index updated → expected PASS.
- **DEP:** 4A.8 dep 4A.7 = `done` → PASS.
- **SCOPE:** only active spec 4A.8; all touched files in its `file_scope`.
  No `WAIVE_*`.

## Next concrete action

- Operator review; on approval: commit 4A.8 close, push, promote 4A.8 →
  already `done` on the spec/BOARD (no separate promote needed — closing story).
- No open Phase 4 work. Deferred follow-ups (backlog placeholders only):
  (1) calibrator re-fit on iv_clean to tighten conservative coverage;
  (2) no-arb flag-count audit on the hybrid checkpoint;
  (3) all-11-OTM-variant robustness study.
