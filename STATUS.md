# STATUS — Phase 4 accuracy bar MET (4A.7); next = 4A.8 close

**Updated:** 2026-06-20
**Branch:** main
**Mode:** local CPU. All GPU work for Phase 4 is complete.

## Headline

**The RBF-prior residual hybrid statistically significantly beats RBF on the
clean OTM substrate** — the first neural-based predictor in the project to do
so. (4A.7, the formal Phase 4 bar.)

## Where things stand

Phase 4 (epic `4A`) in progress. 4A.1–4A.6 `done` (origin `1889c4b`).
**4A.7 (bar adjudication) done locally and `in_review`.** Only 4A.8 (close)
remains — local.

origin/main is at **1889c4b**. The 4A.7 commit below is local, NOT pushed.

## 4A.7 result

Date-clustered bootstrap CI (full test fold), gaussian hybrid (production
head) vs RBF, vs `iv_clean`:
- hybrid **0.006006** vs RBF **0.006132**; mean Δ **−0.000126**;
  **95% CI [−0.000144, −0.000106]** (entirely < 0) → **significant**.
- hi-conf MAE 0.004710 < no-abstention 0.006006 ✓.
- coverage 0.9181 vs iv_true (in-band) / 0.962 vs iv_clean (conservative,
  over-covers — a calibrator-refit-on-iv_clean follow-up, not a blocker).
- Flag count not recomputed locally (needs the model checkpoint, released
  with the pod); hybrid surface = RBF + small smooth residual, so no-arb
  behavior tracks RBF's; deferred.

New tools: `eval_mae_bootstrap_ci.py` (+5 tests) and `run_4a7_decision_layer.py`.

## Push readiness (4A.7 — designed zero-waiver, NOT pushed)

- **PMR (live):** `scripts/` + `tests/` + `results/4/`; journal + progress_log
  updated → PASS.
- **DEP:** 4A.7 dep 4A.6 = `done` → PASS.
- **SCOPE:** only active spec 4A.7; touched files in its `file_scope`
  (`STATUS.md` added) → PASS. **No `WAIVE_*`.**

## Next concrete action

- **Verify gates standalone, stop before push for review.**
- After approval: push; promote 4A.7 → `done`.
- **4A.8** (local): Phase 4 closing memo (`docs/phase4_result_memo.md`) +
  fill ADR 0010 Outcome (bar MET; recommend the RBF-prior hybrid as the
  production surface, with the coverage-refit caveat) + flip epic 4A `done`.
  Cited numbers already committed in `results/4/.../4a_hybrid/` + the 4A.4/4A.5
  manifests.
