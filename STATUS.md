# STATUS — Phase 3D entered & decomposed (3D.1 executed)

**Updated:** 2026-06-15
**Branch:** main
**Mode:** local, documentation/planning only — no GPU, no model/eval/pipeline run

## Where things stand

Phase 3's three accuracy levers (3A / 3B / 3X / 3C) are all `done` and
all missed the ≥ 5 %-below-RBF bar on the clean OTM substrate. **Epic 3D
(closing memo + RBF re-eval) is now `in_progress` and decomposed.**

## What was just completed (this commit, docs/planning only)

1. **Epic 3D → `in_progress`**; **3D.1 → `in_review`** (decomposition
   executed).
2. **Three atomic child stories drafted (`backlog`)**, all local CPU:
   - **3D.2** — `scripts/generate_phase3_results_notebook.py` generator
     scaffold + smoke test.
   - **3D.3** — `docs/phase3_result_memo.md` (verdict vs RBF + vs 2D;
     dirty + clean-OTM restatement; §5 acceptance map).
   - **3D.4** — emit `notebooks/06_phase3_results.ipynb`, finalize ADR
     0009, journal close-out, flip epic 3D `done` (+ optional Phase 4
     placeholder).
   Chain: `3D.2 ∥ 3D.3 → 3D.4`.
3. **ADR 0009 skeleton** created
   (`docs/decisions/0009_phase3_production_predictor_selection.md`,
   `Proposed`): no pure conditional-neural predictor promoted; RBF stays
   the accuracy baseline; reliability layer retained; **Phase 4 =
   RBF-prior hybrid / residual neural**. Outcome filled by 3D.4.
4. Synced BOARD / roadmap §W13 / PHASE3_INDEX / progress_log / STATUS.

## Push readiness (designed zero-waiver — NOT yet pushed)

- Scope gate: only 3D.1 is active; its `file_scope` covers every touched
  path (`3D.*_*.md`, `0009_*.md`, BOARD, roadmap, index, progress_log,
  `STATUS.md`). The child specs are `backlog` (not active). → PASS.
- Dep gate: 3D.1 has no `## Dependencies` section. → PASS.
- PMR gate: docs/planning only. → PASS.
- **Do NOT set `WAIVE_DEPS` / `WAIVE_SCOPE` on this push.** All three
  gates were verified standalone before commit.

## Next concrete action

- **Stopped before push for review** (per instruction). After approval:
  push the commit with a clean env (no waiver vars).
- Then: operator promotes 3D.1 → `done`; run 3D.2 ∥ 3D.3, then 3D.4 to
  close epic 3D / Phase 3.
- Backlog: **M1.6** (waiver-hook fix) when convenient.
