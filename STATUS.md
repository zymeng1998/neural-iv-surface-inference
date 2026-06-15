# STATUS — Phase 3 epic 3C closed (3C.8) + 3B/3D housekeeping

**Updated:** 2026-06-14
**Branch:** main (local == origin/main @ 3960308; this work is staged as one
uncommitted docs/planning change set)
**Mode:** local, documentation/planning only — no GPU, no model/eval/pipeline run

## What was just completed

Operator-directed Phase 3 close-out + housekeeping, all docs/planning:

1. **Closed epic 3C (3C.8 closing addendum).** 3C.3 had already shown the
   `micro_v1` microstructure feature set (ADR 0008) *worsens* test MAE in all
   three heads vs the 3X.9 minimal-feature OTM baseline (gauss +0.00194 /
   quant +0.00187 / point +0.00452; all above the RBF floor 0.00613). Clean
   negative; Phase 3 ≥ 5 %-below-RBF bar **NOT met**. ADR 0008 → **Implemented**
   (Outcome block filled, three deferred questions answered).
2. **Cancelled 3C.4–3C.7.** Because base accuracy already regressed, the
   ensemble / calibrator / decision-layer / comparison chain on the same
   feature set is no longer informative. Marked `cancelled` on BOARD + index +
   roadmap table, with a rationale banner in each spec. No Pod spend.
3. **Resolved dangling 3B in_review.** 3B.1 / 3B.6 / 3B.7 promoted
   `in_review → done`; epic 3B → `done`. The original 3B **dirty-substrate**
   verdict (ANP +2.7 % vs RBF) is now explicitly **superseded by the 3X
   clean-OTM restatement** (gap widened to +61 % best head / +90 % calibrated);
   §W11 addendum preserved for traceability.
4. **Teed up 3D.** All gating epics (3A/3B/3X/3C) `done`. §W13 + 3D row updated:
   Phase 3 closing memo + RBF re-eval, with **Phase 4 framed as an RBF-prior
   hybrid / residual neural model**. (Noted the ADR-number collision: the
   production-selection ADR is **0009**, not 0008 — 3D.1 to fix on execution.)
5. **Honesty fixes** in the same files: PHASE3_INDEX 3X.13 and roadmap §W11.5
   3X.7/3X.8 `in_review → done` drift corrected to match BOARD.

## Files changed (uncommitted working tree, docs/planning only)

- `docs/decisions/0008_microstructure_feature_set_freeze.md` (Implemented + Outcome)
- `docs/roadmaps/phase3_accuracy_push.md` (§W12 addendum, §W13 Phase 4, status block, table drift)
- `docs/tasks/BOARD.md`, `docs/PHASE3_INDEX.md`
- `docs/tasks/specs/3C.4–3C.7` (cancelled), `3C.8` (in_review), `3B.1/3B.6/3B.7` (done)
- `docs/logs/progress_log.md`, `docs/experiments/experiment_journal.md`
- `README.md`
- `STATUS.md` (this file)

## Git state / next concrete action

- **Not yet committed.** Prepare ONE atomic docs/planning commit, then **STOP
  before push** for operator review (per instruction).
- Commit must carry the `Co-authored-by: Claude <noreply@anthropic.com>`
  trailer (project commit-msg hook + CLAUDE.md override).
- The pre-push gates (PMR + dependency + file-scope) only fire on `git push`,
  not on commit. This change set spans multiple stories (operator-directed
  housekeeping), so it intentionally exceeds any single story's `file_scope`;
  expect to clear the file-scope gate with a documented `WAIVE_*` (or split)
  if/when the operator pushes.

## After review (operator)

1. Promote 3C.8 → `done` on BOARD + index.
2. Promote 3D.1 (`backlog → todo`); run the Phase 3D decomposition (closing
   memo + RBF re-eval; ADR 0009 production-selection; frame Phase 4 hybrid).
3. Run `python3 scripts/pmr_prepush_gate.py --verbose --dry-run` before push.
