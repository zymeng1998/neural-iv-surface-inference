# STATUS — Phase 3D: 3D.1 done; 3D.2 (notebook generator) implemented

**Updated:** 2026-06-15 (afternoon)
**Branch:** main
**Mode:** local, CPU only — no GPU, no model/eval/data run

## Where things stand

Epic **3D is `in_progress`**. The decomposition (3D.1) is `done`; the
notebook generator (3D.2) is implemented and `in_review`. Remaining:
3D.3 (closing memo) → 3D.4 (emit notebook + finalize ADR 0009 + close).

## What was just completed (two local commits, NOT yet pushed)

1. **`chore(3D.1)` — promote in_review → done** (`3f342eb`). Operator
   approved the decomposition push (94b75dc) + said "start 3D.2"; this
   unblocks the 3D.2 dependency gate.
2. **`feat(3D.2)` — Phase 3 results notebook generator + smoke test.**
   - `scripts/generate_phase3_results_notebook.py`: 19-cell notebook
     assembled from committed bundles (clean-OTM ladder, 3A ablation,
     3B/3X calibrated reliability + figures, 3C `micro_v1` negative,
     training curves, §5 acceptance map, Phase 2D anchor, Phase 4
     framing). `validate_paths()` checks 21 committed artifacts; `main()`
     fails loudly on any missing path; deterministic cell ids.
   - `tests/test_generate_phase3_results_notebook.py`: 5 tests green.
   - **Committed notebook deliberately NOT created** — that is 3D.4.
     Verified via `--output /tmp/...` dry build.

## Push readiness (designed zero-waiver — NOT yet pushed)

- **DEP:** commit A makes 3D.1 `done`; 3D.2's only declared dep (3D.1)
  resolves `done` → PASS.
- **SCOPE:** commit A has no active in_review/in_progress spec → trivially
  passes; commit B's only active spec is 3D.2, whose `file_scope` covers
  the script, test, BOARD, spec, index, progress_log, and `STATUS.md`
  (added per 3C.2/3C.3/3X.7/3X.8 precedent) → PASS.
- **PMR:** commit B touches `scripts/` + `tests/` (evidence-class) → live
  PMR gate; progress_log + spec keep coverage → PASS.
- **Do NOT set `WAIVE_DEPS` / `WAIVE_SCOPE`.** Verified all gates pass
  standalone before stopping.

## Next concrete action

- **Stopped before push for review** (per the established rhythm). After
  approval: push both commits with a clean env (no waiver vars).
- Then: operator promotes 3D.2 → `done`; run **3D.3** (memo), then
  **3D.4** (emit notebook, finalize ADR 0009, close epic 3D / Phase 3).
- Backlog: **M1.6** (waiver-hook fix) when convenient.
