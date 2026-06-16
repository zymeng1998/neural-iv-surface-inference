# STATUS — PHASE 3 CLOSED (negative on accuracy); next = M1.6, then Phase 4

**Updated:** 2026-06-16
**Branch:** main
**Mode:** local, CPU only — no GPU, no model/eval/data run

## Where things stand

**Phase 3 is closed.** Epic 3D and stories 3D.1–3D.4 are `done`. Verdict:
**negative on accuracy** — no pure conditional-neural variant beats RBF on
the clean OTM substrate (best head ANP point 0.00987 vs RBF floor 0.00613
= **+61 %**; calibrated production +90 %). RBF remains the production
accuracy baseline; the calibrated reliability layer is retained; the
forward direction is **Phase 4 = RBF-prior hybrid / residual neural model**
(epic `4A`, backlog placeholder — not decomposed).

origin/main is at **74bc1bb** (3D.3). The 3D.4 close commit below is
local, NOT yet pushed.

## What was just completed (3D.4, one local commit)

1. Emitted committed `notebooks/06_phase3_results.ipynb` (19 cells);
   `nbconvert --execute` → 0 cell errors.
2. ADR 0009 → **Accepted/Implemented** (Outcome block filled with the
   clean-OTM ladder + locked decision).
3. Phase 3 close-out entry in `docs/experiments/experiment_journal.md`.
4. Promoted 3D.2 + 3D.3 → `done`; flipped 3D.4 + epic 3D → `done`.
5. Added Phase 4 epic placeholder `4A` (backlog) to the BOARD.
6. Synced roadmap §W13, README, PHASE3_INDEX, progress_log, STATUS.

## Push readiness (3D.4 close — designed zero-waiver, NOT pushed)

- **SCOPE:** the commit flips every 3D story + epic to `done`, so there is
  no `in_review`/`in_progress` spec in the diff → scope gate trivially
  passes (no `WAIVE_SCOPE`).
- **DEP:** every story's deps are `done` → PASS (no `WAIVE_DEPS`).
- **PMR:** notebook + journal are evidence-class → live PMR; docs +
  progress_log updated → PASS.
- Single self-contained commit → one normal clean push (no sequencing).

## Next concrete action

- **Verify all three gates standalone, stop before push for operator
  approval** (per the established rhythm).
- After approval: push (clean env, no waiver vars).
- **Then M1.6** (waiver-hook fix) — operator-directed to precede Phase 4 so
  the GPU-heavy cadence isn't slowed by post-commit waiver mutations /
  sequenced pushes.
- Then **Phase 4 (`4A`)** kickoff — roadmap + hybrid-design ADR +
  decomposition; gated on operator go-ahead (reopens Pod spend).
