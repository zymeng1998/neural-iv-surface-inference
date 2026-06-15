# STATUS — 3C closed & finalized; 3D.1 readied (todo); M1.6 filed

**Updated:** 2026-06-14 (late)
**Branch:** main
**Mode:** local, documentation/planning only — no GPU, no model/eval/pipeline run

## Where things stand

Epic **3C is closed** (`micro_v1` negative; bar NOT met; ADR 0008
Implemented). The previous push (5787d2e) is on `origin/main`. This step
finalizes the close-out cleanly and sets up 3D, as **one atomic commit**
designed to push with **no `WAIVE_DEPS` / `WAIVE_SCOPE`**.

## What was just completed (this commit, docs/planning only)

1. **3C.8 `in_review → done`** (operator approved). Rewrote its
   `## Dependencies` to the real satisfied set (3C.3 + 3X.14, both
   `done`); cancelled 3C.6/3C.7 demoted to a prose note. This is what
   lets the dependency gate pass without `WAIVE_DEPS`.
2. **Folded in the post-push waiver-audit lines.** The `DEP/SCOPE WAIVER`
   lines the pre-push hook appended to the 3C.8 spec during the 5787d2e
   push are committed here (bundled with the promotion, never as a
   standalone waiver-only commit).
3. **3D.1 `backlog → todo` (readied, not executed).** Fixed the
   ADR-number collision — production-selection ADR is **0009**, not 0008
   — and wrote in the Phase 4 = RBF-prior hybrid / residual neural
   framing. Running the decomposition (epic 3D → in_progress, child
   specs 3D.2–3D.4) is the next step, deliberately not done here.
4. **M1.6 filed (backlog)** — pre-push waivers must not mutate tracked
   files after the pushed commit; write to a separate generated/untracked
   audit log with a follow-up workflow.
5. Synced BOARD / PHASE3_INDEX / roadmap §W13 / progress_log / STATUS.

## Push plan (NO waivers)

- Scope gate: no changed spec is `in_progress`/`in_review` → no active
  specs → PASS.
- Dep gate: only 3C.8 (`done`) is active; deps now `done`-only → PASS.
- PMR gate: docs-only → PASS.
- **Do NOT set `WAIVE_DEPS` or `WAIVE_SCOPE` on this push.** If a gate
  blocks, fix the doc to satisfy it rather than waiving (the whole point
  of this step). The previous waiver-write-loop must not recur.

## Next concrete action

1. Commit atomically (commit-msg trailer required) and run all three
   gates standalone to confirm PASS, then push with a clean env.
2. After push: run the **3D.1 decomposition** (epic 3D → in_progress;
   draft 3D.2/3D.3/3D.4; ADR 0009; roadmap §W13 + index).
3. Schedule **M1.6** (waiver-hook fix) when convenient.
