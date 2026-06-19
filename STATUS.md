# STATUS — Phase 4 (`4A`) kicked off & decomposed; next = run 4A.2

**Updated:** 2026-06-18
**Branch:** main
**Mode:** local, CPU only — no GPU, no model/eval/data run

## Where things stand

Phase 3 closed (negative on accuracy); **M1.6 (waiver-hook fix) is `done`
and on origin** (`7f5a386`). **Phase 4 (epic `4A`) is now kicked off and
fully decomposed** — roadmap + ADR 0010 + 4A.1 decomposition + the full
4A.2–4A.8 child specs.

origin/main is at **7f5a386**. The Phase 4 kickoff commit below is local,
NOT yet pushed.

## What was just completed (this local commit — docs/planning only)

- `docs/roadmaps/phase4_hybrid_residual.md` (new) — Phase 4 plan.
- `docs/decisions/0010_rbf_prior_residual_hybrid.md` (new, **Proposed**) —
  additive residual hybrid `σ̂ = RBF + f_θ(r=iv−rbf)`; reuse 3B/3X backbone
  via a `target_mode: residual` flag; reuse the 3X.11 calibrator. Backbone
  fork (ANP-residual vs MLP-residual) open for the 4A.4 review.
- `4A.1` decomposition spec (`in_review`) + **4A.2–4A.8** child specs
  (`backlog`, fully populated, each with Compute requirements). Chain mirrors
  3X: builder → residual dataset → train 3 heads → K=5 ensemble → calibrator
  → decision-layer eval + **bootstrap CI vs RBF** → comparison + close.
- Epic 4A → `in_progress`; created `docs/PHASE4_INDEX.md`; updated README +
  BOARD + progress_log.

**Operator decisions captured:** success bar = *any statistically meaningful
gain over RBF (0.00613) + reliability preserved* (negative branch — ship the
reliability layer on RBF — is acceptable); depth = full decomposition now.

## Push readiness (kickoff commit — designed zero-waiver, NOT pushed)

- **SCOPE:** only active spec is 4A.1; its `file_scope` covers every touched
  path (roadmap, ADR 0010, `4A.*_*.md`, BOARD, PHASE4_INDEX, README, STATUS,
  progress_log); 4A.2–4A.8 are `backlog` → PASS.
- **DEP:** 4A.1 has no non-`done` deps → PASS.
- **PMR:** docs/planning only → PASS.
- **Do NOT set `WAIVE_DEPS` / `WAIVE_SCOPE`.**

## Next concrete action

- **Verify all three gates standalone, stop before push for approval.**
- After approval: push (clean env). Then operator promotes 4A.1 → `done`.
- Then run **4A.2** (residual-target builder + `target_mode` flag + tests),
  local CPU. GPU stories (4A.4/4A.5) gated on a Pod go-ahead.
