# STATUS — M1.6 done (waiver-hook fix); next = Phase 4 (4A) kickoff

**Updated:** 2026-06-17
**Branch:** main
**Mode:** local, CPU only — no GPU, no model/eval/data run

## Where things stand

Phase 3 is closed (negative on accuracy; RBF stays production baseline).
**M1.6 (waiver-hook fix) is implemented** and `in_review`. Next is the
**Phase 4 (`4A`) kickoff** (RBF-prior hybrid / residual neural) — planning
only until the operator green-lights GPU spend.

origin/main is at **093adc9** (Phase 3 close). The M1.6 commit below is
local, NOT yet pushed.

## What was just completed (M1.6, one local commit)

- Both pre-push gates now record a fired `WAIVE_DEPS` / `WAIVE_SCOPE`
  bypass to the **untracked, gitignored** `docs/audit/waiver_log.md` (via
  the shared `record_waiver_audit` helper) instead of appending to a
  tracked spec. A waived push no longer dirties the working tree.
- Removed the spec-mutating `append_audit_line`; `evaluate()` unchanged
  (only the audit sink moved). Trigger conditions untouched.
- `.gitignore` ignores `docs/audit/`. ADR 0007 addendum + meta1 roadmap
  follow-up #2 → FIXED. 31 gate tests green.

## Push readiness (M1.6 — designed zero-waiver, NOT pushed)

- **PMR:** touches `scripts/` + `tests/` (evidence-class) → live PMR;
  progress_log + spec updated → PASS.
- **SCOPE:** only active spec is M1.6; `.gitignore` + `STATUS.md` added to
  its `file_scope` → PASS.
- **DEP:** M1.6 has no non-`done` deps → PASS.
- **Do NOT set `WAIVE_DEPS` / `WAIVE_SCOPE`.** (And M1.6 itself now makes
  any future waiver tree-clean.)

## Next concrete action

- **Verify all three gates standalone, stop before push for approval**
  (established rhythm).
- After approval: push (clean env). Then operator promotes M1.6 → `done`.
- Then **Phase 4 (`4A`) kickoff** — roadmap + hybrid-design ADR + `4A.1`
  decomposition; planning only, no training until operator go-ahead.
- Optional: close/park the dormant 2E epic.
