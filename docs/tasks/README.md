# Task System

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

A Jira-style task board for all implementation work. The active-work layer of the
operating model defined in `docs/workflows/ai_human_collaboration.md`.

## Structure

| Path | Role |
|---|---|
| `BOARD.md` | The single canonical board. Every epic and story, with status. **Never deleted.** |
| `specs/` | One spec file per story (`NNNN_short_title.md` or `2A.1_short_title.md`). |
| `_template.md` | Template for new story specs. |

There are no per-status folders. **Status changes happen in place** — you edit the
`status:` field in the spec and the matching row on `BOARD.md`. Files do not move.

## Hierarchy

- **Epic** = a phase (`2A`, `2B`, `2C`, `2D`), defined in `docs/roadmaps/`.
- **Story** = an atomic task under an epic (`2A.1`, `2A.2`, ...), one focused
  session each, specified in `specs/`.

## Status workflow

```text
backlog -> todo -> in_progress -> in_review -> done
                       |
                       v
                    blocked
```

See the status legend in `BOARD.md`. A story moves `backlog -> todo` only when its
spec is complete (see `_template.md` / operating model §4 "definition of ready").

## Progressive decomposition

Epics are decomposed **one phase at a time**, never all up front. The backlog
starts with the four Phase 2 epics as single rows. When you pull an epic in:

1. Set the epic to `in_progress` on `BOARD.md`.
2. Its first story is the decomposition story (e.g. `2A.1 Decompose Phase 2A`),
   run in Plan mode. It writes the remaining stories for that epic into `specs/`
   and adds their rows to `BOARD.md` at `backlog`.
3. You review the new stories and promote the ready ones to `todo`.
4. Implement-mode sessions execute stories one at a time, advancing status in
   place through `in_progress -> in_review -> done`.

Later epics stay at the roadmap level (single `backlog` rows) until entered, so
their stories are informed by earlier results instead of going stale.

## No-delete rule

Nothing is ever removed from the board. Completed stories remain as `done`, so the
full project history is preserved for the entire lifecycle.
