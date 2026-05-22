# Story [ID]: [Short Descriptive Title]

---
id: [e.g. 2A.1]
epic: [e.g. 2A]
status: backlog   # backlog | todo | in_progress | in_review | blocked | done
created_at: YYYY-MM-DDTHH:MM:SS-TZ
last_updated_at: YYYY-MM-DDTHH:MM:SS-TZ
---

> Keep the `status:` field above in sync with this story's row on
> `docs/tasks/BOARD.md`. Status changes happen in place; this file is never moved
> or deleted.

## Purpose

[One paragraph: what this task achieves and why it matters now. Tie it to a
roadmap subtask or workstream.]

## Context files to read first

[Only the files the executing session needs. Keep this short — this list is the
session's entire reading budget.]

- `path/to/file`

## Files likely to touch

- `path/to/file`

## Non-goals

[Explicit list of what this task must NOT do. Prevents scope creep and giant-task
collapse.]

- ...

## Implementation steps

1. ...
2. ...

## Tests to run / add

- [ ] `pytest ...`
- [ ] `python3 scripts/smoke_test.py`

## Expected artifacts

[Files, results CSVs, figures, checkpoints this task should produce. Remember to
commit result artifacts alongside any checkpoints.]

- ...

## Acceptance criteria

[Observable, verifiable conditions for "done". Avoid vague criteria like "it
works". Prefer "tests in X pass", "results.csv has columns [...]", etc.]

- [ ] ...

## Rollback notes

[How to safely undo this change if it proves wrong — e.g. files to revert, no
data mutated, checkpoint preserved.]

---

## Handoff summary (fill on completion)

```text
## Session Summary
- Goal:
- Mode:
- Files inspected:
- Files changed:
- Tests run:
- Results:
- Open questions:
- Risks:
- Recommended next task:
- Exact prompt for next Claude Code session:
```
