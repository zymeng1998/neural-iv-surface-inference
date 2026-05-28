# Story [ID]: [Short Descriptive Title]

---
id: [e.g. 2A.1]
epic: [e.g. 2A]
status: backlog   # backlog | todo | in_progress | in_review | blocked | done
created_at: YYYY-MM-DDTHH:MM:SS-TZ
last_updated_at: YYYY-MM-DDTHH:MM:SS-TZ
parallel_safe_with: []   # list of story IDs whose file_scope is disjoint from this one's
file_scope: []           # hard contract: paths this story may create or modify
---

> Keep the `status:` field above in sync with this story's row on
> `docs/tasks/BOARD.md`. Status changes happen in place; this file is never moved
> or deleted.
>
> `file_scope` is a **hard contract**: if implementation needs to touch a path
> not listed here, stop and either expand `file_scope` (and re-check
> `parallel_safe_with`) or split the new work into a new backlog story.
>
> `parallel_safe_with` lists story IDs that can safely run in another worktree
> at the same time as this one (no overlapping `file_scope`, no implicit
> dependency on the same artifact). Stories not listed here may still be
> parallel-safe — list them when you actually verify it.

## Purpose

[One paragraph: what this task achieves and why it matters now. Tie it to a
roadmap subtask or workstream.]

## Context files to read first

[Only the files the executing session needs. Keep this short — this list is the
session's entire reading budget.]

- `path/to/file`

## Files likely to touch

[Must be a subset of `file_scope` in the front matter. This list may be
narrower (e.g. "we will probably only touch foo.py") but never wider.]

- `path/to/file`

## Non-goals

[Explicit list of what this task must NOT do. Prevents scope creep and giant-task
collapse.]

- ...

## Compute requirements

[Required so the operator can plan remote rentals. Fill every field, even if
the answer is "none" — leaving it blank is not allowed. If the story spans
multiple compute phases (e.g. local test + Pod train), list each phase.]

- **Locale:** local Mac | remote (RunPod) | mixed
- **Hardware class:** CPU-only | single GPU (e.g. RTX 4090 / A40 / A100-40GB / A100-80GB / H100) | multi-GPU
- **Approx. VRAM:** e.g. ~6 GB, ~24 GB, n/a
- **Approx. wall time:** e.g. ~5 min local, ~30 min Pod, ~3 h Pod
- **Disk footprint (new artifacts):** e.g. ~50 MB committed + ~1 GB checkpoints (gitignored)
- **Notes:** anything operator-facing about budgeting (sequential vs parallel runs, idle cost while waiting, recommended Pod tier, etc.)

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

## Last checkpoint

> Updated mid-story by the executing session at logical break points (after
> each significant step, before any long-running command, immediately before
> ending the session). One short block per checkpoint, newest at the top.
> Replaces the need for a transcript replay on resume — a fresh session reads
> only the latest entry to know the exact next concrete action.

### YYYY-MM-DDTHH:MM:SS-TZ — [short label, e.g. "after step 3 / synthetic smoke green"]

- Last completed: [implementation step number + one-sentence summary]
- Last touched: [exact file path(s) — the diff under review, if any]
- Next concrete action: [the single next command or edit to execute]
- Open blocker: [none | one-sentence description]

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
