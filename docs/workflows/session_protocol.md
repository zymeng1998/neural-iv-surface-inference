# Session Protocol

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

The concrete checklist an AI agent (e.g. Claude Code) follows at the start and
end of every working session. The goal is clean resumption after context loss,
session resets, or rate-limit interruptions.

See `docs/workflows/ai_human_collaboration.md` for the broader operating model
(modes, task lifecycle, validation gate, retrospective loop).

---

## Start of session

1. **Orient.** Run `git status --short` and `git log --oneline -5`.
2. **Find the active story.** Open `docs/tasks/BOARD.md`. Work the story the human
   named, or the one already `in_progress`, or the top `todo`. Then read its spec
   in `docs/tasks/specs/`. If there is no actionable story, the session is Explore
   or Plan mode (e.g. an epic decomposition) — confirm with the human.
3. **Read only what the spec lists.** Open only the "context files to read first"
   named in the story spec. Do not load the whole repo.
4. **Restate.** In 1–3 sentences, restate the story, its acceptance criteria, and
   its non-goals. Surface any mismatch between the spec and current repo state.
5. **Confirm mode.** State which mode you are in (Explore / Plan / Implement /
   Review / Retrospective).
6. **Propose a plan.** Give a short step plan. In Plan mode, **wait for approval**
   before any edit. In Implement mode, proceed only on the single approved task.

---

## During the session

- Stay inside the task's declared file scope. New work discovered mid-task is
  captured as a new `backlog` story on `docs/tasks/BOARD.md`, never absorbed into
  the current one.
- Redirect verbose command output to files; bring only summaries into context.
- Do not run expensive training runs or large downloads unless the task spec
  explicitly authorizes it.

---

## End of session

1. **Summarize files changed** (with paths).
2. **Summarize tests run** and their results (numbers / exit codes, not raw logs).
3. **State risks** and anything left unverified.
4. **Run the validation gate** (see operating-model doc §5) if a commit is
   intended.
5. **Update durable state if allowed** (progress log, task spec, roadmap status)
   per the retrospective loop.
6. **Propose exactly one next task**, and write the exact prompt the next session
   should start from, using the handoff template below.

---

## Handoff template

Copy this into the end-of-session message (and into the task spec when archiving).

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

The "exact prompt for next session" is the most important field. It should be
copy-pasteable and self-contained — naming the mode, the single task, and the
files to read — so the next session needs no rediscovery.
