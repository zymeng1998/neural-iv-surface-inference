# Reusable Session Prompts

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

Copy-pasteable prompts for starting a fresh Claude Code session under the
operating model in `docs/workflows/ai_human_collaboration.md`. These are durable
on purpose: any session can be lost, but these prompts let the next one start
cleanly with no rediscovery.

To reuse, copy a template and replace the `<PLACEHOLDERS>`.

---

## 1. Epic decomposition (the first story of every epic)

Run this in **Plan mode** when entering a new epic. It produces that epic's story
specs; it does not implement anything.

Placeholders:
- `<EPIC>` — epic ID, e.g. `2A`, `2B`, `2C`, `2D`
- `<WORKSTREAM>` — matching roadmap workstream: `2A`→`W1`, `2B`→`W2`, `2C`→`W3`, `2D`→`W4+W5`

```
Plan mode. Enter epic <EPIC> for decomposition.

Read first (only these):
- docs/workflows/ai_human_collaboration.md  (§3 modes, §4 board lifecycle + phase-entry decomposition)
- docs/workflows/session_protocol.md
- docs/tasks/README.md
- docs/tasks/_template.md
- docs/roadmaps/phase2_reliability_aware_surface_inference.md  (workstream <WORKSTREAM> / Phase <EPIC>)

Then orient: run git status --short and git log --oneline -5.

Task — decompose epic <EPIC> into atomic story specs:
1. Set epic <EPIC> to in_progress on docs/tasks/BOARD.md.
2. Break <EPIC> into stories small enough for one focused session each
   (<EPIC>.1, <EPIC>.2, ...). Story <EPIC>.1 is this decomposition story itself
   — mark it done when finished.
3. Write one spec per story into docs/tasks/specs/ using _template.md
   (fill purpose, context files, files likely to touch, non-goals, steps, tests,
   artifacts, acceptance criteria, rollback).
4. Add a row for each new story to docs/tasks/BOARD.md at status backlog.
5. Do NOT decompose any other epic. Do NOT implement any story.

Constraints: no code changes, no training runs, no large downloads.
End: run python3 scripts/pmr_prepush_gate.py --verbose --dry-run, give the
end-of-session handoff summary, and stop for my review. Do not commit.
```

---

## 2. Implement a story

Run this in **Implement mode** to execute one already-`todo` story. The prompt is
short because all detail lives in the story spec.

Placeholders:
- `<SPEC>` — path to the story spec, e.g. `docs/tasks/specs/2A.2_uncertainty_metrics.md`

```
Implement mode. Execute <SPEC>.

Follow docs/workflows/session_protocol.md:
1. Run git status --short and git log --oneline -5.
2. Read ONLY the context files the spec lists. Do not load the whole repo.
3. Restate the story, its acceptance criteria, and its non-goals.
4. Set the story to in_progress on docs/tasks/BOARD.md and in the spec.
5. Implement only this story, staying inside its declared file scope.
6. Run the spec's tests plus python3 scripts/smoke_test.py.
7. Run the validation gate (operating model §5) and python3 scripts/pmr_prepush_gate.py --verbose --dry-run.
8. Set the story to in_review, give the end-of-session handoff summary, and stop
   for my review. Do not commit unless I approve.
```

---

## 3. Plan / Explore (no task yet)

Run this for an open-ended planning or inspection session that is not tied to a
single story.

```
<Explore mode | Plan mode>. <one-line goal>.

Orient: run git status --short and git log --oneline -5.
Read docs/tasks/BOARD.md to see current state.
Read only the files needed for this goal — do not load the whole repo.
Do not edit code. <In Plan mode: propose specs/changes and wait for my approval.>
End with a short handoff summary.
```
