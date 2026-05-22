# Human-AI Collaboration Operating Model

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

This document defines how this repository is developed collaboratively between a
human and AI coding agents (e.g. Claude Code) across many sessions. Its purpose
is to make long-running work survive context loss, session resets, and
rate-limit interruptions, by keeping durable state in the repository rather than
in any single conversation.

It complements, and does not replace, the Project Memory Reviewer (PMR) system in
`docs/agent_bootstrap/`. PMR governs *how documentation is updated*; this document
governs *how work is planned, executed, and handed off*.

> Companion doc: `docs/workflows/session_protocol.md` defines the concrete
> start-of-session and end-of-session checklists and the handoff template.

---

## 1. Core principles

1. **Context is a scarce resource.** Do not assume any session can load the whole
   repo history. Every session starts from a small, high-signal entry point.
2. **Durable state lives in the repo, not the chat.** Plans, decisions, task
   specs, results, and handoffs are files. Conversations are disposable.
3. **Small reviewable units.** Work is decomposed into atomic task specs, each
   sized for one focused session.
4. **No giant task collapse.** A task that cannot fit in one session must be
   split before it is started, not abandoned midway.
5. **Separate stable knowledge from active work.** Stable knowledge lives in
   durable docs (roadmaps, ADRs, data lineage); active work lives in task specs.
6. **Safe edits.** AI edits follow explicit modes and respect non-goals,
   validation gates, and secret-safety rules.
7. **Progressive decomposition.** Backlog is written one phase at a time, not all
   phases up front. Decomposing a phase into task specs is itself the first task
   of that phase. Later phases stay at the roadmap level until entered, so their
   specs are informed by the results of earlier phases instead of going stale.

---

## 2. Memory map (where things live)

| Concern | Location | Status |
|---|---|---|
| Persistent AI instructions | `CLAUDE.md` (repo root) | exists |
| Project overview / direction | `README.md` | exists |
| Phase roadmaps | `docs/roadmaps/` | exists |
| Task board (all epics + stories, every status) | `docs/tasks/BOARD.md` | this upgrade |
| Story specs | `docs/tasks/specs/` | this upgrade |
| Design decisions | `docs/decisions/` (ADRs) | exists |
| Chronological progress | `docs/logs/progress_log.md` | exists |
| Experiment results | `docs/experiments/experiment_journal.md` | exists |
| Failures / course corrections | `docs/retrospectives/` | exists |
| Data lineage | `docs/data/data_lineage.md` | exists |
| Documentation update rules | `docs/agent_bootstrap/` (PMR) | exists |
| Session protocol + handoff | `docs/workflows/session_protocol.md` | this upgrade |

`CLAUDE.md` should remain a concise **table of contents** that points at the
active roadmap and the active task index. Do not dump roadmaps or task specs into
it.

---

## 3. Session modes

Every session operates in exactly one mode. The human states the mode (or the
agent infers and confirms it) at the start.

| Mode | Purpose | May edit files? |
|---|---|---|
| **Explore** | Inspect repo, summarize, answer questions | No |
| **Plan** | Propose tasks / decompose work | No, unless explicitly allowed |
| **Implement** | Execute exactly one approved task spec | Yes — only files in the task's scope |
| **Review** | Inspect a diff, run tests, identify risks | No (review only) |
| **Retrospective** | Update progress/docs after work completes | Docs only |

Implement mode executes **one** task spec at a time. If new work is discovered
mid-task, it is captured as a new backlog spec, not absorbed into the current one.

---

## 4. Task board and lifecycle

Work is tracked on a single Jira-style board, `docs/tasks/BOARD.md`. Work is
organized in two levels:

- **Epic** = a phase (`2A`, `2B`, `2C`, `2D`), defined in `docs/roadmaps/`.
- **Story** = an atomic task under an epic (`2A.1`, `2A.2`, ...), one focused
  session each, specified in `docs/tasks/specs/`.

Stories move through these statuses, **changed in place** (edit the `status:`
field in the spec and the matching row on `BOARD.md` — files are never moved):

```text
backlog -> todo -> in_progress -> in_review -> done
                       |
                       v
                    blocked
```

| Status | Meaning |
|---|---|
| `backlog` | Captured but not pulled into active work. Epics start here; undefined new stories start here. |
| `todo` | Fully specified, reviewed, prioritized — safe for a session to pick up cold. |
| `in_progress` | Currently being worked in a session. |
| `in_review` | Implementation done; awaiting diff review + tests. |
| `blocked` | Cannot proceed; blocker recorded in the story's spec. |
| `done` | Reviewed, tests pass, results committed. |

**No-delete rule.** Rows are never removed from the board. Completed stories stay
as `done`, preserving the full project history for the entire lifecycle.

**Definition of ready (`backlog -> todo`).** A story may move to `todo` only when
its spec contains all of: title, status, purpose, context files to read first,
files likely to touch, non-goals, implementation steps, tests to run, expected
artifacts, acceptance criteria, and rollback notes. Use `docs/tasks/_template.md`.

### Phase-entry decomposition (the first story of every epic)

The backlog starts with the four Phase 2 epics (2A–2D) as single `backlog` rows.
Epics are decomposed into stories **one phase at a time**. When an epic is
entered:

1. Set the epic to `in_progress` on `BOARD.md`.
2. Its **first story is the decomposition story** (e.g. `2A.1 Decompose Phase
   2A`), run in Plan mode. It reads the phase definition in `docs/roadmaps/`, the
   task template, and this operating model, then writes the remaining stories for
   that epic into `docs/tasks/specs/` and adds their rows to `BOARD.md` at
   `backlog`.
3. The human reviews the new stories and promotes the ready ones to `todo`.
4. Implement-mode sessions execute stories one at a time, advancing status in
   place through `in_progress -> in_review -> done`.

Do **not** decompose future epics in advance. A later epic stays a single
`backlog` row until it is entered, so its stories can be informed by the results
of earlier epics rather than written against assumptions those results may
invalidate.

---

## 5. Validation gate

Before any commit that touches evidence-source files, the following must hold.
This extends the PMR pre-push documentation gate (`scripts/pmr_prepush_gate.py`),
which is necessary but not sufficient.

- [ ] Formatting / lint pass if configured for the touched files
- [ ] Unit tests pass (`pytest`) and the smoke test passes (`scripts/smoke_test.py`)
- [ ] No large data downloads triggered as a side effect
- [ ] No expensive training runs triggered as a side effect
- [ ] No secrets, tokens, credentials, private paths, or RunPod details in the diff
      (`scripts/security_scan_changed_files.py`)
- [ ] No unexpected artifact churn (only intended artifacts changed)
- [ ] Result artifacts (CSV / tables / figures) committed alongside any checkpoints
      they describe — closes the Phase 1 evidence gap noted in
      `docs/phase1_result_memo.md`
- [ ] All applicable PMR docs updated per `CLAUDE.md` pre-push routine; PMR gate
      passes (`python3 scripts/pmr_prepush_gate.py --verbose --dry-run`)

---

## 6. Retrospective loop

When a task reaches `done`, update durable state so the next session does not
rediscover anything:

1. **Progress log** — append a dated entry to `docs/logs/progress_log.md`
   (what was completed, notes, next actions).
2. **Story spec + board** — record the final handoff summary in the spec, set its
   `status: done`, and update the matching row on `docs/tasks/BOARD.md` to `done`
   (the row is kept, never deleted).
3. **Roadmap** — update the relevant subtask status in the phase roadmap.
4. **Experiment journal** — if the task ran an experiment, append an entry to
   `docs/experiments/experiment_journal.md`.
5. **ADR** — if a design decision was made, create a numbered ADR in
   `docs/decisions/`.
6. **Retrospective** — only on a significant failure or course correction, create
   a numbered doc in `docs/retrospectives/`.

---

## 7. Relationship to existing systems

- **PMR (`docs/agent_bootstrap/`)** is the authority on which docs must be updated
  for a given change event, and enforces it at push time. This operating model
  adds the *task* and *session* layer on top.
- **ADRs, progress log, experiment journal, retrospectives** are unchanged; this
  model defines *when* the task lifecycle triggers writes to them.
- Where guidance overlaps, the PMR update policy
  (`docs/agent_bootstrap/project_memory_update_policy.md`) takes precedence on
  documentation-update questions.
