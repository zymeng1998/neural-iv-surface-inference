# Change Event Taxonomy

> Classification system for change events detected during reviewer runs.

---
created_at: 2026-04-02T00:00:00-04:00
last_updated_at: 2026-04-02T00:00:00-04:00
---

## 1. Code Progress

A feature, pipeline step, script, module, or meaningful code unit was implemented, completed, or substantially improved.

| Field | Value |
|---|---|
| Detection cues | New or modified files in `src/`, `scripts/`, `tests/`; new configs; commit messages mentioning "add", "implement", "complete", "build" |
| What to record | What was completed, key design choices made during implementation, any notable observations |
| Where to record | `docs/logs/progress_log.md` (append entry). Update roadmap subtask status if applicable. |
| What NOT to over-claim | Do not claim a feature "works" unless tests pass or evidence of successful execution exists. Do not claim completion of a milestone unless all its deliverables are present. |
| Timestamp type | **Entry-level** in progress log |

## 2. Routine Experiment Run

A training run, hyperparameter test, ablation study, or evaluation was executed and produced results.

| Field | Value |
|---|---|
| Detection cues | New files in `logs/`, `plots/`, `reports/`; modified training configs; commit messages mentioning "train", "run", "experiment", "eval", "test" (in ML context) |
| What to record | Experiment name, purpose, changed variables, result summary, interpretation, decision impact, next step |
| Where to record | `docs/experiments/experiment_journal.md` (append entry) |
| What NOT to over-claim | Do not claim a model "outperforms" without clear metric evidence. Do not interpret noise as signal. A negative result is a valid result, not a failure. |
| Timestamp type | **Entry-level** in experiment journal |

## 3. Failed Run / Mistake / Unexpected Outcome

Something went wrong in a way that caused a real problem: lost work, wasted time, broken pipeline, wrong assumption that led to bad results, OOM, environment failure.

| Field | Value |
|---|---|
| Detection cues | Error messages in logs, reverted commits, commit messages mentioning "fix", "revert", "broken", "OOM", "fail"; evidence of re-doing work; significant debugging sessions |
| What to record | What happened, why it was a mistake/why it mattered, root cause, what was learned, improvement plan, updated implementation plan, current decision, immediate next action |
| Where to record | `docs/retrospectives/` — new numbered doc using `_template.md` |
| What NOT to over-claim | A hyperparameter that didn't improve metrics is NOT a mistake — it's a normal experiment result. Only create retrospectives for genuine failures that caused real problems or revealed important lessons. |
| Timestamp type | **Document-level** with `event_at` |

## 4. Architecture / Scope Decision

A high-signal decision was made that constrains or directs future work: scope freeze, technology choice, data source selection, methodology commitment.

| Field | Value |
|---|---|
| Detection cues | Commit messages mentioning "decide", "freeze", "choose", "switch to"; new ADR files; changes to roadmap scope; removal or addition of major dependencies |
| What to record | Status, context, decision, rationale, consequences (positive and trade-offs) |
| Where to record | `docs/decisions/` — new numbered ADR doc |
| What NOT to over-claim | Do not create ADRs for trivial implementation choices (e.g., variable naming, minor refactors). Reserve for decisions that would be costly to reverse. |
| Timestamp type | **Document-level** with `created_at` and `event_at` |

## 5. Roadmap Change

Priorities shifted, milestones were added or removed, phase boundaries changed, or the project timeline was materially altered.

| Field | Value |
|---|---|
| Detection cues | Modified roadmap files; commit messages mentioning "reprioritize", "defer", "accelerate", "phase"; changes to the subtask matrix or decision rules |
| What to record | What changed in the roadmap and why |
| Where to record | `docs/roadmaps/` — curated edit of the relevant roadmap doc |
| What NOT to over-claim | A single experiment result does not justify a roadmap change unless it materially affects priorities. Do not rewrite the roadmap for minor tactical noise. |
| Timestamp type | **Document-level** — update `_Last updated_` |

## 6. Milestone Completion

A defined milestone or phase gate was reached: all deliverables present, success criteria met.

| Field | Value |
|---|---|
| Detection cues | Multiple related subtasks marked complete in roadmap; deliverable files present (plots, tables, memos); commit messages mentioning "complete", "finish", "phase done" |
| What to record | Which milestone was completed, key deliverables produced, summary of outcomes |
| Where to record | `docs/logs/progress_log.md` (append entry) AND `docs/roadmaps/` (update subtask status) |
| What NOT to over-claim | Do not declare a milestone complete unless ALL its defined deliverables exist. Partial completion should be recorded as progress, not as milestone completion. |
| Timestamp type | **Entry-level** in progress log, **document-level** in roadmap |

## 7. Infrastructure / Workflow Issue

An environment, tooling, CI, or workflow problem occurred: pod restart, package conflict, SSH failure, disk space, permission issue.

| Field | Value |
|---|---|
| Detection cues | Commit messages mentioning "infra", "env", "setup", "config", "fix environment"; changes to setup docs; changes to `.gitignore` or CI files |
| What to record | What the issue was, how it was resolved, any lasting impact |
| Where to record | Minor (resolved quickly): brief note in `docs/logs/progress_log.md`. Major (caused lost work): new retrospective. Resulted in durable decision: new ADR. |
| What NOT to over-claim | A routine pod restart is not an infrastructure issue. Only record issues that caused friction or required non-trivial resolution. |
| Timestamp type | **Entry-level** for progress log notes, **document-level** for retrospectives/ADRs |
