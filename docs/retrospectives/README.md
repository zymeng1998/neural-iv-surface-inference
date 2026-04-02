# Retrospectives

> This directory contains numbered retrospective documents for major mistakes, failures, course corrections, and lessons learned.

---
created_at: 2026-04-02T00:00:00-04:00
last_updated_at: 2026-04-02T00:00:00-04:00
---

## When to create a new retrospective

Create a new retrospective when:

- A pipeline, script, or training run failed in a way that caused real problems (lost work, wasted significant time, broken assumptions)
- A design or architectural assumption proved wrong and required rework
- An unexpected outcome revealed an important lesson about the project, data, or workflow
- A phase is completed and warrants a postmortem

Do NOT create a retrospective for:

- A hyperparameter that didn't improve metrics (that's a normal experiment — log it in the experiment journal)
- A minor bug that was quickly fixed (note it in the progress log if worth recording)
- Speculative concerns that haven't materialized

## How retrospectives differ from other project-memory files

### Retrospectives vs. progress log (`docs/logs/progress_log.md`)

The progress log records **what was done** — completed work, milestones, observations.
Retrospectives analyze **what went wrong** — the failure, root cause, lessons, and corrective plan.

A progress log entry might mention that a failure occurred and link to the retrospective. The retrospective contains the full analysis.

### Retrospectives vs. ADRs (`docs/decisions/`)

ADRs record **stable decisions** with their rationale and consequences. They are forward-looking: "We decided X because Y."
Retrospectives are backward-looking: "X went wrong because Y, and we learned Z."

A retrospective may result in a new ADR if the lesson leads to a durable decision.

### Retrospectives vs. experiment journal (`docs/experiments/experiment_journal.md`)

The experiment journal records **routine experiment runs** — what was tested, what the results were, what they mean.
Retrospectives are for **genuine failures**, not for experiments that produced negative results.

An experiment with poor metrics is a normal outcome logged in the journal.
A training run that OOMed and crashed the pod is a failure that gets a retrospective.

## Naming convention

```
NNNN_short_descriptive_name.md
```

Increment `NNNN` from the highest existing number. Use underscores, lowercase, and keep the name short but descriptive.

Examples:
- `0001_spy_step3_oom_and_pipeline_fix.md`
- `0002_wrong_spot_price_convention.md`
- `0003_phase1_postmortem.md`

## Required timestamp policy

Each retrospective must include a metadata section near the top with:

```
---
created_at: YYYY-MM-DDTHH:MM:SS-TZ
last_updated_at: YYYY-MM-DDTHH:MM:SS-TZ
event_at: YYYY-MM-DDTHH:MM:SS-TZ
---
```

- `created_at`: when the retrospective document was first written
- `last_updated_at`: when the document was last materially updated
- `event_at`: when the underlying failure or event actually happened
- If event time cannot be reliably inferred, omit `event_at` or write `event_at: unknown`
- For events spanning time, use `event_start_at` and `event_end_at`

## Template

See `_template.md` in this directory for the standard retrospective structure.
