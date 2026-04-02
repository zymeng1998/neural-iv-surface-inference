# Project Memory Update Policy

> Event-to-file update rules for the Project Memory Reviewer system.

---
created_at: 2026-04-02T00:00:00-04:00
last_updated_at: 2026-04-02T01:46:00-04:00
---

## Core principles

1. **Evidence-based only.** Every update must be traceable to observable evidence (git diff, log output, file content). Never infer outcomes from intentions.
2. **Minimal, surgical edits.** Touch only the files justified by the change event. Do not rewrite surrounding content for style consistency.
3. **Completed vs. intended.** Always distinguish work that is done from work that is planned or in progress.
4. **Preserve existing style.** Match the formatting conventions already established in each file.
5. **Timestamp everything.** Follow the timestamp policy defined in the registry.
6. **Never duplicate.** Before writing any entry or document, verify that a semantically equivalent one does not already exist. See duplicate-prevention rules below.

---

## Event-to-file update rules

### 1. Code progress (feature implemented, pipeline step completed, script added)

| Field | Value |
|---|---|
| What to update | `docs/logs/progress_log.md` |
| How | Append a new dated entry with Completed / Notes / Next Actions |
| Timestamp | Entry-level |
| What NOT to do | Do not log every tiny file touch. Only log meaningful work units. |
| Also consider | Update roadmap subtask status if a tracked milestone was completed |

### 2. Routine experiment run (training run, hyperparameter test, ablation)

| Field | Value |
|---|---|
| What to update | `docs/experiments/experiment_journal.md` |
| How | Append a new timestamped entry with experiment name, purpose, variables, result, interpretation, decision impact, next step |
| Timestamp | Entry-level |
| What NOT to do | Do not write experiment entries in the progress log. Do not create a retrospective for a routine experiment unless it revealed a significant failure. |

### 3. Failed run / mistake / unexpected outcome

| Field | Value |
|---|---|
| What to update | `docs/retrospectives/` — create a new numbered retrospective doc |
| How | Use `docs/retrospectives/_template.md`. Follow existing numbering: `NNNN_short_name.md` |
| Timestamp | Document-level with `event_at` |
| What NOT to do | Do not overload the progress log with failure analysis. Do not create a retrospective for minor expected failures (e.g., a hyperparameter that didn't improve metrics). |
| Also consider | Add a brief mention in `docs/logs/progress_log.md` linking to the retrospective |

### 4. Architecture / scope decision

| Field | Value |
|---|---|
| What to update | `docs/decisions/` — create a new numbered ADR doc |
| How | Use the existing ADR format: Status / Date / Context / Decision / Rationale / Consequences |
| Timestamp | Document-level with `created_at`, `event_at` |
| What NOT to do | Do not create ADRs for trivial implementation choices. Reserve for decisions that constrain future work. |

### 5. Roadmap change (priority shift, milestone added/removed, phase transition)

| Field | Value |
|---|---|
| What to update | `docs/roadmaps/phase1_structural_roadmap.md` (or the relevant phase roadmap) |
| How | Curated edit of the affected sections. Update `_Last updated_` line. |
| Timestamp | Document-level |
| What NOT to do | Do not rewrite the roadmap for minor tactical noise. Only update when priorities, sequencing, or milestones materially change. |
| Also consider | Update Mermaid charts if the dependency or timeline structure changed |

### 6. Milestone completion (phase gate reached, deliverable produced)

| Field | Value |
|---|---|
| What to update | `docs/logs/progress_log.md` AND `docs/roadmaps/` |
| How | Progress log: append entry marking the milestone. Roadmap: update subtask status to "Completed". |
| Timestamp | Entry-level in progress log, document-level in roadmap |
| Also consider | If the milestone closes a phase, consider a phase postmortem retrospective |

### 7. Infrastructure / workflow issue (environment problem, tooling change, CI issue)

| Field | Value |
|---|---|
| What to update | Depends on severity |
| Minor (resolved quickly) | Brief note in `docs/logs/progress_log.md` |
| Major (caused lost work, blocked progress) | New retrospective in `docs/retrospectives/` |
| Resulted in a durable decision | New ADR in `docs/decisions/` |
| What NOT to do | Do not auto-edit `docs/private/` or `docs/setup/` files unless the run is specifically about infrastructure and the evidence strongly supports it |

---

## Duplicate-prevention rules

Before writing to any project-memory file, the reviewer MUST check for existing equivalent content.

### For append-only logs

Target files: `docs/logs/progress_log.md`, `docs/experiments/experiment_journal.md`

Check signals:
- **Timestamp match**: Does an entry with the same date/timestamp already exist?
- **Commit hash match**: Does an existing entry reference the same commit(s)?
- **Semantic match**: Does an existing entry describe the same work unit, milestone, or experiment?

Rules:
- If an entry covers the same event → do not append a duplicate.
- If an existing entry is incomplete → update it in place.
- If uncertain whether overlap exists → report the ambiguity; do not write.

### For numbered documents

Target directories: `docs/retrospectives/`, `docs/decisions/`

Check signals:
- **Known IDs**: Check `reviewer_state.json` → `known_retrospective_ids` and `known_decision_ids`.
- **Event match**: Does an existing doc have the same `event_at`, related files, or root cause?
- **Title/topic match**: Does an existing doc cover the same underlying event or decision?

Rules:
- If a doc already covers the same event → do not create a new one.
- If the existing doc needs updating → edit it with a `last_updated_at` bump.
- If uncertain → report the ambiguity; do not create.

### For curated-edit files

Target files: roadmaps, project plans

Check signals:
- **Last updated match**: Was the file already updated in this review cycle?
- **Content match**: Does the file already reflect the change being recorded?

Rules:
- If the file already reflects the current state → do not edit.
- If the file needs a minor update → apply the minimum change needed.

### State tracking

After every run (including no-ops), update `docs/agent_bootstrap/reviewer_state.json` with:
- `last_review_run_at`
- `last_processed_git_head`
- `last_processed_diff_signature`
- Any newly created retrospective/decision IDs
- Updated entry counts for append-only logs

---

## Files the reviewer should NEVER auto-edit

| Path | Reason |
|---|---|
| `src/`, `tests/`, `scripts/` (except the review packet builder) | Source code — not project memory |
| `configs/` | Configuration — not project memory |
| `notebooks/` | Analysis artifacts — not project memory |
| `docs/private/` | Private/local-only notes |
| `docs/setup/private_runbook_template.md` | Template intended to remain generic |
| `.claude/`, `.gitignore` | Tool/system configuration |
| Root-level plan/notes files | Read-only reference unless fundamental direction changes |

---

## Timestamp policy summary

### Document-level timestamps (standalone docs)

Required metadata fields near the top of the document:

```
---
created_at: 2026-04-02T15:41:00-04:00
last_updated_at: 2026-04-02T15:41:00-04:00
event_at: 2026-04-02T14:00:00-04:00
---
```

- `created_at` = when the document was first created. Never overwritten.
- `last_updated_at` = when the document was last materially updated. Always updated on material change.
- `event_at` = when the underlying event actually happened. Omit or mark as `unknown` if not reliably inferrable.
- For time spans, use `event_start_at` and `event_end_at` instead of `event_at`.

Applies to: ADRs, retrospectives, major roadmap docs, phase plans, postmortems.

### Entry-level timestamps (append-only logs)

Each entry heading includes its own timestamp:

```
## 2026-04-02T15:41:00-04:00 — Short description of what happened
```

Or with dual timestamps when the entry is recorded after the event:

```
event_at: 2026-04-02T14:00:00-04:00
recorded_at: 2026-04-02T15:41:00-04:00
```

Applies to: progress log, experiment journal.

### Enforcement scope

- All newly created project-memory files must follow this policy.
- Existing files are updated to follow this policy only when they are next materially touched by a reviewer run.
- Do NOT mass-rewrite the repo to add timestamps retroactively.
