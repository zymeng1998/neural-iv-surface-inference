# Project Memory Registry

> Canonical index of all project-memory files, their roles, update rules, and timestamp policies.

---
created_at: 2026-04-02T00:00:00-04:00
last_updated_at: 2026-04-02T01:46:00-04:00
---

## A. System-of-Record Memory Files

These are the primary project-memory surfaces. Future reviewer runs update these.

### 1. `docs/logs/progress_log.md`

| Field | Value |
|---|---|
| Role | Chronological record of major progress milestones and meaningful work units |
| Update frequency | After each significant work session or milestone |
| Update mode | Append new dated entry |
| Timestamp policy | **Entry-level** — each entry heading includes its own date |
| Existing format | `## YYYY-MM-DD` with Completed / Notes / Open Items / Next Actions subsections |
| Likely evidence inputs | git log, git diff, completed tasks, resolved issues |
| What should never be written here | Tiny file touches, speculative plans, detailed failure analysis (use retrospectives), experiment details (use experiment journal) |

### 2. `docs/decisions/` (numbered ADR docs)

| Field | Value |
|---|---|
| Role | Stable, high-signal architectural and scope decisions with rationale |
| Update frequency | When a major decision is made or revised |
| Update mode | New numbered doc (`NNNN_short_name.md`) for each major decision |
| Timestamp policy | **Document-level** — `created_at`, `last_updated_at`, `event_at` in metadata block |
| Existing format | Status / Date / Context / Decision / Rationale / Consequences |
| Likely evidence inputs | Discussion outcomes, scope changes, architecture choices, trade-off analysis |
| What should never be written here | Implementation details, experiment results, progress updates, routine tactical choices |

### 3. `docs/retrospectives/` (numbered retrospective docs)

| Field | Value |
|---|---|
| Role | Primary sink for major mistakes, failures, course corrections, lessons learned, and phase postmortems |
| Update frequency | After a significant failure, mistake, course correction, or phase completion |
| Update mode | New numbered doc (`NNNN_short_name.md`) for each major retrospective |
| Timestamp policy | **Document-level** — `created_at`, `last_updated_at`, `event_at` (or `event_start_at`/`event_end_at` for spans) |
| Existing format | Numbered sections: What happened / Why mistake / Root cause / What we learned / Improvement plan / Updated implementation plan / Current decision / Immediate next action / One-sentence summary |
| Likely evidence inputs | Failed runs, OOM errors, bad assumptions, pipeline breakages, wrong architectural bets |
| What should never be written here | Routine progress, successful experiment results, minor tactical adjustments |

### 4. `docs/roadmaps/phase1_structural_roadmap.md`

| Field | Value |
|---|---|
| Role | Phase-level structural roadmap with subtask matrix and decision rules |
| Update frequency | When priorities, sequencing, or milestones materially change |
| Update mode | Curated edit of existing content |
| Timestamp policy | **Document-level** — `_Last updated: ..._` line near top (existing convention); should migrate to ISO 8601 with timezone when next touched |
| Likely evidence inputs | Completed milestones, new phase entries, priority shifts, blocked tasks |
| What should never be written here | Minor tactical noise, individual experiment results, daily progress |

### 5. `docs/roadmaps/` (Mermaid charts)

| Field | Value |
|---|---|
| Files | `flowchart_deps.mmd`, `gantt_chart.mmd`, and their `.png` renders |
| Role | Visual dependency and timeline representations |
| Update frequency | When the roadmap structure materially changes |
| Update mode | Curated edit of `.mmd` source, re-render `.png` |
| Timestamp policy | Inherits from parent roadmap document |
| What should never be written here | Prose, experiment results, progress log entries |

### 6. `docs/experiments/experiment_journal.md`

| Field | Value |
|---|---|
| Role | Append-only log of routine experiment runs, observations, and interpretations |
| Update frequency | After each meaningful experiment run |
| Update mode | Append new timestamped entry |
| Timestamp policy | **Entry-level** — each entry heading includes ISO 8601 timestamp |
| Likely evidence inputs | Training logs, metric outputs, config diffs, plots |
| What should never be written here | Major failure analysis (use retrospectives), scope decisions (use ADRs), infrastructure issues |

### 7. `ml_finance_iv_surface_project_plan_notes.md` (repo root)

| Field | Value |
|---|---|
| Role | Strategic project plan, problem framing, phase map, and execution priorities |
| Update frequency | Rarely — only when fundamental project direction changes |
| Update mode | Read-only reference in normal operation; curated edit only with strong justification |
| Timestamp policy | **Document-level** when touched |
| Likely evidence inputs | Major scope pivots, fundamental reframing |
| What should never be written here | Daily progress, experiment results, tactical changes |

### 8. `phase1_actions_remote_workstation_plan.md` (repo root)

| Field | Value |
|---|---|
| Role | Detailed Phase 1 execution plan and action checklist |
| Update frequency | Rarely — operational reference |
| Update mode | Read-only reference in normal operation |
| Timestamp policy | **Document-level** when touched |
| What should never be written here | Results, retrospectives, new decisions |

## B. Reference/Context Files (read-only for reviewer)

These provide context but are not updated by the project memory reviewer unless the run is specifically about their topic.

| Path | Role | Notes |
|---|---|---|
| `README.md` | Project readme | Update only when project structure materially changes |
| `docs/data_assumptions_and_cleaning.md` | Data pipeline rules | Update only when cleaning rules change |
| `docs/data/data_lineage.md` | Data lineage (raw → processed → modeling-ready) | Update when pipeline structure changes |
| `docs/setup/remote_dev.md` | Remote dev setup guide | Infra reference |
| `docs/setup/ai_agent_runpod_manual.md` | RunPod agent manual | Infra reference |
| `docs/setup/private_runbook_template.md` | Private runbook template | Template — do not fill in |

## C. Evidence Sources (inputs to reviewer, never directly edited by reviewer)

| Path | Role |
|---|---|
| `src/neural_iv_surface_inference/` | Main Python package |
| `scripts/` | Entry-point scripts |
| `tests/` | Test suite |
| `notebooks/` | Exploratory notebooks |
| `configs/` | YAML configurations |
| `data_raw/`, `data_processed/` | Data directories |
| `plots/`, `reports/` | Output artifacts |
| `logs/` | Runtime/training logs |

## D. Protected / Low-Touch Files

| Path | Reason |
|---|---|
| `docs/private/infra_local.md` | Local-only infrastructure notes |
| `docs/private/.gitkeep` | Placeholder |
| `.claude/settings.local.json` | Claude Code local config |
| `.gitignore` | Git config |

## E. Bootstrap System Files (this system)

| Path | Role |
|---|---|
| `docs/agent_bootstrap/project_memory_registry.md` | This file |
| `docs/agent_bootstrap/project_memory_update_policy.md` | Event-to-file update rules |
| `docs/agent_bootstrap/project_memory_runbook.md` | Reviewer operating sequence |
| `docs/agent_bootstrap/change_event_taxonomy.md` | Change event classification |
| `docs/agent_bootstrap/repo_tree.txt` | Full repo tree |
| `docs/agent_bootstrap/repo_tree_summary.md` | Structural summary |
| `docs/agent_bootstrap/reviewer_state.json` | Machine-readable reviewer state (idempotence tracking) |
| `docs/agent_bootstrap/reviewer_state.md` | Human-readable reviewer state summary |
| `docs/retrospectives/README.md` | Retrospective guidance |
| `docs/retrospectives/_template.md` | Retrospective template |
| `scripts/build_project_memory_review_packet.py` | Review packet builder (with fingerprinting) |
| `scripts/pmr_prepush_gate.py` | Pre-push gate script |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
