# Task Board

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

The single canonical board for all work on this project. Every epic and story
ever created lives here, for the entire project lifecycle. **Rows are never
deleted** — completed work stays on the board marked `done`.

See `docs/workflows/ai_human_collaboration.md` for the operating model and
`docs/tasks/README.md` for board conventions.

## Status legend

| Status | Meaning |
|---|---|
| `backlog` | Captured but not yet pulled into active work. Epics start here; an epic stays `backlog` until it is entered and decomposed. Newly created (undefined) stories also start here. |
| `todo` | Reviewed, fully specified, and prioritized — safe for a session to pick up cold. |
| `in_progress` | Currently being worked in a session. |
| `in_review` | Implementation done; awaiting diff review + tests. |
| `blocked` | Cannot proceed; blocker noted in the story's spec. |
| `done` | Reviewed, tests pass, results committed. Never removed from the board. |

## Hierarchy

- **Epic** = a phase (`2A`, `2B`, `2C`, `2D`). Defined in `docs/roadmaps/`.
- **Story** = an atomic task under an epic (`2A.1`, `2A.2`, ...). Each story has a
  spec in `docs/tasks/specs/`.

Epics are decomposed into stories **one phase at a time** (progressive
decomposition). The first story of any epic is always its decomposition story.

## Board

| ID | Type | Title | Status | Spec / Definition | Updated |
|---|---|---|---|---|---|
| 2A | Epic | Reliability evaluation infrastructure | `in_progress` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W1 / §5) | 2026-05-22 |
| 2A.1 | Story | Decompose Phase 2A | `done` | `docs/tasks/specs/2A.1_decompose_phase_2a.md` | 2026-05-22 |
| 2A.2 | Story | Model-agnostic predictor interface | `done` | `docs/tasks/specs/2A.2_predictor_interface.md` | 2026-05-22 |
| 2A.3 | Story | Core uncertainty-evaluation metrics | `done` | `docs/tasks/specs/2A.3_core_uncertainty_metrics.md` | 2026-05-22 |
| 2A.4 | Story | Abstention / selective-prediction curves | `done` | `docs/tasks/specs/2A.4_abstention_curves.md` | 2026-05-22 |
| 2A.5 | Story | Uncertainty-evaluation runner + artifacts | `backlog` | `docs/tasks/specs/2A.5_evaluation_runner_artifacts.md` | 2026-05-22 |
| 2B | Epic | Sensitivity & structure diagnostics | `backlog` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W2 / §5) | 2026-05-22 |
| 2C | Epic | Conditional neural surface model | `backlog` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W3 / §5) | 2026-05-22 |
| 2D | Epic | Uncertainty-aware inference & decision layer | `backlog` | `docs/roadmaps/phase2_reliability_aware_surface_inference.md` (W4+W5 / §5) | 2026-05-22 |

> When an epic is entered, set it to `in_progress`, add its decomposition story
> (e.g. `2A.1`), then add the resulting stories as new rows. Do not delete or
> renumber existing rows.
