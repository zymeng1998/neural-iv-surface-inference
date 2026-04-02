# Repo Structure Summary

**Root:** `Neural IV Surface inference/`

## Top-level folders

| Folder / File | Purpose |
|---|---|
| `configs/` | YAML configuration files for model baselines and data pipelines |
| `data_processed/` | Cleaned/transformed data (currently SPY options data) |
| `data_raw/` | Raw source data before processing |
| `docs/` | Project documentation: decisions, retrospectives, roadmaps, setup guides, progress logs |
| `logs/` | Runtime/training logs (currently empty) |
| `notebooks/` | Jupyter notebooks for exploratory analysis |
| `plots/` | Generated plot outputs (currently empty) |
| `reports/` | Generated reports (currently empty) |
| `scripts/` | Runnable entry-point scripts: data prep, training, smoke tests |
| `src/` | Main Python package `neural_iv_surface_inference` with features, models, training, and utils submodules |
| `tests/` | Test suite (data pipeline tests, smoke tests) |
| `.claude/` | Claude Code local settings |

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Project readme |
| `.gitignore` | Git ignore rules |
| `ml_finance_iv_surface_project_plan_notes.md` | Project plan and notes |
| `phase1_actions_remote_workstation_plan.md` | Phase 1 action plan for remote workstation setup |

## Candidate project-memory files

Files/folders whose names suggest roadmap, progress, retrospective, decisions, plans, or notes:

- `docs/decisions/0001_remote_dev_stack.md` — ADR: remote dev stack choice
- `docs/decisions/0002_phase1_scope_freeze.md` — ADR: phase 1 scope freeze
- `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` — Retrospective: OOM fix in Step 3
- `docs/roadmaps/phase1_structural_roadmap.md` — Phase 1 structural roadmap
- `docs/roadmaps/flowchart_deps.mmd` — Dependency flowchart (Mermaid)
- `docs/roadmaps/gantt_chart.mmd` — Gantt chart (Mermaid)
- `docs/logs/progress_log.md` — Progress log
- `ml_finance_iv_surface_project_plan_notes.md` — Root-level project plan and notes
- `phase1_actions_remote_workstation_plan.md` — Root-level phase 1 action plan

---
*Generated: 2026-04-02*
