# Project Instructions — Neural IV Surface Inference

## Pre-Push Routine (MANDATORY)

Before any commit that touches evidence-source files (`src/`, `scripts/`, `tests/`, `configs/`, `notebooks/`), you MUST run the full PMR documentation update — not just enough to pass the gate.

### Step-by-step

1. **Identify all changed evidence files.** Run:
   ```
   git diff --name-only HEAD  (unstaged)
   git diff --cached --name-only  (staged)
   ```
   Plus any new untracked files you created.

2. **For each changed evidence file, evaluate which PMR docs need updating.** Use this checklist:

   | Change type | Docs to update |
   |---|---|
   | New module / script / feature implemented | `docs/logs/progress_log.md` (append entry), `docs/roadmaps/phase1_structural_roadmap.md` (update subtask status) |
   | Data pipeline changed | `docs/data/data_lineage.md` (update pipeline flow, file paths, open questions) |
   | Cleaning rules or thresholds changed | `docs/data_assumptions_and_cleaning.md` |
   | Major failure or course correction | `docs/retrospectives/` (new numbered doc) |
   | Scope or architecture decision | `docs/decisions/` (new ADR) |
   | Experiment run completed | `docs/experiments/experiment_journal.md` (append entry) |
   | Config structure changed | `docs/data/data_lineage.md` (governing references section) |

3. **Update ALL applicable docs**, not just the minimum to pass the gate. The gate checks *whether any* PMR doc was touched — it does not verify completeness. You are responsible for completeness.

4. **Verify the gate passes:**
   ```
   python3 scripts/pmr_prepush_gate.py --verbose --dry-run
   ```

5. **Check for stale claims.** When updating a PMR doc, scan it for assertions that are now incorrect given your changes. Common traps:
   - Subtask status in the roadmap marked "Completed" for stubs
   - Data lineage "open questions" that have been resolved
   - "Future" pipeline steps that are now implemented
   - "What Each Work Item Fulfilled" tables with wrong attributions

### Reference: PMR system docs
- Registry: `docs/agent_bootstrap/project_memory_registry.md`
- Update policy: `docs/agent_bootstrap/project_memory_update_policy.md`
- Change event taxonomy: `docs/agent_bootstrap/change_event_taxonomy.md`

## Project Structure

- **Data pipeline scripts**: `src/data/01_ingest_*.py` through `04_build_*.py`
- **Pipeline config (source of truth)**: `src/data/config.py` (NOT `configs/data.yaml`)
- **Benchmark task config**: `configs/benchmark_tasks.yaml`
- **Package modules**: `src/neural_iv_surface_inference/`
- **Tests**: `tests/`
- **Raw data**: only on RunPod (`data_raw/spy/` is empty locally)

## Key Conventions

- All data files are gitignored; pipeline is designed to re-run from scratch
- Time-based splits are strictly chronological (no shuffling)
- `spot = close` (unadjusted), not `adjusted_close`
- Benchmark datasets are named: `spy_phase1_{strategy}{pct}_noise{level}`
