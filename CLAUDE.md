# Project Instructions — Neural IV Surface Inference

> **READ `AGENTS.md` FIRST.** It is the canonical multi-agent entry
> point (router + hard rules + cold-start prompt). This file
> (`CLAUDE.md`) contains only Claude-specific overrides on top of
> `AGENTS.md`. Per [ADR 0007](docs/decisions/0007_multi_agent_handoff.md),
> the hard rules in `AGENTS.md` §3 are enforced by git hooks installed
> via `bash scripts/install_hooks.sh` — they bind Claude exactly as
> they bind any other agent. Do not bend a rule without a `WAIVE_*`
> env-var; the gate will block you and that is intentional.
>
> Claude-specific note: commits authored during a Claude Code session
> must carry a `Co-authored-by: Claude <noreply@anthropic.com>` trailer.
> This overrides the global "attribution disabled" preference in
> `~/.claude/CLAUDE.md` for this project (ADR 0007 documents the
> override).

## Communication Language

Default to English-only communication with the user.

Do not use Chinese-English mixed explanations by default. Even if the user writes instructions in Chinese, understand the request but respond in concise English unless the user explicitly requests Chinese or bilingual output.

Avoid translating technical terms into both Chinese and English unless explicitly requested. For normal repository work, coding, debugging, planning, reviews, and progress updates, use clear and compact English.

This rule is mainly for token efficiency. Chinese-English mixed responses are too token-heavy for this workflow.

Exception: if the user explicitly asks for Chinese, bilingual output, translation, lecture notes, or a user-facing document that must be in Chinese, follow that task-specific language requirement.

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

## Operator's Personal Working Notes (do not touch unprompted)

Two files in `docs/` are the operator's personal notebooks. They are **gitignored** (`.gitignore` lines 127–132) and must never be committed, pushed, or referenced in committed documentation.

- `docs/neural_iv_surface_project_personal_notes.md`
- `docs/neural_iv_surface_research_methodology_narrative.md`

### Rules

- **Do not auto-read.** Never open either file as part of an exploration sweep, a "read all docs" pattern, or PMR routine. Read only when the operator explicitly names a file or says "the personal notes" / "the methodology narrative".
- **Update only on explicit request.** Wait for the operator to say "update the personal notes with X" or equivalent. Do not infer that recent work should land there.
- **Do not quote or summarize in committed artifacts.** Nothing from these files may appear in commits, PRs, BOARD entries, roadmaps, ADRs, progress log, experiment journal, README, or any other tracked file. If something from the notes belongs in committed evidence, the operator will say so and you will port the content explicitly into the appropriate PMR doc.
- **Do not include in PMR pre-push routine.** The pre-push gate already ignores these files (gitignored). Do not flag them as "PMR docs that need updating" when other evidence changes.
- **Do not stage.** `git add docs/` or similar broad staging is safe because the files are gitignored, but never run `git add -f` on either path.
