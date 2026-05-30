# AGENTS.md — Multi-Agent Entry Point

> Canonical bootstrap file for any coding agent (Claude Code, Cursor,
> Codex, Aider, …). **Read this first, then read only what it points
> you at.** Do NOT broadly grep the `docs/` tree on cold start.
>
> Authority: [ADR 0007](docs/decisions/0007_multi_agent_handoff.md).

## 1. What this project is (one paragraph)

Neural IV surface inference research repo. Pipeline scripts live in
`src/data/`, package code in `src/neural_iv_surface_inference/`, raw
data only on RunPod (gitignored). Work is organised by epics (e.g.
`3X`, `M1`) on `docs/tasks/BOARD.md`, each epic decomposed into atomic
stories with a spec under `docs/tasks/specs/<ID>.md`.

## 2. Routing — for the task at hand, read ONLY:

| Task | Read these files (in order) |
|---|---|
| Picking up story `<ID>` | `docs/tasks/BOARD.md` (find row) → `docs/tasks/specs/<ID>.md` → only the files listed under that spec's `## Context files to read first` |
| Writing a new spec | An existing spec as template (`docs/tasks/specs/3X.4_...md`) + `docs/agent_bootstrap/project_memory_registry.md` |
| Updating PMR docs after a change | `docs/agent_bootstrap/project_memory_update_policy.md` + `CLAUDE.md` §"Pre-Push Routine" |
| Making an architectural decision | `docs/decisions/` (numbered ADRs) — write a new one |
| Recording an experiment run | `docs/experiments/experiment_journal.md` (append) |
| Major failure / course correction | `docs/retrospectives/` (new numbered doc) |

If the task does not fit any row above, ask the operator before reading more.

## 3. Hard rules (also enforced by `git push` hooks)

These are not aspirational — `bash scripts/install_hooks.sh` installs
gates that block pushes violating them. See
[ADR 0007](docs/decisions/0007_multi_agent_handoff.md) for rationale.

1. **Story-dependency rule.** A spec's `Dependencies:` list must all be
   `done` on the BOARD before the spec moves to `in_progress`.
   Override: `WAIVE_DEPS="<id>:<actual-status>:<reason>"`
   (writes an audit line into the spec checkpoint).
   Enforced by: `scripts/check_story_dependencies.py`.
2. **File-scope contract.** Every spec declares `file_scope:`. Pushes
   touching active specs may not modify paths outside the scope union.
   Override: `WAIVE_SCOPE="<reason>"`.
   Enforced by: `scripts/check_file_scope.py`.
3. **Agent-trailer rule.** Agent-driven commits must carry a
   `Co-authored-by: <Agent> <email>` trailer.
   Override: `WAIVE_TRAILER="<reason>"`.
   Enforced by: `scripts/check_commit_trailer.py`.
4. **PMR coverage rule.** Changes to evidence-source files
   (`src/`, `scripts/`, `tests/`, `configs/`, `notebooks/`) must be
   paired with PMR doc updates.
   Enforced by: `scripts/pmr_prepush_gate.py`.

## 4. First-time setup (per clone)

```bash
bash scripts/install_hooks.sh
```

Installs `pre-commit`, `pre-push`, `commit-msg` from `scripts/hooks/`
into `.git/hooks/`. Re-run after `git clone` on a new machine / pod.

## 5. Per-agent shims (do not duplicate rules here)

- **Claude Code:** [`CLAUDE.md`](CLAUDE.md) — Claude-specific overrides
  only. Pre-Push Routine details live there.
- **Cursor:** [`.cursor/rules/000-bootstrap.mdc`](.cursor/rules/000-bootstrap.mdc)
  redirects to this file with `alwaysApply: true`.
- **New agents:** add a shim under your tool's native config dir
  pointing at `AGENTS.md`. Do not copy rules.

## 6. Operator personal notebooks — DO NOT READ unless named

Two files in `docs/` are gitignored personal notes:
`docs/neural_iv_surface_project_personal_notes.md` and
`docs/neural_iv_surface_research_methodology_narrative.md`. Never auto-
read, never quote in tracked artifacts. Details in
[`CLAUDE.md`](CLAUDE.md) §"Operator's Personal Working Notes".

---

## 7. Cold-start prompt (copy-paste to any fresh agent)

```text
You are picking up work on the Neural IV Surface inference repo
(<absolute path to repo>). Bootstrap rules:

1. Read AGENTS.md first. Follow its routing table — do NOT broadly
   grep docs/.
2. Do NOT read docs/neural_iv_surface_project_personal_notes.md or
   docs/neural_iv_surface_research_methodology_narrative.md unless I
   explicitly name them.
3. Run `bash scripts/install_hooks.sh` once if this is a fresh clone.
4. Confirm in one sentence which story (BOARD ID) you understand
   yourself to be picking up, then wait for my go-ahead before
   editing any file.
5. When you commit, include a `Co-authored-by:` trailer naming
   yourself (Cursor / Claude / Codex / Aider).
6. All `file_scope`, `Dependencies`, and PMR rules in AGENTS.md §3
   are enforced by git hooks. Bending them requires an explicit
   env-var waiver, never silent override.

The story I want you to pick up is: <STORY_ID>.
```
