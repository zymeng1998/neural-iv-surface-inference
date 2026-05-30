# ADR 0007: Multi-Agent Collaboration Rules — Discovery + Enforcement

## Status

Accepted

## Date

2026-05-30

## Context

This project is increasingly worked by two different coding agents (Claude
Code, Cursor) plus the human operator, often within the same week and
sometimes within the same epic. The 5-hour Claude rate-limit window
forces handoffs across agents whether we plan for them or not.

The 3X.4 commit ([`ee11f69`](../../)) surfaced the asymmetry. Cursor
executed 3X.4 while its declared dependency 3X.2 was still `in_review`
(spec required `done`). The deviation was operator-authorized and
self-disclosed in the spec checkpoint, but nothing in the repository
*stopped* it from happening — the dependency rule was prose in a spec
file, not an enforced check.

The PMR pre-push gate ([`scripts/pmr_prepush_gate.py`](../../scripts/pmr_prepush_gate.py))
works precisely because it is executable: it cannot be bent by an agent
that does not read it. By contrast, the prose rules in
[`CLAUDE.md`](../../CLAUDE.md), individual story specs, and
[`docs/agent_bootstrap/`](../agent_bootstrap/) depend on the agent
voluntarily reading and obeying them.

## Decision

Adopt a two-layer multi-agent collaboration system, both layers
project-scoped (lives in this repo, travels with `git clone`):

### Layer 1 — Discovery (router, not manual)

- A single canonical agent entry-point file at the repo root:
  [`AGENTS.md`](../../AGENTS.md). Open standard read natively by Cursor
  (0.45+), Codex, Aider, and most modern agents.
- `AGENTS.md` is a **router**, capped at ~120 lines. It points to the
  one or two files needed for a given kind of task. It does **not**
  re-state rules that live elsewhere.
- Tool-specific shims point at `AGENTS.md`, they do not duplicate it:
  - [`CLAUDE.md`](../../CLAUDE.md): first line directs the agent to read
    `AGENTS.md`. Claude-specific overrides only below.
  - [`.cursor/rules/000-bootstrap.mdc`](../../.cursor/rules/000-bootstrap.mdc):
    `alwaysApply: true`, redirects to `AGENTS.md`.
- A cold-start prompt for any agent (Cursor, fresh Claude session, etc.)
  lives at the bottom of `AGENTS.md` so the operator can paste it
  verbatim. The agent then bootstraps itself.

### Layer 2 — Enforcement (executable gates below the agent)

Every collaboration rule that *can* be bent must be enforced by a script
that runs in a git hook, so it fires regardless of which agent (or
human) is driving git:

| Gate | Script | Hook | Replaces prose rule |
|---|---|---|---|
| Story dep status | `scripts/check_story_dependencies.py` | pre-push | "Dependencies: 3X.2 `done`" in spec |
| File-scope contract | `scripts/check_file_scope.py` | pre-push | "`file_scope` is a hard contract" in spec |
| Agent-trailer trace | `scripts/check_commit_trailer.py` | commit-msg | Convention only |
| PMR doc coverage | `scripts/pmr_prepush_gate.py` (existing) | pre-push | — |

All gates are **bypassable with an explicit waiver** (env var, not a
flag in a tracked file). The waiver requires a free-text reason and is
recorded in the story's checkpoint as an audit line. Bending the rule
is allowed; bending it silently is not.

Hook installation is via a tracked installer script
[`scripts/install_hooks.sh`](../../scripts/install_hooks.sh) that copies
from [`scripts/hooks/`](../../scripts/hooks/) into `.git/hooks/`.
Operator runs it once per clone.

### Out of scope for this ADR

- Promoting any of this to user-level (`~/.claude/`) or
  machine-global config. That is a future move once the project-level
  pattern is proven.
- Cross-agent task-queue / lock systems. Sequencing is still operator-
  driven; the gates only prevent silent rule violations, not race
  conditions.

## Rationale

- **Prose rules get bent; executable rules don't.** The 3X.4 incident
  is the immediate evidence; the broader principle is general.
- **One discovery file beats N tool-specific files.** Duplicating rules
  per agent is the failure mode the existing `~/.claude/CLAUDE.md` and
  per-project `CLAUDE.md` were already drifting toward.
- **Project-scoped first.** Iteration speed matters more than coverage
  at this stage. Once the gates are stable, the operator can lift them
  into a cookiecutter / template for future projects.
- **Waivers, not bypasses.** A rule that cannot be bent under any
  circumstance will be either disabled or worked around. A rule that
  requires `WAIVE_DEPS="reason"` to bend leaves a paper trail and
  forces conscious choice.

## Consequences

- Both Claude and Cursor (and any future agent) are bound by the same
  gates. Claude included — the rule that 3X.4 bent will, after M1.3
  ships, block Claude in the equivalent situation just as it would have
  blocked Cursor.
- The `Co-authored-by:` trailer becomes mandatory on agent-driven
  commits. This conflicts with the global "attribution disabled" rule
  in [`~/.claude/CLAUDE.md`](../../) — the project-level rule wins
  (CLAUDE.md will be updated to reflect this exception).
- New stories must declare `file_scope:` and `Dependencies:` in
  parseable form (existing spec template already does both).
- New contributors / new clones need to run
  `bash scripts/install_hooks.sh` once. Documented in
  [`README.md`](../../README.md) and [`AGENTS.md`](../../AGENTS.md).
- **`.gitignore` exception.** Commit `a3d3ee1` (2026-05-19, "hide AI
  assistant config files from repo") had broadly ignored `CLAUDE.md`,
  `AGENTS.md`, and `.cursor/`. That policy is incompatible with the
  "travels with `git clone`" requirement here, so `.gitignore` now
  carries a narrow negation that **commits exactly three discovery
  files** (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules/000-bootstrap.mdc`)
  while every other AI-assistant artifact (`.claude/`, `.aider*`,
  `CLAUDE.local.md`, all other `.cursor/` contents, …) stays local-only.
  The three committed files must contain **no machine-specific paths or
  secrets** — they are public-by-design routers, not local context.

## References

- Epic `M1` on [`docs/tasks/BOARD.md`](../tasks/BOARD.md)
- Roadmap [`docs/roadmaps/meta1_agent_collaboration.md`](../roadmaps/meta1_agent_collaboration.md)
- Stories `M1.1` … `M1.5` under [`docs/tasks/specs/`](../tasks/specs/)
