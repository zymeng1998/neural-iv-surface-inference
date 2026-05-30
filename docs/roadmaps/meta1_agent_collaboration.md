# Meta-1 Roadmap — Multi-Agent Collaboration Infrastructure

---
created_at: 2026-05-30T13:00:00-04:00
last_updated_at: 2026-05-30T13:00:00-04:00
epic: M1
status: in_progress
---

## Goal

Make multi-agent collaboration (Claude Code + Cursor + future agents)
seamless and rule-bending-resistant **at the project level**, so the
operator can hand off across the 5-hour Claude rate-limit window
without losing rule discipline. Project-scoped first; promote to
user-level later once proven.

## Why this epic exists

See [ADR 0007](../decisions/0007_multi_agent_handoff.md). Short
version: prose rules in `CLAUDE.md` and story specs get bent
(3X.4 ran while 3X.2 was `in_review`); only executable gates hold.

## Subtask matrix

| ID | Title | Locale | Status |
|---|---|---|---|
| M1.1 | Decompose M1 + ADR 0007 + 4 sub-specs | local | in_progress |
| M1.2 | `AGENTS.md` router + Cursor bootstrap + CLAUDE.md re-point | local | backlog |
| M1.3 | `check_story_dependencies.py` + tests + `install_hooks.sh` | local | backlog |
| M1.4 | `check_file_scope.py` + tests | local | backlog |
| M1.5 | `commit-msg` trailer hook (agent-trailer enforcement) | local | backlog |

All stories are `parallel_safe_with: ["3X.*", "3B.*", "3C.*", "3D.*"]`
because they touch only workflow infra (`scripts/`, `docs/`,
`AGENTS.md`, `.cursor/`), never `src/` or `data_processed/`.

## Sequencing

```
M1.1  →  M1.2  →  M1.3  →  M1.4  →  M1.5
                    │
                    └─ first gate that would have caught 3X.4 deviation
```

M1.2 ships before M1.3 so the next agent (Cursor or Claude) reading
`AGENTS.md` already knows about the gates being installed.

## Acceptance for the epic

- `git push` from a clone of this repo runs PMR + dep + file-scope
  gates; bending any requires an explicit env-var waiver.
- A fresh agent (Cursor, new Claude session) given the cold-start
  prompt in `AGENTS.md` can pick up the next story without
  reading the whole `docs/` tree.
- The 3X.4-style deviation ("dep is `in_review`, not `done`") is
  blocked by default and produces a clear actionable error message.

## Out of scope

- Migrating any of this to `~/.claude/` global config.
- Cross-agent task locking / queueing.
- CI-side enforcement (gates run locally; CI is a future iteration).
