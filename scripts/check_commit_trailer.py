#!/usr/bin/env python3
"""
Commit-message agent-trailer gate (M1.5).

When the commit is detected as agent-driven (CLAUDECODE, CURSOR_TRACE_ID,
CODEX_*, AIDER_*, …), require a ``Co-authored-by:`` trailer.

Per ADR 0007.

Usage (commit-msg hook):
    python3 scripts/check_commit_trailer.py "$1"

Waiver
------
``WAIVE_TRAILER="<reason>"`` bypasses the check (logged to stderr).

Exit codes
----------
0  pass
1  fail
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List

# Env-var prefixes / exact names that indicate an agent-driven session.
# Extend this list when a new agent is added.
AGENT_ENV_EXACT = {"CLAUDECODE", "CURSOR_TRACE_ID"}
AGENT_ENV_PREFIXES = ("CODEX_", "AIDER_")

TRAILER_RE = re.compile(
    r"^Co-authored-by:\s+\S.*<\S+@\S+>\s*$", re.MULTILINE
)


def detect_agent() -> List[str]:
    """Return the list of env-var names that indicate an agent session."""
    detected: List[str] = []
    for name in AGENT_ENV_EXACT:
        if os.environ.get(name):
            detected.append(name)
    for name in os.environ:
        if any(name.startswith(p) for p in AGENT_ENV_PREFIXES):
            detected.append(name)
    return sorted(set(detected))


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: check_commit_trailer.py <commit-msg-file>", file=sys.stderr)
        return 1

    msg_path = Path(argv[1])
    try:
        msg = msg_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[trailer-gate] cannot read {msg_path}: {e}", file=sys.stderr)
        return 1

    # Strip git comment lines (those starting with '#' after newline).
    body = "\n".join(
        line for line in msg.splitlines() if not line.startswith("#")
    )

    agents = detect_agent()
    if not agents:
        return 0  # human commit — nothing to enforce

    if TRAILER_RE.search(body):
        return 0

    waiver = os.environ.get("WAIVE_TRAILER")
    if waiver:
        print(
            f"[trailer-gate] WAIVED: agent={agents} reason={waiver!r}",
            file=sys.stderr,
        )
        return 0

    print(
        "[trailer-gate] BLOCKED — agent-driven commit missing Co-authored-by trailer.\n"
        f"  detected agent env: {agents}\n"
        '  add a line like:  Co-authored-by: Cursor <cursoragent@cursor.com>\n'
        '  to bypass once:   WAIVE_TRAILER="<reason>" git commit ...',
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
