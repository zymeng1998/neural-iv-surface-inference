#!/usr/bin/env bash
# Install repo-managed git hooks into .git/hooks/.
# Per ADR 0007. Run once per clone (e.g. fresh laptop, fresh Pod).
#
# Hooks installed:
#   pre-push    — PMR + dep + (optional) scope gates
#   pre-commit  — no-op placeholder
#   commit-msg  — agent-trailer enforcement (installed if present)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/scripts/hooks"
DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$DST" ]; then
    echo "error: $DST does not exist — are you inside a git repo?" >&2
    exit 1
fi

for hook in pre-push pre-commit commit-msg; do
    src_path="$SRC/$hook"
    if [ -f "$src_path" ]; then
        install -m 0755 "$src_path" "$DST/$hook"
        echo "installed: $hook"
    else
        echo "skip: $hook (no $src_path)"
    fi
done

echo "done. Hooks installed under $DST."
