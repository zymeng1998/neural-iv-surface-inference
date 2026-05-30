#!/usr/bin/env python3
"""
File-scope gate (M1.4).

For every story spec touched by the current diff whose status is
``in_progress`` or ``in_review``, refuse the push if the diff also
touches any path outside the *union* of those specs'
``file_scope:`` glob lists.

Per ADR 0007. Mirrors the UX of
``scripts/check_story_dependencies.py``.

Exit codes
----------
0  pass
1  fail

Waiver
------
``WAIVE_SCOPE="<reason>"`` bypasses the check and appends an audit
line listing the out-of-scope paths to each affected spec's last
checkpoint block.

Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Re-use the dep-gate helpers where possible.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_story_dependencies import (  # noqa: E402
    SPEC_DIR,
    changed_files_from_git,
    append_audit_line,
)

ACTIVE_STATUSES = {"in_progress", "in_review"}
TZ = timezone(timedelta(hours=-4))


def parse_spec_with_scope(path: Path) -> Optional[Dict[str, object]]:
    """Return {'id', 'status', 'scope': [globs]} or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    fm_match = re.search(r"^---\n(.*?)\n---\n", text, re.DOTALL | re.MULTILINE)
    if not fm_match:
        return None
    fm = fm_match.group(1)

    id_match = re.search(r"^id:\s*([0-9A-Z]+\.[0-9]+)\s*$", fm, re.MULTILINE)
    status_match = re.search(r"^status:\s*([a-z_]+)\b", fm, re.MULTILINE)
    if not id_match or not status_match:
        return None

    # file_scope is a YAML list; we accept either inline or block form.
    scope: List[str] = []
    scope_match = re.search(
        r"^file_scope:\s*\n((?:\s+-\s+.+\n?)+)", fm, re.MULTILINE
    )
    if scope_match:
        for line in scope_match.group(1).splitlines():
            m = re.match(r'\s+-\s+"?([^"\n]+?)"?\s*$', line)
            if m:
                scope.append(m.group(1).strip())
    return {
        "id": id_match.group(1),
        "status": status_match.group(1),
        "scope": scope,
        "path": path,
    }


def path_matches_any(path_str: str, globs: Iterable[str]) -> bool:
    """fnmatch on each glob; '**' is treated as recursive wildcard."""
    for g in globs:
        # Translate '**' to fnmatch-compatible: replace '**' with '*'
        # then check; also accept prefix match for dir-style globs.
        norm = g.replace("**", "*")
        if fnmatch.fnmatch(path_str, norm):
            return True
        # Allow "dir/**" to also match "dir/anything/below".
        if g.endswith("/**") and path_str.startswith(g[:-3] + "/"):
            return True
    return False


def evaluate(
    changed: List[Path],
    active_specs: List[Dict[str, object]],
    waiver_reason: Optional[str],
    verbose: bool = False,
) -> Tuple[List[str], List[Tuple[Path, str]]]:
    """Return (violations, audit_writes)."""
    if not active_specs:
        if verbose:
            print("[scope-gate] no active specs in diff — PASS")
        return [], []

    scope_union: List[str] = []
    for s in active_specs:
        scope_union.extend(s["scope"])
    if verbose:
        print(f"[scope-gate] scope union ({len(scope_union)} globs) from "
              f"{[s['id'] for s in active_specs]}")

    out_of_scope: List[str] = []
    for p in changed:
        s = p.as_posix()
        if not path_matches_any(s, scope_union):
            out_of_scope.append(s)

    audit_writes: List[Tuple[Path, str]] = []
    if out_of_scope and waiver_reason:
        now = datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S%z")
        line = (
            f"- SCOPE WAIVER ({now}): out-of-scope paths "
            f"[{', '.join(out_of_scope)}] → bypassed ({waiver_reason})"
        )
        for s in active_specs:
            audit_writes.append((s["path"], line))
        return [], audit_writes

    violations = []
    if out_of_scope:
        violations.append(
            f"paths outside the scope union of active specs "
            f"{[s['id'] for s in active_specs]}: "
            f"{out_of_scope}"
        )
    return violations, audit_writes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="File-scope gate (M1.4).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Override: treat these paths as the diff.")
    args = parser.parse_args(argv)

    if args.files is not None:
        changed = [Path(p) for p in args.files]
    else:
        changed = changed_files_from_git()

    active_specs: List[Dict[str, object]] = []
    for p in changed:
        if p.suffix == ".md" and SPEC_DIR.as_posix() in p.as_posix():
            spec = parse_spec_with_scope(p)
            if spec and spec["status"] in ACTIVE_STATUSES:
                active_specs.append(spec)

    waiver = os.environ.get("WAIVE_SCOPE")
    violations, audit_writes = evaluate(
        changed, active_specs, waiver_reason=waiver, verbose=args.verbose,
    )

    for path, line in audit_writes:
        if args.dry_run:
            print(f"[scope-gate] DRY-RUN would append to {path}: {line}")
        else:
            append_audit_line(path, line)
            if args.verbose:
                print(f"[scope-gate] appended scope waiver to {path}")

    if violations:
        print("[scope-gate] BLOCKED — file-scope violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            '\nTo bypass: WAIVE_SCOPE="<reason>" git push',
            file=sys.stderr,
        )
        return 1
    if args.verbose:
        print(f"[scope-gate] PASS ({len(active_specs)} active spec(s); "
              f"{len(changed)} changed file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
