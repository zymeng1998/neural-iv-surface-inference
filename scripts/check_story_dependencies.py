#!/usr/bin/env python3
"""
Story-dependency gate (M1.3).

For every story spec touched by the current diff whose own status is
active (``in_progress`` / ``in_review`` / ``done``), verify that every
story declared under its ``## Dependencies`` section is marked ``done``
on ``docs/tasks/BOARD.md``.

Per ADR 0007. Mirrors the UX of ``scripts/pmr_prepush_gate.py``.

Exit codes
----------
0  pass (push allowed)
1  fail (push blocked; an offending dep was identified)

Waiver
------
Set ``WAIVE_DEPS`` to a comma-separated list of
``<story_id>:<actual_status>:<reason>`` entries to bypass the check
for those specific (id, status) pairs. The waiver appends one audit
line to the last ``### <timestamp>`` checkpoint block of every
affected spec.

Usage
-----
    python3 scripts/check_story_dependencies.py [--verbose] [--dry-run]
    python3 scripts/check_story_dependencies.py --files PATH [PATH ...]

The hook entry-point passes no args; --files is for tests and local
ad-hoc checks.

Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SPEC_DIR = Path("docs/tasks/specs")
BOARD_PATH = Path("docs/tasks/BOARD.md")

STORY_ID_RE = re.compile(r"^[0-9A-Z]+\.[0-9]+$")
DEP_LINE_RE = re.compile(r"^\s*-\s+`?([0-9A-Z]+\.[0-9]+)`?")
BOARD_ROW_RE = re.compile(
    r"^\|\s*([0-9A-Z]+\.[0-9]+)\s*\|[^|]*\|[^|]*\|\s*`([a-z_]+)`\s*\|"
)
ACTIVE_STATUSES = {"in_progress", "in_review", "done"}
TZ = timezone(timedelta(hours=-4))  # repo convention: -04:00


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_spec(path: Path) -> Optional[Dict[str, object]]:
    """Return {'id', 'status', 'deps': [ids]} or None if not a story spec."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if not text.startswith("#"):
        return None

    fm_match = re.search(r"^---\n(.*?)\n---\n", text, re.DOTALL | re.MULTILINE)
    if not fm_match:
        return None
    fm = fm_match.group(1)

    id_match = re.search(r"^id:\s*([0-9A-Z]+\.[0-9]+)\s*$", fm, re.MULTILINE)
    status_match = re.search(r"^status:\s*([a-z_]+)\b", fm, re.MULTILINE)
    if not id_match or not status_match:
        return None

    spec_id = id_match.group(1)
    status = status_match.group(1)

    # Find the Dependencies section (until next "## " header).
    deps: List[str] = []
    dep_section = re.search(
        r"^##\s+Dependencies\s*\n(.*?)(?=^##\s|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if dep_section:
        for line in dep_section.group(1).splitlines():
            m = DEP_LINE_RE.match(line)
            if m and STORY_ID_RE.match(m.group(1)):
                dep_id = m.group(1)
                if dep_id != spec_id and dep_id not in deps:
                    deps.append(dep_id)

    return {"id": spec_id, "status": status, "deps": deps, "path": path}


def parse_board(board_path: Path) -> Dict[str, str]:
    """Return {story_id: status} from BOARD.md table rows."""
    if not board_path.is_file():
        return {}
    rows: Dict[str, str] = {}
    for line in board_path.read_text(encoding="utf-8").splitlines():
        m = BOARD_ROW_RE.match(line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


# ---------------------------------------------------------------------------
# Diff discovery
# ---------------------------------------------------------------------------

def changed_files_from_git() -> List[Path]:
    """Files changed vs origin/HEAD if available, else vs HEAD."""
    candidates = [
        ["git", "diff", "--name-only", "--cached"],
        ["git", "diff", "--name-only", "HEAD"],
    ]
    seen: Set[str] = set()
    for cmd in candidates:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip()
            if line:
                seen.add(line)
    # Untracked files
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            line = line.strip()
            if line:
                seen.add(line)
    except subprocess.CalledProcessError:
        pass
    return [Path(p) for p in sorted(seen)]


# ---------------------------------------------------------------------------
# Waiver
# ---------------------------------------------------------------------------

def parse_waiver(raw: Optional[str]) -> Dict[Tuple[str, str], str]:
    """Parse WAIVE_DEPS into {(dep_id, actual_status): reason}."""
    out: Dict[Tuple[str, str], str] = {}
    if not raw:
        return out
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 2)
        if len(parts) != 3:
            print(
                f"[dep-gate] malformed WAIVE_DEPS entry (need id:status:reason): {entry!r}",
                file=sys.stderr,
            )
            continue
        out[(parts[0].strip(), parts[1].strip())] = parts[2].strip()
    return out


def append_audit_line(spec_path: Path, line: str) -> None:
    """Append `line` under the last `### <timestamp> ...` checkpoint block."""
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError:
        return
    # Insert just before the next "### " heading after the last one, or at EOF.
    matches = list(re.finditer(r"^### [^\n]*$", text, re.MULTILINE))
    if not matches:
        # No checkpoint section; append a fresh one.
        suffix = (
            f"\n\n### {datetime.now(TZ).strftime('%Y-%m-%dT%H:%M:%S%z')}"
            f" — dep waiver appended\n\n{line}\n"
        )
        spec_path.write_text(text + suffix, encoding="utf-8")
        return
    last = matches[-1]
    # Find end of that section (next "### " or EOF).
    after = text[last.end():]
    next_match = re.search(r"^### ", after, re.MULTILINE)
    insert_at = last.end() + (next_match.start() if next_match else len(after))
    new_text = text[:insert_at].rstrip() + "\n" + line + "\n\n" + text[insert_at:].lstrip()
    spec_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    spec_paths: Iterable[Path],
    board_status: Dict[str, str],
    waivers: Dict[Tuple[str, str], str],
    verbose: bool = False,
) -> Tuple[List[str], List[Tuple[Path, str]]]:
    """Return (violations, waiver_audit_writes)."""
    violations: List[str] = []
    audit_writes: List[Tuple[Path, str]] = []
    now = datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S%z")

    for path in spec_paths:
        spec = parse_spec(path)
        if not spec:
            continue
        if spec["status"] not in ACTIVE_STATUSES:
            if verbose:
                print(f"[dep-gate] skip (status={spec['status']}): {path}")
            continue
        for dep_id in spec["deps"]:
            actual = board_status.get(dep_id, "<not-on-board>")
            if actual == "done":
                continue
            key = (dep_id, actual)
            if key in waivers:
                reason = waivers[key]
                audit_writes.append((
                    path,
                    f"- DEP WAIVER ({now}): {dep_id} {actual} → bypassed ({reason})",
                ))
                if verbose:
                    print(f"[dep-gate] waived: {spec['id']} dep {dep_id} ({actual})")
                continue
            violations.append(
                f"{path}: story {spec['id']} (status {spec['status']}) "
                f"depends on {dep_id} which is `{actual}` on BOARD (need `done`)"
            )
    return violations, audit_writes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Story-dependency gate (M1.3).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not write audit lines for waivers; report only.",
    )
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Override: check these spec paths instead of git diff.",
    )
    parser.add_argument(
        "--board", default=str(BOARD_PATH),
        help=f"Path to BOARD.md (default {BOARD_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.files is not None:
        candidates = [Path(p) for p in args.files]
    else:
        candidates = changed_files_from_git()

    spec_paths = [
        p for p in candidates
        if p.suffix == ".md" and SPEC_DIR.as_posix() in p.as_posix()
    ]
    if not spec_paths:
        if args.verbose:
            print("[dep-gate] no story specs in diff — PASS")
        return 0

    board_status = parse_board(Path(args.board))
    waivers = parse_waiver(os.environ.get("WAIVE_DEPS"))

    violations, audit_writes = evaluate(
        spec_paths, board_status, waivers, verbose=args.verbose,
    )

    for path, line in audit_writes:
        if args.dry_run:
            print(f"[dep-gate] DRY-RUN would append to {path}: {line}")
        else:
            append_audit_line(path, line)
            if args.verbose:
                print(f"[dep-gate] appended waiver audit line to {path}")

    if violations:
        print("[dep-gate] BLOCKED — story dependency violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            '\nTo bypass: WAIVE_DEPS="<id>:<actual_status>:<reason>" '
            "git push  (comma-separated for multiple).",
            file=sys.stderr,
        )
        return 1

    if args.verbose:
        print(f"[dep-gate] {len(spec_paths)} active spec(s) checked — PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
