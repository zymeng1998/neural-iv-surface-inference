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
for those specific (id, status) pairs. The waiver records one audit
line per bypass to the untracked ``docs/audit/waiver_log.md`` (M1.6 /
ADR 0007) — it never modifies a tracked file, so a waived push leaves
the working tree clean.

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
# Untracked (gitignored) sink for waiver audit trails. The pre-push gates
# fire AFTER the commit being pushed is created, so writing the audit trail
# into a tracked spec would leave the working tree dirty post-push (M1.6 /
# ADR 0007). Writing here keeps the tree clean while preserving a durable,
# operator-reviewable record. `docs/audit/` is gitignored.
WAIVER_LOG = Path("docs/audit/waiver_log.md")

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

def _git(args: List[str]) -> str:
    """Run a git command, returning stdout (empty string on failure)."""
    try:
        return subprocess.check_output(
            ["git"] + args, text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return ""


def changed_files_from_git() -> List[Path]:
    """Files in the commits about to be pushed, plus working-tree changes.

    At ``git push`` time the working tree is clean (everything committed),
    so the meaningful set is the commit range ``<upstream>..HEAD`` — the
    same range the PMR gate inspects. We deliberately do NOT read the
    pre-push stdin here: when several gates are chained in one hook the
    first script (the PMR gate) drains stdin, leaving none for us. The
    ``<upstream>..HEAD`` diff is independent of stdin and therefore
    robust to gate ordering.

    Working-tree (staged + unstaged + untracked) changes are also
    included so the gate is useful when run standalone *before* a commit.
    """
    seen: Set[str] = set()

    # 1. Commits being pushed: <upstream>..HEAD (the push range).
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    upstream = _git(
        ["rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"]
    ).strip()
    if upstream:
        rng = _git(["diff", "--name-only", f"{upstream}..HEAD"])
    else:
        # No upstream configured (e.g. brand-new branch) — fall back to
        # the last few commits so a first push is still inspected.
        rng = _git(["diff", "--name-only", "HEAD~5..HEAD"]) or _git(
            ["show", "--name-only", "--pretty=format:", "HEAD"]
        )
    for line in rng.splitlines():
        line = line.strip()
        if line:
            seen.add(line)

    # 2. Working-tree changes (standalone pre-commit usage).
    for cmd in (
        ["diff", "--name-only", "--cached"],
        ["diff", "--name-only", "HEAD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        for line in _git(cmd).splitlines():
            line = line.strip()
            if line:
                seen.add(line)

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


_WAIVER_LOG_HEADER = (
    "# Waiver audit log\n"
    "\n"
    "> **Untracked / gitignored.** Appended by the pre-push gates"
    " (`check_story_dependencies.py`, `check_file_scope.py`) whenever a\n"
    "> documented `WAIVE_DEPS` / `WAIVE_SCOPE` bypass fires. It lives here —"
    " not in a tracked spec — so a waived push never leaves the working\n"
    "> tree dirty (M1.6 / ADR 0007).\n"
    ">\n"
    "> **Follow-up workflow:** review with `cat docs/audit/waiver_log.md`."
    " If a waiver belongs in a story's permanent record, fold it into that\n"
    "> spec's `## Last checkpoint` as a *normal pre-commit edit* — never let"
    " a gate write to a tracked file.\n"
    "\n"
)


def record_waiver_audit(
    entries: Iterable[Tuple[Path, str]],
    *,
    dry_run: bool = False,
) -> Optional[Path]:
    """Append waiver audit `entries` to the untracked `WAIVER_LOG`.

    `entries` is an iterable of ``(related_spec_path, audit_line)``. Returns
    the log path if anything was written, else ``None``. Never mutates a
    tracked file — this is the whole point of M1.6.
    """
    entries = list(entries)
    if not entries or dry_run:
        return None
    WAIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not WAIVER_LOG.exists()
    with WAIVER_LOG.open("a", encoding="utf-8") as fh:
        if new_file:
            fh.write(_WAIVER_LOG_HEADER)
        for spec_path, line in entries:
            fh.write(f"{line}  [spec: {Path(spec_path).as_posix()}]\n")
    return WAIVER_LOG


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

    if audit_writes:
        if args.dry_run:
            for path, line in audit_writes:
                print(f"[dep-gate] DRY-RUN would record waiver to {WAIVER_LOG}: {line}")
        else:
            record_waiver_audit(audit_writes)
            print(f"[dep-gate] recorded {len(audit_writes)} waiver(s) to "
                  f"{WAIVER_LOG} (untracked — no tracked file modified)")
            if args.verbose:
                for path, line in audit_writes:
                    print(f"[dep-gate]   {line}  [spec: {path.as_posix()}]")

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
