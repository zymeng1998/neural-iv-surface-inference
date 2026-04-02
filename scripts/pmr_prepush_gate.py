#!/usr/bin/env python3
"""
PMR Pre-Push Gate — Project Memory Reviewer pre-push check.

Checks whether meaningful evidence-source changes have been pushed without
corresponding project-memory documentation updates.

This hook is designed for the architecture where:
  - Git push happens from a remote clone (e.g. RunPod)
  - PMR review/update is done via Claude Code or Cursor operating directly
    on that same remote repository
  - There is NO default local-pull / local-review / push-back loop

Exit codes:
  0 = pass (push allowed)
  1 = fail (push blocked — PMR attention likely required)

Usage:
  Called by git pre-push hook. Can also be run standalone:
    python scripts/pmr_prepush_gate.py [--verbose] [--dry-run]

Dependencies: Python 3.7+ standard library only.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories whose changes are "evidence sources" — they may require PMR updates
EVIDENCE_DIRS = {
    "src/",
    "scripts/",
    "tests/",
    "configs/",
    "notebooks/",
}

# Individual evidence files outside the dirs above
EVIDENCE_FILES = {
    "docs/data_assumptions_and_cleaning.md",
    "docs/data/data_lineage.md",
}

# PMR documentation directories — if evidence changed, these should also show changes
PMR_DOC_PATHS = {
    "docs/logs/progress_log.md",
    "docs/experiments/experiment_journal.md",
    "docs/decisions/",
    "docs/retrospectives/",
    "docs/roadmaps/",
}

# Files that are part of the PMR system itself — changes here don't require PMR updates
PMR_SYSTEM_FILES = {
    "docs/agent_bootstrap/",
    "scripts/build_project_memory_review_packet.py",
    "scripts/pmr_prepush_gate.py",
    "reports/project_memory/",
    ".pre-commit-config.yaml",
}

# Files that are always trivial — never trigger PMR
TRIVIAL_PATTERNS = {
    ".gitignore",
    ".gitkeep",
    "README.md",
    ".claude/",
    "docs/private/",
    "docs/setup/",
}

# Minimum number of evidence-source file changes to trigger PMR check
MIN_EVIDENCE_CHANGES = 1

# Reviewer state file
REVIEWER_STATE_PATH = "docs/agent_bootstrap/reviewer_state.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run a command and return stdout."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, cwd=cwd
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def get_repo_root() -> Path:
    """Get the git repo root."""
    root = run(["git", "rev-parse", "--show-toplevel"])
    if not root:
        print("ERROR: Not inside a git repository.", file=sys.stderr)
        sys.exit(1)
    return Path(root)


def get_commits_being_pushed(repo_root: Path) -> List[str]:
    """
    Determine which commits are about to be pushed.

    In a pre-push hook, stdin provides lines of:
      <local ref> <local sha> <remote ref> <remote sha>

    We compute commits in local..remote range.
    If not running as a hook (standalone), compare HEAD vs origin/current-branch.
    """
    commits = []

    # Try reading from stdin (pre-push hook mode)
    if not sys.stdin.isatty():
        for line in sys.stdin:
            parts = line.strip().split()
            if len(parts) >= 4:
                local_sha = parts[1]
                remote_sha = parts[3]
                # 0000... means new branch or deleted ref
                if remote_sha.startswith("0000"):
                    # New branch — check last 20 commits
                    log = run(
                        ["git", "log", "--oneline", "-20", local_sha],
                        cwd=str(repo_root),
                    )
                else:
                    log = run(
                        ["git", "log", "--oneline", f"{remote_sha}..{local_sha}"],
                        cwd=str(repo_root),
                    )
                if log:
                    commits.extend(log.splitlines())
        return commits

    # Standalone mode — compare against remote tracking branch
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root))
    remote_ref = run(
        ["git", "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"],
        cwd=str(repo_root),
    )
    if remote_ref:
        log = run(
            ["git", "log", "--oneline", f"{remote_ref}..HEAD"],
            cwd=str(repo_root),
        )
        if log:
            commits = log.splitlines()
    else:
        # No upstream — check last 5 commits
        log = run(["git", "log", "--oneline", "-5"], cwd=str(repo_root))
        if log:
            commits = log.splitlines()

    return commits


def get_changed_files(repo_root: Path) -> Set[str]:
    """Get files changed in commits being pushed + any staged/unstaged changes."""
    files = set()

    # Files changed in unpushed commits vs remote
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root))
    remote_ref = run(
        ["git", "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"],
        cwd=str(repo_root),
    )
    if remote_ref:
        diff = run(
            ["git", "diff", "--name-only", f"{remote_ref}..HEAD"],
            cwd=str(repo_root),
        )
        if diff:
            files.update(diff.splitlines())

    # Also check staged + unstaged working tree changes
    staged = run(["git", "diff", "--cached", "--name-only"], cwd=str(repo_root))
    if staged:
        files.update(staged.splitlines())

    unstaged = run(["git", "diff", "--name-only"], cwd=str(repo_root))
    if unstaged:
        files.update(unstaged.splitlines())

    return files


def classify_files(files: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Classify changed files into evidence, PMR docs, and trivial."""
    evidence = set()
    pmr_docs = set()
    trivial = set()

    for f in files:
        # Check trivial first
        if any(f.startswith(t) or f == t for t in TRIVIAL_PATTERNS):
            trivial.add(f)
            continue

        # Check PMR system files
        if any(f.startswith(s) for s in PMR_SYSTEM_FILES):
            trivial.add(f)  # PMR system changes don't require PMR review
            continue

        # Check PMR doc paths
        is_pmr = False
        for p in PMR_DOC_PATHS:
            if f == p or f.startswith(p):
                pmr_docs.add(f)
                is_pmr = True
                break
        if is_pmr:
            continue

        # Check evidence dirs
        is_evidence = False
        for d in EVIDENCE_DIRS:
            if f.startswith(d):
                evidence.add(f)
                is_evidence = True
                break
        if is_evidence:
            continue

        # Check evidence files
        if f in EVIDENCE_FILES:
            evidence.add(f)
            continue

        # Unclassified — treat as trivial (don't block on unknown files)
        trivial.add(f)

    return evidence, pmr_docs, trivial


def check_reviewer_state(repo_root: Path, current_head: str) -> bool:
    """Check if PMR has been run for the current HEAD."""
    state_file = repo_root / REVIEWER_STATE_PATH
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        last_head = state.get("last_processed_git_head", "")
        return last_head == current_head
    except (json.JSONDecodeError, OSError):
        return False


def build_review_packet(repo_root: Path) -> bool:
    """Try to build the review packet. Returns True on success."""
    script = repo_root / "scripts" / "build_project_memory_review_packet.py"
    if not script.exists():
        return False
    result = run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
    )
    return bool(result)


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

REMEDIATION_MESSAGE = """
╔══════════════════════════════════════════════════════════════════════╗
║  PMR PRE-PUSH GATE: Project Memory Review likely required          ║
╚══════════════════════════════════════════════════════════════════════╝

Evidence-source files changed without corresponding PMR doc updates.

Changed evidence files:
{evidence_list}

To resolve:

  1. Open Claude Code or Cursor against THIS repository
     (you can connect remotely — no need to pull locally)

  2. Run the PMR ongoing reviewer prompt:

     > Run the Project Memory Reviewer for the current repository state.
     > Follow the repo-local rules in docs/agent_bootstrap/.

  3. Stage and commit the PMR documentation updates:

     git add docs/logs/ docs/experiments/ docs/retrospectives/ docs/decisions/ docs/roadmaps/
     git add docs/agent_bootstrap/reviewer_state.json docs/agent_bootstrap/reviewer_state.md
     git commit -m "docs: update project memory for recent changes"

  4. Retry the push:

     git push

To bypass (use sparingly):

  git push --no-verify

Or set the environment variable:

  SKIP_PMR_GATE=1 git push
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="PMR pre-push gate")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run checks but always exit 0",
    )
    args = parser.parse_args()

    # Allow explicit skip via environment variable
    if os.environ.get("SKIP_PMR_GATE", "").strip() in ("1", "true", "yes"):
        if args.verbose:
            print("PMR gate: skipped via SKIP_PMR_GATE env var")
        return 0

    repo_root = get_repo_root()
    current_head = run(["git", "rev-parse", "HEAD"], cwd=str(repo_root))

    # Step 1 — Try to refresh review packet (best-effort)
    build_review_packet(repo_root)

    # Step 2 — Get changed files
    changed = get_changed_files(repo_root)

    if not changed:
        if args.verbose:
            print("PMR gate: no changed files detected — PASS")
        return 0

    # Step 3 — Classify
    evidence, pmr_docs, trivial = classify_files(changed)

    if args.verbose:
        print(f"PMR gate: {len(evidence)} evidence, {len(pmr_docs)} PMR docs, {len(trivial)} trivial")
        if evidence:
            print(f"  Evidence: {sorted(evidence)}")
        if pmr_docs:
            print(f"  PMR docs: {sorted(pmr_docs)}")

    # Step 4 — If no evidence-source changes, pass
    if len(evidence) < MIN_EVIDENCE_CHANGES:
        if args.verbose:
            print("PMR gate: no meaningful evidence-source changes — PASS")
        return 0

    # Step 5 — Evidence changed. Check if PMR docs were also updated.
    if pmr_docs:
        if args.verbose:
            print("PMR gate: evidence changed AND PMR docs updated — PASS")
        return 0

    # Step 6 — Check reviewer state as fallback
    if check_reviewer_state(repo_root, current_head):
        if args.verbose:
            print("PMR gate: reviewer_state.json shows HEAD was already reviewed — PASS")
        return 0

    # Step 7 — Fail: evidence changed but no PMR docs updated
    evidence_list = "\n".join(f"  - {f}" for f in sorted(evidence))
    message = REMEDIATION_MESSAGE.format(evidence_list=evidence_list)
    print(message, file=sys.stderr)

    if args.dry_run:
        print("(dry-run mode — would have blocked push)", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
