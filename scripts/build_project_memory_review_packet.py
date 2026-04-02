#!/usr/bin/env python3
"""
Build a project-memory review packet.

Gathers git status, recent commits, changed files, diff stats, and
candidate memory files into a concise markdown summary for the
Project Memory Reviewer.

Usage:
    python scripts/build_project_memory_review_packet.py

Output:
    reports/project_memory/latest_review_packet.md

Dependencies: Python 3.7+ standard library only (subprocess, pathlib, datetime).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run a shell command and return stdout, or an error message."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"[timeout running: {' '.join(cmd)}]"
    except FileNotFoundError:
        return f"[command not found: {cmd[0]}]"


def find_repo_root() -> Path:
    """Find the git repo root."""
    root = run(["git", "rev-parse", "--show-toplevel"])
    if root.startswith("["):
        print(f"Error: {root}", file=sys.stderr)
        sys.exit(1)
    return Path(root)


def compute_fingerprint(repo_root: Path) -> dict:
    """Compute a stable fingerprint of the current repo state."""
    git_head = run(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
    git_head_short = run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root))
    diff_names = run(["git", "diff", "--name-only", "HEAD"], cwd=str(repo_root))
    diff_stat = run(["git", "diff", "--stat", "HEAD"], cwd=str(repo_root))
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=str(repo_root))

    # Build a stable string from the repo state for hashing
    state_string = f"HEAD:{git_head}\nDIFF_NAMES:{diff_names}\nUNTRACKED:{untracked}"
    state_hash = hashlib.sha256(state_string.encode("utf-8")).hexdigest()[:16]

    # Build a human-readable diff signature
    diff_parts = []
    if diff_names:
        for line in diff_names.splitlines():
            diff_parts.append(line.strip())
    diff_signature = ";".join(diff_parts) if diff_parts else "(clean)"

    return {
        "git_head": git_head,
        "git_head_short": git_head_short,
        "diff_signature": diff_signature,
        "packet_fingerprint": state_hash,
    }


def build_packet(repo_root: Path) -> tuple:
    """Build the review packet markdown content and fingerprint data."""
    now = datetime.now().astimezone()
    timestamp = now.isoformat(timespec="seconds")

    fp = compute_fingerprint(repo_root)

    sections = []

    # Header
    sections.append(f"# Project Memory Review Packet")
    sections.append(f"")
    sections.append(f"Generated: {timestamp}")
    sections.append(f"Repo root: `{repo_root.name}`")
    sections.append(f"")

    # Git status
    sections.append("## Git status")
    sections.append("")
    sections.append("```")
    sections.append(run(["git", "status", "--short"], cwd=str(repo_root)) or "(clean)")
    sections.append("```")
    sections.append("")

    # Current branch
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root))
    sections.append(f"**Branch:** `{branch}`")
    sections.append("")

    # Changed tracked files (staged + unstaged)
    sections.append("## Changed tracked files")
    sections.append("")
    diff_names = run(["git", "diff", "--name-only", "HEAD"], cwd=str(repo_root))
    if diff_names:
        sections.append("```")
        sections.append(diff_names)
        sections.append("```")
    else:
        sections.append("_(no changes)_")
    sections.append("")

    # Diff stats (staged + unstaged)
    sections.append("## Diff stats")
    sections.append("")
    diff_stat = run(["git", "diff", "--stat", "HEAD"], cwd=str(repo_root))
    if diff_stat:
        sections.append("```")
        sections.append(diff_stat)
        sections.append("```")
    else:
        sections.append("_(no changes)_")
    sections.append("")

    # Staged diff stats (if different from above)
    staged_stat = run(["git", "diff", "--cached", "--stat"], cwd=str(repo_root))
    if staged_stat:
        sections.append("## Staged diff stats")
        sections.append("")
        sections.append("```")
        sections.append(staged_stat)
        sections.append("```")
        sections.append("")

    # Recent commits
    sections.append("## Recent commits (last 15)")
    sections.append("")
    sections.append("```")
    log = run(
        ["git", "log", "--oneline", "--no-decorate", "-15"],
        cwd=str(repo_root),
    )
    sections.append(log or "(no commits)")
    sections.append("```")
    sections.append("")

    # Candidate memory files
    sections.append("## Candidate memory files")
    sections.append("")
    memory_paths = [
        "docs/logs/progress_log.md",
        "docs/experiments/experiment_journal.md",
        "docs/roadmaps/phase1_structural_roadmap.md",
        "ml_finance_iv_surface_project_plan_notes.md",
        "phase1_actions_remote_workstation_plan.md",
    ]

    # Dynamically find decisions and retrospectives
    decisions_dir = repo_root / "docs" / "decisions"
    if decisions_dir.is_dir():
        for f in sorted(decisions_dir.glob("*.md")):
            rel = f.relative_to(repo_root)
            if str(rel) not in memory_paths:
                memory_paths.append(str(rel))

    retro_dir = repo_root / "docs" / "retrospectives"
    if retro_dir.is_dir():
        for f in sorted(retro_dir.glob("[0-9]*.md")):
            rel = f.relative_to(repo_root)
            if str(rel) not in memory_paths:
                memory_paths.append(str(rel))

    for p in memory_paths:
        full = repo_root / p
        status = "exists" if full.exists() else "MISSING"
        sections.append(f"- `{p}` — {status}")
    sections.append("")

    # Fingerprint
    sections.append("## Packet fingerprint")
    sections.append("")
    sections.append(f"- **git HEAD:** `{fp['git_head_short']}` (`{fp['git_head']}`)")
    sections.append(f"- **diff signature:** `{fp['diff_signature']}`")
    sections.append(f"- **packet fingerprint:** `{fp['packet_fingerprint']}`")
    sections.append("")

    # Timestamp
    sections.append("---")
    sections.append(f"_Packet built at {timestamp}_")

    return "\n".join(sections), fp, timestamp


def main():
    repo_root = find_repo_root()
    packet, fp, timestamp = build_packet(repo_root)

    output_dir = repo_root / "reports" / "project_memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest_review_packet.md"

    output_file.write_text(packet + "\n", encoding="utf-8")
    print(f"Review packet written to: {output_file}")

    # Write machine-readable sidecar
    sidecar = {
        "generated_at": timestamp,
        "git_head": fp["git_head"],
        "git_head_short": fp["git_head_short"],
        "diff_signature": fp["diff_signature"],
        "packet_fingerprint": fp["packet_fingerprint"],
    }
    sidecar_file = output_dir / "latest_review_packet_fingerprint.json"
    sidecar_file.write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fingerprint sidecar written to: {sidecar_file}")


if __name__ == "__main__":
    main()
