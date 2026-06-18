"""Tests for scripts/check_story_dependencies.py (M1.3)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_story_dependencies as cs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPEC_TEMPLATE = """\
# Story {id}

---
id: {id}
epic: TEST
status: {status}
parallel_safe_with: []
file_scope:
  - "docs/tasks/specs/{id}.md"
---

## Dependencies

{deps_block}

## Last checkpoint

### 2026-05-30T13:00:00-0400 — registered

- Last completed: spec authored.
- Next: implement.
"""


def write_spec(tmp_path: Path, story_id: str, status: str, deps: list[str]) -> Path:
    spec_dir = tmp_path / "docs" / "tasks" / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    deps_block = "\n".join(f"- {d} `done`" for d in deps) if deps else "- None."
    text = SPEC_TEMPLATE.format(id=story_id, status=status, deps_block=deps_block)
    p = spec_dir / f"{story_id}.md"
    p.write_text(text)
    return p


def write_board(tmp_path: Path, rows: dict[str, str]) -> Path:
    board_dir = tmp_path / "docs" / "tasks"
    board_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task Board\n",
        "| ID | Type | Title | Status | Spec | Updated |",
        "|---|---|---|---|---|---|",
    ]
    for sid, status in rows.items():
        lines.append(f"| {sid} | Story | t | `{status}` | s | 2026-05-30 |")
    p = board_dir / "BOARD.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# parse_spec
# ---------------------------------------------------------------------------

def test_parse_spec_extracts_id_status_deps(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2", "3X.3"])
    parsed = cs.parse_spec(spec)
    assert parsed["id"] == "M1.9"
    assert parsed["status"] == "in_progress"
    assert parsed["deps"] == ["3X.2", "3X.3"]


def test_parse_spec_ignores_prose_bullets(tmp_path):
    spec_dir = tmp_path / "docs" / "tasks" / "specs"
    spec_dir.mkdir(parents=True)
    text = (
        "# X\n\n---\nid: 3X.4\nepic: 3X\nstatus: in_review\n---\n\n"
        "## Dependencies\n\n"
        "- 3X.2 `done` (the builder)\n"
        "- Dirty strict file already on the Pod volume.\n"
    )
    p = spec_dir / "3X.4.md"
    p.write_text(text)
    parsed = cs.parse_spec(p)
    assert parsed["deps"] == ["3X.2"]


def test_parse_spec_returns_none_for_non_spec(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("# Just a readme\n")
    assert cs.parse_spec(p) is None


# ---------------------------------------------------------------------------
# parse_board
# ---------------------------------------------------------------------------

def test_parse_board_reads_status_column(tmp_path):
    b = write_board(tmp_path, {"3X.2": "in_review", "3X.4": "in_review", "M1.1": "done"})
    parsed = cs.parse_board(b)
    assert parsed == {"3X.2": "in_review", "3X.4": "in_review", "M1.1": "done"}


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_all_deps_done_passes(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = {"3X.2": "done"}
    v, a = cs.evaluate([spec], board, waivers={})
    assert v == []
    assert a == []


def test_evaluate_blocks_when_dep_not_done(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = {"3X.2": "in_review"}
    v, _ = cs.evaluate([spec], board, waivers={})
    assert len(v) == 1
    assert "3X.2" in v[0] and "in_review" in v[0]


def test_evaluate_skips_inactive_specs(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "backlog", ["3X.2"])
    board = {"3X.2": "in_review"}
    v, _ = cs.evaluate([spec], board, waivers={})
    assert v == []


def test_evaluate_waiver_passes_and_records_audit(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = {"3X.2": "in_review"}
    waivers = {("3X.2", "in_review"): "operator approved"}
    v, audits = cs.evaluate([spec], board, waivers=waivers)
    assert v == []
    assert len(audits) == 1
    assert audits[0][0] == spec
    assert "DEP WAIVER" in audits[0][1] and "3X.2" in audits[0][1]


def test_evaluate_dep_not_on_board_blocks(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.99"])
    v, _ = cs.evaluate([spec], {}, waivers={})
    assert len(v) == 1
    assert "not-on-board" in v[0]


# ---------------------------------------------------------------------------
# record_waiver_audit (M1.6: write to the untracked log, never the spec)
# ---------------------------------------------------------------------------

def test_record_waiver_audit_writes_untracked_log_not_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    before = spec.read_text()
    line = "- DEP WAIVER (2026-06-17T13:30:00-0400): 3X.2 in_review → bypassed (test)"
    log = cs.record_waiver_audit([(spec, line)])
    # The waiver lands in the untracked log...
    assert log == cs.WAIVER_LOG
    log_text = (tmp_path / cs.WAIVER_LOG).read_text()
    assert line in log_text
    assert "M1.9.md" in log_text  # the spec ref is recorded for traceability
    # ...and the tracked spec is left byte-for-byte unchanged.
    assert spec.read_text() == before


def test_record_waiver_audit_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    assert cs.record_waiver_audit([(spec, "x")], dry_run=True) is None
    assert not (tmp_path / cs.WAIVER_LOG).exists()


# ---------------------------------------------------------------------------
# parse_waiver
# ---------------------------------------------------------------------------

def test_parse_waiver_handles_multiple_entries():
    raw = "3X.2:in_review:reason one, 3X.3:in_review:reason two"
    w = cs.parse_waiver(raw)
    assert w == {
        ("3X.2", "in_review"): "reason one",
        ("3X.3", "in_review"): "reason two",
    }


def test_parse_waiver_handles_none():
    assert cs.parse_waiver(None) == {}
    assert cs.parse_waiver("") == {}


# ---------------------------------------------------------------------------
# main (CLI) — end-to-end with --files
# ---------------------------------------------------------------------------

def test_main_passes_when_no_specs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cs.main(["--files", "README.md", "--verbose"])
    assert rc == 0


def test_main_blocks_on_violation(tmp_path, monkeypatch, capsys):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = write_board(tmp_path, {"3X.2": "in_review"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WAIVE_DEPS", raising=False)
    rc = cs.main(["--files", str(spec.relative_to(tmp_path)),
                  "--board", str(board.relative_to(tmp_path))])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "3X.2" in err


def test_main_passes_with_waiver_logs_untracked_and_leaves_spec_clean(tmp_path, monkeypatch):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = write_board(tmp_path, {"3X.2": "in_review"})
    spec_before = spec.read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WAIVE_DEPS", "3X.2:in_review:operator approved")
    rc = cs.main(["--files", str(spec.relative_to(tmp_path)),
                  "--board", str(board.relative_to(tmp_path))])
    assert rc == 0
    # M1.6: the waiver is recorded in the untracked log, NOT the spec.
    assert "DEP WAIVER" in (tmp_path / cs.WAIVER_LOG).read_text()
    assert spec.read_text() == spec_before


# ---------------------------------------------------------------------------
# changed_files_from_git — commit-range detection (regression for the
# push-time no-op bug: gates must inspect <upstream>..HEAD, not just the
# working tree, because the tree is clean at push time).
# ---------------------------------------------------------------------------

def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _init_repo_with_upstream(tmp_path):
    """Create a work repo W tracking a bare origin, return W path."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    _run(["git", "init", "--bare", "-b", "main", str(origin)], cwd=tmp_path)
    _run(["git", "init", "-b", "main", str(work)], cwd=tmp_path)
    _run(["git", "config", "user.email", "t@t.t"], cwd=work)
    _run(["git", "config", "user.name", "t"], cwd=work)
    _run(["git", "remote", "add", "origin", str(origin)], cwd=work)
    (work / "seed.txt").write_text("seed\n")
    _run(["git", "add", "-A"], cwd=work)
    _run(["git", "commit", "-m", "seed"], cwd=work)
    _run(["git", "push", "-u", "origin", "main"], cwd=work)
    return work


def test_changed_files_sees_unpushed_commit(tmp_path, monkeypatch):
    work = _init_repo_with_upstream(tmp_path)
    # New commit touching a spec, NOT yet pushed and NOT in the work tree
    # as an uncommitted change.
    spec_dir = work / "docs" / "tasks" / "specs"
    spec_dir.mkdir(parents=True)
    (spec_dir / "M9.9.md").write_text("# x\n")
    _run(["git", "add", "-A"], cwd=work)
    _run(["git", "commit", "-m", "add spec"], cwd=work)

    monkeypatch.chdir(work)
    changed = {p.as_posix() for p in cs.changed_files_from_git()}
    # The committed-but-unpushed spec MUST be detected (this is the bug:
    # working tree is clean here, only the commit range carries it).
    assert "docs/tasks/specs/M9.9.md" in changed


def test_changed_files_clean_after_push_is_empty(tmp_path, monkeypatch):
    work = _init_repo_with_upstream(tmp_path)
    monkeypatch.chdir(work)
    # Everything pushed, tree clean → no changed files.
    assert cs.changed_files_from_git() == []
