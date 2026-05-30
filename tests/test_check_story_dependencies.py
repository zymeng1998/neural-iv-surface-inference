"""Tests for scripts/check_story_dependencies.py (M1.3)."""

from __future__ import annotations

import os
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
# append_audit_line
# ---------------------------------------------------------------------------

def test_append_audit_line_lands_under_last_checkpoint(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    line = "- DEP WAIVER (2026-05-30T13:30:00-0400): 3X.2 in_review → bypassed (test)"
    cs.append_audit_line(spec, line)
    text = spec.read_text()
    assert line in text
    # Audit line must appear after the "### ... — registered" heading.
    assert text.index("### 2026-05-30T13:00:00-0400") < text.index(line)


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


def test_main_passes_with_waiver_and_writes_audit(tmp_path, monkeypatch):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["3X.2"])
    board = write_board(tmp_path, {"3X.2": "in_review"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WAIVE_DEPS", "3X.2:in_review:operator approved")
    rc = cs.main(["--files", str(spec.relative_to(tmp_path)),
                  "--board", str(board.relative_to(tmp_path))])
    assert rc == 0
    assert "DEP WAIVER" in spec.read_text()
