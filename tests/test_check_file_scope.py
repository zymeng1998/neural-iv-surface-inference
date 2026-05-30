"""Tests for scripts/check_file_scope.py (M1.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_file_scope as cs  # noqa: E402


SPEC_TEMPLATE = """\
# Story {id}

---
id: {id}
epic: TEST
status: {status}
file_scope:
{scope_block}
---

## Last checkpoint

### 2026-05-30T13:00:00-0400 — x

- y
"""


def write_spec(tmp_path: Path, story_id: str, status: str, scope: list[str]) -> Path:
    spec_dir = tmp_path / "docs" / "tasks" / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    scope_block = "\n".join(f'  - "{g}"' for g in scope)
    text = SPEC_TEMPLATE.format(id=story_id, status=status, scope_block=scope_block)
    p = spec_dir / f"{story_id}.md"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# parse_spec_with_scope
# ---------------------------------------------------------------------------

def test_parse_spec_extracts_scope_globs(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", [
        "scripts/check_x.py", "tests/test_check_x.py", "docs/tasks/BOARD.md",
    ])
    parsed = cs.parse_spec_with_scope(spec)
    assert parsed["id"] == "M1.9"
    assert parsed["status"] == "in_progress"
    assert parsed["scope"] == [
        "scripts/check_x.py", "tests/test_check_x.py", "docs/tasks/BOARD.md",
    ]


# ---------------------------------------------------------------------------
# path_matches_any
# ---------------------------------------------------------------------------

def test_path_matches_exact():
    assert cs.path_matches_any("scripts/foo.py", ["scripts/foo.py"])


def test_path_matches_glob_double_star():
    assert cs.path_matches_any("artifacts/runs/3X4/manifest.json",
                               ["artifacts/runs/3X4/**"])


def test_path_does_not_match():
    assert not cs.path_matches_any("src/data/foo.py", ["scripts/*.py"])


def test_path_matches_simple_star():
    assert cs.path_matches_any("scripts/foo.py", ["scripts/*.py"])


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_no_active_specs_passes(tmp_path):
    v, a = cs.evaluate([Path("src/data/anything.py")], [], waiver_reason=None)
    assert v == [] and a == []


def test_evaluate_all_in_scope_passes(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress",
                      ["scripts/foo.py", "tests/test_foo.py"])
    parsed = cs.parse_spec_with_scope(spec)
    changed = [Path("scripts/foo.py"), Path("tests/test_foo.py")]
    v, a = cs.evaluate(changed, [parsed], waiver_reason=None)
    assert v == [] and a == []


def test_evaluate_out_of_scope_blocks(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["scripts/foo.py"])
    parsed = cs.parse_spec_with_scope(spec)
    changed = [Path("scripts/foo.py"), Path("src/data/sneaky.py")]
    v, _ = cs.evaluate(changed, [parsed], waiver_reason=None)
    assert len(v) == 1
    assert "src/data/sneaky.py" in v[0]


def test_evaluate_waiver_passes_and_records_audit(tmp_path):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["scripts/foo.py"])
    parsed = cs.parse_spec_with_scope(spec)
    changed = [Path("scripts/foo.py"), Path("src/data/sneaky.py")]
    v, audits = cs.evaluate(changed, [parsed], waiver_reason="urgent")
    assert v == []
    assert len(audits) == 1
    assert audits[0][0] == spec
    assert "SCOPE WAIVER" in audits[0][1]
    assert "src/data/sneaky.py" in audits[0][1]


def test_evaluate_scope_union_across_specs(tmp_path):
    s1 = write_spec(tmp_path, "M1.9", "in_progress", ["scripts/a.py"])
    s2 = write_spec(tmp_path, "M1.10", "in_review", ["scripts/b.py"])
    parsed = [cs.parse_spec_with_scope(s1), cs.parse_spec_with_scope(s2)]
    changed = [Path("scripts/a.py"), Path("scripts/b.py")]
    v, _ = cs.evaluate(changed, parsed, waiver_reason=None)
    assert v == []


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------

def test_main_no_active_spec_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = cs.main(["--files", "README.md"])
    assert rc == 0


def test_main_blocks_on_out_of_scope(tmp_path, monkeypatch, capsys):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["scripts/foo.py"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WAIVE_SCOPE", raising=False)
    rc = cs.main(["--files",
                  str(spec.relative_to(tmp_path)),
                  "src/data/sneaky.py", "scripts/foo.py"])
    assert rc == 1
    assert "BLOCKED" in capsys.readouterr().err


def test_main_passes_with_waiver(tmp_path, monkeypatch):
    spec = write_spec(tmp_path, "M1.9", "in_progress", ["scripts/foo.py"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WAIVE_SCOPE", "urgent fix")
    rc = cs.main(["--files",
                  str(spec.relative_to(tmp_path)),
                  "src/data/sneaky.py", "scripts/foo.py"])
    assert rc == 0
    assert "SCOPE WAIVER" in spec.read_text()
