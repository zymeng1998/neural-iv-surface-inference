"""Tests for scripts/check_commit_trailer.py (M1.5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_commit_trailer as ct  # noqa: E402


@pytest.fixture(autouse=True)
def clear_agent_env(monkeypatch):
    for name in list(ct.AGENT_ENV_EXACT):
        monkeypatch.delenv(name, raising=False)
    for name in [k for k in list(__import__("os").environ)
                 if any(k.startswith(p) for p in ct.AGENT_ENV_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WAIVE_TRAILER", raising=False)


def write_msg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "COMMIT_EDITMSG"
    p.write_text(body, encoding="utf-8")
    return p


def test_human_commit_no_trailer_passes(tmp_path, monkeypatch):
    msg = write_msg(tmp_path, "feat: human change\n\nSome body.\n")
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 0


def test_agent_commit_with_trailer_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    msg = write_msg(
        tmp_path,
        "feat: M1 bootstrap\n\nbody\n\n"
        "Co-authored-by: Claude <noreply@anthropic.com>\n",
    )
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 0


def test_agent_commit_without_trailer_blocks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CURSOR_TRACE_ID", "abc-123")
    msg = write_msg(tmp_path, "feat: cursor change\n\nbody\n")
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err
    assert "CURSOR_TRACE_ID" in err


def test_agent_commit_waiver_passes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("WAIVE_TRAILER", "rebase fixup")
    msg = write_msg(tmp_path, "feat: x\n")
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 0
    assert "WAIVED" in capsys.readouterr().err


def test_codex_prefix_env_triggers(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_SESSION_ID", "xyz")
    msg = write_msg(tmp_path, "feat: codex change\n")
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 1


def test_comment_lines_ignored(tmp_path, monkeypatch):
    """A `# Co-authored-by:` line in git comments must NOT count."""
    monkeypatch.setenv("CLAUDECODE", "1")
    msg = write_msg(
        tmp_path,
        "feat: x\n\nbody\n\n# Co-authored-by: Claude <fake@anthropic.com>\n",
    )
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 1


def test_trailer_regex_requires_email(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    msg = write_msg(tmp_path, "feat: x\n\nCo-authored-by: Claude only\n")
    rc = ct.main(["check_commit_trailer.py", str(msg)])
    assert rc == 1


def test_missing_msg_file_errors(tmp_path, capsys):
    rc = ct.main(["check_commit_trailer.py", str(tmp_path / "nope")])
    assert rc == 1
