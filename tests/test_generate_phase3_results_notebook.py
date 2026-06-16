"""Smoke tests for scripts/generate_phase3_results_notebook.py (story 3D.2).

These assert the generator builds a structurally valid notebook from the
committed Phase 3 bundles, references only paths that exist, and is
idempotent. They do NOT execute the notebook — that is 3D.4's job
(`jupyter nbconvert --execute`)."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import generate_phase3_results_notebook as gen  # noqa: E402


def test_referenced_paths_nonempty():
    paths = gen.referenced_paths()
    assert paths, "generator must declare its referenced bundle paths"
    # No prediction dumps (gitignored) should be hard-required.
    assert not any(p.endswith(".parquet") for p in paths)


def test_all_referenced_paths_exist():
    """Every committed artifact the notebook depends on must be present."""
    missing = gen.validate_paths(REPO)
    assert missing == [], f"missing committed artifacts: {missing}"


def test_build_notebook_is_structurally_valid():
    nb = gen.build_notebook()
    assert len(nb.cells) > 0
    # Raises nbformat.ValidationError on a malformed notebook.
    nbf.validate(nb)
    # Sanity: a mix of markdown and code cells.
    kinds = {c.cell_type for c in nb.cells}
    assert {"markdown", "code"} <= kinds


def test_build_is_idempotent():
    a = nbf.writes(gen.build_notebook())
    b = nbf.writes(gen.build_notebook())
    assert a == b, "two builds must produce byte-identical notebook JSON"


def test_main_writes_to_tmp_and_exits_zero(tmp_path):
    out = tmp_path / "_phase3_check.ipynb"
    rc = gen.main(["--output", str(out)])
    assert rc == 0
    assert out.exists()
    nb = nbf.read(str(out), as_version=4)
    assert len(nb.cells) > 0
