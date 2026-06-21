"""Tests for scripts/eval_mae_bootstrap_ci.py (story 4A.7).

The paired date-clustered bootstrap CI must: (1) bracket a known delta;
(2) flag a clearly-better model as significant (CI entirely < 0); (3) NOT
flag two equal arms as significant; (4) be deterministic under a fixed seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import eval_mae_bootstrap_ci as ci  # noqa: E402


def _setup(n_groups=200, per=50, model_edge=0.0, seed=0):
    """Build (err_model, err_base, groups). The arms are INDEPENDENT abs-normal
    error draws (so the equal case gives a non-degenerate CI straddling 0);
    ``model_edge`` > 0 subtracts a genuine accuracy edge from the model arm."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    base = np.abs(rng.normal(0.0, 0.01, size=n_groups * per))
    model = np.clip(np.abs(rng.normal(0.0, 0.01, size=n_groups * per)) - model_edge, 0, None)
    return model, base, groups


def test_ci_brackets_known_delta():
    model, base, groups = _setup(model_edge=0.002, seed=1)
    r = ci.bootstrap_mae_delta_ci(model, base, groups, n_boot=500, seed=1)
    assert r["ci_low"] <= r["mean_delta"] <= r["ci_high"]
    # mean delta ≈ -0.002 (model better by the edge).
    assert -0.0025 < r["mean_delta"] < -0.0015


def test_clear_model_edge_is_significant():
    model, base, groups = _setup(model_edge=0.003, seed=2)
    r = ci.bootstrap_mae_delta_ci(model, base, groups, n_boot=800, seed=2)
    assert r["significant"] is True
    assert r["ci_high"] < 0.0


def test_equal_arms_not_significant():
    model, base, groups = _setup(model_edge=0.0, seed=3)
    r = ci.bootstrap_mae_delta_ci(model, base, groups, n_boot=800, seed=3)
    assert r["significant"] is False
    assert r["ci_low"] < 0.0 < r["ci_high"]


def test_deterministic_under_seed():
    model, base, groups = _setup(model_edge=0.001, seed=4)
    a = ci.bootstrap_mae_delta_ci(model, base, groups, n_boot=300, seed=7)
    b = ci.bootstrap_mae_delta_ci(model, base, groups, n_boot=300, seed=7)
    assert a == b


def test_main_writes_json(tmp_path, monkeypatch):
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.normal(0.2, 0.05, size=n)
    df = pd.DataFrame({
        "date": np.repeat(np.arange(100), n // 100),
        "iv_clean": y,
        "mu": y + rng.normal(0, 0.004, size=n),       # model close
        "rbf_pred": y + rng.normal(0, 0.006, size=n),  # baseline looser
    })
    csv = tmp_path / "preds.csv"; df.to_csv(csv, index=False)
    out = tmp_path / "ci.json"
    rc = ci.main(["--pred-csv", str(csv), "--pred-col", "mu",
                  "--baseline-col", "rbf_pred", "--target-col", "iv_clean",
                  "--group-col", "date", "--n-boot", "300", "--seed", "0",
                  "--output", str(out)])
    assert rc == 0 and out.exists()
    import json
    r = json.loads(out.read_text())
    assert "ci_low" in r and "ci_high" in r and "significant" in r
