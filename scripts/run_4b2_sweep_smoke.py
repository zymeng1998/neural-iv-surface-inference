#!/usr/bin/env python3
"""Synthetic smoke for the Phase 4B sparsity-sweep harness (story 4B.2).

Builds a tiny synthetic single-date surface, runs all four sparsity regimes,
checks the harness invariants (fixed query, monotone nesting, dense rung ==
full context), and writes a JSON manifest to ``artifacts/runs/4B2/``. No
benchmark data, no RBF, no model, no GPU — this only exercises the construction.

Usage
-----
    PYTHONPATH=src python3 scripts/run_4b2_sweep_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neural_iv_surface_inference.diagnostics.sparsity_sweep import (
    REGIMES,
    build_all_regimes,
    sweep_manifest,
)

OUT_DIR = Path("artifacts/runs/4B2")
LADDER = (0.0, 0.25, 0.5, 0.75)
SEED = 20260621


def _synthetic_date(seed: int = 0) -> pd.DataFrame:
    """One date: a 9-moneyness x 5-maturity grid, ~70% observed (context)."""
    rng = np.random.default_rng(seed)
    moneyness = np.linspace(-0.4, 0.4, 9)
    taus = np.array([0.05, 0.15, 0.35, 0.6, 1.0])
    rows = [
        (exp_id, float(tau), float(k))
        for exp_id, tau in enumerate(taus)
        for k in moneyness
    ]
    df = pd.DataFrame(rows, columns=["expiration", "tau", "log_moneyness"])
    df["date"] = pd.Timestamp("2024-01-02")
    df["iv_true"] = 0.2 + 0.5 * df["log_moneyness"] ** 2 + 0.1 * df["tau"]
    df["observed"] = rng.random(len(df)) < 0.7
    return df


def main() -> int:
    df = _synthetic_date()
    base_observed = df["observed"].to_numpy()
    query = ~base_observed

    sweeps = build_all_regimes(df, {r: LADDER for r in REGIMES}, seed=SEED)

    # Invariant checks (belt-and-braces alongside the unit tests).
    for regime, sweep in sweeps.items():
        assert np.array_equal(sweep.query_mask, query), regime
        np.testing.assert_array_equal(sweep.rungs[0].retained_observed, base_observed)
        counts = [int(r.retained_observed.sum()) for r in sweep.rungs]
        assert counts == sorted(counts, reverse=True), (regime, counts)
        for prev, cur in zip(sweep.rungs, sweep.rungs[1:]):
            assert np.all(cur.retained_observed <= prev.retained_observed), regime

    manifest = sweep_manifest(sweeps)
    manifest["smoke"] = {
        "seed": SEED,
        "ladder": list(LADDER),
        "n_points": int(len(df)),
        "n_observed": int(base_observed.sum()),
        "n_query": int(query.sum()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    print(f"[4B.2 smoke] wrote {out}")
    for regime in REGIMES:
        rec = manifest["regimes"][regime]
        retained = [r["n_retained"] for r in rec["rungs"]]
        print(
            f"  {regime:>22}: observed={rec['n_base_observed']} "
            f"query={rec['n_query']} retained@rungs={retained}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
