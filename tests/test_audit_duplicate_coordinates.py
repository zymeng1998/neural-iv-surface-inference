"""Lightweight smoke tests for ``scripts/audit_duplicate_coordinates.py``.

We don't have local SPY data — these tests verify the audit logic on a
hand-built parquet with a known duplicate structure, so a regression in
the streaming / grouping / leakage maths is caught locally without Pod
data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_duplicate_coordinates as audit  # noqa: E402


def _strict_df() -> pd.DataFrame:
    """Two dates. Date A: one call-put dup at (K=400, exp=t1). Date B: clean."""
    rows = [
        # Date A: clean call
        dict(date="2024-01-02", expiration="2024-02-16", strike=350.0,
             type="call", log_moneyness=-0.10, tau=0.12,
             implied_volatility=0.22),
        # Date A: a call AND a put at the same contract → duplicate.
        dict(date="2024-01-02", expiration="2024-02-16", strike=400.0,
             type="call", log_moneyness=0.05, tau=0.12,
             implied_volatility=0.18),
        dict(date="2024-01-02", expiration="2024-02-16", strike=400.0,
             type="put",  log_moneyness=0.05, tau=0.12,
             implied_volatility=0.21),
        # Date B: no duplicates
        dict(date="2024-01-03", expiration="2024-02-16", strike=380.0,
             type="call", log_moneyness=0.00, tau=0.115,
             implied_volatility=0.20),
        dict(date="2024-01-03", expiration="2024-02-16", strike=420.0,
             type="put",  log_moneyness=0.10, tau=0.115,
             implied_volatility=0.24),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _bench_df() -> pd.DataFrame:
    """One date with an observed call and a held-out put at the same coord."""
    rows = [
        dict(date="2024-01-02", log_moneyness=0.05, tau=0.12,
             implied_volatility=0.18, iv_clean=0.18,
             observed=True, split="test"),
        dict(date="2024-01-02", log_moneyness=0.05, tau=0.12,
             implied_volatility=0.21, iv_clean=0.21,
             observed=False, split="test"),
        dict(date="2024-01-02", log_moneyness=-0.10, tau=0.12,
             implied_volatility=0.22, iv_clean=0.22,
             observed=True, split="test"),
        dict(date="2024-01-02", log_moneyness=-0.30, tau=0.12,
             implied_volatility=0.30, iv_clean=0.30,
             observed=False, split="test"),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def test_strict_audit_counts_call_put_duplicate(tmp_path: Path) -> None:
    strict_path = tmp_path / "strict.parquet"
    _write_parquet(_strict_df(), strict_path)

    accs = audit.audit_strict(strict_path, chunk_rows=10)
    by_name = {a.key_name: a for a in accs}
    contract = by_name["date+expiration+strike"]

    assert contract.total_rows == 5
    assert contract.n_dup_groups == 1
    assert contract.rows_in_dup_groups == 2
    assert contract.n_groups_call_put_mix == 1
    assert contract.n_groups_same_type == 0
    # iv_range for the one mixed group is |0.21 − 0.18| = 0.03.
    assert contract.iv_ranges_mixed == pytest.approx([0.03], rel=1e-6)


def test_strict_audit_coordinate_key_matches_contract_for_this_fixture(
    tmp_path: Path,
) -> None:
    strict_path = tmp_path / "strict.parquet"
    _write_parquet(_strict_df(), strict_path)

    accs = audit.audit_strict(strict_path, chunk_rows=10)
    coord_10 = next(a for a in accs if "round(lm,10)" in a.key_name)
    # Same call+put pair shares (log_m, tau) → exactly one dup group at every
    # rounding tolerance.
    assert coord_10.n_dup_groups == 1
    assert coord_10.rows_in_dup_groups == 2


def test_benchmark_leakage_detects_exact_observed_twin(tmp_path: Path) -> None:
    bench_path = tmp_path / "bench.parquet"
    _write_parquet(_bench_df(), bench_path)

    leak_rows, density = audit.audit_benchmark(bench_path, chunk_rows=10)

    # Only one hidden row sits at an observed twin's coord (the put leg).
    twinned = [r for r in leak_rows.values() if r.n_hidden_with_obs_twin > 0]
    untwinned = [r for r in leak_rows.values() if r.n_hidden_with_obs_twin == 0]
    assert len(twinned) >= 1
    # At least one hidden row had no twin (the deep-put-wing one).
    assert any(r.n_hidden_rows > 0 for r in untwinned)

    # Density audit: with naive nearest-distance the twin hidden row has
    # distance 0 (it shares coord with observed). With exclude_self_dup it
    # should rise above 0.
    by_mode = {acc.mode: acc for acc in density.values()}
    assert by_mode["naive"].n_zero_distance >= 1
    assert by_mode["exclude_self_dup"].n_zero_distance < by_mode["naive"].n_zero_distance
