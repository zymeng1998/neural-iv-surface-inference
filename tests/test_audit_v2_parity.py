"""Parity tests: v2 vectorised audit must match v1 on shared fixtures.

We re-use the v1 test fixtures (built inline here to keep this test file
self-contained) and assert v2's accumulators are equal to v1's:

- per-key total_rows / rows_in_dup_groups / n_dup_groups
- per-key call-put mix counts
- per-group iv_range values (≤ 1e-9 tolerance)
- benchmark leakage rows match key-by-key
- density mode counts (naive / dedup_obs / exclude_self_dup) match
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

import audit_duplicate_coordinates as v1  # noqa: E402
import audit_duplicate_coordinates_v2 as v2  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures (mirror tests/test_audit_duplicate_coordinates.py)
# ----------------------------------------------------------------------

def _strict_df() -> pd.DataFrame:
    rows = [
        dict(date="2024-01-02", expiration="2024-02-16", strike=350.0,
             type="call", log_moneyness=-0.10, tau=0.12,
             implied_volatility=0.22),
        dict(date="2024-01-02", expiration="2024-02-16", strike=400.0,
             type="call", log_moneyness=0.05, tau=0.12,
             implied_volatility=0.18),
        dict(date="2024-01-02", expiration="2024-02-16", strike=400.0,
             type="put",  log_moneyness=0.05, tau=0.12,
             implied_volatility=0.21),
        # Date A also: a same-type same-coord dup to exercise the same_type path.
        dict(date="2024-01-02", expiration="2024-02-16", strike=410.0,
             type="call", log_moneyness=0.07, tau=0.12,
             implied_volatility=0.19),
        dict(date="2024-01-02", expiration="2024-02-16", strike=410.0,
             type="call", log_moneyness=0.07, tau=0.12,
             implied_volatility=0.20),
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


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def _by_name(accs):
    return {a.key_name: a for a in accs}


def test_strict_parity(tmp_path: Path) -> None:
    p = tmp_path / "strict.parquet"
    _write_parquet(_strict_df(), p)

    v1_accs = _by_name(v1.audit_strict(p, chunk_rows=10))
    v2_accs = _by_name(v2.audit_strict(p, chunk_rows=10))

    assert set(v1_accs.keys()) == set(v2_accs.keys())
    for k, a1 in v1_accs.items():
        a2 = v2_accs[k]
        assert a2.total_rows == a1.total_rows, k
        assert a2.rows_in_dup_groups == a1.rows_in_dup_groups, k
        assert a2.n_dup_groups == a1.n_dup_groups, k
        assert a2.n_groups_call_put_mix == a1.n_groups_call_put_mix, k
        assert a2.n_groups_same_type == a1.n_groups_same_type, k
        assert dict(a2.group_size_hist) == dict(a1.group_size_hist), k
        assert sorted(a2.iv_ranges_mixed) == pytest.approx(
            sorted(a1.iv_ranges_mixed), abs=1e-9
        ), k
        assert sorted(a2.iv_ranges_same) == pytest.approx(
            sorted(a1.iv_ranges_same), abs=1e-9
        ), k


def test_benchmark_parity(tmp_path: Path) -> None:
    p = tmp_path / "bench.parquet"
    _write_parquet(_bench_df(), p)

    leak_v1, dens_v1 = v1.audit_benchmark(p, chunk_rows=10)
    leak_v2, dens_v2 = v2.audit_benchmark(p, chunk_rows=10)

    # Leakage: same keys, same field values.
    assert set(leak_v1.keys()) == set(leak_v2.keys())
    for k, r1 in leak_v1.items():
        r2 = leak_v2[k]
        assert r2.n_hidden_rows == r1.n_hidden_rows
        assert r2.n_hidden_with_obs_twin == r1.n_hidden_with_obs_twin
        assert r2.abs_err_sum == pytest.approx(r1.abs_err_sum, abs=1e-9)
        assert r2.iv_range_sum == pytest.approx(r1.iv_range_sum, abs=1e-9)

    # Density: same modes, same counts and same distance distribution.
    assert set(dens_v1.keys()) == set(dens_v2.keys())
    for k, d1 in dens_v1.items():
        d2 = dens_v2[k]
        assert d2.n_total == d1.n_total
        assert d2.n_zero_distance == d1.n_zero_distance
        assert d2.n_held_out_with_exact_obs_dup == d1.n_held_out_with_exact_obs_dup
        assert sorted(d2.distances) == pytest.approx(
            sorted(d1.distances), abs=1e-9
        )
