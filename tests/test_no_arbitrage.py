"""Tests for the no-arbitrage diagnostics (Story 2B.3).

Constructed single-date surfaces with known structure — no model or data files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.diagnostics.no_arbitrage import (
    ViolationResult,
    calendar_violations,
    convexity_violations,
    monotonicity_violations,
    no_arb_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid_surface(ks, taus, sigma_fn) -> pd.DataFrame:
    """Full (k, tau) grid; sigma_fn(k, tau) -> implied vol."""
    rows = []
    for t in taus:
        for k in ks:
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-02"),
                    "log_moneyness": float(k),
                    "tau": float(t),
                    "implied_volatility": float(sigma_fn(k, t)),
                }
            )
    return pd.DataFrame(rows)


KS = [-0.2, -0.1, 0.0, 0.1, 0.2]
TAUS = [0.1, 0.3, 0.6, 1.0]


def flat_surface(sigma=0.2):
    return grid_surface(KS, TAUS, lambda k, t: sigma)


# ---------------------------------------------------------------------------
# Arbitrage-free baseline
# ---------------------------------------------------------------------------

class TestArbitrageFree:
    def test_flat_surface_no_violations(self):
        df = flat_surface()
        diag = no_arb_diagnostics(df)
        for name in ("calendar", "monotonicity", "convexity"):
            assert diag[name].n_violations == 0, name
            assert diag[name].severity == pytest.approx(0.0), name
        assert diag["summary"]["total_violations"] == 0
        assert diag["summary"]["overall_rate"] == pytest.approx(0.0)

    def test_each_check_evaluates_expected_counts(self):
        df = flat_surface()
        cal = calendar_violations(df)
        mon = monotonicity_violations(df)
        con = convexity_violations(df)
        # calendar: 5 k-groups x (4 taus -> 3 pairs) = 15
        assert cal.n_evaluated == 5 * 3
        # monotonicity: 4 tau-groups x (5 ks -> 4 pairs) = 16
        assert mon.n_evaluated == 4 * 4
        # convexity: 4 tau-groups x (5 ks -> 3 interior triples) = 12
        assert con.n_evaluated == 4 * 3


# ---------------------------------------------------------------------------
# Injected violations
# ---------------------------------------------------------------------------

class TestCalendar:
    def test_inverted_term_structure_detected(self):
        # total variance drops with tau: sigma falls fast enough.
        # w = sigma^2 * tau; pick sigma so w decreases.
        sig = {0.1: 0.6, 0.3: 0.3, 0.6: 0.15, 1.0: 0.1}
        df = grid_surface(KS, TAUS, lambda k, t: sig[t])
        res = calendar_violations(df)
        assert isinstance(res, ViolationResult)
        # every k-group, every consecutive tau-pair inverts
        assert res.n_violations == 5 * 3
        assert res.severity > 0.0
        assert res.rate == pytest.approx(1.0)

    def test_severity_is_magnitude_of_drop(self):
        # one k, two taus; w1=0.6^2*0.1=0.036, w2=0.1^2*1.0=0.01 -> drop 0.026
        df = grid_surface([0.0], [0.1, 1.0], lambda k, t: 0.6 if t == 0.1 else 0.1)
        res = calendar_violations(df)
        assert res.n_violations == 1
        assert res.severity == pytest.approx(0.036 - 0.01, abs=1e-9)


class TestMonotonicity:
    def test_increasing_price_in_strike_detected(self):
        # vol explodes with strike -> far-OTM call price rises with strike.
        df = grid_surface(KS, [0.5], lambda k, t: 0.1 + 5.0 * max(k, 0.0))
        res = monotonicity_violations(df)
        assert res.n_violations > 0
        assert res.severity > 0.0

    def test_flat_is_monotone(self):
        df = grid_surface(KS, [0.5], lambda k, t: 0.2)
        res = monotonicity_violations(df)
        assert res.n_violations == 0


class TestConvexity:
    def test_spike_breaks_convexity(self):
        # sharp vol spike at the central strike inflates its call price,
        # producing a concave (butterfly-arbitrage) dip.
        df = grid_surface(KS, [0.5], lambda k, t: 3.0 if k == 0.0 else 0.15)
        res = convexity_violations(df)
        assert res.n_violations > 0
        assert res.severity > 0.0

    def test_flat_is_convex(self):
        df = grid_surface(KS, [0.5], lambda k, t: 0.2)
        res = convexity_violations(df)
        assert res.n_violations == 0


# ---------------------------------------------------------------------------
# Sparse axes / NaN handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_sparse_axis_yields_no_evaluations(self):
        # one point per tau and per k -> no group can form a pair/triple.
        df = grid_surface([0.0], [0.5], lambda k, t: 0.2)
        cal = calendar_violations(df)
        mon = monotonicity_violations(df)
        con = convexity_violations(df)
        assert cal.n_evaluated == 0 and np.isnan(cal.rate)
        assert mon.n_evaluated == 0 and np.isnan(mon.rate)
        assert con.n_evaluated == 0 and np.isnan(con.rate)
        assert cal.severity == 0.0

    def test_nan_iv_rows_dropped(self):
        df = flat_surface()
        df.loc[df.index[0], "implied_volatility"] = np.nan
        diag = no_arb_diagnostics(df)
        # dropping one point reduces evaluated counts but stays violation-free
        assert diag["calendar"].n_violations == 0
        assert diag["monotonicity"].n_violations == 0
        assert diag["convexity"].n_violations == 0

    def test_custom_iv_col(self):
        df = flat_surface()
        df = df.rename(columns={"implied_volatility": "iv_pred"})
        res = calendar_violations(df, iv_col="iv_pred")
        assert res.n_violations == 0
        assert res.meta["iv_col"] == "iv_pred"


class TestAggregate:
    def test_summary_keys_and_overall_rate(self):
        df = flat_surface()
        diag = no_arb_diagnostics(df)
        s = diag["summary"]
        assert set(s) == {
            "total_evaluated",
            "total_violations",
            "overall_rate",
            "total_severity",
            "per_check_severity",
        }
        assert s["total_evaluated"] == 15 + 16 + 12
        assert s["total_severity"] == pytest.approx(0.0)

    def test_empty_surface_overall_rate_nan(self):
        df = grid_surface([0.0], [0.5], lambda k, t: 0.2)
        diag = no_arb_diagnostics(df)
        assert diag["summary"]["total_evaluated"] == 0
        assert np.isnan(diag["summary"]["overall_rate"])
