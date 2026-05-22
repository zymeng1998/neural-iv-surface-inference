"""Tests for risk-flag synthesis + region heatmaps (Story 2B.4).

Constructed surfaces with known violation regions — no model or data files.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.diagnostics.no_arbitrage import (  # noqa: E402
    no_arb_diagnostics,
)
from neural_iv_surface_inference.diagnostics.risk_flags import (  # noqa: E402
    RiskFlagConfig,
    RiskFlagResult,
    bin_to_regions,
    derive_risk_flags,
)
from neural_iv_surface_inference.viz.diagnostic_plots import (  # noqa: E402
    plot_instability_heatmap,
    plot_risk_region_heatmap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid_surface(ks, taus, sigma_fn) -> pd.DataFrame:
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
# derive_risk_flags
# ---------------------------------------------------------------------------

class TestDeriveRiskFlags:
    def test_clean_surface_no_structural_flags(self):
        diag = no_arb_diagnostics(flat_surface())
        res = derive_risk_flags(diag)
        assert isinstance(res, RiskFlagResult)
        assert res.n_points == len(flat_surface())
        assert not res.no_arb_risk_flag.any()
        assert not res.struct_flag.any()
        np.testing.assert_array_equal(res.struct_count, 0)
        np.testing.assert_allclose(res.risk_score, 0.0)

    def test_violation_flags_stay_localized(self):
        # A single sharp vol spike at (k=0.0, tau=0.6) trips structural checks
        # near that region; points far in BOTH k and tau must stay unflagged.
        df = grid_surface(
            KS, TAUS,
            lambda k, t: 3.0 if (k == 0.0 and t == 0.6) else 0.2,
        )
        diag = no_arb_diagnostics(df)
        res = derive_risk_flags(diag)
        flag = res.no_arb_risk_flag
        assert flag.any()
        km = df["log_moneyness"].to_numpy()
        tm = df["tau"].to_numpy()
        # far corner (deep strike, short tau) is untouched by the spike
        far = (km == -0.2) & (tm == 0.1)
        assert not flag[far].any()
        far2 = (km == 0.2) & (tm == 0.1)
        assert not flag[far2].any()

    def test_instability_threshold_monotone(self):
        diag = no_arb_diagnostics(flat_surface())  # no structural flags
        n = len(flat_surface())
        instab = np.linspace(0.0, 1.0, n)
        counts = []
        for thr in (0.9, 0.5, 0.1):
            res = derive_risk_flags(
                diag, instability=instab,
                config=RiskFlagConfig(instability_threshold=thr),
            )
            counts.append(int(res.no_arb_risk_flag.sum()))
        # lower threshold -> at least as many flags
        assert counts[0] <= counts[1] <= counts[2]
        assert counts[2] > counts[0]

    def test_risk_score_combines_struct_and_instability(self):
        diag = no_arb_diagnostics(flat_surface())
        n = len(flat_surface())
        instab = np.full(n, 0.5)
        res = derive_risk_flags(
            diag, instability=instab,
            config=RiskFlagConfig(struct_weight=1.0, instability_weight=2.0),
        )
        # no structural violation -> score driven by instability only.
        # instab normalized by its max (0.5) -> 1.0; weighted by 2.0 -> 2.0
        np.testing.assert_allclose(res.risk_score, 2.0)
        assert res.meta["used_instability"] is True

    def test_nan_instability_is_safe(self):
        diag = no_arb_diagnostics(flat_surface())
        n = len(flat_surface())
        instab = np.full(n, np.nan)
        res = derive_risk_flags(
            diag, instability=instab,
            config=RiskFlagConfig(instability_threshold=0.0),
        )
        # all-nan -> no instability flags, finite score
        assert not res.no_arb_risk_flag.any()
        assert np.all(np.isfinite(res.risk_score))

    def test_length_mismatch_raises(self):
        diag = no_arb_diagnostics(flat_surface())
        with pytest.raises(ValueError):
            derive_risk_flags(diag, instability=np.zeros(3))


# ---------------------------------------------------------------------------
# bin_to_regions
# ---------------------------------------------------------------------------

class TestBinToRegions:
    def test_shape_and_labels(self):
        df = flat_surface()
        grid = bin_to_regions(
            np.ones(len(df)),
            df["log_moneyness"].to_numpy(),
            df["tau"].to_numpy(),
        )
        assert grid.shape == (3, 5)  # 3 tau buckets x 5 moneyness buckets
        assert list(grid.columns) == [
            "deep_itm", "itm", "atm", "otm", "deep_otm"
        ]
        assert list(grid.index) == ["short", "medium", "long"]

    def test_bucket_membership_matches_eval_edges(self):
        # one ATM/medium point with value 5 -> only that cell is 5, rest nan.
        df = pd.DataFrame(
            {"log_moneyness": [0.0], "tau": [0.1]}  # 0.1y ~ 36.5d -> medium
        )
        grid = bin_to_regions(
            np.array([5.0]), df["log_moneyness"].to_numpy(), df["tau"].to_numpy()
        )
        assert grid.loc["medium", "atm"] == pytest.approx(5.0)
        # all other cells nan
        other = grid.to_numpy()
        finite = np.isfinite(other)
        assert finite.sum() == 1

    def test_fraction_agg(self):
        # two ATM points, one flagged -> fraction 0.5
        df = pd.DataFrame(
            {"log_moneyness": [0.0, 0.01], "tau": [0.5, 0.5]}
        )
        grid = bin_to_regions(
            np.array([True, False]),
            df["log_moneyness"].to_numpy(),
            df["tau"].to_numpy(),
            agg="fraction",
        )
        assert grid.loc["long", "atm"] == pytest.approx(0.5)

    def test_empty_cells_are_nan(self):
        df = flat_surface()
        grid = bin_to_regions(
            np.ones(len(df)),
            df["log_moneyness"].to_numpy(),
            df["tau"].to_numpy(),
        )
        # No k < -0.2 -> deep_itm column empty; and TAUS min 0.1 > 30/365
        # so the "short" maturity row is empty too. (k=0.2 lands in deep_otm
        # since its lower edge is inclusive, so deep_otm is NOT empty.)
        assert np.isnan(grid["deep_itm"]).all()
        assert np.isnan(grid.loc["short"]).all()

    def test_invalid_agg_and_length(self):
        with pytest.raises(ValueError):
            bin_to_regions(np.ones(2), np.zeros(2), np.zeros(2), agg="bogus")
        with pytest.raises(ValueError):
            bin_to_regions(np.ones(2), np.zeros(3), np.zeros(2))


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------

class TestHeatmaps:
    def test_risk_heatmap_returns_figure(self):
        df = flat_surface()
        grid = bin_to_regions(
            np.linspace(0, 1, len(df)),
            df["log_moneyness"].to_numpy(),
            df["tau"].to_numpy(),
        )
        fig = plot_risk_region_heatmap(grid)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_instability_heatmap_returns_figure(self):
        df = flat_surface()
        grid = bin_to_regions(
            np.linspace(0, 0.05, len(df)),
            df["log_moneyness"].to_numpy(),
            df["tau"].to_numpy(),
        )
        fig = plot_instability_heatmap(grid, annotate=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
