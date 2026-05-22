"""End-to-end tests for the W2 structure-diagnostics runner (Story 2B.5).

Exercises diagnostics/report.py table assembly and the
scripts/run_structure_diagnostics.py runner on a tiny synthetic grid benchmark —
no model training, no data files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_iv_surface_inference.eval.adapters import InterpolationPredictor
from neural_iv_surface_inference.diagnostics.report import (
    REGION_COLUMNS,
    SUMMARY_COLUMNS,
    diagnose_date,
    region_table,
)

import run_structure_diagnostics as runner


EXPECTED_SUMMARY_COLUMNS = set(SUMMARY_COLUMNS)
EXPECTED_REGION_COLUMNS = set(REGION_COLUMNS)


@pytest.fixture(scope="module")
def predictor():
    return InterpolationPredictor(method="rbf")


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

class TestSyntheticBenchmark:
    def test_standard_columns_and_grid(self):
        df = runner.make_synthetic_benchmark(n_dates=4)
        for col in ("date", "tau", "log_moneyness", "iv_clean",
                    "implied_volatility", "observed", "split", "noise_sigma"):
            assert col in df.columns
        # shared grid per date -> repeated (k, tau) coordinates across dates
        per_date = df[df["date"] == df["date"].iloc[0]]
        assert per_date["log_moneyness"].nunique() == 7
        assert per_date["tau"].nunique() == 5

    def test_chronological_splits(self):
        df = runner.make_synthetic_benchmark(n_dates=10)
        # train dates strictly precede val precede test
        for a, b in (("train", "val"), ("val", "test")):
            if (df["split"] == a).any() and (df["split"] == b).any():
                assert df[df["split"] == a]["date"].max() < \
                       df[df["split"] == b]["date"].min()


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

class TestReport:
    def test_diagnose_date_returns_aligned_outputs(self, predictor):
        df = runner.make_synthetic_benchmark(n_dates=2)
        df_date = df[df["date"] == df["date"].iloc[0]].reset_index(drop=True)
        instab, diag, flags, pred = diagnose_date(predictor, df_date, n_draws=5)
        n = len(df_date)
        assert len(instab) == n
        assert flags.n_points == n
        assert len(pred) == n
        assert set(diag) >= {"calendar", "monotonicity", "convexity", "summary"}

    def test_region_table_shape_and_columns(self, predictor):
        df = runner.make_synthetic_benchmark(n_dates=2)
        df_date = df[df["date"] == df["date"].iloc[0]].reset_index(drop=True)
        instab, diag, flags, _ = diagnose_date(predictor, df_date, n_draws=5)
        tbl = region_table(
            "interp_rbf", "train", flags.risk_score, instab.std,
            flags.no_arb_risk_flag,
            df_date["log_moneyness"].to_numpy(), df_date["tau"].to_numpy(),
        )
        assert set(tbl.columns) == EXPECTED_REGION_COLUMNS
        assert len(tbl) == 3 * 5  # tau buckets x moneyness buckets
        assert tbl["n_points"].sum() == len(df_date)


# ---------------------------------------------------------------------------
# run_diagnostics
# ---------------------------------------------------------------------------

class TestRunDiagnostics:
    def test_summary_and_region_tables(self, predictor):
        df = runner.make_synthetic_benchmark(n_dates=6)
        df_splits = runner.split_frames(df)
        summary, regions = runner.run_diagnostics(
            predictor, df_splits, model_name="interp_rbf", n_draws=5,
        )
        assert set(summary.columns) == EXPECTED_SUMMARY_COLUMNS
        assert set(regions.columns) == EXPECTED_REGION_COLUMNS
        # arbitrage-free smooth grid -> no structural violations
        assert summary["calendar_violations"].sum() == 0
        assert summary["monotonicity_violations"].sum() == 0
        assert summary["convexity_violations"].sum() == 0
        # masking still produces positive instability
        assert summary["mean_instability"].max() > 0
        # one summary row per (split, date)
        assert len(summary) == df["date"].nunique()


# ---------------------------------------------------------------------------
# write_artifacts (end-to-end)
# ---------------------------------------------------------------------------

class TestWriteArtifacts:
    def test_full_path_creates_artifacts(self, predictor, tmp_path):
        df = runner.make_synthetic_benchmark(n_dates=6)
        df_splits = runner.split_frames(df)
        summary, regions = runner.run_diagnostics(
            predictor, df_splits, model_name="interp_rbf", n_draws=5,
        )
        paths = runner.write_artifacts(summary, regions, tmp_path, "test_demo")

        assert paths["summary"].exists()
        assert paths["regions"].exists()
        # CSVs reload with the documented columns
        s = pd.read_csv(paths["summary"])
        r = pd.read_csv(paths["regions"])
        assert set(s.columns) == EXPECTED_SUMMARY_COLUMNS
        assert set(r.columns) == EXPECTED_REGION_COLUMNS
        # at least one region heatmap per split present
        png = list(tmp_path.glob("structure_diagnostics_test_demo_*risk.png"))
        assert len(png) >= 1
