"""End-to-end tests for the uncertainty-evaluation runner (Story 2A.5).

Exercises report.py table assembly and the scripts/run_uncertainty_eval.py
runner on a tiny synthetic benchmark — no model training, no data files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_iv_surface_inference.eval.predictor import PredictionResult
from neural_iv_surface_inference.eval.adapters import InterpolationPredictor
from neural_iv_surface_inference.eval.report import (
    metrics_row,
    metrics_table,
    risk_coverage_table,
    confidence_from_result,
)

import run_uncertainty_eval as runner


EXPECTED_METRIC_COLUMNS = {
    "model", "split", "n",
    "overall_mae", "overall_rmse", "observed_mae", "unobserved_mae",
    "interval_coverage", "mean_interval_width",
    "err_unc_pearson", "err_unc_spearman",
    "confidence_source", "aurc", "hc_mae_keep0.5", "hc_mae_keep0.8",
}
EXPECTED_CURVE_COLUMNS = {
    "model", "split", "confidence_source", "coverage", "retained_mae",
    "n_retained",
}


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

class TestReport:
    def test_confidence_oracle_when_no_uncertainty(self):
        ae = np.array([0.1, 0.5, 0.2])
        result = PredictionResult(pred=np.zeros(3))
        conf, source = confidence_from_result(result, ae)
        assert source == "oracle_error"
        np.testing.assert_allclose(conf, -ae)

    def test_confidence_uses_uncertainty_when_present(self):
        unc = np.array([0.3, 0.1, 0.9])
        result = PredictionResult(pred=np.zeros(3), uncertainty=unc)
        conf, source = confidence_from_result(result, np.zeros(3))
        assert source == "uncertainty"
        np.testing.assert_allclose(conf, -unc)

    def test_metrics_row_columns_and_nan_calibration(self):
        n = 50
        rng = np.random.default_rng(0)
        y = rng.uniform(0.1, 0.4, n)
        pred = y + rng.normal(0, 0.01, n)
        observed = rng.random(n) < 0.5
        result = PredictionResult(pred=pred, meta={"model": "x"})
        row = metrics_row(result, y, observed, model_name="x", split="test")
        assert EXPECTED_METRIC_COLUMNS.issubset(row.keys())
        # no uncertainty / bounds → calibration + correlation are NaN
        assert np.isnan(row["interval_coverage"])
        assert np.isnan(row["mean_interval_width"])
        assert np.isnan(row["err_unc_pearson"])
        assert row["confidence_source"] == "oracle_error"
        # oracle abstention retains lower error than the full set
        assert row["hc_mae_keep0.5"] <= row["overall_mae"] + 1e-12

    def test_metrics_row_with_intervals_reports_coverage(self):
        n = 1000
        rng = np.random.default_rng(1)
        y = rng.standard_normal(n)
        pred = np.zeros(n)
        lower = np.full(n, -1.6449)
        upper = np.full(n, 1.6449)
        unc = np.abs(pred - y)  # perfectly informative
        result = PredictionResult(pred=pred, uncertainty=unc,
                                  lower=lower, upper=upper)
        row = metrics_row(result, y, np.ones(n, dtype=bool),
                          model_name="g", split="test")
        assert row["interval_coverage"] == pytest.approx(0.90, abs=0.03)
        assert row["mean_interval_width"] == pytest.approx(2 * 1.6449, abs=1e-6)
        assert row["confidence_source"] == "uncertainty"

    def test_risk_coverage_table_shape(self):
        n = 200
        rng = np.random.default_rng(2)
        y = rng.uniform(0.1, 0.4, n)
        result = PredictionResult(pred=y + rng.normal(0, 0.02, n))
        tbl = risk_coverage_table(result, y, model_name="x", split="test",
                                  n_points=10)
        assert EXPECTED_CURVE_COLUMNS == set(tbl.columns)
        assert tbl["coverage"].iloc[-1] == pytest.approx(1.0)
        assert (tbl["coverage"].diff().dropna() > 0).all()

    def test_metrics_table_stacks_rows(self):
        rows = [{"model": "a", "split": "test", "overall_mae": 0.1},
                {"model": "a", "split": "val", "overall_mae": 0.2}]
        df = metrics_table(rows)
        assert len(df) == 2
        assert list(df["split"]) == ["test", "val"]


# ---------------------------------------------------------------------------
# runner helpers
# ---------------------------------------------------------------------------

class TestRunnerHelpers:
    def test_synthetic_benchmark_has_required_columns(self):
        df = runner.make_synthetic_benchmark(n_dates=10, n_points=30, seed=0)
        for col in ("date", "tau", "log_moneyness", "iv_clean",
                    "implied_volatility", "observed", "split"):
            assert col in df.columns
        assert set(df["split"].unique()) <= {"train", "val", "test"}

    def test_split_frames_partitions(self):
        df = runner.make_synthetic_benchmark(n_dates=10, n_points=30, seed=0)
        parts = runner.split_frames(df)
        assert sum(len(v) for v in parts.values()) == len(df)

    def test_evaluate_predictor_returns_tables(self):
        df = runner.make_synthetic_benchmark(n_dates=10, n_points=40, seed=1)
        parts = runner.split_frames(df)
        predictor = InterpolationPredictor(method="linear")
        metrics_df, curve_df = runner.evaluate_predictor(
            predictor, parts, model_name="interp_linear", n_curve_points=20,
        )
        assert EXPECTED_METRIC_COLUMNS.issubset(set(metrics_df.columns))
        assert EXPECTED_CURVE_COLUMNS == set(curve_df.columns)
        assert set(metrics_df["split"]) <= {"train", "val", "test"}
        assert (metrics_df["overall_mae"] >= 0).all()


# ---------------------------------------------------------------------------
# end-to-end artifact write
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_write_artifacts_creates_files(self, tmp_path):
        df = runner.make_synthetic_benchmark(n_dates=12, n_points=40, seed=3)
        parts = runner.split_frames(df)
        predictor = InterpolationPredictor(method="linear")
        metrics_df, curve_df = runner.evaluate_predictor(
            predictor, parts, model_name="interp_linear", n_curve_points=20,
        )
        paths = runner.write_artifacts(metrics_df, curve_df, tmp_path, "t")

        assert paths["metrics"].exists()
        assert paths["curve"].exists()
        assert paths["figure"].exists()
        assert paths["figure"].stat().st_size > 0  # non-empty PNG

        reloaded = pd.read_csv(paths["metrics"])
        assert EXPECTED_METRIC_COLUMNS.issubset(set(reloaded.columns))
        # interpolation runs the full interface → metrics → artifacts path
        assert (reloaded["overall_mae"] >= 0).all()
        assert reloaded["confidence_source"].eq("oracle_error").all()

    def test_build_predictor_interp(self):
        predictor, name = runner.build_predictor("interp", "rbf")
        assert name == "interp_rbf"
        assert hasattr(predictor, "predict")

    def test_build_predictor_rejects_unknown(self):
        with pytest.raises(ValueError):
            runner.build_predictor("mlp", "rbf")
