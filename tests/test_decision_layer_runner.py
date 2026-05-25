"""Synthetic-smoke test for the decision-layer runner (story 2D.6).

Builds a deterministic synthetic surface, a dummy ``Predictor`` that exposes
the full 2D.4 contract (``confidence_score``, ``(lower, upper)``,
``uncertainty``), and a stub ``flags_fn`` so the W2 stack is bypassed. The
runner is invoked end-to-end against a tmp dir and we assert every expected
file is written, ``comparison_summary.csv`` carries the documented columns,
and per-pair CSVs are shape-aligned with the input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from neural_iv_surface_inference.eval.decision_layer import DecisionConfig  # noqa: E402
from neural_iv_surface_inference.eval.predictor import PredictionResult  # noqa: E402

import run_decision_layer_eval as runner  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic(n_dates: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ks = np.linspace(-0.3, 0.3, 5)
    taus = np.array([7, 30, 90, 180], dtype=float) / 365.0
    grid_k, grid_t = np.meshgrid(ks, taus)
    grid_k = grid_k.ravel()
    grid_t = grid_t.ravel()
    per_date = len(grid_k)

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    frames = []
    for d in dates:
        iv_clean = np.clip(0.2 + 0.15 * grid_k**2 + 0.02 * grid_t, 0.01, 3.0)
        observed = rng.random(per_date) < 0.5
        iv_noisy = iv_clean.copy()
        iv_noisy[observed] += rng.normal(0, 0.01, size=per_date)[observed]
        frames.append(pd.DataFrame({
            "date": d,
            "tau": grid_t,
            "log_moneyness": grid_k,
            "iv_clean": iv_clean,
            "implied_volatility": np.clip(iv_noisy, 1e-4, 10.0),
            "observed": observed,
            "noise_sigma": np.full(per_date, 0.01),
        }))
    df = pd.concat(frames, ignore_index=True)
    split = np.array(["test"] * len(df), dtype=object)
    n_train = dates[int(n_dates * 0.5)]
    n_val = dates[int(n_dates * 0.75)]
    split[df["date"].to_numpy() < n_train] = "train"
    split[(df["date"].to_numpy() >= n_train) & (df["date"].to_numpy() < n_val)] = "val"
    df["split"] = split
    return df


class _DummyPredictor:
    """Deterministic predictor with the full 2D.4 contract.

    ``pred`` is the clean surface plus a tiny offset so the calibrated band
    bracketing test is non-trivial. Confidence is a smooth function of
    ``tau`` so the abstention curve is non-degenerate.
    """

    def __init__(self, noise_scale: float = 0.005) -> None:
        self.noise_scale = float(noise_scale)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        n = len(df)
        y = df["iv_clean"].to_numpy(dtype=float)
        # Deterministic seed from row count keeps the test reproducible.
        rng = np.random.default_rng(n)
        offset = rng.normal(0.0, self.noise_scale, size=n)
        mu = y + offset
        sigma = np.full(n, max(self.noise_scale, 1e-3))
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        tau = df["tau"].to_numpy(dtype=float)
        # Confidence: higher for short maturities, dipping in the wings.
        log_m = df["log_moneyness"].to_numpy(dtype=float)
        confidence = np.clip(
            0.9 - 0.5 * tau - 0.3 * np.abs(log_m), 0.05, 0.99
        )
        return PredictionResult(
            pred=mu, uncertainty=sigma, lower=lower, upper=upper,
            meta={"confidence_score": confidence},
        )


def _dummy_flags_fn(predictor, df_split, budget):
    n = len(df_split)
    # A handful of synthetic structural violations to exercise the forbidden
    # path: flag rows where |log_m| is in the top decile.
    log_m = df_split["log_moneyness"].to_numpy(dtype=float)
    threshold = np.quantile(np.abs(log_m), 0.9) if n else 0.0
    cal = np.abs(log_m) >= threshold
    conv = np.zeros(n, dtype=bool)
    mono = np.zeros(n, dtype=bool)
    return {
        "calendar_violation": cal,
        "monotonicity_violation": mono,
        "convexity_violation": conv,
        "no_arb_risk_flag": cal,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def decision_config() -> DecisionConfig:
    return DecisionConfig.from_yaml(REPO_ROOT / "configs" / "decision_layer.yaml")


def test_evaluate_pair_writes_expected_files(tmp_path, decision_config):
    df = _make_synthetic()
    df_splits = runner.split_frames(df)
    predictor = _DummyPredictor()

    pr = runner.evaluate_pair(
        predictor=predictor,
        df_splits=df_splits,
        dataset="synthetic",
        predictor_name="dummy_predictor",
        decision_config=decision_config,
        results_root=tmp_path,
        diagnostics_budget=runner.DiagnosticsBudget(n_draws=2, max_dates_per_split=None),
        flags_fn=_dummy_flags_fn,
    )

    pair_dir = tmp_path / "synthetic" / "dummy_predictor"
    assert pair_dir.is_dir()
    for fname in (
        "predictions_decisions.csv",
        "metrics_summary.csv",
        "region_tradability.csv",
        "abstention_curve.png",
        "calibration_plot.png",
    ):
        p = pair_dir / fname
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"empty artifact: {p}"

    # Per-row CSV is shape-aligned with the input.
    per_row = pd.read_csv(pair_dir / "predictions_decisions.csv")
    assert len(per_row) == len(df)
    expected_cols = {
        "dataset", "predictor", "split", "date", "log_moneyness", "tau",
        "iv_true", "iv_pred", "uncertainty", "lower", "upper",
        "confidence_score", "abstain_flag", "tradability_score",
        "decision_reason",
        "calendar_violation", "monotonicity_violation", "convexity_violation",
    }
    assert expected_cols.issubset(per_row.columns)

    metrics_df = pd.read_csv(pair_dir / "metrics_summary.csv")
    assert {"split", "n", "mae", "abstain_rate",
            "mean_tradability"}.issubset(metrics_df.columns)
    assert set(metrics_df["split"]) == set(df["split"].unique())

    # comparison_row is populated from the test split.
    assert pr.comparison_row["dataset"] == "synthetic"
    assert pr.comparison_row["predictor"] == "dummy_predictor"
    assert np.isfinite(pr.comparison_row["test_mae"])
    assert 0.0 <= pr.comparison_row["abstain_rate"] <= 1.0
    assert 0.0 <= pr.comparison_row["mean_tradability"] <= 1.0


def test_runner_writes_comparison_summary(tmp_path, decision_config):
    """Drive two synthetic pairs and assert the top-level summary parses."""
    df = _make_synthetic()
    df_splits = runner.split_frames(df)

    rows = []
    for predictor_name in ("dummy_a", "dummy_b"):
        predictor = _DummyPredictor(noise_scale=0.005 if predictor_name == "dummy_a" else 0.02)
        pr = runner.evaluate_pair(
            predictor=predictor,
            df_splits=df_splits,
            dataset="synthetic",
            predictor_name=predictor_name,
            decision_config=decision_config,
            results_root=tmp_path,
            diagnostics_budget=runner.DiagnosticsBudget(n_draws=2, max_dates_per_split=None),
            flags_fn=_dummy_flags_fn,
        )
        rows.append(pr.comparison_row)

    path = runner.write_comparison_summary(rows, tmp_path / "comparison_summary.csv")
    assert path.exists()

    summary = pd.read_csv(path)
    assert len(summary) == 2
    assert list(summary.columns) == list(runner.COMPARISON_COLUMNS)
    # Ensure runner never wrote to the canonical results/2D path.
    assert not (REPO_ROOT / "results" / "2D" / "synthetic").exists() or \
        not any((REPO_ROOT / "results" / "2D" / "synthetic").iterdir())


def test_results_root_flag_is_respected(tmp_path, decision_config):
    """`evaluate_pair` writes only under the supplied results_root."""
    df = _make_synthetic()
    df_splits = runner.split_frames(df)
    runner.evaluate_pair(
        predictor=_DummyPredictor(),
        df_splits=df_splits,
        dataset="synthetic",
        predictor_name="dummy_predictor",
        decision_config=decision_config,
        results_root=tmp_path,
        diagnostics_budget=runner.DiagnosticsBudget(n_draws=2, max_dates_per_split=None),
        flags_fn=_dummy_flags_fn,
    )
    # All files live under tmp_path.
    written = list(tmp_path.rglob("*"))
    assert written, "runner produced no files"
    for p in written:
        assert tmp_path in p.parents or p == tmp_path
