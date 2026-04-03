"""Tests for task construction modules: masking, noise, splits.

These tests use synthetic data — no real market data required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.data.masking import (
    random_uniform_mask,
    drop_maturity_mask,
    drop_moneyness_mask,
    realistic_liquidity_mask,
    apply_mask,
)
from neural_iv_surface_inference.data.noise import (
    add_gaussian_noise,
    add_heteroscedastic_noise,
    inject_noise,
    NOISE_REGIMES,
)
from neural_iv_surface_inference.data.splits import (
    time_split,
    split_summary,
    benchmark_name,
    save_benchmark,
    load_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_surface_df():
    """Create a synthetic surface-point DataFrame resembling Step 3 output."""
    rng = np.random.default_rng(0)
    n_dates = 50
    n_points_per_date = 200
    n = n_dates * n_points_per_date

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    date_col = np.repeat(dates, n_points_per_date)

    tau = rng.uniform(1 / 365, 2.0, size=n)
    log_moneyness = rng.uniform(-0.5, 0.5, size=n)
    iv = 0.2 + 0.1 * np.abs(log_moneyness) + rng.normal(0, 0.01, size=n)
    iv = np.clip(iv, 0.01, 3.0)

    return pd.DataFrame({
        "date": date_col,
        "tau": tau,
        "log_moneyness": log_moneyness,
        "implied_volatility": iv,
        "strike": 100 * np.exp(log_moneyness),
        "spot": np.full(n, 100.0),
        "bid": iv * 100 * 0.95,
        "ask": iv * 100 * 1.05,
        "mid": iv * 100,
    })


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Masking tests
# ---------------------------------------------------------------------------

class TestRandomUniformMask:
    def test_returns_correct_length(self, rng):
        mask = random_uniform_mask(1000, 0.5, rng)
        assert len(mask) == 1000
        assert mask.dtype == bool

    def test_approximate_fraction(self, rng):
        mask = random_uniform_mask(10000, 0.3, rng)
        frac = mask.mean()
        assert 0.25 < frac < 0.35, f"Expected ~0.3, got {frac}"

    def test_keep_frac_1_keeps_all(self, rng):
        mask = random_uniform_mask(100, 1.0, rng)
        assert mask.all()

    def test_invalid_keep_frac_raises(self, rng):
        with pytest.raises(ValueError):
            random_uniform_mask(100, 0.0, rng)
        with pytest.raises(ValueError):
            random_uniform_mask(100, 1.5, rng)

    def test_reproducibility(self):
        m1 = random_uniform_mask(100, 0.5, np.random.default_rng(99))
        m2 = random_uniform_mask(100, 0.5, np.random.default_rng(99))
        assert np.array_equal(m1, m2)


class TestDropMaturityMask:
    def test_drop_short(self, rng):
        tau = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
        mask = drop_maturity_mask(tau, "short", rng, tau_short_threshold=0.1)
        # tau < 0.1 should be dropped
        assert not mask[0]
        assert not mask[1]
        assert mask[3]  # 0.5 is kept

    def test_drop_long(self, rng):
        tau = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        mask = drop_maturity_mask(tau, "long", rng, tau_long_threshold=1.0)
        assert mask[0]
        assert mask[1]
        assert mask[2]  # 1.0 is not > 1.0
        assert not mask[3]
        assert not mask[4]

    def test_drop_random_slice(self, rng):
        tau = np.linspace(0.01, 2.0, 1000)
        mask = drop_maturity_mask(tau, "random_slice", rng)
        # Should drop roughly 10% (one of 10 buckets)
        drop_frac = 1 - mask.mean()
        assert 0.05 < drop_frac < 0.25

    def test_unknown_region_raises(self, rng):
        with pytest.raises(ValueError):
            drop_maturity_mask(np.array([0.5]), "invalid", rng)


class TestDropMoneynessMask:
    def test_drop_wings(self, rng):
        lm = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
        mask = drop_moneyness_mask(lm, "wings", rng, wing_threshold=0.3)
        assert not mask[0]  # -0.5 is wing
        assert mask[1]      # -0.1 is not
        assert mask[2]
        assert mask[3]
        assert not mask[4]  # 0.5 is wing

    def test_drop_atm(self, rng):
        lm = np.array([-0.5, -0.02, 0.0, 0.02, 0.5])
        mask = drop_moneyness_mask(lm, "atm", rng, atm_threshold=0.05)
        assert mask[0]
        assert not mask[1]
        assert not mask[2]
        assert not mask[3]
        assert mask[4]


class TestRealisticLiquidityMask:
    def test_atm_more_observed_than_wings(self, rng):
        n = 5000
        lm = np.concatenate([
            np.full(n, 0.0),     # ATM
            np.full(n, 0.5),     # OTM wing
        ])
        tau = np.full(2 * n, 60 / 365)  # ~60 days
        mask = realistic_liquidity_mask(lm, tau, 0.4, rng)
        atm_frac = mask[:n].mean()
        wing_frac = mask[n:].mean()
        assert atm_frac > wing_frac, (
            f"ATM obs rate ({atm_frac:.2f}) should exceed "
            f"wing obs rate ({wing_frac:.2f})"
        )


class TestApplyMask:
    def test_adds_observed_column(self, synthetic_surface_df):
        result = apply_mask(synthetic_surface_df, "random", keep_frac=0.5)
        assert "observed" in result.columns
        assert result["observed"].dtype == bool

    def test_all_strategies_work(self, synthetic_surface_df):
        strategies = [
            "random", "drop_short_maturity", "drop_long_maturity",
            "drop_random_maturity", "drop_wings", "drop_atm", "realistic",
        ]
        for s in strategies:
            result = apply_mask(synthetic_surface_df, s, keep_frac=0.4)
            assert "observed" in result.columns, f"Strategy {s} failed"

    def test_does_not_mutate_input(self, synthetic_surface_df):
        cols_before = set(synthetic_surface_df.columns)
        apply_mask(synthetic_surface_df, "random", keep_frac=0.5)
        assert set(synthetic_surface_df.columns) == cols_before


# ---------------------------------------------------------------------------
# Noise tests
# ---------------------------------------------------------------------------

class TestAddGaussianNoise:
    def test_no_noise_returns_copy(self, rng):
        iv = np.array([0.2, 0.3, 0.4])
        result = add_gaussian_noise(iv, 0.0, rng)
        np.testing.assert_array_equal(result, iv)

    def test_noisy_values_differ(self, rng):
        iv = np.full(1000, 0.2)
        result = add_gaussian_noise(iv, 0.03, rng)
        assert not np.allclose(result, iv)

    def test_positive_floor(self, rng):
        iv = np.full(100, 0.001)  # very small IVs
        result = add_gaussian_noise(iv, 0.1, rng)  # big noise
        assert (result >= 1e-4).all()


class TestAddHeteroscedasticNoise:
    def test_wings_get_more_noise(self, rng):
        n = 5000
        iv = np.full(n, 0.2)
        lm_atm = np.zeros(n)
        lm_wing = np.full(n, 0.5)
        tau = np.full(n, 60 / 365)

        noisy_atm = add_heteroscedastic_noise(iv, lm_atm, tau, 0.03, rng)
        rng2 = np.random.default_rng(42)
        noisy_wing = add_heteroscedastic_noise(iv, lm_wing, tau, 0.03, rng2)

        std_atm = np.std(noisy_atm - iv)
        std_wing = np.std(noisy_wing - iv)
        assert std_wing > std_atm


class TestInjectNoise:
    def test_only_observed_points_noised(self, synthetic_surface_df):
        df = apply_mask(synthetic_surface_df, "random", keep_frac=0.5, seed=0)
        result = inject_noise(df, regime="medium", seed=1)

        unobs = result[~result["observed"]]
        np.testing.assert_array_equal(
            unobs["implied_volatility"].values,
            unobs["iv_clean"].values,
        )

    def test_iv_clean_preserved(self, synthetic_surface_df):
        df = apply_mask(synthetic_surface_df, "random", keep_frac=0.5, seed=0)
        original_iv = df["implied_volatility"].values.copy()
        result = inject_noise(df, regime="high", seed=1)
        np.testing.assert_array_equal(result["iv_clean"].values, original_iv)

    def test_no_noise_regime(self, synthetic_surface_df):
        df = apply_mask(synthetic_surface_df, "random", keep_frac=0.5, seed=0)
        result = inject_noise(df, regime="none", seed=1)
        np.testing.assert_array_equal(
            result["implied_volatility"].values,
            result["iv_clean"].values,
        )

    def test_all_regimes(self, synthetic_surface_df):
        df = apply_mask(synthetic_surface_df, "random", keep_frac=0.5, seed=0)
        for regime in NOISE_REGIMES:
            result = inject_noise(df, regime=regime, seed=1)
            assert "iv_clean" in result.columns
            assert "noise_sigma" in result.columns


# ---------------------------------------------------------------------------
# Splits tests
# ---------------------------------------------------------------------------

class TestTimeSplit:
    def test_all_rows_assigned(self, synthetic_surface_df):
        result = time_split(synthetic_surface_df)
        assert "split" in result.columns
        assert result["split"].isin(["train", "val", "test"]).all()

    def test_chronological_order(self, synthetic_surface_df):
        result = time_split(synthetic_surface_df)
        train_max = result[result["split"] == "train"]["date"].max()
        val_min = result[result["split"] == "val"]["date"].min()
        val_max = result[result["split"] == "val"]["date"].max()
        test_min = result[result["split"] == "test"]["date"].min()

        assert train_max <= val_min, "Train dates must precede val dates"
        assert val_max <= test_min, "Val dates must precede test dates"

    def test_approximate_fractions(self, synthetic_surface_df):
        result = time_split(synthetic_surface_df, train_frac=0.7, val_frac=0.15)
        n_dates = result["date"].nunique()
        n_train = result[result["split"] == "train"]["date"].nunique()
        n_val = result[result["split"] == "val"]["date"].nunique()
        n_test = result[result["split"] == "test"]["date"].nunique()

        assert abs(n_train / n_dates - 0.7) < 0.05
        assert abs(n_val / n_dates - 0.15) < 0.05
        assert abs(n_test / n_dates - 0.15) < 0.05

    def test_invalid_fractions_raise(self, synthetic_surface_df):
        with pytest.raises(ValueError):
            time_split(synthetic_surface_df, train_frac=0.8, val_frac=0.3)


class TestSplitSummary:
    def test_returns_three_rows(self, synthetic_surface_df):
        df = time_split(synthetic_surface_df)
        summary = split_summary(df)
        assert len(summary) == 3
        assert set(summary["split"]) == {"train", "val", "test"}


class TestBenchmarkName:
    def test_basic(self):
        assert benchmark_name("random", 0.2, "none") == "spy_phase1_random20_noise0"

    def test_with_noise(self):
        assert benchmark_name("random", 0.4, "low") == "spy_phase1_random40_noiselow"

    def test_hetero(self):
        name = benchmark_name("realistic", 0.4, "medium", heteroscedastic=True)
        assert name == "spy_phase1_realistic40_noisemed_hetero"


class TestSaveLoadBenchmark:
    def test_roundtrip(self, synthetic_surface_df, tmp_path):
        df = apply_mask(synthetic_surface_df, "random", keep_frac=0.5, seed=0)
        df = inject_noise(df, regime="low", seed=1)
        df = time_split(df)

        path = save_benchmark(
            df, tmp_path, "test_benchmark",
            metadata={"strategy": "random", "keep_frac": "0.5"},
        )

        loaded_df, meta = load_benchmark(path)
        assert len(loaded_df) == len(df)
        assert "benchmark_name" in meta
        assert meta["benchmark_name"] == "test_benchmark"
        assert meta["strategy"] == "random"
