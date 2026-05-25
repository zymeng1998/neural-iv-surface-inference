"""Unit tests for the W4 decision layer (story 2D.5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.decision_layer import (
    REASON_LOW_CONFIDENCE,
    REASON_NO_ARB,
    REASON_OK,
    REASON_WIDE_INTERVAL,
    DecisionConfig,
    TradabilityWeights,
    apply_decision_layer,
)
from neural_iv_surface_inference.eval.predictor import PredictionResult


def _make_pred(
    confidence: np.ndarray,
    rel_width: np.ndarray,
    sigma: float = 1.0,
) -> PredictionResult:
    """Build a PredictionResult with prescribed confidence and relative width."""
    n = len(confidence)
    sigma_arr = np.full(n, sigma, dtype=float)
    width = rel_width * sigma_arr
    mu = np.zeros(n)
    return PredictionResult(
        pred=mu,
        uncertainty=sigma_arr,
        lower=mu - width / 2.0,
        upper=mu + width / 2.0,
        meta={"confidence_score": np.asarray(confidence, dtype=float)},
    )


def _default_config(**overrides) -> DecisionConfig:
    base = dict(
        confidence_threshold=0.5,
        max_relative_width=2.0,
        forbid_flags=("calendar_violation", "convexity_violation"),
        tradability_weights=TradabilityWeights(0.6, 0.3, 0.1),
    )
    base.update(overrides)
    return DecisionConfig(**base)


# ---------------------------------------------------------------------------
# Threshold behavior
# ---------------------------------------------------------------------------


def test_confidence_threshold_flip():
    # Two points: one just below, one just above the threshold. No structural
    # flags, narrow band so only the confidence rule can fire.
    conf = np.array([0.49, 0.51])
    rel_w = np.array([0.1, 0.1])
    pred = _make_pred(conf, rel_w)
    flags = {"calendar_violation": np.zeros(2, dtype=bool)}
    cfg = _default_config(confidence_threshold=0.5)
    out = apply_decision_layer(pred, flags, cfg)
    assert out.abstain_flag.tolist() == [True, False]
    assert out.decision_reason[0] == REASON_LOW_CONFIDENCE
    assert out.decision_reason[1] == REASON_OK


def test_wide_interval_flip():
    # Confidence high, but relative width above max -> abstain via wide rule.
    conf = np.array([0.9, 0.9])
    rel_w = np.array([1.9, 2.1])
    pred = _make_pred(conf, rel_w)
    flags: dict[str, np.ndarray] = {}
    cfg = _default_config(max_relative_width=2.0)
    out = apply_decision_layer(pred, flags, cfg)
    assert out.abstain_flag.tolist() == [False, True]
    assert out.decision_reason[1] == REASON_WIDE_INTERVAL


# ---------------------------------------------------------------------------
# Monotonicity of tradability_score
# ---------------------------------------------------------------------------


def test_tradability_monotone_in_confidence():
    # Increasing confidence with all else fixed -> non-decreasing score.
    conf = np.linspace(0.05, 0.95, 20)
    rel_w = np.full_like(conf, 0.5)
    pred = _make_pred(conf, rel_w)
    flags: dict[str, np.ndarray] = {}
    out = apply_decision_layer(pred, flags, _default_config())
    diffs = np.diff(out.tradability_score)
    assert np.all(diffs >= -1e-12)


def test_tradability_monotone_in_relative_width():
    # Increasing relative width with all else fixed -> non-increasing score.
    rel_w = np.linspace(0.0, 2.0, 20)
    conf = np.full_like(rel_w, 0.8)
    pred = _make_pred(conf, rel_w)
    flags: dict[str, np.ndarray] = {}
    out = apply_decision_layer(pred, flags, _default_config())
    diffs = np.diff(out.tradability_score)
    assert np.all(diffs <= 1e-12)


# ---------------------------------------------------------------------------
# Flag propagation
# ---------------------------------------------------------------------------


def test_forbidden_flag_forces_abstain_regardless_of_confidence():
    conf = np.array([0.99, 0.99, 0.99])
    rel_w = np.array([0.05, 0.05, 0.05])
    pred = _make_pred(conf, rel_w)
    flags = {
        "calendar_violation": np.array([False, True, False]),
        "convexity_violation": np.array([False, False, True]),
        # Unlisted flag must be ignored.
        "monotonicity_violation": np.array([True, False, False]),
    }
    cfg = _default_config()
    out = apply_decision_layer(pred, flags, cfg)
    # Row 0 has only the unlisted flag -> not abstain. Rows 1, 2 have listed
    # flags -> abstain regardless of high confidence.
    assert out.abstain_flag.tolist() == [False, True, True]
    assert out.decision_reason[1] == REASON_NO_ARB
    assert out.decision_reason[2] == REASON_NO_ARB
    assert out.decision_reason[0] == REASON_OK


def test_unknown_forbid_flag_name_is_ignored():
    pred = _make_pred(np.array([0.8]), np.array([0.1]))
    cfg = _default_config(forbid_flags=("ghost_flag",))
    out = apply_decision_layer(pred, {}, cfg)
    assert out.abstain_flag.tolist() == [False]


# ---------------------------------------------------------------------------
# Reason prioritization
# ---------------------------------------------------------------------------


def test_reason_priority_no_arb_beats_low_confidence_beats_wide():
    # Row 0: all three triggers active -> reason = no_arb_violation
    # Row 1: low conf + wide              -> reason = low_confidence
    # Row 2: wide only                    -> reason = wide_interval
    # Row 3: nothing                      -> reason = ok
    conf = np.array([0.1, 0.1, 0.9, 0.9])
    rel_w = np.array([5.0, 5.0, 5.0, 0.1])
    pred = _make_pred(conf, rel_w)
    flags = {
        "calendar_violation": np.array([True, False, False, False]),
    }
    cfg = _default_config(confidence_threshold=0.5, max_relative_width=2.0)
    out = apply_decision_layer(pred, flags, cfg)
    assert out.decision_reason[0] == REASON_NO_ARB
    assert out.decision_reason[1] == REASON_LOW_CONFIDENCE
    assert out.decision_reason[2] == REASON_WIDE_INTERVAL
    assert out.decision_reason[3] == REASON_OK


# ---------------------------------------------------------------------------
# Config + result shape contract
# ---------------------------------------------------------------------------


def test_result_shapes_and_score_range():
    n = 16
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=n)
    rel_w = rng.uniform(0, 3, size=n)
    pred = _make_pred(conf, rel_w)
    flags = {"calendar_violation": rng.random(n) < 0.2}
    out = apply_decision_layer(pred, flags, _default_config())
    assert out.abstain_flag.shape == (n,)
    assert out.tradability_score.shape == (n,)
    assert out.decision_reason.shape == (n,)
    assert out.tradability_score.min() >= 0.0
    assert out.tradability_score.max() <= 1.0
    assert out.abstain_flag.dtype == bool


def test_config_from_yaml(tmp_path):
    yaml_text = """
confidence_threshold: 0.7
max_relative_width: 1.25
forbid_flags:
  - calendar_violation
tradability_weights:
  confidence: 0.5
  width: 0.4
  structure: 0.1
"""
    p = tmp_path / "decision.yaml"
    p.write_text(yaml_text)
    cfg = DecisionConfig.from_yaml(p)
    assert cfg.confidence_threshold == 0.7
    assert cfg.max_relative_width == 1.25
    assert cfg.forbid_flags == ("calendar_violation",)
    assert cfg.tradability_weights.width == 0.4


def test_missing_confidence_score_raises():
    pred = PredictionResult(
        pred=np.zeros(3),
        uncertainty=np.ones(3),
        lower=np.zeros(3),
        upper=np.ones(3),
        meta={},
    )
    with pytest.raises(ValueError, match="confidence_score"):
        apply_decision_layer(pred, {}, _default_config())


def test_missing_band_raises():
    pred = PredictionResult(
        pred=np.zeros(3),
        meta={"confidence_score": np.array([0.5, 0.5, 0.5])},
    )
    with pytest.raises(ValueError, match="lower, upper"):
        apply_decision_layer(pred, {}, _default_config())
