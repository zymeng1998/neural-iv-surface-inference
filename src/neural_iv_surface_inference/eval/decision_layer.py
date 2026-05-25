"""W4 decision layer (story 2D.5).

Converts the calibrated reliability signals from 2D.4 (per-point
``confidence_score`` + ``(lower, upper)`` band on a :class:`PredictionResult`)
plus the structural-risk flags from 2B.4 (per-flag boolean arrays such as
``calendar_violation``, ``monotonicity_violation``, ``convexity_violation``)
into two auditable per-point outputs:

* ``abstain_flag``        — hard refuse-to-predict bit.
* ``tradability_score``   — continuous ``[0, 1]`` grade of the served region.

Rules are intentionally simple, explicit, and config-driven (see
:class:`DecisionConfig` / ``configs/decision_layer.yaml``) so the operating
point is auditable and tunable without code edits. This module is pure
NumPy — no model state, no training, no I/O beyond config loading.

Abstain rule (OR of three triggers):

    abstain = (confidence < confidence_threshold)
              | (relative_width > max_relative_width)
              | any_forbidden_flag_set

where ``relative_width = (upper - lower) / max(uncertainty, eps)``.

Decision-reason priority (first match wins, top to bottom):

    1. ``no_arb_violation``   — any flag in ``forbid_flags`` is set
    2. ``low_confidence``     — confidence below threshold
    3. ``wide_interval``      — relative width above max
    4. ``ok``                 — none of the above

Tradability formula (weights from config, weights need not sum to 1; the
score is clipped to ``[0, 1]``):

    conf_term   = clip(confidence, 0, 1)
    width_term  = 1 - clip(relative_width / max_relative_width, 0, 1)
    struct_term = 0.0 if any forbidden flag else 1.0
    tradability = clip( w_conf * conf_term
                        + w_width * width_term
                        + w_struct * struct_term,
                        0, 1 )

``tradability_score`` is monotone non-decreasing in ``confidence`` and
monotone non-increasing in ``relative_width``, and is forced toward 0 when a
forbidden structural flag is set (subject to the other weighted terms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from neural_iv_surface_inference.eval.predictor import PredictionResult


# ---------------------------------------------------------------------------
# Reason codes — single source of truth, priority order (first match wins).
# ---------------------------------------------------------------------------

REASON_NO_ARB = "no_arb_violation"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_WIDE_INTERVAL = "wide_interval"
REASON_OK = "ok"

REASON_PRIORITY: tuple[str, ...] = (
    REASON_NO_ARB,
    REASON_LOW_CONFIDENCE,
    REASON_WIDE_INTERVAL,
    REASON_OK,
)


# ---------------------------------------------------------------------------
# Config + result containers (immutable).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradabilityWeights:
    confidence: float = 0.6
    width: float = 0.3
    structure: float = 0.1


@dataclass(frozen=True)
class DecisionConfig:
    """Operating-point knobs for :func:`apply_decision_layer`.

    Fields
    ------
    confidence_threshold : float
        Per-point ``confidence_score`` strictly below this triggers abstain.
    max_relative_width : float
        Relative band width ``(upper - lower) / uncertainty`` strictly above
        this triggers abstain. Use ``inf`` to disable the rule.
    forbid_flags : tuple[str, ...]
        Names of structural-risk flags in ``no_arb_risk_flags`` whose boolean
        being ``True`` triggers abstain. Unknown names are ignored (with a
        single-shot warning would normally fire — kept silent here to keep the
        layer dependency-free).
    tradability_weights : TradabilityWeights
        Non-negative weights on the confidence / width / structure terms.
    epsilon : float
        Lower clamp on ``uncertainty`` when computing relative width to avoid
        divide-by-zero for collapsed bands.
    """

    confidence_threshold: float = 0.5
    max_relative_width: float = 0.5
    forbid_flags: tuple[str, ...] = (
        "calendar_violation",
        "convexity_violation",
    )
    tradability_weights: TradabilityWeights = field(
        default_factory=TradabilityWeights
    )
    epsilon: float = 1e-9

    @staticmethod
    def from_yaml(path: str | Path) -> "DecisionConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return DecisionConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: Mapping) -> "DecisionConfig":
        weights_raw = raw.get("tradability_weights", {}) or {}
        weights = TradabilityWeights(
            confidence=float(weights_raw.get("confidence", 0.6)),
            width=float(weights_raw.get("width", 0.3)),
            structure=float(weights_raw.get("structure", 0.1)),
        )
        forbid = tuple(raw.get("forbid_flags", ()) or ())
        return DecisionConfig(
            confidence_threshold=float(raw.get("confidence_threshold", 0.5)),
            max_relative_width=float(raw.get("max_relative_width", 0.5)),
            forbid_flags=forbid,
            tradability_weights=weights,
            epsilon=float(raw.get("epsilon", 1e-9)),
        )


@dataclass(frozen=True)
class DecisionResult:
    """Per-point decision outputs aligned to the input ``PredictionResult``.

    Fields
    ------
    abstain_flag : np.ndarray[bool], shape (n,)
    tradability_score : np.ndarray[float], shape (n,), values in [0, 1]
    decision_reason : np.ndarray[object], shape (n,), one of the
        ``REASON_*`` strings; priority order is :data:`REASON_PRIORITY`.
    meta : dict
        Echoes the resolved config + any diagnostic counts.
    """

    abstain_flag: np.ndarray
    tradability_score: np.ndarray
    decision_reason: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.abstain_flag)


# ---------------------------------------------------------------------------
# Core transform.
# ---------------------------------------------------------------------------


def _extract_confidence(prediction_result: PredictionResult, n: int) -> np.ndarray:
    cs = prediction_result.meta.get("confidence_score")
    if cs is None:
        raise ValueError(
            "PredictionResult.meta['confidence_score'] is required for the "
            "decision layer; run the calibrated predictor (2D.4) first."
        )
    cs = np.asarray(cs, dtype=float)
    if cs.shape != (n,):
        raise ValueError(
            f"confidence_score has shape {cs.shape}, expected ({n},)"
        )
    return cs


def _extract_relative_width(
    prediction_result: PredictionResult, n: int, epsilon: float
) -> np.ndarray:
    lo, hi = prediction_result.lower, prediction_result.upper
    sigma = prediction_result.uncertainty
    if lo is None or hi is None or sigma is None:
        raise ValueError(
            "PredictionResult must populate lower, upper, and uncertainty for "
            "the decision layer; run the calibrated predictor (2D.4) first."
        )
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), epsilon)
    rel = width / sigma_safe
    if rel.shape != (n,):
        raise ValueError(
            f"derived relative_width has shape {rel.shape}, expected ({n},)"
        )
    return rel


def _any_forbidden_flag(
    no_arb_risk_flags: Mapping[str, np.ndarray],
    forbid_names: tuple[str, ...],
    n: int,
) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for name in forbid_names:
        arr = no_arb_risk_flags.get(name)
        if arr is None:
            continue
        arr_b = np.asarray(arr, dtype=bool)
        if arr_b.shape != (n,):
            raise ValueError(
                f"risk flag '{name}' has shape {arr_b.shape}, expected ({n},)"
            )
        out |= arr_b
    return out


def apply_decision_layer(
    prediction_result: PredictionResult,
    no_arb_risk_flags: Mapping[str, np.ndarray],
    config: DecisionConfig,
) -> DecisionResult:
    """Combine 2D.4 reliability + 2B.4 structural risk into a decision.

    Parameters
    ----------
    prediction_result : PredictionResult
        Must carry ``lower``, ``upper``, ``uncertainty`` and
        ``meta['confidence_score']`` — i.e. the output of the calibrated
        predictor from 2D.4.
    no_arb_risk_flags : Mapping[str, np.ndarray]
        Per-flag boolean arrays of length ``len(prediction_result)``. Names
        not listed in ``config.forbid_flags`` are ignored.
    config : DecisionConfig

    Returns
    -------
    DecisionResult
    """
    n = len(prediction_result)
    confidence = _extract_confidence(prediction_result, n)
    rel_width = _extract_relative_width(prediction_result, n, config.epsilon)
    forbidden = _any_forbidden_flag(
        no_arb_risk_flags, config.forbid_flags, n
    )

    low_conf = confidence < config.confidence_threshold
    wide = rel_width > config.max_relative_width
    abstain = forbidden | low_conf | wide

    # Reason — priority order, first match wins.
    reason = np.full(n, REASON_OK, dtype=object)
    reason[wide] = REASON_WIDE_INTERVAL
    reason[low_conf] = REASON_LOW_CONFIDENCE
    reason[forbidden] = REASON_NO_ARB

    # Tradability score (continuous, clipped to [0, 1]).
    w = config.tradability_weights
    conf_term = np.clip(confidence, 0.0, 1.0)
    if np.isfinite(config.max_relative_width) and config.max_relative_width > 0:
        width_term = 1.0 - np.clip(
            rel_width / config.max_relative_width, 0.0, 1.0
        )
    else:
        width_term = np.ones(n)
    struct_term = np.where(forbidden, 0.0, 1.0)
    tradability = np.clip(
        w.confidence * conf_term
        + w.width * width_term
        + w.structure * struct_term,
        0.0,
        1.0,
    )

    meta = {
        "n": n,
        "n_abstain": int(abstain.sum()),
        "n_low_confidence": int(low_conf.sum()),
        "n_wide_interval": int(wide.sum()),
        "n_no_arb_violation": int(forbidden.sum()),
        "confidence_threshold": config.confidence_threshold,
        "max_relative_width": config.max_relative_width,
        "forbid_flags": list(config.forbid_flags),
        "tradability_weights": {
            "confidence": w.confidence,
            "width": w.width,
            "structure": w.structure,
        },
    }
    return DecisionResult(
        abstain_flag=abstain,
        tradability_score=tradability,
        decision_reason=reason,
        meta=meta,
    )
