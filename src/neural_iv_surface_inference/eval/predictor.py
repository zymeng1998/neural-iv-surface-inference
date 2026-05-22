"""Model-agnostic predictor interface for the W1 uncertainty-evaluation layer.

Defines the stable contract that every current and future predictor
(interpolation, MLP, conditional neural models) plugs into so that the
uncertainty-metric modules (2A.3/2A.4) and the evaluation runner (2A.5) can be
written without knowing the underlying model.

Two pieces:
  - ``PredictionResult``: an immutable container for what a prediction carries.
  - ``Predictor``: a structural Protocol describing ``predict(df)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Immutable container for a single prediction over a benchmark frame.

    Fields
    ------
    pred : np.ndarray, shape (n,)
        Point predictions (one per row of the input frame).
    uncertainty : np.ndarray | None
        Per-point uncertainty signal (e.g. predictive std). ``None`` for the
        Phase 1 baselines — real signals arrive in W4.
    lower, upper : np.ndarray | None
        Optional prediction-interval bounds. Both ``None`` for the baselines.
    meta : dict
        Free-form metadata (model name, method, etc.).

    All non-``None`` arrays must share the same length as ``pred``; this is
    validated at construction time.
    """

    pred: np.ndarray
    uncertainty: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.pred)
        for name in ("uncertainty", "lower", "upper"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"PredictionResult.{name} has length {len(arr)}, "
                    f"expected {n} to match pred"
                )

    def __len__(self) -> int:
        return len(self.pred)


@runtime_checkable
class Predictor(Protocol):
    """Structural contract for anything that produces a ``PredictionResult``.

    A predictor consumes a benchmark frame carrying the standard columns
    (``log_moneyness``, ``tau``, ``observed``, ``iv_clean``, ...) and returns a
    :class:`PredictionResult` aligned to ``df`` row order.

    Marked ``runtime_checkable`` so tests can assert ``isinstance(obj,
    Predictor)``. Note this only verifies that a ``predict`` attribute exists,
    not its signature — the behavioral test that calls ``predict`` and checks
    the result is the stronger conformance check.
    """

    def predict(self, df: pd.DataFrame) -> PredictionResult: ...
