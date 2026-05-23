"""Model-agnostic uncertainty-evaluation layer (Phase 2 W1).

Public surface for 2A.2: the predictor contract and the Phase 1 baseline
adapters. Uncertainty metrics (2A.3/2A.4) and the evaluation runner (2A.5)
build on this.
"""

from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
    InterpolationPredictor,
    MLPPredictor,
)
from neural_iv_surface_inference.eval.predictor import (
    PredictionResult,
    Predictor,
)

__all__ = [
    "PredictionResult",
    "Predictor",
    "InterpolationPredictor",
    "MLPPredictor",
    "ConditionalSurfacePredictor",
]
