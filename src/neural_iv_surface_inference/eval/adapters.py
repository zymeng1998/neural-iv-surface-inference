"""Adapters wrapping the Phase 1 baselines in the model-agnostic interface.

Both adapters return ``uncertainty=None``: the interpolation and MLP baselines
carry no genuine uncertainty signal. Real signals (ensemble, heteroscedastic,
masking-based) arrive in W4 — see the Phase 2 roadmap.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import pandas as pd

from neural_iv_surface_inference.data.loaders import IVSurfaceDataset
from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.interpolation import (
    run_interpolation_baseline,
)
from neural_iv_surface_inference.training.train import predict_mlp

from neural_iv_surface_inference.eval.predictor import PredictionResult


class InterpolationPredictor:
    """Wraps ``run_interpolation_baseline`` as a :class:`Predictor`."""

    def __init__(self, method: str = "rbf", verbose: bool = False):
        self.method = method
        self.verbose = verbose

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        pred = run_interpolation_baseline(
            df, method=self.method, verbose=self.verbose
        )
        return PredictionResult(
            pred=pred,
            uncertainty=None,
            meta={"model": "interpolation", "method": self.method},
        )


class MLPPredictor:
    """Wraps a trained ``BaselineMLP`` + ``predict_mlp`` as a :class:`Predictor`.

    Builds a non-shuffled DataLoader from the input frame so predictions stay
    aligned to ``df`` row order.
    """

    def __init__(
        self,
        model: BaselineMLP,
        device: torch.device | None = None,
        batch_size: int = 256,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        loader = DataLoader(
            IVSurfaceDataset(df),
            batch_size=self.batch_size,
            shuffle=False,
        )
        pred = predict_mlp(self.model, loader, device=self.device)
        return PredictionResult(
            pred=pred,
            uncertainty=None,
            meta={"model": "baseline_mlp"},
        )
