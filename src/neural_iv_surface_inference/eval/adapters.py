"""Adapters wrapping the Phase 1 baselines in the model-agnostic interface.

Both adapters return ``uncertainty=None``: the interpolation and MLP baselines
carry no genuine uncertainty signal. Real signals (ensemble, heteroscedastic,
masking-based) arrive in W4 — see the Phase 2 roadmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import pandas as pd

from neural_iv_surface_inference.data.loaders import IVSurfaceDataset
from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
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


class ConditionalSurfacePredictor:
    """Wraps a trained :class:`ConditionalSurfaceModel` as a :class:`Predictor`.

    Builds each date's context from its ``observed == True`` rows (the 2C.2
    contract), decodes at that date's frame rows, and re-aligns outputs to the
    input ``df`` row order — mirroring ``MLPPredictor``'s non-shuffled
    alignment guarantee.

    Like the Phase 1 baselines, the conditional model carries no genuine
    uncertainty signal in this story; ``uncertainty=None``. Real signals
    arrive in epic 2D.
    """

    def __init__(
        self,
        model: ConditionalSurfaceModel,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device | None = None,
    ) -> "ConditionalSurfacePredictor":
        """Construct from a 2C.4 ``best_conditional.pt`` checkpoint."""
        ckpt = torch.load(
            str(checkpoint_path),
            map_location=device if device else "cpu",
            weights_only=False,
        )
        cfg = ckpt.get("config", {})
        model = ConditionalSurfaceModel(
            context_dim=int(cfg.get("context_dim", 3)),
            coord_dim=int(cfg.get("coord_dim", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            latent_dim=int(cfg.get("latent_dim", 32)),
            n_elem_layers=int(cfg.get("n_elem_layers", 2)),
            n_post_layers=int(cfg.get("n_post_layers", 1)),
            n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        required = {"date", "log_moneyness", "tau", "implied_volatility", "observed"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"ConditionalSurfacePredictor.predict: missing columns {sorted(missing)}"
            )

        out = np.zeros(len(df), dtype=np.float32)
        # Use the original row index to scatter predictions back in order.
        df_indexed = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})

        for _date, group in df_indexed.groupby("date", sort=False):
            obs_mask = group["observed"].values.astype(bool)
            if obs_mask.sum() == 0:
                # No observed context for this date: leave preds at 0; eval
                # downstream still gets a well-formed array and can flag the
                # zero predictions if needed. This mirrors how the masked
                # training contract treats empty-context cases (which 2C.2
                # rejects on the *training* side).
                continue

            ctx = group.loc[
                obs_mask, ["log_moneyness", "tau", "implied_volatility"]
            ].to_numpy(dtype=np.float32)
            query = group[["log_moneyness", "tau"]].to_numpy(dtype=np.float32)

            ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(self.device)
            ctx_mask_t = torch.ones(1, ctx_t.shape[1], dtype=torch.bool, device=self.device)
            q_t = torch.from_numpy(query).unsqueeze(0).to(self.device)

            pred = self.model(ctx_t, ctx_mask_t, q_t).squeeze(0).cpu().numpy()
            out[group["_orig_idx"].values] = pred

        return PredictionResult(
            pred=out,
            uncertainty=None,
            meta={"model": "conditional_surface"},
        )
