"""Adapters wrapping the Phase 1 baselines in the model-agnostic interface.

Both Phase-1 adapters return ``uncertainty=None``: the interpolation and MLP
baselines carry no genuine uncertainty signal. Real W4 signals arrive in
epic 2D: 2D.3 wires the deep-ensemble disagreement adapter
(:class:`EnsembleConditionalPredictor`), and 2D.4 wires the calibrated
fusion on top of the heteroscedastic / quantile head (2D.2).
"""

from __future__ import annotations

import json
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
        head_cfg = cfg.get("head") or {"kind": "point"}
        model = ConditionalSurfaceModel(
            context_dim=int(cfg.get("context_dim", 3)),
            coord_dim=int(cfg.get("coord_dim", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            latent_dim=int(cfg.get("latent_dim", 32)),
            n_elem_layers=int(cfg.get("n_elem_layers", 2)),
            n_post_layers=int(cfg.get("n_post_layers", 1)),
            n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
            head=dict(head_cfg),
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

            forward_out = self.model(ctx_t, ctx_mask_t, q_t)
            mu = forward_out["mu"] if isinstance(forward_out, dict) else forward_out
            pred = mu.squeeze(0).cpu().numpy()
            out[group["_orig_idx"].values] = pred

        return PredictionResult(
            pred=out,
            uncertainty=None,
            meta={"model": "conditional_surface"},
        )


# ── Deep-ensemble adapter (W4 / 2D.3) ─────────────────────────────────────


class EnsembleConditionalPredictor:
    """Aggregate K independently trained conditional models into one Predictor.

    Loads its members from a ``members.json`` manifest written by
    ``scripts/run_ensemble_train.py``. Member predictions are pooled per query
    point into:

    * ``pred`` — arithmetic **mean** of the per-member ``mu`` predictions.
      This is also exposed via ``PredictionResult.pred`` and corresponds to
      ``sigma_hat = mean_k sigma_k`` in the roadmap notation (member k's IV).
    * ``uncertainty`` — **population standard deviation** of the per-member
      ``mu`` predictions (``disagreement = std_k sigma_k``). This is the raw,
      **uncalibrated** disagreement signal; calibration into a probability
      lives in 2D.4.

    ``lower``/``upper`` are left ``None`` — no calibrated interval is emitted
    by this story.

    Manifest schema (``members.json``)
    ----------------------------------
    .. code-block:: json

        {
          "version": 1,
          "ensemble_size": 2,
          "config_hash": "...",
          "members": [
            {"seed": 101, "val_loss": 0.0034,
             "checkpoint": "seed_101/best_conditional.pt"},
            ...
          ]
        }

    Checkpoint paths in ``members[*].checkpoint`` are resolved relative to the
    manifest's parent directory.
    """

    def __init__(
        self,
        members: list[ConditionalSurfacePredictor],
        manifest_path: Path | None = None,
        meta_extra: dict | None = None,
    ):
        if not members:
            raise ValueError(
                "EnsembleConditionalPredictor requires at least one member"
            )
        self.members = members
        self.manifest_path = manifest_path
        self._meta_extra = dict(meta_extra or {})

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        device: torch.device | None = None,
    ) -> "EnsembleConditionalPredictor":
        """Eager-load all members from a ``members.json`` manifest.

        Raises ``FileNotFoundError`` with a clear message if any referenced
        checkpoint is missing — the manifest is the authoritative source of
        truth and an incomplete ensemble is treated as a hard error.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"ensemble manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        members_meta = manifest.get("members") or []
        if not members_meta:
            raise ValueError(
                f"ensemble manifest {manifest_path} has no members"
            )

        loaded: list[ConditionalSurfacePredictor] = []
        for entry in members_meta:
            ckpt_rel = entry.get("checkpoint")
            if not ckpt_rel:
                raise ValueError(
                    f"manifest member missing 'checkpoint' field: {entry!r}"
                )
            ckpt_path = (manifest_path.parent / ckpt_rel).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"ensemble member checkpoint missing: {ckpt_path} "
                    f"(seed={entry.get('seed', '?')})"
                )
            loaded.append(
                ConditionalSurfacePredictor.from_checkpoint(
                    ckpt_path, device=device
                )
            )

        return cls(
            members=loaded,
            manifest_path=manifest_path,
            meta_extra={"ensemble_size": len(loaded)},
        )

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if len(self.members) == 0:
            raise RuntimeError("ensemble has no members")
        per_member: list[np.ndarray] = []
        for member in self.members:
            res = member.predict(df)
            per_member.append(np.asarray(res.pred, dtype=np.float64))
        stacked = np.stack(per_member, axis=0)  # (K, N)
        mean_pred = stacked.mean(axis=0).astype(np.float32)
        # Population std (ddof=0) — documented choice; for K small the
        # difference vs sample std is non-trivial but the metric is treated
        # as a *raw, uncalibrated* signal that 2D.4 will calibrate against
        # empirical coverage, so the ddof choice is absorbed by calibration.
        disagreement = stacked.std(axis=0, ddof=0).astype(np.float32)

        meta = {
            "model": "ensemble_conditional",
            "ensemble_size": len(self.members),
            "disagreement_ddof": 0,
        }
        meta.update(self._meta_extra)
        return PredictionResult(
            pred=mean_pred,
            uncertainty=disagreement,
            lower=None,
            upper=None,
            meta=meta,
        )
