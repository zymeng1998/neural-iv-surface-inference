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

from neural_iv_surface_inference.data.conditional_loaders import (
    DEFAULT_FEATURE_SET,
    build_context_matrix,
)
from neural_iv_surface_inference.data.loaders import IVSurfaceDataset
from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.models.interpolation import (
    run_interpolation_baseline,
)
from neural_iv_surface_inference.training.train import predict_mlp

from neural_iv_surface_inference.eval.calibration import Calibrator
from neural_iv_surface_inference.eval.predictor import PredictionResult, Predictor


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
        feature_set: str | None = None,
    ):
        self.model = model
        # ADR 0008: the feature set governs which context columns ``predict``
        # builds. Prefer the value the model was constructed with; fall back
        # to the explicit arg, then to the legacy ``minimal`` default.
        self.feature_set = str(
            feature_set
            if feature_set is not None
            else getattr(model, "feature_set", DEFAULT_FEATURE_SET)
        )
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
        # 3A.2 / 3B.2 fields are optional — defaults preserve pre-Phase-3
        # checkpoint behavior (raw coord encoding, deepsets decoder).
        coord_encoding_cfg = cfg.get("coord_encoding")
        decoder_kind = str(cfg.get("decoder_kind", "deepsets"))
        anp_cfg = cfg.get("anp") or {}
        # 3C.2 / ADR 0008: feature_set is authoritative for the context dim.
        # Absent on legacy checkpoints → "minimal" (3-dim), bit-identical.
        feature_set = str(cfg.get("feature_set", DEFAULT_FEATURE_SET))
        context_dim = (
            int(cfg["context_dim"]) if "context_dim" in cfg else None
        )
        model = ConditionalSurfaceModel(
            context_dim=context_dim,
            coord_dim=int(cfg.get("coord_dim", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            latent_dim=int(cfg.get("latent_dim", 32)),
            n_elem_layers=int(cfg.get("n_elem_layers", 2)),
            n_post_layers=int(cfg.get("n_post_layers", 1)),
            n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
            head=dict(head_cfg),
            coord_encoding=coord_encoding_cfg,
            decoder_kind=decoder_kind,  # type: ignore[arg-type]
            anp=dict(anp_cfg),
            feature_set=feature_set,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, device=device, feature_set=feature_set)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        required = {"date", "log_moneyness", "tau", "implied_volatility", "observed"}
        if self.feature_set == "micro_v1":
            # Raw AV columns needed to materialise the ADR 0008 micro tuple.
            required |= {"bid", "ask", "volume", "open_interest", "type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"ConditionalSurfacePredictor.predict: missing columns {sorted(missing)}"
            )

        n = len(df)
        out = np.zeros(n, dtype=np.float32)
        head_kind = getattr(self.model, "head_kind", "point")
        sigma_out = np.full(n, np.nan, dtype=np.float32) if head_kind == "gaussian" else None
        q_lo_out = q_hi_out = None
        quantile_levels: tuple[float, ...] | None = None
        if head_kind == "quantile":
            q_lo_out = np.full(n, np.nan, dtype=np.float32)
            q_hi_out = np.full(n, np.nan, dtype=np.float32)

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

            ctx = build_context_matrix(
                group.loc[obs_mask], self.feature_set
            )
            query = group[["log_moneyness", "tau"]].to_numpy(dtype=np.float32)

            ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(self.device)
            ctx_mask_t = torch.ones(1, ctx_t.shape[1], dtype=torch.bool, device=self.device)
            q_t = torch.from_numpy(query).unsqueeze(0).to(self.device)

            forward_out = self.model(ctx_t, ctx_mask_t, q_t)
            mu = forward_out["mu"] if isinstance(forward_out, dict) else forward_out
            idx = group["_orig_idx"].values
            out[idx] = mu.squeeze(0).cpu().numpy()
            if head_kind == "gaussian" and isinstance(forward_out, dict) and "sigma" in forward_out:
                sigma_out[idx] = forward_out["sigma"].squeeze(0).cpu().numpy()
            elif head_kind == "quantile" and isinstance(forward_out, dict) and "quantiles" in forward_out:
                q = forward_out["quantiles"].squeeze(0).cpu().numpy()  # (N, K)
                # `quantiles` is sorted at inference inside MultiOutputDecoder
                q_lo_out[idx] = q[:, 0]
                q_hi_out[idx] = q[:, -1]
                if quantile_levels is None and "quantile_levels" in forward_out:
                    quantile_levels = tuple(
                        float(x)
                        for x in forward_out["quantile_levels"].detach().cpu().numpy().tolist()
                    )

        meta: dict = {"model": "conditional_surface", "head_kind": head_kind}
        if quantile_levels is not None:
            meta["quantile_levels"] = list(quantile_levels)
        return PredictionResult(
            pred=out,
            uncertainty=sigma_out,
            lower=q_lo_out,
            upper=q_hi_out,
            meta=meta,
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


# ── Calibrated conditional predictor (W4 / 2D.4) ─────────────────────────


class CalibratedConditionalPredictor:
    """Fuse and calibrate the W4 uncertainty sources into one Predictor.

    Wraps a *base* :class:`Predictor` (typically a Gaussian- or quantile-head
    :class:`ConditionalSurfacePredictor`) plus a fitted
    :class:`~neural_iv_surface_inference.eval.calibration.Calibrator`, and
    optionally an *ensemble* :class:`Predictor`
    (:class:`EnsembleConditionalPredictor`) whose ``uncertainty`` carries
    per-member disagreement std, and a *masking* callable that returns the 2B.2
    per-point masking-sensitivity std for a frame.

    The emitted :class:`PredictionResult` fills all four slots:

    * ``pred``       — point prediction ``mu`` from the base predictor.
    * ``uncertainty``— fused sigma-unit uncertainty (quadrature of the
      calibrated primary head's sigma plus the calibrated auxiliary signals).
    * ``lower``, ``upper`` — calibrated nominal-``alpha`` interval from the
      primary head (temperature-scaled Gaussian or split-conformal quantile).

    The dict ``meta`` carries ``confidence_score`` so consumers can plug the
    per-point score into 2A.4 abstention curves without re-fusing.
    """

    def __init__(
        self,
        base: Predictor,
        calibrator: Calibrator,
        ensemble: Predictor | None = None,
        masking_fn=None,
    ) -> None:
        self.base = base
        self.calibrator = calibrator
        self.ensemble = ensemble
        self.masking_fn = masking_fn

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        head_kind = self.calibrator.fit_params.head_kind
        base_res = self.base.predict(df)
        mu = np.asarray(base_res.pred, dtype=float)
        kwargs: dict = {"mu": mu}

        if head_kind == "gaussian":
            if base_res.uncertainty is None:
                raise ValueError(
                    "CalibratedConditionalPredictor: base predictor must emit "
                    "sigma in PredictionResult.uncertainty for a gaussian head"
                )
            kwargs["sigma"] = np.asarray(base_res.uncertainty, dtype=float)
        else:  # quantile
            if base_res.lower is None or base_res.upper is None:
                raise ValueError(
                    "CalibratedConditionalPredictor: base predictor must emit "
                    "q_lo/q_hi in PredictionResult.lower/upper for a quantile head"
                )
            kwargs["q_lo"] = np.asarray(base_res.lower, dtype=float)
            kwargs["q_hi"] = np.asarray(base_res.upper, dtype=float)

        if self.ensemble is not None:
            ens_res = self.ensemble.predict(df)
            if ens_res.uncertainty is None:
                raise ValueError(
                    "ensemble Predictor must populate uncertainty (disagreement std)"
                )
            kwargs["disagreement"] = np.asarray(ens_res.uncertainty, dtype=float)

        if self.masking_fn is not None:
            kwargs["masking_std"] = np.asarray(self.masking_fn(df), dtype=float)

        out = self.calibrator.apply(**kwargs)
        meta = {
            "model": "calibrated_conditional",
            "head_kind": head_kind,
            "nominal_alpha": self.calibrator.fit_params.nominal_alpha,
            "confidence_score": out["confidence_score"],
            "base_meta": dict(base_res.meta),
        }
        return PredictionResult(
            pred=out["pred"],
            uncertainty=out["uncertainty"],
            lower=out["lower"],
            upper=out["upper"],
            meta=meta,
        )
