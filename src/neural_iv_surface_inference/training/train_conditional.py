"""Training loop for the W3 conditional surface model.

Trains ``ConditionalSurfaceModel`` (2C.3) over date-grouped batches produced
by ``ConditionalIVSurfaceDataset`` + ``collate_conditional`` (2C.2). Mirrors
the baseline ``train_mlp`` conventions — config-driven hyperparameters,
best-val checkpointing, per-epoch logging, early stopping, seed control — so
the conditional model is trained and reported on the same footing as the
Phase 1 baselines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_iv_surface_inference.models.conditional_surface import (
    DEFAULT_QUANTILES,
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.models.losses import (
    gaussian_nll_loss,
    pinball_loss,
)
from neural_iv_surface_inference.utils.seed import set_seed


def masked_query_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over query points, ignoring padded positions.

    Parameters
    ----------
    pred   : (B, Q) float tensor
    target : (B, Q) float tensor
    mask   : (B, Q) bool tensor — True for real query points
    """
    m = mask.to(pred.dtype)
    denom = m.sum().clamp_min(1.0)
    sq = ((pred - target) ** 2) * m
    return sq.sum() / denom


def compute_head_loss(
    out: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    head_kind: str,
) -> torch.Tensor:
    """Dispatch the per-head loss for the W4 conditional model (2D.2).

    ``point``     → masked MSE on ``out["mu"]``.
    ``gaussian``  → masked Gaussian NLL on ``out["mu"], out["sigma"]``.
    ``quantile``  → masked pinball loss on ``out["quantiles"]`` with the
                     per-decoder ``out["quantile_levels"]``.
    """
    if head_kind == "point":
        return masked_query_mse(out["mu"], target, mask)
    if head_kind == "gaussian":
        return gaussian_nll_loss(out["mu"], out["sigma"], target, mask)
    if head_kind == "quantile":
        return pinball_loss(
            out["quantiles"], target, mask, out["quantile_levels"]
        )
    raise ValueError(f"unsupported head_kind: {head_kind!r}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_kind: str = "point",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        out = model(batch["context"], batch["context_mask"], batch["query"])
        loss = compute_head_loss(
            out, batch["target"], batch["query_mask"], head_kind
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_kind: str = "point",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_obs_ae = 0.0
    total_unobs_ae = 0.0
    n_obs = 0
    n_unobs = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        out = model(batch["context"], batch["context_mask"], batch["query"])
        loss = compute_head_loss(
            out, batch["target"], batch["query_mask"], head_kind
        )
        total_loss += float(loss.detach().cpu())
        n_batches += 1

        mu = out["mu"]
        diff = (mu - batch["target"]).abs()
        qmask = batch["query_mask"]
        obs_mask = qmask & batch["query_observed"]
        unobs_mask = qmask & (~batch["query_observed"])
        if obs_mask.any():
            total_obs_ae += float(diff[obs_mask].sum().cpu())
            n_obs += int(obs_mask.sum().cpu())
        if unobs_mask.any():
            total_unobs_ae += float(diff[unobs_mask].sum().cpu())
            n_unobs += int(unobs_mask.sum().cpu())

    return {
        "loss": total_loss / max(n_batches, 1),
        "obs_mae": total_obs_ae / max(n_obs, 1),
        "unobs_mae": total_unobs_ae / max(n_unobs, 1),
        "n_obs": n_obs,
        "n_unobs": n_unobs,
    }


def train_conditional(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device | None = None,
    checkpoint_dir: Path | None = None,
    log_every: int = 5,
) -> tuple[ConditionalSurfaceModel, list[dict]]:
    """Train the conditional surface model.

    Parameters
    ----------
    train_loader, val_loader
        DataLoaders built from ``ConditionalIVSurfaceDataset`` +
        ``collate_conditional``.
    config
        Hyperparameters. Recognized keys: ``context_dim``, ``coord_dim``,
        ``hidden_dim``, ``latent_dim``, ``n_elem_layers``, ``n_post_layers``,
        ``n_decoder_layers``, ``learning_rate``, ``weight_decay``, ``epochs``,
        ``patience``, ``seed``.
    device
        Defaults to CUDA if available, else CPU.
    checkpoint_dir
        Where to write ``best_conditional.pt``. Created on demand.
    log_every
        Print a progress line every N epochs (set to a large value to silence).

    Returns
    -------
    (model, history) — the trained model (loaded from best checkpoint if
    written) and per-epoch metric records.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    head_cfg_in = config.get("head") or {"kind": "point"}
    head_cfg = dict(head_cfg_in)
    head_kind = str(head_cfg.get("kind", "point"))
    if head_kind == "quantile" and "quantiles" not in head_cfg:
        head_cfg["quantiles"] = list(DEFAULT_QUANTILES)

    coord_encoding_cfg = config.get("coord_encoding")
    decoder_kind = str(config.get("decoder_kind", "deepsets"))
    anp_cfg = config.get("anp")
    # 3C.2 / ADR 0008: feature_set is authoritative for the context dim and is
    # persisted into the saved checkpoint ``config`` block below so
    # ``ConditionalSurfacePredictor.from_checkpoint`` reconstructs it without
    # an operator override. Defaults to "minimal" (legacy 3-dim context).
    feature_set = str(config.get("feature_set", "minimal"))
    context_dim_cfg = config.get("context_dim")
    # Record the resolved feature_set on the config that gets persisted into
    # the checkpoint, so it is always present even when the caller defaulted.
    config = {**config, "feature_set": feature_set}

    model = ConditionalSurfaceModel(
        context_dim=int(context_dim_cfg) if context_dim_cfg is not None else None,
        coord_dim=int(config.get("coord_dim", 2)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        latent_dim=int(config.get("latent_dim", 32)),
        n_elem_layers=int(config.get("n_elem_layers", 2)),
        n_post_layers=int(config.get("n_post_layers", 1)),
        n_decoder_layers=int(config.get("n_decoder_layers", 3)),
        head=head_cfg,
        coord_encoding=coord_encoding_cfg,
        decoder_kind=decoder_kind,  # type: ignore[arg-type]
        anp=anp_cfg,
        feature_set=feature_set,
    ).to(device)

    # 3A.3: optionally seed the encoder from a saved checkpoint
    # (loads only ``encoder.*`` keys from a ConditionalSurfaceModel state dict).
    encoder_init_from = config.get("encoder_init_from")
    if encoder_init_from:
        src_path = Path(encoder_init_from)
        if not src_path.exists():
            raise FileNotFoundError(
                f"encoder_init_from checkpoint not found: {src_path}"
            )
        payload = torch.load(str(src_path), map_location=device, weights_only=False)
        src_state = payload.get("model_state_dict", payload)
        enc_state = {
            k[len("encoder.") :]: v
            for k, v in src_state.items()
            if k.startswith("encoder.")
        }
        if not enc_state:
            raise ValueError(
                f"no encoder.* keys found in {src_path}"
            )
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=True)
        print(
            f"  Encoder seeded from {src_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # 3A.3: optionally freeze encoder parameters (decoder-only retrain).
    freeze_encoder = bool(config.get("freeze_encoder", False))
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"  Encoder frozen — training {sum(p.numel() for p in trainable):,} "
            f"of {sum(p.numel() for p in model.parameters()):,} parameters"
        )
    else:
        trainable = list(model.parameters())

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 10))

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Training Conditional: {n_params:,} params, device={device}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, head_kind=head_kind
        )
        val_metrics = evaluate(
            model, val_loader, device, head_kind=head_kind
        )
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_obs_mae": val_metrics["obs_mae"],
            "val_unobs_mae": val_metrics["unobs_mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        }
        history.append(record)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"val_obs_mae={val_metrics['obs_mae']:.6f}  "
                f"val_unobs_mae={val_metrics['unobs_mae']:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.1f}s)"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_dir is not None:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / "best_conditional.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    ckpt_path,
                )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / "best_conditional.pt"
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            print(f"  Loaded best checkpoint from epoch {payload['epoch']}")

    return model, history
