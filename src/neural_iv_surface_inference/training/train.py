"""Training loop for the masked MLP baseline.

Trains the BaselineMLP on observed points from benchmark datasets.
Supports:
  - Config-driven hyperparameters (via YAML)
  - Checkpointing (best val loss)
  - Per-epoch logging with train/val metrics
  - Early stopping
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.losses import MaskedMSELoss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device).unsqueeze(1)
        mask = batch["observed"].to(device)

        # Skip batch if no observed points
        if mask.sum() == 0:
            continue

        pred = model(features)
        loss = criterion(pred, target, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on val/test set, return loss on observed and MAE on unobserved."""
    model.eval()
    total_obs_loss = 0.0
    total_unobs_ae = 0.0
    n_obs = 0
    n_unobs = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device).unsqueeze(1)
        mask = batch["observed"].to(device)

        pred = model(features)

        # Loss on observed
        if mask.sum() > 0:
            obs_loss = ((pred[mask] - target[mask]) ** 2).sum().item()
            total_obs_loss += obs_loss
            n_obs += mask.sum().item()

        # MAE on unobserved
        unobs_mask = ~mask
        if unobs_mask.sum() > 0:
            unobs_ae = (pred[unobs_mask] - target[unobs_mask]).abs().sum().item()
            total_unobs_ae += unobs_ae
            n_unobs += unobs_mask.sum().item()

    return {
        "obs_mse": total_obs_loss / max(n_obs, 1),
        "unobs_mae": total_unobs_ae / max(n_unobs, 1),
        "n_obs": n_obs,
        "n_unobs": n_unobs,
    }


def train_mlp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[BaselineMLP, list[dict]]:
    """Full training loop for the masked MLP baseline.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
        From ``load_benchmark_splits``.
    config : dict
        Must contain: hidden_dim, n_layers, dropout, learning_rate,
        weight_decay, epochs, patience (for early stopping).
    device : torch.device, optional
        Defaults to CUDA if available.
    checkpoint_dir : Path, optional
        Where to save best model checkpoint.

    Returns
    -------
    (model, history) — trained model and per-epoch metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineMLP(
        input_dim=config.get("input_dim", 2),
        hidden_dim=config.get("hidden_dim", 128),
        n_layers=config.get("n_layers", 3),
        dropout=config.get("dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    criterion = MaskedMSELoss()
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"  Training MLP: {sum(p.numel() for p in model.parameters()):,} params, "
          f"device={device}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["obs_mse"]

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_obs_mse": val_loss,
            "val_unobs_mae": val_metrics["unobs_mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        }
        history.append(record)

        # Logging
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.6f}  "
                f"val_obs_mse={val_loss:.6f}  "
                f"val_unobs_mae={val_metrics['unobs_mae']:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.1f}s)"
            )

        # Checkpointing (best val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / "best_mlp.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                }, ckpt_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best checkpoint if saved
    if checkpoint_dir:
        ckpt_path = checkpoint_dir / "best_mlp.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded best checkpoint from epoch {ckpt['epoch']}")

    return model, history


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run inference and return predictions as numpy array."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds = []
    for batch in loader:
        features = batch["features"].to(device)
        pred = model(features).cpu().numpy().flatten()
        preds.append(pred)
    return np.concatenate(preds)
