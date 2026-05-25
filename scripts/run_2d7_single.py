#!/usr/bin/env python3
"""2D.7 single-config runner: train + score val/test + emit CSVs + manifest.

Wraps the existing ``train_conditional`` entrypoint (no behavioral change to
2D.2 code) and adds the prediction-emission + manifest layer that downstream
stories (2D.4, 2D.5, 2D.9) consume.

Usage:
    python scripts/run_2d7_single.py --config configs/conditional_2D7_point_control.yaml
    python scripts/run_2d7_single.py --config configs/conditional_2D7_gaussian.yaml
    python scripts/run_2d7_single.py --config configs/conditional_2D7_quantile.yaml

Outputs (under config.paths.results_dir):
    checkpoints/best_conditional.pt   (Pod-side only — not pulled back)
    training_curve.csv
    val_predictions.csv
    test_predictions.csv
    manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)
from neural_iv_surface_inference.utils.io import load_config


def _build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ConditionalIVSurfaceDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_conditional,
        num_workers=0,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _cfg_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_model_from_cfg(cfg: dict, head_cfg: dict) -> ConditionalSurfaceModel:
    return ConditionalSurfaceModel(
        context_dim=int(cfg.get("context_dim", 3)),
        coord_dim=int(cfg.get("coord_dim", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        n_elem_layers=int(cfg.get("n_elem_layers", 2)),
        n_post_layers=int(cfg.get("n_post_layers", 1)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        head=dict(head_cfg),
    )


@torch.no_grad()
def _score_split(
    model: ConditionalSurfaceModel,
    df: pd.DataFrame,
    head_kind: str,
    device: torch.device,
) -> pd.DataFrame:
    """Per-row predictions on a benchmark frame.

    Iterates by date (matching ConditionalSurfacePredictor) and pokes the
    model directly so all head outputs are captured.
    """
    model.eval()
    n = len(df)
    mu_out = np.full(n, np.nan, dtype=np.float32)
    sigma_out = np.full(n, np.nan, dtype=np.float32)
    log_sigma2_out = np.full(n, np.nan, dtype=np.float32)
    # quantile heads use these (filled only if applicable)
    q_lo = np.full(n, np.nan, dtype=np.float32)
    q_med = np.full(n, np.nan, dtype=np.float32)
    q_hi = np.full(n, np.nan, dtype=np.float32)

    df_indexed = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    for _date, group in df_indexed.groupby("date", sort=False):
        obs = group["observed"].values.astype(bool)
        if obs.sum() == 0:
            continue

        ctx = group.loc[
            obs, ["log_moneyness", "tau", "implied_volatility"]
        ].to_numpy(dtype=np.float32)
        query = group[["log_moneyness", "tau"]].to_numpy(dtype=np.float32)
        idx = group["_orig_idx"].values

        ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
        ctx_mask_t = torch.ones(1, ctx_t.shape[1], dtype=torch.bool, device=device)
        q_t = torch.from_numpy(query).unsqueeze(0).to(device)

        out = model(ctx_t, ctx_mask_t, q_t)
        if isinstance(out, dict):
            mu = out["mu"].squeeze(0).cpu().numpy()
            mu_out[idx] = mu
            if head_kind == "gaussian":
                sigma_out[idx] = out["sigma"].squeeze(0).cpu().numpy()
                log_sigma2_out[idx] = out["log_sigma2"].squeeze(0).cpu().numpy()
            elif head_kind == "quantile":
                q = out["quantiles"].squeeze(0).cpu().numpy()  # (N, K)
                # Already sorted at eval time per model code.
                q_lo[idx] = q[:, 0]
                q_med[idx] = q[:, q.shape[1] // 2]
                q_hi[idx] = q[:, -1]
        else:
            mu_out[idx] = out.squeeze(0).cpu().numpy()

    pred_df = pd.DataFrame(
        {
            "date": df["date"].values,
            "log_moneyness": df["log_moneyness"].values.astype(np.float32),
            "tau": df["tau"].values.astype(np.float32),
            "observed": df["observed"].values.astype(bool),
            "iv_true": df["implied_volatility"].values.astype(np.float32),
            "iv_clean": df["iv_clean"].values.astype(np.float32)
            if "iv_clean" in df.columns
            else np.full(n, np.nan, dtype=np.float32),
            "mu": mu_out,
        }
    )
    if head_kind == "gaussian":
        pred_df["sigma"] = sigma_out
        pred_df["log_sigma2"] = log_sigma2_out
    elif head_kind == "quantile":
        pred_df["q_lo"] = q_lo
        pred_df["q_med"] = q_med
        pred_df["q_hi"] = q_hi

    return pred_df


def _mae(pred_df: pd.DataFrame, target_col: str = "iv_true") -> float:
    err = (pred_df["mu"] - pred_df[target_col]).abs()
    return float(err.dropna().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))
    head_cfg = dict(cond_cfg.get("head", {"kind": "point"}))
    head_kind = str(head_cfg.get("kind", "point"))

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/2D7/_unknown"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[2D.7] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2D.7] head_kind={head_kind}  device={device}  benchmark={bench_file}")

    df_all = pd.read_parquet(bench_file)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(
        f"[2D.7] rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    train_loader = _build_loader(train_df, batch_size, shuffle=True)
    val_loader = _build_loader(val_df, batch_size, shuffle=False)

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cond_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    train_elapsed = time.time() - t0
    print(f"[2D.7] training done in {train_elapsed:.1f}s  epochs={len(history)}")

    # Reload best checkpoint
    ckpt_path = checkpoint_dir / "best_conditional.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model_from_cfg(ckpt["config"], head_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Score ────────────────────────────────────────────────────────
    t_score = time.time()
    val_preds = _score_split(model, val_df, head_kind, device)
    test_preds = _score_split(model, test_df, head_kind, device)
    score_elapsed = time.time() - t_score

    val_csv = results_dir / "val_predictions.csv"
    test_csv = results_dir / "test_predictions.csv"
    val_preds.to_csv(val_csv, index=False)
    test_preds.to_csv(test_csv, index=False)

    # ── Training curve ───────────────────────────────────────────────
    curve = pd.DataFrame(history)
    curve_csv = results_dir / "training_curve.csv"
    curve.to_csv(curve_csv, index=False)

    val_mae = _mae(val_preds)
    test_mae = _mae(test_preds)
    final = history[-1]

    # Quantile monotonicity sanity (must hold on test set after sort).
    qmono_ok = True
    if head_kind == "quantile":
        qmono_ok = bool(
            (
                (test_preds["q_lo"] <= test_preds["q_med"])
                & (test_preds["q_med"] <= test_preds["q_hi"])
            )
            .dropna()
            .all()
        )

    manifest = {
        "story": "2D.7",
        "head_kind": head_kind,
        "git_sha": _git_sha(),
        "config_hash": _cfg_hash(cond_cfg),
        "benchmark_file": str(bench_file),
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "epochs_completed": len(history),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "best_val_loss": float(min(r["val_loss"] for r in history)),
        "val_mae_mu": val_mae,
        "test_mae_mu": test_mae,
        "quantile_monotonicity_ok": qmono_ok,
        "train_wall_clock_s": train_elapsed,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
        "outputs": {
            "checkpoint": str(ckpt_path),
            "val_predictions": str(val_csv),
            "test_predictions": str(test_csv),
            "training_curve": str(curve_csv),
        },
    }
    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[2D.7] head={head_kind}  test_MAE_mu={test_mae:.6f}  "
        f"val_MAE_mu={val_mae:.6f}  qmono_ok={qmono_ok}  manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
