#!/usr/bin/env python3
"""3A.3 single-config runner: decoder-only retrain on the frozen 2D.7 encoder.

Mirrors ``scripts/run_2d7_single.py`` but:

  * passes ``coord_encoding`` through to ``ConditionalSurfaceModel``;
  * loads encoder weights from ``conditional.encoder_init_from`` and
    freezes them when ``conditional.freeze_encoder: true``;
  * verifies, at the end of training, that the encoder weights in the
    best-checkpoint are bit-for-bit equal to the source encoder weights;
  * writes ``training_curves.csv`` and parquet predictions (per spec),
    plus a 3A-shaped manifest with the encoder init flags and source
    checkpoint SHA-256.

Usage:
    python scripts/run_3a_decoder_only.py --config configs/conditional_3A3_fourier.yaml
    python scripts/run_3a_decoder_only.py --config configs/conditional_3A3_raw.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


def _cfg_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_model_from_cfg(
    cfg: dict[str, Any], head_cfg: dict[str, Any]
) -> ConditionalSurfaceModel:
    return ConditionalSurfaceModel(
        context_dim=int(cfg.get("context_dim", 3)),
        coord_dim=int(cfg.get("coord_dim", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        n_elem_layers=int(cfg.get("n_elem_layers", 2)),
        n_post_layers=int(cfg.get("n_post_layers", 1)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        head=dict(head_cfg),
        coord_encoding=cfg.get("coord_encoding"),
    )


@torch.no_grad()
def _score_split(
    model: ConditionalSurfaceModel,
    df: pd.DataFrame,
    head_kind: str,
    device: torch.device,
) -> pd.DataFrame:
    """Per-row predictions, iterated by date (parity with run_2d7_single)."""
    model.eval()
    n = len(df)
    mu_out = np.full(n, np.nan, dtype=np.float32)
    sigma_out = np.full(n, np.nan, dtype=np.float32)
    log_sigma2_out = np.full(n, np.nan, dtype=np.float32)

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
        mu = out["mu"].squeeze(0).cpu().numpy()
        mu_out[idx] = mu
        if head_kind == "gaussian":
            sigma_out[idx] = out["sigma"].squeeze(0).cpu().numpy()
            log_sigma2_out[idx] = out["log_sigma2"].squeeze(0).cpu().numpy()

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
    return pred_df


def _mae(pred_df: pd.DataFrame, target_col: str = "iv_true") -> float:
    err = (pred_df["mu"] - pred_df[target_col]).abs()
    return float(err.dropna().mean())


def _encoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k[len("encoder.") :]: v.detach().cpu()
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }


def _assert_encoder_equal(
    trained_state: dict[str, torch.Tensor],
    src_state: dict[str, torch.Tensor],
) -> bool:
    trained_enc = _encoder_state(trained_state)
    src_enc = _encoder_state(src_state)
    if set(trained_enc.keys()) != set(src_enc.keys()):
        return False
    for k, v in trained_enc.items():
        if not torch.equal(v, src_enc[k]):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))
    head_cfg = dict(cond_cfg.get("head", {"kind": "point"}))
    head_kind = str(head_cfg.get("kind", "point"))
    coord_encoding_cfg = dict(cond_cfg.get("coord_encoding", {"kind": "raw"}))
    freeze_encoder = bool(cond_cfg.get("freeze_encoder", False))
    encoder_init_from = cond_cfg.get("encoder_init_from")

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/3A/_unknown"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[3A.3] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    encoder_src = Path(encoder_init_from) if encoder_init_from else None
    encoder_src_sha = None
    if encoder_src is not None:
        if not encoder_src.exists():
            print(
                f"[3A.3] encoder_init_from not found: {encoder_src}", file=sys.stderr
            )
            return 3
        encoder_src_sha = _file_sha256(encoder_src)
        print(f"[3A.3] encoder_init_from={encoder_src}  sha256={encoder_src_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[3A.3] coord_encoding={coord_encoding_cfg.get('kind', 'raw')}  "
        f"freeze_encoder={freeze_encoder}  head_kind={head_kind}  device={device}"
    )

    df_all = pd.read_parquet(bench_file)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(
        f"[3A.3] rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    train_loader = _build_loader(train_df, batch_size, shuffle=True)
    val_loader = _build_loader(val_df, batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cond_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    train_elapsed = time.time() - t0
    print(f"[3A.3] training done in {train_elapsed:.1f}s  epochs={len(history)}")

    ckpt_path = checkpoint_dir / "best_conditional.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model_from_cfg(ckpt["config"], head_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    encoder_equal = None
    if encoder_src is not None:
        src_payload = torch.load(str(encoder_src), map_location="cpu", weights_only=False)
        src_state = src_payload.get("model_state_dict", src_payload)
        encoder_equal = _assert_encoder_equal(model.state_dict(), src_state)
        if not encoder_equal:
            print(
                "[3A.3] WARNING: encoder weights in trained checkpoint "
                "differ from source — freeze_encoder may not be honoured",
                file=sys.stderr,
            )

    t_score = time.time()
    val_preds = _score_split(model, val_df, head_kind, device)
    test_preds = _score_split(model, test_df, head_kind, device)
    score_elapsed = time.time() - t_score

    val_parquet = results_dir / "predictions_val.parquet"
    test_parquet = results_dir / "predictions_test.parquet"
    val_preds.to_parquet(val_parquet, index=False)
    test_preds.to_parquet(test_parquet, index=False)

    curve = pd.DataFrame(history)
    curve_csv = results_dir / "training_curves.csv"
    curve.to_csv(curve_csv, index=False)

    val_mae = _mae(val_preds)
    test_mae = _mae(test_preds)
    final = history[-1]

    manifest = {
        "story": "3A.3",
        "head_kind": head_kind,
        "coord_encoding": coord_encoding_cfg,
        "freeze_encoder": freeze_encoder,
        "encoder_init_from": str(encoder_src) if encoder_src else None,
        "encoder_init_sha256": encoder_src_sha,
        "encoder_weights_equal_source": encoder_equal,
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
        "train_wall_clock_s": train_elapsed,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
        "outputs": {
            "checkpoint": str(ckpt_path),
            "predictions_val": str(val_parquet),
            "predictions_test": str(test_parquet),
            "training_curves": str(curve_csv),
        },
    }
    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[3A.3] coord={coord_encoding_cfg.get('kind', 'raw')}  "
        f"test_MAE_mu={test_mae:.6f}  val_MAE_mu={val_mae:.6f}  "
        f"encoder_equal={encoder_equal}  manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
