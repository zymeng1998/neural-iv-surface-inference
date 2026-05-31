#!/usr/bin/env python3
"""3X.7 runner: masked-MLP baseline on the OTM benchmark.

Trains the fixed Phase-1 masked MLP (``(log_m, tau) -> sigma``, no
conditioning on the observed chain) on ``random40_noiselow_otm`` and emits
the 3X.7 bundle:

    artifacts/runs/3X7/mlp_otm/
        checkpoints/best_mlp.pt          (Pod-side only)
        training_curve.csv
        predictions_val.parquet
        predictions_test.parquet
        manifest.json

This reuses the proven Phase-1 training entrypoints (``train_mlp`` /
``predict_mlp``) verbatim — only the prediction-emission + manifest layer
is new, mirroring the run_2d7_single.py schema so the dirty-vs-OTM MLP MAE
delta is directly comparable.

Usage:
    python scripts/run_3x7_mlp.py --config configs/baseline_3X7_mlp_otm.yaml
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

from neural_iv_surface_inference.utils.io import load_config
from neural_iv_surface_inference.utils.seed import set_seed
from neural_iv_surface_inference.data.loaders import load_benchmark_splits
from neural_iv_surface_inference.training.train import train_mlp, predict_mlp


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


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _predictions_frame(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "date": df["date"].values,
            "log_moneyness": df["log_moneyness"].values.astype(np.float32),
            "tau": df["tau"].values.astype(np.float32),
            "observed": df["observed"].values.astype(bool),
            "iv_true": df["implied_volatility"].values.astype(np.float32),
            "iv_clean": df["iv_clean"].values.astype(np.float32)
            if "iv_clean" in df.columns
            else np.full(n, np.nan, dtype=np.float32),
            "mu": preds.astype(np.float32),
        }
    )


def _mae_block(pred_df: pd.DataFrame, target_col: str) -> dict:
    err = (pred_df["mu"] - pred_df[target_col]).abs()
    obs = pred_df["observed"].values.astype(bool)
    return {
        "overall_mae": float(err.dropna().mean()),
        "observed_mae": float(err[obs].dropna().mean()),
        "unobserved_mae": float(err[~obs].dropna().mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    mlp_cfg = dict(config.get("mlp", {}))
    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/3X7/mlp_otm"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[3X.7] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config.get("data", {}).get("batch_size", 256))
    print(f"[3X.7] device={device}  benchmark={bench_file}  batch_size={batch_size}")

    data = load_benchmark_splits(
        bench_file,
        batch_size=batch_size,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
    )
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    print(
        f"[3X.7] rows: train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,}"
    )

    # ── Train (from scratch — D8) ────────────────────────────────────
    t0 = time.time()
    model, history = train_mlp(
        data["train_loader"],
        data["val_loader"],
        config=mlp_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    train_elapsed = time.time() - t0
    print(f"[3X.7] training done in {train_elapsed:.1f}s  epochs={len(history)}")

    # ── Score (val/test loaders are unshuffled -> row-aligned) ───────
    t_score = time.time()
    val_pred = predict_mlp(model, data["val_loader"], device=device)
    test_pred = predict_mlp(model, data["test_loader"], device=device)
    score_elapsed = time.time() - t_score

    if len(val_pred) != len(val_df) or len(test_pred) != len(test_df):
        print(
            f"[3X.7] prediction/row mismatch: "
            f"val {len(val_pred)}!={len(val_df)} or "
            f"test {len(test_pred)}!={len(test_df)}",
            file=sys.stderr,
        )
        return 3

    val_preds = _predictions_frame(val_df, val_pred)
    test_preds = _predictions_frame(test_df, test_pred)

    finite_ok = bool(
        np.isfinite(val_preds["mu"]).all() and np.isfinite(test_preds["mu"]).all()
    )

    val_parquet = results_dir / "predictions_val.parquet"
    test_parquet = results_dir / "predictions_test.parquet"
    val_preds.to_parquet(val_parquet, index=False)
    test_preds.to_parquet(test_parquet, index=False)

    curve = pd.DataFrame(history)
    curve_csv = results_dir / "training_curve.csv"
    curve.to_csv(curve_csv, index=False)

    # MAE against both the noisy target (iv_true == implied_volatility,
    # matches the 2D.7 / RBF-floor convention) and the clean target.
    val_mae_true = _mae_block(val_preds, "iv_true")
    test_mae_true = _mae_block(test_preds, "iv_true")
    val_mae_clean = _mae_block(val_preds, "iv_clean")
    test_mae_clean = _mae_block(test_preds, "iv_clean")

    final = history[-1]
    manifest = {
        "story": "3X.7",
        "model": "masked_mlp_baseline",
        "git_sha": _git_sha(),
        "config_hash": _cfg_hash(config),
        "benchmark_file": str(bench_file),
        "benchmark_sha256_16": _file_sha(bench_file),
        "seed": seed,
        "from_scratch": True,
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "epochs_completed": len(history),
        "final_train_loss": float(final["train_loss"]),
        "final_val_obs_mse": float(final["val_obs_mse"]),
        "best_val_obs_mse": float(min(r["val_obs_mse"] for r in history)),
        "val_mae_vs_iv_true": val_mae_true,
        "test_mae_vs_iv_true": test_mae_true,
        "val_mae_vs_iv_clean": val_mae_clean,
        "test_mae_vs_iv_clean": test_mae_clean,
        "predictions_finite": finite_ok,
        "train_wall_clock_s": train_elapsed,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
        "outputs": {
            "checkpoint": str(checkpoint_dir / "best_mlp.pt"),
            "predictions_val": str(val_parquet),
            "predictions_test": str(test_parquet),
            "training_curve": str(curve_csv),
        },
    }
    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[3X.7] DONE  test_MAE(iv_true)={test_mae_true['overall_mae']:.6f}  "
        f"val_MAE(iv_true)={val_mae_true['overall_mae']:.6f}  "
        f"test_MAE(iv_clean)={test_mae_clean['overall_mae']:.6f}  "
        f"finite={finite_ok}  manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
