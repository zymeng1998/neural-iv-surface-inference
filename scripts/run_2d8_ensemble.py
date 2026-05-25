#!/usr/bin/env python3
"""2D.8 K-seed ensemble: train K members + score ensemble + emit CSVs.

Composes the existing 2D.3 ``scripts/run_ensemble_train.py`` (training pass)
with a new scoring pass that emits the per-row ensemble val/test predictions
(mean + disagreement_std + per-member columns) and a top-level manifest.

Usage:
    python scripts/run_2d8_ensemble.py --config configs/conditional_2D8_ensemble.yaml

Outputs (under config.paths.results_dir):
    checkpoints/ensemble/seed_<seed>/best_conditional.pt  (Pod-side only)
    checkpoints/ensemble/members.json                     (Pod-side; mirrored)
    val_predictions.csv
    test_predictions.csv
    training_curves.csv
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
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)
from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
)
from neural_iv_surface_inference.utils.io import load_config


MANIFEST_VERSION = 1


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


def _score_ensemble(
    members: list[ConditionalSurfacePredictor],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-row ensemble predictions: mean, disagreement_std, member_0..K-1."""
    per_member: list[np.ndarray] = []
    for m in members:
        res = m.predict(df)
        per_member.append(np.asarray(res.pred, dtype=np.float64))
    stacked = np.stack(per_member, axis=0)  # (K, N)
    mean_pred = stacked.mean(axis=0).astype(np.float32)
    disagreement = stacked.std(axis=0, ddof=0).astype(np.float32)

    out = pd.DataFrame(
        {
            "date": df["date"].values,
            "log_moneyness": df["log_moneyness"].values.astype(np.float32),
            "tau": df["tau"].values.astype(np.float32),
            "observed": df["observed"].values.astype(bool),
            "iv_true": df["implied_volatility"].values.astype(np.float32),
            "iv_clean": df["iv_clean"].values.astype(np.float32)
            if "iv_clean" in df.columns
            else np.full(len(df), np.nan, dtype=np.float32),
            "mu": mean_pred,
            "disagreement_std": disagreement,
        }
    )
    for k in range(stacked.shape[0]):
        out[f"member_{k}"] = stacked[k, :].astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))
    ens_cfg = dict(config.get("ensemble", {}))
    size = int(ens_cfg.get("size", 5))
    all_seeds = list(ens_cfg.get("seeds", []))
    if len(all_seeds) < size:
        raise ValueError(f"ensemble.seeds has {len(all_seeds)} entries < size={size}")
    seeds = [int(s) for s in all_seeds[:size]]

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/2D8"))
    checkpoint_root = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    ensemble_dir = checkpoint_root / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[2D.8] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2D.8] device={device}  size={size}  seeds={seeds}  benchmark={bench_file}")

    df_all = pd.read_parquet(bench_file)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(
        f"[2D.8] rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    train_loader = _build_loader(train_df, batch_size, shuffle=True)
    val_loader = _build_loader(val_df, batch_size, shuffle=False)

    cfg_hash = _cfg_hash(cond_cfg)

    # ── Train K members ──────────────────────────────────────────────
    members_meta: list[dict] = []
    curves_frames: list[pd.DataFrame] = []
    train_total_s = 0.0
    for seed in seeds:
        member_cfg = dict(cond_cfg)
        member_cfg["seed"] = seed
        member_dir = ensemble_dir / f"seed_{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _, history = train_conditional(
            train_loader=train_loader,
            val_loader=val_loader,
            config=member_cfg,
            device=device,
            checkpoint_dir=member_dir,
        )
        elapsed = time.time() - t0
        train_total_s += elapsed
        best_val = float(min(r["val_loss"] for r in history))
        members_meta.append(
            {
                "seed": seed,
                "val_loss": best_val,
                "checkpoint": f"seed_{seed}/best_conditional.pt",
                "epochs": len(history),
                "wall_clock_s": elapsed,
            }
        )
        curve = pd.DataFrame(history)
        curve.insert(0, "seed", seed)
        curves_frames.append(curve)
        print(
            f"[2D.8] seed={seed} done  best_val={best_val:.6f}  "
            f"epochs={len(history)}  ({elapsed:.1f}s)"
        )

    manifest_seed = {
        "version": MANIFEST_VERSION,
        "ensemble_size": len(members_meta),
        "config_hash": cfg_hash,
        "members": members_meta,
    }
    members_json = ensemble_dir / "members.json"
    with open(members_json, "w") as f:
        json.dump(manifest_seed, f, indent=2)
    print(f"[2D.8] wrote {members_json}")

    pd.concat(curves_frames, ignore_index=True).to_csv(
        results_dir / "training_curves.csv", index=False
    )

    # ── Score ensemble ───────────────────────────────────────────────
    print("[2D.8] loading members for scoring …")
    member_predictors: list[ConditionalSurfacePredictor] = []
    for meta in members_meta:
        ckpt = ensemble_dir / meta["checkpoint"]
        member_predictors.append(
            ConditionalSurfacePredictor.from_checkpoint(ckpt, device=device)
        )

    t_score = time.time()
    val_preds = _score_ensemble(member_predictors, val_df)
    test_preds = _score_ensemble(member_predictors, test_df)
    score_elapsed = time.time() - t_score

    val_csv = results_dir / "val_predictions.csv"
    test_csv = results_dir / "test_predictions.csv"
    val_preds.to_csv(val_csv, index=False)
    test_preds.to_csv(test_csv, index=False)

    val_mae = float((val_preds["mu"] - val_preds["iv_true"]).abs().mean())
    test_mae = float((test_preds["mu"] - test_preds["iv_true"]).abs().mean())
    disagreement_stats = {
        "min": float(test_preds["disagreement_std"].min()),
        "mean": float(test_preds["disagreement_std"].mean()),
        "max": float(test_preds["disagreement_std"].max()),
        "any_positive": bool((test_preds["disagreement_std"] > 0).any()),
        "any_negative": bool((test_preds["disagreement_std"] < 0).any()),
    }

    manifest = {
        "story": "2D.8",
        "git_sha": _git_sha(),
        "config_hash": cfg_hash,
        "benchmark_file": str(bench_file),
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "ensemble_size": len(members_meta),
        "seeds": seeds,
        "members": members_meta,
        "ensemble_val_mae": val_mae,
        "ensemble_test_mae": test_mae,
        "disagreement_test_stats": disagreement_stats,
        "train_total_wall_clock_s": train_total_s,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
        "outputs": {
            "members_json": str(members_json),
            "val_predictions": str(val_csv),
            "test_predictions": str(test_csv),
            "training_curves": str(results_dir / "training_curves.csv"),
        },
    }
    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[2D.8] ensemble test_MAE={test_mae:.6f}  val_MAE={val_mae:.6f}  "
        f"disagreement={disagreement_stats}  manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
