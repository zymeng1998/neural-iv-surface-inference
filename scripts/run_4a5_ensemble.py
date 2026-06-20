#!/usr/bin/env python3
"""4A.5 K=5 deep ensemble of the RESIDUAL point head (ADR 0010).

Clone of ``scripts/run_3x10_anp_ensemble.py`` adapted to the RBF-prior
residual hybrid:
  - each member trains with ``target_mode="residual"`` (fits r = iv_clean − rbf_pred);
  - members are scored with the validated 4A.4 ``_score_split`` (point head →
    residual ``mu``); the ensemble takes the mean over members' residual
    predictions, adds ``rbf_pred`` to form the hybrid ``sigma_hat``, and the
    disagreement is the std of the per-member residual prediction (= std of
    sigma_hat, since rbf_pred is a per-row constant shift);
  - the ensemble MAE is reported vs ``iv_clean`` (RBF-comparable) and
    ``iv_true`` (3X.10-comparable).

Seeds / recipe / K identical to 3X.10. From-scratch members.

Usage:
    python scripts/run_4a5_ensemble.py --config configs/conditional_4A5_residual_ensemble_otm.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.training.train_conditional import train_conditional
from neural_iv_surface_inference.utils.io import load_config

# Reuse the validated 4A.4 helpers (residual model build + hybrid scoring).
from run_4a4_residual import (  # noqa: E402
    _build_model_from_cfg,
    _cfg_hash,
    _git_sha,
    _score_split,
    RBF_OTM_TEST_MAE,
)

ANP_3X10_TEST_MAE = 0.012200304307043552  # 3X.10 ensemble point, vs iv_true
ANP_4A4_POINT_TEST_MAE = 0.006138         # 4A.4 single residual point, vs iv_clean


def _build_loader(df, batch_size, shuffle, target_mode):
    return DataLoader(
        ConditionalIVSurfaceDataset(df, target_mode=target_mode),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_conditional, num_workers=0,
    )


def _score_ensemble(member_models, df, device):
    """Mean over members' residual predictions → hybrid σ̂ = rbf_pred + mean.

    Disagreement = std of the per-member residual prediction across members.
    """
    resid_stack = []
    for m in member_models:
        pred = _score_split(m, df, "point", device)  # mu_residual + rbf in pred["mu"]
        resid_stack.append(pred["mu_residual"].to_numpy(dtype=np.float64))
    stacked = np.stack(resid_stack, axis=0)  # (K, N) residual preds
    mean_resid = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0, ddof=0).astype(np.float32)
    rbf = df["rbf_pred"].to_numpy(dtype=np.float64)
    out = pd.DataFrame({
        "date": df["date"].values,
        "log_moneyness": df["log_moneyness"].values.astype(np.float32),
        "tau": df["tau"].values.astype(np.float32),
        "observed": df["observed"].values.astype(bool),
        "iv_true": df["implied_volatility"].values.astype(np.float32),
        "iv_clean": df["iv_clean"].values.astype(np.float32),
        "rbf_pred": rbf.astype(np.float32),
        "mu_residual_mean": mean_resid.astype(np.float32),
        "mu": (rbf + mean_resid).astype(np.float32),  # hybrid σ̂
        "disagreement_std": disagreement,
    })
    for k in range(stacked.shape[0]):
        out[f"member_resid_{k}"] = stacked[k, :].astype(np.float32)
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
    seeds = [int(s) for s in list(ens_cfg.get("seeds", []))[:size]]
    if len(seeds) < size:
        raise ValueError(f"ensemble.seeds has < size={size} entries")
    target_mode = str(config.get("data", {}).get("target_mode", "residual"))

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/4A5"))
    ensemble_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints")) / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[4A.5] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4A.5] device={device}  size={size}  seeds={seeds}  "
          f"target_mode={target_mode}  benchmark={bench_file}")

    df_all = pd.read_parquet(bench_file)
    if "rbf_pred" not in df_all.columns:
        print("[4A.5] benchmark missing 'rbf_pred' — use the residual parquet (4A.3)",
              file=sys.stderr)
        return 2
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(f"[4A.5] rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    train_loader = _build_loader(train_df, batch_size, True, target_mode)
    val_loader = _build_loader(val_df, batch_size, False, target_mode)
    cfg_hash = _cfg_hash(cond_cfg)
    head_cfg = dict(cond_cfg.get("head", {"kind": "point"}))

    members_meta = []
    curves = []
    train_total_s = 0.0
    for seed in seeds:
        member_cfg = dict(cond_cfg)
        member_cfg["seed"] = seed
        member_dir = ensemble_dir / f"seed_{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _, history = train_conditional(
            train_loader=train_loader, val_loader=val_loader,
            config=member_cfg, device=device, checkpoint_dir=member_dir,
        )
        elapsed = time.time() - t0
        train_total_s += elapsed
        best_val = float(min(r["val_loss"] for r in history))
        members_meta.append({"seed": seed, "val_loss": best_val,
                             "checkpoint": f"seed_{seed}/best_conditional.pt",
                             "epochs": len(history), "wall_clock_s": elapsed})
        c = pd.DataFrame(history); c.insert(0, "seed", seed); curves.append(c)
        print(f"[4A.5] seed={seed} done  best_val={best_val:.6f}  epochs={len(history)}  ({elapsed:.1f}s)")

    pd.concat(curves, ignore_index=True).to_csv(results_dir / "training_curves.csv", index=False)

    # Load members for scoring (residual point models).
    member_models = []
    for meta in members_meta:
        ckpt = torch.load(str(ensemble_dir / meta["checkpoint"]),
                          map_location=device, weights_only=False)
        m = _build_model_from_cfg(ckpt["config"], head_cfg).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        member_models.append(m)

    t_score = time.time()
    val_preds = _score_ensemble(member_models, val_df, device)
    test_preds = _score_ensemble(member_models, test_df, device)
    score_elapsed = time.time() - t_score
    val_preds.to_csv(results_dir / "val_predictions.csv", index=False)
    test_preds.to_csv(results_dir / "test_predictions.csv", index=False)

    test_mae = float((test_preds["mu"] - test_preds["iv_clean"]).abs().mean())
    val_mae = float((val_preds["mu"] - val_preds["iv_clean"]).abs().mean())
    test_mae_ivtrue = float((test_preds["mu"] - test_preds["iv_true"]).abs().mean())
    dis = test_preds["disagreement_std"]
    disagreement_stats = {"min": float(dis.min()), "mean": float(dis.mean()),
                          "max": float(dis.max()), "any_positive": bool((dis > 0).any())}

    manifest = {
        "story": "4A.5",
        "head_kind": "point",
        "target_mode": target_mode,
        "hybrid": "sigma_hat = rbf_pred + mean_k(f_theta^k residual)",
        "decoder_kind": str(cond_cfg.get("decoder_kind", "anp")),
        "anp": cond_cfg.get("anp"),
        "git_sha": _git_sha(),
        "config_hash": cfg_hash,
        "benchmark_file": str(bench_file),
        "splits": {"train_rows": int(len(train_df)), "val_rows": int(len(val_df)),
                   "test_rows": int(len(test_df))},
        "ensemble_size": len(members_meta),
        "seeds": seeds,
        "members": members_meta,
        "ensemble_val_mae": val_mae,                # vs iv_clean (primary)
        "ensemble_test_mae": test_mae,              # vs iv_clean (RBF-comparable)
        "ensemble_test_mae_vs_iv_true": test_mae_ivtrue,
        "rbf_floor_test_mae": RBF_OTM_TEST_MAE,
        "delta_vs_rbf_test": test_mae - RBF_OTM_TEST_MAE,
        "delta_vs_4a4_single_point": test_mae - ANP_4A4_POINT_TEST_MAE,
        "anp_3x10_ensemble_test_mae_ivtrue": ANP_3X10_TEST_MAE,
        "disagreement_test_stats": disagreement_stats,
        "train_total_wall_clock_s": train_total_s,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
    }
    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[4A.5] ensemble hybrid_test_MAE(vs iv_clean)={test_mae:.6f}  "
          f"(vs iv_true)={test_mae_ivtrue:.6f}  RBF_floor={RBF_OTM_TEST_MAE:.6f}  "
          f"delta_vs_RBF={test_mae - RBF_OTM_TEST_MAE:+.6f}  "
          f"disagreement_mean={disagreement_stats['mean']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
