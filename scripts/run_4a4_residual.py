#!/usr/bin/env python3
"""4A.4 single-config runner: RBF-prior residual hybrid (ADR 0010).

Clone of ``scripts/run_3x9_anp.py``. The only substantive changes:
  - the dataset is built with ``target_mode="residual"`` so the ANP model
    ``f_theta`` fits the additive residual ``r = iv_clean - rbf_pred``;
  - scoring reconstructs the hybrid ``sigma_hat = rbf_pred + f_theta`` (and
    shifts the quantile bands by ``rbf_pred``) before computing MAE;
  - the hybrid MAE is reported both vs ``iv_clean`` (apples-to-apples with
    the RBF floor 0.00613, 3X.6) and vs ``iv_true`` (comparable to 3X.9).

Architecture / hparams / seed are identical to 3X.9 — do NOT change.

Usage:
    python scripts/run_4a4_residual.py --config configs/conditional_4A4_residual_gaussian_otm.yaml
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

# Reference floors for the diagnostic deltas (committed bundles).
RBF_OTM_TEST_MAE = 0.006131847035635011   # 3X.6 rbf/metrics_summary.csv (vs iv_clean)
RBF_OTM_VAL_MAE = 0.006151118707710642
ANP_3X9_TEST_MAE = {"gaussian": 0.01440, "quantile": 0.01175, "point": 0.00987}  # vs iv_true


def _build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool,
                  target_mode: str) -> DataLoader:
    return DataLoader(
        ConditionalIVSurfaceDataset(df, target_mode=target_mode),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_conditional,
        num_workers=0,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
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
        coord_encoding=cfg.get("coord_encoding"),
        decoder_kind=str(cfg.get("decoder_kind", "deepsets")),  # type: ignore[arg-type]
        anp=cfg.get("anp"),
    )


@torch.no_grad()
def _score_split(
    model: ConditionalSurfaceModel,
    df: pd.DataFrame,
    head_kind: str,
    device: torch.device,
) -> pd.DataFrame:
    """Score a split. The model predicts the RESIDUAL; this reconstructs the
    hybrid ``sigma_hat = rbf_pred + residual_pred`` (mu) and shifts quantile
    bands by ``rbf_pred``. ``df`` must carry an ``rbf_pred`` column."""
    if "rbf_pred" not in df.columns:
        raise ValueError("4A.4 scoring requires an 'rbf_pred' column (residual parquet)")
    model.eval()
    n = len(df)
    mu_out = np.full(n, np.nan, dtype=np.float32)
    sigma_out = np.full(n, np.nan, dtype=np.float32)
    log_sigma2_out = np.full(n, np.nan, dtype=np.float32)
    q_lo = np.full(n, np.nan, dtype=np.float32)
    q_med = np.full(n, np.nan, dtype=np.float32)
    q_hi = np.full(n, np.nan, dtype=np.float32)

    df_indexed = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    for _date, group in df_indexed.groupby("date", sort=False):
        obs = group["observed"].values.astype(bool)
        if obs.sum() == 0:
            continue
        ctx = group.loc[obs, ["log_moneyness", "tau", "implied_volatility"]].to_numpy(dtype=np.float32)
        query = group[["log_moneyness", "tau"]].to_numpy(dtype=np.float32)
        idx = group["_orig_idx"].values
        ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
        ctx_mask_t = torch.ones(1, ctx_t.shape[1], dtype=torch.bool, device=device)
        q_t = torch.from_numpy(query).unsqueeze(0).to(device)
        out = model(ctx_t, ctx_mask_t, q_t)
        if isinstance(out, dict):
            mu_out[idx] = out["mu"].squeeze(0).cpu().numpy()
            if head_kind == "gaussian":
                sigma_out[idx] = out["sigma"].squeeze(0).cpu().numpy()
                log_sigma2_out[idx] = out["log_sigma2"].squeeze(0).cpu().numpy()
            elif head_kind == "quantile":
                q = out["quantiles"].squeeze(0).cpu().numpy()
                q_lo[idx] = q[:, 0]
                q_med[idx] = q[:, q.shape[1] // 2]
                q_hi[idx] = q[:, -1]
        else:
            mu_out[idx] = out.squeeze(0).cpu().numpy()

    rbf = df["rbf_pred"].values.astype(np.float32)
    pred_df = pd.DataFrame(
        {
            "date": df["date"].values,
            "log_moneyness": df["log_moneyness"].values.astype(np.float32),
            "tau": df["tau"].values.astype(np.float32),
            "observed": df["observed"].values.astype(bool),
            "iv_true": df["implied_volatility"].values.astype(np.float32),
            "iv_clean": df["iv_clean"].values.astype(np.float32),
            "rbf_pred": rbf,
            "mu_residual": mu_out,
            "mu": mu_out + rbf,            # hybrid sigma_hat
        }
    )
    if head_kind == "gaussian":
        pred_df["sigma"] = sigma_out      # scale unchanged by the additive shift
        pred_df["log_sigma2"] = log_sigma2_out
    elif head_kind == "quantile":
        pred_df["q_lo"] = q_lo + rbf
        pred_df["q_med"] = q_med + rbf
        pred_df["q_hi"] = q_hi + rbf
    return pred_df


def _mae(pred_df: pd.DataFrame, target_col: str) -> float:
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
    target_mode = str(config.get("data", {}).get("target_mode", "residual"))

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/4A4/_unknown"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(config.get("data", {}).get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[4A.4] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4A.4] head_kind={head_kind}  target_mode={target_mode}  "
          f"device={device}  benchmark={bench_file}")

    df_all = pd.read_parquet(bench_file)
    if "rbf_pred" not in df_all.columns:
        print("[4A.4] benchmark is missing 'rbf_pred' — use the residual parquet (4A.3)",
              file=sys.stderr)
        return 2
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(f"[4A.4] rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    train_loader = _build_loader(train_df, batch_size, True, target_mode)
    val_loader = _build_loader(val_df, batch_size, False, target_mode)

    t0 = time.time()
    model, history = train_conditional(
        train_loader=train_loader, val_loader=val_loader,
        config=cond_cfg, device=device, checkpoint_dir=checkpoint_dir,
    )
    train_elapsed = time.time() - t0
    print(f"[4A.4] training done in {train_elapsed:.1f}s  epochs={len(history)}")

    ckpt = torch.load(str(checkpoint_dir / "best_conditional.pt"),
                      map_location=device, weights_only=False)
    model = _build_model_from_cfg(ckpt["config"], head_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    t_score = time.time()
    val_preds = _score_split(model, val_df, head_kind, device)
    test_preds = _score_split(model, test_df, head_kind, device)
    score_elapsed = time.time() - t_score

    val_preds.to_csv(results_dir / "val_predictions.csv", index=False)
    test_preds.to_csv(results_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(results_dir / "training_curve.csv", index=False)

    # Primary metric: hybrid vs iv_clean (apples-to-apples with the RBF floor).
    val_mae = _mae(val_preds, "iv_clean")
    test_mae = _mae(test_preds, "iv_clean")
    val_mae_ivtrue = _mae(val_preds, "iv_true")
    test_mae_ivtrue = _mae(test_preds, "iv_true")

    qmono_ok = True
    if head_kind == "quantile":
        qmono_ok = bool(
            ((test_preds["q_lo"] <= test_preds["q_med"])
             & (test_preds["q_med"] <= test_preds["q_hi"])).dropna().all()
        )

    manifest = {
        "story": "4A.4",
        "head_kind": head_kind,
        "target_mode": target_mode,
        "hybrid": "sigma_hat = rbf_pred + f_theta(residual)",
        "decoder_kind": str(cond_cfg.get("decoder_kind", "deepsets")),
        "anp": cond_cfg.get("anp"),
        "coord_encoding": cond_cfg.get("coord_encoding"),
        "seed": int(cond_cfg.get("seed", 42)),
        "git_sha": _git_sha(),
        "config_hash": _cfg_hash(cond_cfg),
        "benchmark_file": str(bench_file),
        "splits": {"train_rows": int(len(train_df)), "val_rows": int(len(val_df)),
                   "test_rows": int(len(test_df))},
        "epochs_completed": len(history),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "best_val_loss": float(min(r["val_loss"] for r in history)),
        # Primary: hybrid vs iv_clean (RBF-comparable).
        "val_mae_mu": val_mae,
        "test_mae_mu": test_mae,
        "val_mae_vs_iv_true": val_mae_ivtrue,
        "test_mae_vs_iv_true": test_mae_ivtrue,
        "rbf_floor_test_mae": RBF_OTM_TEST_MAE,
        "delta_vs_rbf_test": test_mae - RBF_OTM_TEST_MAE,
        "anp_3x9_test_mae": ANP_3X9_TEST_MAE.get(head_kind),
        "delta_vs_3x9_test_ivtrue": (test_mae_ivtrue - ANP_3X9_TEST_MAE[head_kind])
        if head_kind in ANP_3X9_TEST_MAE else None,
        "quantile_monotonicity_ok": qmono_ok,
        "train_wall_clock_s": train_elapsed,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
    }
    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[4A.4] head={head_kind}  hybrid_test_MAE(vs iv_clean)={test_mae:.6f}  "
          f"(vs iv_true)={test_mae_ivtrue:.6f}  RBF_floor={RBF_OTM_TEST_MAE:.6f}  "
          f"delta_vs_RBF={test_mae - RBF_OTM_TEST_MAE:+.6f}  qmono_ok={qmono_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
