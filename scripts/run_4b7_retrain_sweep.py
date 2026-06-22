#!/usr/bin/env python3
"""Fair per-(regime,rung) retrain of the RBF-prior hybrid (story 4B.7, remote GPU).

Resolves the 4B.5 `ambiguous` gate by removing the eval-time OOD confound: instead
of scoring a full-context checkpoint on sparse context (4B.4), this **retrains**
the hybrid on each rung's *sparse* context and re-evaluates. The architecture,
hyperparameters, seed, and success bar mirror 4A.4 **exactly**; the only thing
that varies per cell is the training context.

Per (regime, rung) cell — full sweep = 4 regimes × 4 non-dense rungs = 16 cells
(the dense rung, intensity 0, is the 4A.4 anchor and is not retrained):

  1. Re-derive the rung's sparse mask on **every split** with the 4B.2 harness and
     the same seed as 4B.3/4B.4 (so the *test* masking is identical to 4B.4 —
     apples-to-apples: only the model differs).
  2. Rebuild residual targets with RBF re-fit on the **sparse** context
     (`add_rbf_pred_column` with `observed := observed_rung`).
  3. Train the hybrid on the sparse train/val (mirror 4A.4), early-stopped on val.
  4. Score the sparse test context → σ̂ = RBF_sparse + f_θ(sparse).
  5. Record the cell's fair hybrid MAE, RBF MAE (the rung baseline), relative
     edge, and a date-clustered paired-bootstrap 95 % CI on the per-query delta.

Resumable: each cell writes `artifacts/runs/4B7/cells/{regime}_r{rung}.json`
(+ a pod-side per-cell test-prediction parquet, gitignored) and is skipped if its
JSON already exists. Aggregate the fair trajectory + resolve the gate with
`--aggregate` (or the companion `run_4b5`-style recompute) once all cells exist.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from neural_iv_surface_inference.data.conditional_loaders import (  # noqa: E402
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.data.residual_targets import (  # noqa: E402
    add_rbf_pred_column,
)
from neural_iv_surface_inference.diagnostics.sparsity_sweep import build_sweep  # noqa: E402
from neural_iv_surface_inference.training.train_conditional import (  # noqa: E402
    train_conditional,
)
from neural_iv_surface_inference.utils.io import load_config  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

RBF_FLOOR = 0.006131847035635011
HYBRID_FLOOR = 0.006006201729178429
_DATE_SEED_MULT = 1_000_003


def _rung_mask(df_split: pd.DataFrame, regime: str, intensity: float,
               mask_seed: int, maturity_col: str) -> np.ndarray:
    """Sparse observed mask for one split (df_split must carry a 0..n-1 RangeIndex).

    Per date (sorted), reuse the 4B.2 harness at the single rung intensity with
    the per-date seed fold — identical to 4B.3/4B.4 (the rng draw depends only on
    (seed, n_rows), not ladder position, so a single-intensity call reproduces
    the nested-ladder mask)."""
    out = np.zeros(len(df_split), dtype=bool)
    for di, date in enumerate(sorted(df_split["date"].unique())):
        g = df_split[df_split["date"] == date]
        date_seed = (int(mask_seed) * _DATE_SEED_MULT + di) % (2 ** 63)
        sweep = build_sweep(g, regime, (intensity,), seed=date_seed,
                            maturity_col=maturity_col)
        out[g.index.to_numpy()] = sweep.rungs[0].retained_observed
    return out


def _build_sparse_split(df_split: pd.DataFrame, regime: str, intensity: float,
                        mask_seed: int, maturity_col: str) -> pd.DataFrame:
    """Sub-mask a split to the rung's sparse context and rebuild RBF + residual."""
    work = df_split.reset_index(drop=True).copy()  # 0..n-1 RangeIndex for _rung_mask
    work["observed"] = _rung_mask(work, regime, intensity, mask_seed, maturity_col)
    return add_rbf_pred_column(work)  # recomputes rbf_pred + residual_target (sparse)


def _loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ConditionalIVSurfaceDataset(df, target_mode="residual"),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_conditional, num_workers=0,
    )


@torch.no_grad()
def _score_test(model, df_test: pd.DataFrame, device) -> np.ndarray:
    """Hybrid σ̂ = rbf_pred + f_θ(sparse context) at every test row."""
    model.eval()
    n = len(df_test)
    mu = np.full(n, np.nan, np.float32)
    d = df_test.reset_index(drop=True)
    for _date, g in d.groupby("date", sort=False):
        obs = g["observed"].to_numpy(bool)
        if obs.sum() == 0:
            continue
        ctx = torch.from_numpy(
            g.loc[obs, ["log_moneyness", "tau", "implied_volatility"]].to_numpy(np.float32)
        ).unsqueeze(0).to(device)
        mask = torch.ones(1, ctx.shape[1], dtype=torch.bool, device=device)
        q = torch.from_numpy(g[["log_moneyness", "tau"]].to_numpy(np.float32)).unsqueeze(0).to(device)
        out = model(ctx, mask, q)
        mu[g.index.to_numpy()] = out["mu"].squeeze(0).cpu().numpy()
    return mu + d["rbf_pred"].to_numpy(np.float32)  # σ̂


def _bootstrap_delta_ci(delta, groups, n_boot=2000, seed=0, ci=0.95):
    uniq, inv = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    gsum = np.bincount(inv, weights=delta, minlength=n_g)
    gcnt = np.bincount(inv, minlength=n_g).astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        boot[b] = gsum[pick].sum() / gcnt[pick].sum()
    return {"mean_delta": float(delta.mean()),
            "ci_low": float(np.quantile(boot, (1 - ci) / 2)),
            "ci_high": float(np.quantile(boot, 1 - (1 - ci) / 2)),
            "significant": bool(np.quantile(boot, 1 - (1 - ci) / 2) < 0.0)}


def run_cell(df_all, regime, intensity, *, cond_cfg, batch_size, mask_seed,
             maturity_col, device, ckpt_dir, pred_dir) -> dict:
    t0 = time.time()
    sp = {s: _build_sparse_split(df_all[df_all["split"] == s], regime, intensity,
                                 mask_seed, maturity_col)
          for s in ("train", "val", "test")}
    model, history = train_conditional(
        train_loader=_loader(sp["train"], batch_size, True),
        val_loader=_loader(sp["val"], batch_size, False),
        config=cond_cfg, device=device, checkpoint_dir=ckpt_dir,
    )
    test = sp["test"]
    sigma_hat = _score_test(model, test, device)
    y = test["iv_clean"].to_numpy(np.float64)
    rbf = test["rbf_pred"].to_numpy(np.float64)
    err_h = np.abs(sigma_hat.astype(np.float64) - y)
    err_r = np.abs(rbf - y)
    rbf_mae, hyb_mae = float(err_r.mean()), float(err_h.mean())
    ci = _bootstrap_delta_ci(err_h - err_r, test["date"].to_numpy())
    if pred_dir is not None:
        pd.DataFrame({"date": test["date"].to_numpy(), "iv_clean": y,
                      "rbf_pred": rbf, "sigma_hat": sigma_hat,
                      "observed_rung": test["observed"].to_numpy(bool)}).to_parquet(
            Path(pred_dir) / f"{regime}_r{intensity}.parquet", index=False)
    return {
        "regime": regime, "intensity": float(intensity),
        "rbf_mae": rbf_mae, "hybrid_fair_mae": hyb_mae,
        "abs_delta": hyb_mae - rbf_mae,
        "rel_edge_pct": (rbf_mae - hyb_mae) / rbf_mae * 100.0,
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "significant": ci["significant"],
        "n_context": int(test["observed"].sum()), "n_test_rows": int(len(test)),
        "epochs": len(history), "best_val_loss": float(min(r["val_loss"] for r in history)),
        "wall_seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="4B.7 fair per-(regime,rung) retrain sweep.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/conditional_4B7_retrain_gaussian_otm.yaml"))
    ap.add_argument("--input", default=None, help="Residual parquet (default: config data.benchmark_file).")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "artifacts/runs/4B7"))
    ap.add_argument("--ckpt-root", default=None, help="Pod dir for per-cell checkpoints.")
    ap.add_argument("--pred-dir", default=None, help="Pod dir for per-cell test-pred parquets.")
    ap.add_argument("--only-regime", default=None, help="Restrict to one regime (debug).")
    ap.add_argument("--only-intensity", type=float, default=None, help="Restrict to one rung (debug).")
    ap.add_argument("--max-dates", type=int, default=None, help="Smoke: first N dates per split.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cond_cfg = dict(cfg["conditional"]); cond_cfg.setdefault("seed", cfg.get("seed", 42))
    batch_size = int(cfg["data"].get("batch_size", 32))
    sweep = cfg["sweep"]
    intensities = [float(x) for x in sweep["intensities"]]
    regimes = list(sweep["regimes"])
    mask_seed = int(sweep.get("mask_seed", 4023))
    maturity_col = str(sweep.get("maturity_col", "tau"))
    if args.only_regime:
        regimes = [args.only_regime]
    if args.only_intensity is not None:
        intensities = [args.only_intensity]

    input_path = args.input or cfg["data"]["benchmark_file"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4b7] device={device} input={input_path} regimes={regimes} rungs={intensities}", flush=True)

    df_all = pd.read_parquet(input_path)
    if args.max_dates is not None:
        keep = {s: sorted(df_all[df_all["split"] == s]["date"].unique())[: args.max_dates]
                for s in ("train", "val", "test")}
        df_all = df_all[df_all.apply(lambda r: r["date"] in keep.get(r["split"], []), axis=1)].copy()

    outdir = Path(args.outdir); (outdir / "cells").mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else outdir / "checkpoints"
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    cells, t0 = [], time.time()
    for regime in regimes:
        for intensity in intensities:
            cell_json = outdir / "cells" / f"{regime}_r{intensity}.json"
            if cell_json.exists():
                print(f"[4b7] skip (done) {regime} r{intensity}", flush=True)
                cells.append(json.loads(cell_json.read_text())); continue
            print(f"[4b7] === retrain {regime} intensity={intensity} ===", flush=True)
            ckpt_dir = ckpt_root / f"{regime}_r{intensity}"; ckpt_dir.mkdir(parents=True, exist_ok=True)
            res = run_cell(df_all, regime, intensity, cond_cfg=cond_cfg, batch_size=batch_size,
                           mask_seed=mask_seed, maturity_col=maturity_col, device=device,
                           ckpt_dir=ckpt_dir, pred_dir=pred_dir)
            cell_json.write_text(json.dumps(res, indent=2))
            cells.append(res)
            print(f"[4b7] {regime} r{intensity}: fair_hybrid_mae={res['hybrid_fair_mae']:.6f} "
                  f"rbf={res['rbf_mae']:.6f} rel_edge={res['rel_edge_pct']:+.2f}% "
                  f"sig={res['significant']} ({res['wall_seconds']:.0f}s)", flush=True)

    stats = pd.DataFrame(cells).sort_values(["regime", "intensity"])
    stats.to_csv(outdir / "fair_retrain_stats.csv", index=False)
    (outdir / "manifest.json").write_text(json.dumps({
        "story": "4B.7", "input": str(input_path), "config": str(args.config),
        "regimes": regimes, "intensities": intensities, "mask_seed": mask_seed,
        "n_cells": len(cells), "wall_seconds": round(time.time() - t0, 1),
        "rbf_floor_ref": RBF_FLOOR, "hybrid_floor_ref": HYBRID_FLOOR,
        "note": "fair per-(regime,rung) retrain; dense rung = 4A.4 anchor (not retrained). "
                "Trajectory recompute + gate resolution: aggregate vs eval-time 4B.5.",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2))
    print(f"\n[4b7] {len(cells)} cells done; wrote {outdir}/fair_retrain_stats.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
