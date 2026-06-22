#!/usr/bin/env python3
"""Run the 4A RBF-prior hybrid checkpoint forward across the 4B sparsity sweep
(story 4B.4, remote).

For each regime x rung of the 4B.2 sweep, the **existing 4A checkpoints** (the
gaussian residual hybrid from 4A.4 + the 5-seed point ensemble from 4A.5) are
run **forward** over the rung's sparser observed context, and the residual
prediction is summed with that rung's RBF to form the hybrid

    sigma_hat = RBF_rung(k, tau) + f_theta(sparse context).

This is the **eval-time** half of the staged design (ADR 0011): **no retrain** —
a checkpoint trained on full context is stress-tested on thinner context. That
is the documented fairness caveat (a labelled robustness read, not a fair
per-regime retrain — that escalation is the conditional 4B.7).

Mirroring the 4A factorisation (4A.4 / 4A.5 emit *raw* mu / sigma / ensemble;
4A.7 applies the 4A.6 calibrator), this story emits **raw** sigma_hat, gaussian
sigma, and ensemble disagreement per rung. The calibrator + coverage / CIs /
gate verdict are applied locally in **4B.5** (the calibrator path is recorded in
the manifest for that step).

The rung masks are regenerated deterministically from the 4B.2 harness (same
seed as 4B.3) and the per-rung RBF is recomputed verbatim from
``models/interpolation.py`` (the dense rung reuses the committed ``rbf_pred``
column and is asserted == the 4A RBF floor 0.006132). Self-contained: no fragile
join against the 4B.3 parquet.

Acceptance: finite sigma_hat for every rung; the dense rung reproduces the 4A
hybrid test MAE 0.006006 (checkpoint provenance).
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

from neural_iv_surface_inference.diagnostics.sparsity_sweep import (  # noqa: E402
    REGIMES,
    build_all_regimes,
)
from neural_iv_surface_inference.models.conditional_surface import (  # noqa: E402
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.models.interpolation import (  # noqa: E402
    interpolate_surface_date,
)

# Provenance anchors (committed).
RBF_FLOOR = 0.006131847035635011        # 3X.6 rbf overall test MAE vs iv_clean
HYBRID_FLOOR = 0.006006201729178429     # 4A.4 gaussian test_mae_mu vs iv_clean
DEFAULT_INTENSITIES = (0.0, 0.2, 0.4, 0.6, 0.8)
_DATE_SEED_MULT = 1_000_003


def _build_model(cfg: dict, head: dict) -> ConditionalSurfaceModel:
    return ConditionalSurfaceModel(
        context_dim=int(cfg.get("context_dim", 3)),
        coord_dim=int(cfg.get("coord_dim", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        n_elem_layers=int(cfg.get("n_elem_layers", 2)),
        n_post_layers=int(cfg.get("n_post_layers", 1)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        head=dict(head),
        coord_encoding=cfg.get("coord_encoding"),
        decoder_kind=str(cfg.get("decoder_kind", "deepsets")),  # type: ignore[arg-type]
        anp=cfg.get("anp"),
    )


def _load(ckpt_path: str, head: dict, device: torch.device) -> ConditionalSurfaceModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model(ck["config"], head).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _forward_mu(model, ctx_np, query_np, device, want_sigma):
    """One date's forward. Returns (mu_residual, sigma_or_None)."""
    ctx = torch.from_numpy(ctx_np).unsqueeze(0).to(device)
    mask = torch.ones(1, ctx.shape[1], dtype=torch.bool, device=device)
    q = torch.from_numpy(query_np).unsqueeze(0).to(device)
    out = model(ctx, mask, q)
    if isinstance(out, dict):
        mu = out["mu"].squeeze(0).cpu().numpy()
        sig = out["sigma"].squeeze(0).cpu().numpy() if (want_sigma and "sigma" in out) else None
        return mu, sig
    return out.squeeze(0).cpu().numpy(), None


class _Acc:
    """Running MAE / finiteness accumulator for one (regime, rung)."""

    __slots__ = ("sum_abs", "n", "n_query", "sum_abs_q", "n_nonfinite", "n_retained")

    def __init__(self):
        self.sum_abs = 0.0
        self.n = 0
        self.sum_abs_q = 0.0
        self.n_query = 0
        self.n_nonfinite = 0
        self.n_retained = 0

    def update(self, sigma_hat, truth, query_mask, n_retained):
        finite = np.isfinite(sigma_hat)
        self.n_nonfinite += int((~finite).sum())
        err = np.abs(sigma_hat - truth)
        self.sum_abs += float(np.nansum(err))
        self.n += int(sigma_hat.shape[0])
        self.sum_abs_q += float(np.nansum(err[query_mask]))
        self.n_query += int(query_mask.sum())
        self.n_retained += int(n_retained)

    @property
    def mae(self):
        return self.sum_abs / self.n if self.n else float("nan")

    @property
    def mae_query(self):
        return self.sum_abs_q / self.n_query if self.n_query else float("nan")


def run_sweep(
    df, *, gauss, ensemble, intensities, seed, device, parquet_out,
    maturity_col="tau", verbose=False,
):
    """Forward the hybrid over every regime x rung; stream per-query parquet.

    Returns accumulators keyed by ``(regime, rung_index)``.
    """
    accs = {(r, i): _Acc() for r in REGIMES for i in range(len(intensities))}
    ladders = {r: intensities for r in REGIMES}
    dates = sorted(df["date"].unique())

    import pyarrow as pa
    import pyarrow.parquet as pq
    Path(parquet_out).parent.mkdir(parents=True, exist_ok=True)
    writer = None

    def _emit(date_arr, regime, rung, lm, tau, obs_rung, qmask, iv_true,
              iv_clean, rbf, mu_hat, sigma, ens_dis):
        nonlocal writer
        tbl = pa.Table.from_pandas(pd.DataFrame({
            "date": date_arr, "regime": regime,
            "rung_index": np.int16(rung.rung_index),
            "intensity": np.float32(rung.intensity),
            "log_moneyness": lm, "tau": tau,
            "observed_rung": obs_rung, "is_query_fixed": qmask,
            "iv_true": iv_true, "iv_clean": iv_clean, "rbf_pred": rbf,
            "mu_hybrid": mu_hat, "sigma": sigma, "ens_disagreement": ens_dis,
        }), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_out, tbl.schema, compression="zstd")
        writer.write_table(tbl)

    try:
        for di, date in enumerate(dates):
            g = df[df["date"] == date]
            date_arr = g["date"].to_numpy()
            lm = g["log_moneyness"].to_numpy(np.float32)
            tau = g["tau"].to_numpy(np.float32)
            iv_true = g["implied_volatility"].to_numpy(np.float32)
            iv_clean = g["iv_clean"].to_numpy(np.float32)
            base_observed = g["observed"].to_numpy(bool)
            query_mask = ~base_observed
            dense_rbf = g["rbf_pred"].to_numpy(np.float32)  # committed dense RBF
            query_np = g[["log_moneyness", "tau"]].to_numpy(np.float32)
            date_seed = (int(seed) * _DATE_SEED_MULT + di) % (2 ** 63)
            sweeps = build_all_regimes(g, ladders, seed=date_seed, maturity_col=maturity_col)

            # Cache the dense (rung 0) forward once — identical across regimes.
            dense_cache = None
            for regime in REGIMES:
                for rung in sweeps[regime].rungs:
                    if rung.rung_index == 0:
                        if dense_cache is None:
                            ctx0 = g.loc[base_observed,
                                         ["log_moneyness", "tau", "implied_volatility"]
                                         ].to_numpy(np.float32)
                            mu_res, sigma = _forward_mu(gauss, ctx0, query_np, device, True)
                            member = np.stack([
                                _forward_mu(m, ctx0, query_np, device, False)[0]
                                for m in ensemble
                            ]) if ensemble else None
                            ens_dis = (member.std(axis=0).astype(np.float32)
                                       if member is not None
                                       else np.zeros(len(g), np.float32))
                            dense_cache = (dense_rbf + mu_res.astype(np.float32),
                                           sigma.astype(np.float32), ens_dis, dense_rbf,
                                           int(base_observed.sum()))
                        mu_hat, sigma, ens_dis, rbf_r, n_ret = dense_cache
                    else:
                        obs_r = rung.retained_observed
                        gg = g.assign(observed=obs_r)
                        rbf_r = interpolate_surface_date(gg, method="rbf").astype(np.float32)
                        ctx = g.loc[obs_r,
                                    ["log_moneyness", "tau", "implied_volatility"]
                                    ].to_numpy(np.float32)
                        if ctx.shape[0] == 0:
                            mu_res = np.zeros(len(g), np.float32)
                            sigma = np.full(len(g), np.nan, np.float32)
                            ens_dis = np.zeros(len(g), np.float32)
                        else:
                            mu_res, sigma = _forward_mu(gauss, ctx, query_np, device, True)
                            mu_res = mu_res.astype(np.float32)
                            sigma = sigma.astype(np.float32)
                            if ensemble:
                                member = np.stack([
                                    _forward_mu(m, ctx, query_np, device, False)[0]
                                    for m in ensemble
                                ])
                                ens_dis = member.std(axis=0).astype(np.float32)
                            else:
                                ens_dis = np.zeros(len(g), np.float32)
                        mu_hat = (rbf_r + mu_res).astype(np.float32)
                        n_ret = int(obs_r.sum())

                    accs[(regime, rung.rung_index)].update(mu_hat, iv_clean, query_mask, n_ret)
                    _emit(date_arr, regime, rung, lm, tau,
                          (rung.retained_observed if rung.rung_index else base_observed),
                          query_mask, iv_true, iv_clean, rbf_r, mu_hat, sigma, ens_dis)

            if verbose and (di + 1) % 100 == 0:
                print(f"  [{di + 1}/{len(dates)}] dates forwarded", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return accs


def main(argv=None):
    ap = argparse.ArgumentParser(description="4B.4 hybrid forward sweep.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gaussian-ckpt", required=True)
    ap.add_argument("--ensemble-dir", default=None,
                    help="Dir with seed_*/best_conditional.pt (4A.5). Omit / --no-ensemble to skip.")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--calibrator", default="artifacts/calibration/4A6_hybrid.json",
                    help="4A.6 calibrator path (recorded for 4B.5; not applied here).")
    ap.add_argument("--intensities", default=",".join(str(x) for x in DEFAULT_INTENSITIES))
    ap.add_argument("--seed", type=int, default=4023)
    ap.add_argument("--maturity-col", default="tau")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "artifacts/runs/4B4"))
    ap.add_argument("--parquet-out", required=True)
    ap.add_argument("--hybrid-floor", type=float, default=HYBRID_FLOOR)
    ap.add_argument("--floor-tol", type=float, default=5e-5)
    ap.add_argument("--no-floor-check", action="store_true")
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    intensities = tuple(float(x) for x in args.intensities.split(",") if x.strip())
    if intensities[0] != 0.0 or list(intensities) != sorted(intensities):
        print("[4b4] intensities must be ascending and start at 0.0", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gauss = _load(args.gaussian_ckpt, {"kind": "gaussian"}, device)
    ensemble = []
    if not args.no_ensemble and args.ensemble_dir:
        seeds = sorted(Path(args.ensemble_dir).glob("seed_*/best_conditional.pt"))
        ensemble = [_load(str(p), {"kind": "point"}, device) for p in seeds]
    print(f"[4b4] device={device} gaussian=1 ensemble={len(ensemble)} "
          f"calibrator={args.calibrator} (recorded, applied in 4B.5)", flush=True)

    cols = ["date", "log_moneyness", "tau", "implied_volatility", "iv_clean",
            "observed", "split", "rbf_pred"]
    df = pd.read_parquet(args.input, columns=cols)
    df = df[df["split"] == args.split].copy()
    if args.max_dates is not None:
        keep = sorted(df["date"].unique())[: args.max_dates]
        df = df[df["date"].isin(keep)].copy()
    n_dates, n_rows = int(df["date"].nunique()), int(len(df))
    print(f"[4b4] split={args.split} dates={n_dates} rows={n_rows} rungs={intensities}", flush=True)

    t0 = time.time()
    accs = run_sweep(
        df, gauss=gauss, ensemble=ensemble, intensities=intensities, seed=args.seed,
        device=device, parquet_out=args.parquet_out, maturity_col=args.maturity_col,
        verbose=args.verbose,
    )
    wall = time.time() - t0

    rows = []
    for regime in REGIMES:
        for i, intensity in enumerate(intensities):
            a = accs[(regime, i)]
            rows.append({
                "regime": regime, "rung_index": i, "intensity": intensity,
                "hybrid_mae_overall": a.mae, "hybrid_mae_query_fixed": a.mae_query,
                "n_overall": a.n, "n_query": a.n_query,
                "n_retained_context": a.n_retained, "n_nonfinite": a.n_nonfinite,
            })
    stats = pd.DataFrame(rows)

    total_nonfinite = int(stats["n_nonfinite"].sum())
    dense_overall = float(stats[stats["rung_index"] == 0]["hybrid_mae_overall"].iloc[0])
    floor_err = abs(dense_overall - args.hybrid_floor)
    floor_ok = floor_err <= args.floor_tol
    monotone = {}
    for regime in REGIMES:
        seq = stats[stats["regime"] == regime].sort_values("rung_index")["hybrid_mae_overall"].to_numpy()
        d = np.diff(seq)
        monotone[regime] = {"non_decreasing": bool(np.all(d >= -1e-6)),
                            "max_decrease": float(max(0.0, -d.min())) if d.size else 0.0}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(outdir / "hybrid_sweep_stats.csv", index=False)
    manifest = {
        "story": "4B.4", "input": str(args.input), "split": args.split,
        "gaussian_ckpt": str(args.gaussian_ckpt), "ensemble_members": len(ensemble),
        "calibrator_ref": args.calibrator, "calibrator_applied_here": False,
        "calibration_note": "raw mu/sigma/disagreement emitted; calibrator applied in 4B.5 (mirrors 4A.7)",
        "intensities": list(intensities), "seed": args.seed,
        "n_dates": n_dates, "n_rows": n_rows,
        "hybrid_floor_ref": args.hybrid_floor, "rbf_floor_ref": RBF_FLOOR,
        "dense_rung_hybrid_mae": dense_overall, "dense_rung_floor_abs_err": floor_err,
        "dense_rung_reproduces_floor": floor_ok, "floor_tol": args.floor_tol,
        "total_nonfinite": total_nonfinite, "monotone_hybrid": monotone,
        "wall_seconds": round(wall, 1), "parquet_out": args.parquet_out,
        "fairness_caveat": "eval-time only; checkpoint trained on full context, "
                           "stress-tested on sparser context (no retrain; 4B.7 is the "
                           "conditional fair-retrain escalation)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n[4b4] per-(regime,rung) hybrid MAE (overall):", flush=True)
    for regime in REGIMES:
        sub = stats[stats["regime"] == regime].sort_values("rung_index")
        ov = ", ".join(f"{m:.6f}" for m in sub["hybrid_mae_overall"])
        print(f"  {regime:<22} [{ov}]  monotone={monotone[regime]['non_decreasing']}")
    print(f"\n[4b4] dense hybrid MAE = {dense_overall:.6f} "
          f"(4A floor {args.hybrid_floor:.6f}, |err|={floor_err:.2e}, ok={floor_ok})")
    print(f"[4b4] total non-finite = {total_nonfinite}  wall = {wall:.1f}s")
    print(f"[4b4] wrote {outdir/'manifest.json'} + {outdir/'hybrid_sweep_stats.csv'}")
    print(f"[4b4] streamed predictions -> {args.parquet_out}")

    if total_nonfinite:
        print("[4b4] BLOCKED — non-finite sigma_hat present", file=sys.stderr)
        return 1
    if not args.no_floor_check and not floor_ok:
        print(f"[4b4] BLOCKED — dense hybrid MAE {dense_overall:.6f} != 4A floor "
              f"{args.hybrid_floor:.6f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
