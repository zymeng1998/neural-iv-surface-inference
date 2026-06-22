#!/usr/bin/env python3
"""Edge-vs-sparsity trajectory + date-clustered bootstrap CIs + ADR 0011 gate
adjudication (story 4B.5, local CPU — the decision-bearing story).

Consumes the cached 4B.4 per-query predictions
(``artifacts/runs/4B4/swept_hybrid_test.parquet``: per (date, regime, rung) the
rung RBF ``rbf_pred`` and the hybrid ``mu_hybrid`` vs ``iv_clean``) and, per
regime x rung, computes:

  - RBF MAE, hybrid sigma_hat MAE (overall + query-fixed), the absolute and
    **relative** edge (``(RBF - hybrid)/RBF`` — positive = hybrid better);
  - a **date-clustered paired-bootstrap 95 % CI** on the per-query MAE delta
    (``|hybrid - iv_clean| - |rbf - iv_clean|``), resampling whole dates —
    the same statistic as 4A.7 (``scripts/eval_mae_bootstrap_ci.py``), here
    computed via per-group sum/count for speed (algebraically identical: same
    seed -> same date picks -> same CI);
  - (secondary, non-gating) calibrated 90 % coverage vs ``iv_clean`` using the
    frozen 4A.6 calibrator, as a reliability read of the dense-calibrated band
    under sparsity.

It then **adjudicates the ADR 0011 gate** against a rule **frozen in this file
before the trajectory is computed** (see ``PREREGISTERED_*`` constants), writing
``accuracy_survives in {true, false, ambiguous}`` to ``gate_verdict.json``.

Pure analysis from cached inputs — no model, no RBF, no pod (mirrors 4A.7/4A.8).
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

from neural_iv_surface_inference.diagnostics.sparsity_sweep import REGIMES  # noqa: E402

# Dense-end provenance anchors (committed 4A results).
RBF_FLOOR = 0.006131847035635011        # 3X.6 rbf overall test MAE
HYBRID_FLOOR = 0.006006201729178429     # 4A.4 gaussian test_mae_mu
DELTA_4A7 = -0.00012564530645658208     # 4A.7 committed mean_delta (dense)

# --------------------------------------------------------------------------
# PRE-REGISTERED GATE RULE (frozen from ADR 0011 / phase4b roadmap BEFORE the
# trajectory is computed; do NOT tune after seeing the numbers).
# --------------------------------------------------------------------------
PREREGISTERED_REL_EDGE_DENSE_BAND_PCT = 2.0   # the known dense ~2 % edge band
PREREGISTERED_REL_EDGE_SURVIVE_PCT = 5.0      # sparse-end "material" bar
PREREGISTERED_EDGE_GROWTH_MIN_PP = 1.0        # min dense->sparse increase to "grow"
PREREGISTERED_WING_SENSITIVE = ("thin_wings", "combined_quotes_wings")
PREREGISTERED_RULE_TEXT = (
    "survives: for every wing-sensitive regime {thin_wings, combined_quotes_wings} "
    "the relative edge INCREASES dense->sparse by >= EDGE_GROWTH_MIN_PP, the "
    "sparse-end 95% CI is entirely < 0, and the sparse-end relative edge >= "
    "REL_EDGE_SURVIVE_PCT. retired: the relative edge stays within the dense "
    "band (<= REL_EDGE_DENSE_BAND_PCT) across ALL regimes x rungs (no material "
    "growth anywhere). ambiguous: anything else (trend present but below the "
    "survive bar, or CIs do not separate, or the eval-time fairness caveat "
    "clouds the wing read) -> triggers the conditional retrain escalation 4B.7."
)


def _bootstrap_delta_ci(delta, groups, n_boot=2000, seed=0, ci=0.95):
    """Date-clustered bootstrap CI on mean(delta), resampling whole groups.

    Algebraically identical to ``scripts/eval_mae_bootstrap_ci`` (same seed ->
    same date picks -> same CI), but O(n_groups) per draw via per-group
    sum/count instead of O(n_rows).
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)
    gsum = np.bincount(inv, weights=delta, minlength=n_groups)
    gcnt = np.bincount(inv, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        boot[b] = gsum[pick].sum() / gcnt[pick].sum()
    lo = float(np.quantile(boot, (1 - ci) / 2))
    hi = float(np.quantile(boot, 1 - (1 - ci) / 2))
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": lo, "ci_high": hi, "ci_level": ci,
        "n_boot": n_boot, "n_groups": int(n_groups),
        "significant": bool(hi < 0.0),
    }


def _coverage(df_rung, calibrator_path):
    """Calibrated 90% coverage vs iv_clean for one rung (None on any failure)."""
    try:
        from neural_iv_surface_inference.eval.calibration import Calibrator
        from neural_iv_surface_inference.eval.uncertainty_metrics import interval_coverage
        cal = Calibrator.load(Path(calibrator_path))
        out = cal.apply(
            mu=df_rung["mu_hybrid"].to_numpy(np.float64),
            sigma=df_rung["sigma"].to_numpy(np.float64),
            disagreement=df_rung["ens_disagreement"].to_numpy(np.float64),
        )
        return float(interval_coverage(
            df_rung["iv_clean"].to_numpy(np.float64),
            np.asarray(out["lower"], np.float64),
            np.asarray(out["upper"], np.float64),
        ))
    except Exception as exc:  # noqa: BLE001 — coverage is secondary / non-gating
        print(f"[4b5] coverage skipped ({exc})", file=sys.stderr)
        return None


def build_trajectory(df, *, calibrator_path, n_boot, seed, with_coverage):
    rows = []
    for regime in REGIMES:
        sub_r = df[df["regime"] == regime]
        for rung_index, g in sub_r.groupby("rung_index", sort=True):
            y = g["iv_clean"].to_numpy(np.float64)
            err_h = np.abs(g["mu_hybrid"].to_numpy(np.float64) - y)
            err_r = np.abs(g["rbf_pred"].to_numpy(np.float64) - y)
            qf = g["is_query_fixed"].to_numpy(bool)
            rbf_mae = float(err_r.mean())
            hyb_mae = float(err_h.mean())
            rel_edge = (rbf_mae - hyb_mae) / rbf_mae * 100.0 if rbf_mae else float("nan")
            ci = _bootstrap_delta_ci(err_h - err_r, g["date"].to_numpy(),
                                     n_boot=n_boot, seed=seed)
            rbf_mae_q = float(err_r[qf].mean()) if qf.any() else float("nan")
            hyb_mae_q = float(err_h[qf].mean()) if qf.any() else float("nan")
            rel_edge_q = ((rbf_mae_q - hyb_mae_q) / rbf_mae_q * 100.0
                          if rbf_mae_q else float("nan"))
            cov = _coverage(g, calibrator_path) if with_coverage else None
            rows.append({
                "regime": regime,
                "rung_index": int(rung_index),
                "intensity": float(g["intensity"].iloc[0]),
                "rbf_mae": rbf_mae,
                "hybrid_mae": hyb_mae,
                "abs_delta": hyb_mae - rbf_mae,
                "rel_edge_pct": rel_edge,
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "significant": ci["significant"],
                "rel_edge_pct_query": rel_edge_q,
                "coverage_90_calibrated": cov,
                "n_rows": int(len(g)),
                "n_dates": ci["n_groups"],
            })
            print(f"  {regime:<22} r{rung_index} rel_edge={rel_edge:+.2f}% "
                  f"CI=[{ci['ci_low']:+.2e},{ci['ci_high']:+.2e}] sig={ci['significant']}",
                  flush=True)
    return pd.DataFrame(rows)


def adjudicate(traj):
    """Apply the frozen rule. Returns (verdict, reasoning, per_regime)."""
    per = {}
    for regime in REGIMES:
        s = traj[traj["regime"] == regime].sort_values("rung_index")
        dense = s.iloc[0]
        sparse = s.iloc[-1]
        grows = (sparse["rel_edge_pct"] - dense["rel_edge_pct"]) >= PREREGISTERED_EDGE_GROWTH_MIN_PP
        per[regime] = {
            "edge_dense_pct": float(dense["rel_edge_pct"]),
            "edge_sparse_pct": float(sparse["rel_edge_pct"]),
            "grows": bool(grows),
            "sparse_significant": bool(sparse["significant"]),
            "sparse_material": bool(sparse["rel_edge_pct"] >= PREREGISTERED_REL_EDGE_SURVIVE_PCT),
            "max_edge_pct": float(s["rel_edge_pct"].max()),
        }

    survives = all(
        per[w]["grows"] and per[w]["sparse_significant"] and per[w]["sparse_material"]
        for w in PREREGISTERED_WING_SENSITIVE
    )
    retired = bool(traj["rel_edge_pct"].max() <= PREREGISTERED_REL_EDGE_DENSE_BAND_PCT)

    if survives:
        verdict = "true"
        reason = ("wing-sensitive regimes show a growing, significant, >=5% relative "
                  "edge at the sparse end — accuracy survives sparsity.")
    elif retired:
        verdict = "false"
        reason = ("relative edge stays within the ~0-2% dense band across all regimes "
                  "x rungs — no material growth; retire accuracy chasing, Phase 5 "
                  "reliability-first becomes primary.")
    else:
        verdict = "ambiguous"
        reason = ("a trend is present (relative edge exceeds the 2% dense band in at "
                  "least one regime) but the wing-sensitive regimes do not clear the "
                  ">=5% survive bar; the eval-time fairness caveat (full-context "
                  "checkpoint scored on thinner, partly OOD context) clouds the wing "
                  "read -> trigger the conditional fair-retrain escalation 4B.7.")
    return verdict, reason, per


def main(argv=None):
    ap = argparse.ArgumentParser(description="4B.5 edge-vs-sparsity trajectory + gate.")
    ap.add_argument("--pred", default=str(REPO_ROOT / "artifacts/runs/4B4/swept_hybrid_test.parquet"))
    ap.add_argument("--calibrator", default=str(REPO_ROOT / "artifacts/calibration/4A6_hybrid.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "results/4/spy_phase1_random40_noiselow_otm/4b_sweep"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "artifacts/runs/4B5"))
    args = ap.parse_args(argv)

    cols = ["date", "regime", "rung_index", "intensity", "iv_clean",
            "rbf_pred", "mu_hybrid", "sigma", "ens_disagreement", "is_query_fixed"]
    print(f"[4b5] loading {args.pred}", flush=True)
    df = pd.read_parquet(args.pred, columns=cols)
    print(f"[4b5] rows={len(df):,} regimes={sorted(df['regime'].unique())} "
          f"rungs={sorted(df['rung_index'].unique())}", flush=True)

    t0 = time.time()
    traj = build_trajectory(df, calibrator_path=args.calibrator, n_boot=args.n_boot,
                            seed=args.seed, with_coverage=not args.no_coverage)
    wall = time.time() - t0

    # --- dense-rung provenance tie-back to 4A -----------------------------
    dense = traj[traj["rung_index"] == 0]
    dense_rbf = float(dense["rbf_mae"].iloc[0])
    dense_hyb = float(dense["hybrid_mae"].iloc[0])
    dense_delta = float(dense["abs_delta"].iloc[0])
    prov = {
        "dense_rbf_mae": dense_rbf, "dense_hybrid_mae": dense_hyb,
        "dense_abs_delta": dense_delta,
        "rbf_floor_ref": RBF_FLOOR, "hybrid_floor_ref": HYBRID_FLOOR,
        "delta_4a7_ref": DELTA_4A7,
        "rbf_matches": bool(abs(dense_rbf - RBF_FLOOR) <= 1e-5),
        "hybrid_matches": bool(abs(dense_hyb - HYBRID_FLOOR) <= 1e-5),
        "delta_matches_4a7": bool(abs(dense_delta - DELTA_4A7) <= 5e-6),
    }

    verdict, reason, per_regime = adjudicate(traj)

    # --- write the committed bundle ---------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    traj.to_csv(results_dir / "trajectory.csv", index=False)

    # wide markdown: relative edge % matrix (regime x rung)
    wide = traj.pivot(index="regime", columns="rung_index", values="rel_edge_pct")
    lines = ["# 4B sparsity-sweep — relative edge % (RBF − hybrid)/RBF, overall test MAE",
             "", "Positive = hybrid better. Rung 0 = dense (full context).", ""]
    header = "| regime | " + " | ".join(f"r{c} (i={traj[traj['rung_index']==c]['intensity'].iloc[0]:.1f})"
                                        for c in wide.columns) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(wide.columns) + 1))
    for regime in REGIMES:
        cells = " | ".join(f"{wide.loc[regime, c]:+.2f}%" for c in wide.columns)
        lines.append(f"| {regime} | {cells} |")
    lines += ["", f"**Gate verdict:** `accuracy_survives = {verdict}`", "", reason, ""]
    (results_dir / "trajectory_wide.md").write_text("\n".join(lines))

    verdict_obj = {
        "story": "4B.5",
        "accuracy_survives": verdict,
        "reason": reason,
        "preregistered_rule": {
            "text": PREREGISTERED_RULE_TEXT,
            "rel_edge_dense_band_pct": PREREGISTERED_REL_EDGE_DENSE_BAND_PCT,
            "rel_edge_survive_pct": PREREGISTERED_REL_EDGE_SURVIVE_PCT,
            "edge_growth_min_pp": PREREGISTERED_EDGE_GROWTH_MIN_PP,
            "wing_sensitive_regimes": list(PREREGISTERED_WING_SENSITIVE),
        },
        "per_regime": per_regime,
        "dense_provenance": prov,
        "fairness_caveat": ("eval-time only: a full-context-trained checkpoint is "
                            "scored on thinner, partly-OOD context; a fair per-regime "
                            "retrain is 4B.7 (fired iff verdict == ambiguous)."),
        "triggers_4b7": bool(verdict == "ambiguous"),
        "n_boot": args.n_boot, "seed": args.seed,
    }
    (results_dir / "gate_verdict.json").write_text(json.dumps(verdict_obj, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(json.dumps({
        "story": "4B.5", "pred": str(args.pred), "calibrator": str(args.calibrator),
        "n_rows": int(len(df)), "n_boot": args.n_boot, "seed": args.seed,
        "verdict": verdict, "dense_provenance": prov, "wall_seconds": round(wall, 1),
        "results_dir": str(results_dir),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
    }, indent=2))

    # --- report -----------------------------------------------------------
    print("\n[4b5] relative edge % (regime x rung):", flush=True)
    for regime in REGIMES:
        s = traj[traj["regime"] == regime].sort_values("rung_index")
        seq = ", ".join(f"{e:+.2f}%" for e in s["rel_edge_pct"])
        print(f"  {regime:<22} [{seq}]")
    print(f"\n[4b5] dense provenance: rbf={dense_rbf:.6f}(={prov['rbf_matches']}) "
          f"hybrid={dense_hyb:.6f}(={prov['hybrid_matches']}) "
          f"delta={dense_delta:+.6f}(4A.7 match={prov['delta_matches_4a7']})")
    print(f"[4b5] GATE VERDICT: accuracy_survives = {verdict}  (triggers_4b7={verdict=='ambiguous'})")
    print(f"[4b5] {reason}")
    print(f"[4b5] wall={wall:.1f}s  wrote {results_dir}/ + {outdir}/manifest.json")

    if not (prov["rbf_matches"] and prov["hybrid_matches"]):
        print("[4b5] BLOCKED — dense rung does not tie back to 4A floors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
