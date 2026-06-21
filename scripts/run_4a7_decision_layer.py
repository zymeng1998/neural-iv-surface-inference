#!/usr/bin/env python3
"""4A.7 reliability metrics for the calibrated RBF-prior hybrid (ADR 0010).

Applies the 4A.6 calibrator to the cached hybrid test predictions (4A.4
gaussian mu/sigma + 4A.5 ensemble disagreement) and computes the
decision-layer reliability metrics on the **full** OTM test fold — coverage,
mean interval width, high-confidence MAE (keep_fraction=0.8 by calibrated
confidence) and the no-abstention MAE — all vs ``iv_clean`` (the target
``evaluate_pair`` uses, and the one the RBF floor 0.00613 is measured
against). Q2 invariant: the calibrator + nominal level are inherited from
3X.11/3X.12 unchanged; nothing is retuned.

Fully local from cached predictions — no model checkpoint, no GPU. The
model-based no-arb flag count + width-driven abstain rate (which require
re-prediction under masking perturbations) are NOT recomputed here; see the
4A.7 spec / journal for how that is handled.

Usage:
    python scripts/run_4a7_decision_layer.py \
        --gaussian artifacts/runs/4A4/gaussian/test_predictions.csv \
        --ensemble artifacts/runs/4A5/test_predictions.csv \
        --calibrator artifacts/calibration/4A6_hybrid.json \
        --output results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/metrics_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_iv_surface_inference.eval.calibration import Calibrator
from neural_iv_surface_inference.eval.uncertainty_metrics import interval_coverage

RBF_OTM_TEST_MAE = 0.006131847035635011  # 3X.6 floor (vs iv_clean)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian", required=True, help="4A.4 gaussian test predictions CSV")
    ap.add_argument("--ensemble", required=True, help="4A.5 ensemble test predictions CSV")
    ap.add_argument("--calibrator", required=True, help="4A.6 calibrator JSON")
    ap.add_argument("--keep-fraction", type=float, default=0.8)
    ap.add_argument("--nominal", type=float, default=0.90)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    g = pd.read_csv(args.gaussian)
    e = pd.read_csv(args.ensemble, usecols=["disagreement_std"])
    if len(g) != len(e):
        print(f"[4A.7] row mismatch gaussian={len(g)} ensemble={len(e)}", file=sys.stderr)
        return 2

    cal = Calibrator.load(Path(args.calibrator))
    out = cal.apply(
        mu=g["mu"].to_numpy(dtype=np.float64),
        sigma=g["sigma"].to_numpy(dtype=np.float64),
        disagreement=e["disagreement_std"].to_numpy(dtype=np.float64),
    )
    lower = np.asarray(out["lower"], dtype=np.float64)
    upper = np.asarray(out["upper"], dtype=np.float64)
    conf = np.asarray(out["confidence_score"], dtype=np.float64)

    y = g["iv_clean"].to_numpy(dtype=np.float64)        # evaluate_pair / RBF-floor target
    mu = g["mu"].to_numpy(dtype=np.float64)
    rbf = g["rbf_pred"].to_numpy(dtype=np.float64)
    abs_err = np.abs(mu - y)

    coverage = float(interval_coverage(y, lower, upper))   # vs iv_clean (3X.12 convention)
    iv_true = g["iv_true"].to_numpy(dtype=np.float64)
    coverage_ivtrue = float(interval_coverage(iv_true, lower, upper))  # calibrator's fit target
    mean_width = float(np.mean(upper - lower))
    no_abst_mae = float(abs_err.mean())

    # High-confidence MAE: keep the top keep_fraction by calibrated confidence.
    k = max(1, int(round(args.keep_fraction * len(conf))))
    keep_idx = np.argsort(-conf)[:k]
    hi_conf_mae = float(abs_err[keep_idx].mean())

    metrics = {
        "split": "test",
        "n": int(len(g)),
        "mae_vs_iv_clean": no_abst_mae,
        "mae_vs_iv_true": float(np.abs(mu - g["iv_true"].to_numpy(np.float64)).mean()),
        "rbf_floor_mae": RBF_OTM_TEST_MAE,
        "delta_vs_rbf": no_abst_mae - RBF_OTM_TEST_MAE,
        "coverage_90": coverage,
        "coverage_within_2pp": bool(abs(coverage - args.nominal) <= 0.02),
        "coverage_90_vs_iv_true": coverage_ivtrue,
        "coverage_vs_iv_true_within_2pp": bool(abs(coverage_ivtrue - args.nominal) <= 0.02),
        "mean_width": mean_width,
        f"hi_conf_mae_keep{args.keep_fraction}": hi_conf_mae,
        "no_abstention_mae": no_abst_mae,
        "hi_conf_below_no_abstention": bool(hi_conf_mae < no_abst_mae),
        "calibrator": str(args.calibrator),
        "note": ("reliability on the FULL test fold from cached predictions; "
                 "no-arb flag count + width-abstain require model re-prediction "
                 "and are not recomputed here (see 4A.7 spec)."),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.output, index=False)
    Path(args.output).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(f"[4A.7] coverage={coverage:.4f} (within2pp={metrics['coverage_within_2pp']}) "
          f"hi_conf_mae={hi_conf_mae:.6f} < no_abst_mae={no_abst_mae:.6f} "
          f"={metrics['hi_conf_below_no_abstention']}  mae_vs_RBF={no_abst_mae:.6f}/"
          f"{RBF_OTM_TEST_MAE:.6f} (Δ {metrics['delta_vs_rbf']:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
