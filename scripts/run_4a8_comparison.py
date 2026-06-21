#!/usr/bin/env python3
"""4A.8 — Phase 4 hybrid-vs-RBF-vs-pure-neural comparison assembler (local, CPU).

Reads committed source bundles for the matched clean OTM substrate
`spy_phase1_random40_noiselow_otm` and emits the Phase 4 comparison:

- the RBF floor (3X.6),
- the best pure-neural references (3X.9 single heads, 3X.12 calibrated),
- the RBF-prior residual hybrid (4A.4 gaussian production head, calibrated at
  4A.6, adjudicated at 4A.7),

as a long-format CSV, a wide human-readable Markdown table, and a headline
JSON carrying the bar verdict.

No model is re-run; every number is cited from a committed file path. The
script asserts each source exists and that the hybrid CI verdict it reports
matches the committed `mae_delta_ci.json` exactly (no drift).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUB = "spy_phase1_random40_noiselow_otm"

# ---- Source bundles (all committed) -------------------------------------
RBF_METRICS = REPO / f"results/3/{SUB}/rbf/metrics_summary.csv"
NEURAL_WIDE = REPO / f"results/3/{SUB}/3x_compare/comparison_wide.csv"
CALIB_METRICS = REPO / f"results/3/{SUB}/3x_anp/metrics_summary.csv"
HYBRID_METRICS = REPO / f"results/4/{SUB}/4a_hybrid/metrics_summary.csv"
HYBRID_CI = REPO / f"results/4/{SUB}/4a_hybrid/mae_delta_ci.json"

OUT_DIR = REPO / f"results/4/{SUB}/4a_compare"
OUT_LONG = OUT_DIR / "comparison.csv"
OUT_WIDE_MD = OUT_DIR / "comparison_wide.md"
OUT_HEADLINE = OUT_DIR / "headline.json"

LONG_COLUMNS = ["family", "head", "split", "metric", "value", "source"]

# ADR 0010 §5 reliability bar.
COVERAGE_TARGET = 0.90
COVERAGE_TOL = 0.02


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"[4A.8] missing committed source: {rel(path)}")
    return path


def load_metrics_summary(path: Path) -> dict[str, dict[str, str]]:
    """Return {split: {col: raw_str}} for a long metrics_summary.csv."""
    out: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["split"]] = row
    return out


def rbf_test_mae(path: Path) -> float:
    """RBF metrics file uses Phase 1 wide format (overall_mae per split row)."""
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                return float(row["overall_mae"])
    raise SystemExit("[4A.8] no test split in RBF metrics")


def neural_test_mae(path: Path) -> dict[tuple[str, str], float]:
    """Pull pure-neural OTM test MAE from the 3X comparison wide CSV."""
    out: dict[tuple[str, str], float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            otm = row.get("otm_test_mae", "")
            if otm:
                out[(row["family"], row["head"])] = float(otm)
    return out


def main() -> int:
    for p in (RBF_METRICS, NEURAL_WIDE, CALIB_METRICS, HYBRID_METRICS, HYBRID_CI):
        require(p)

    rows: list[dict] = []

    def push(family, head, split, metric, value, source: Path):
        if value is None or value == "":
            raise SystemExit(f"[4A.8] empty value {family}/{head}/{split}/{metric}")
        rows.append({
            "family": family, "head": head, "split": split,
            "metric": metric, "value": value, "source": rel(source),
        })

    # ---- RBF floor (3X.6) ------------------------------------------------
    rbf_floor = rbf_test_mae(RBF_METRICS)
    push("rbf", "interp", "test", "mae", rbf_floor, RBF_METRICS)

    # ---- Pure-neural references (3X.9 single heads) ----------------------
    neural = neural_test_mae(NEURAL_WIDE)
    for head in ("point", "quantile", "gaussian"):
        key = ("anp_single", head)
        if key in neural:
            push("anp_single", head, "test", "mae", neural[key], NEURAL_WIDE)

    # ---- Pure-neural calibrated production (3X.12) -----------------------
    calib = load_metrics_summary(CALIB_METRICS)["test"]
    push("anp_calibrated", "fused", "test", "mae", float(calib["mae"]), CALIB_METRICS)
    push("anp_calibrated", "fused", "test", "coverage_90",
         float(calib["coverage_90"]), CALIB_METRICS)
    push("anp_calibrated", "fused", "test", "hi_conf_mae_keep0.8",
         float(calib["hi_conf_mae_keep0.8"]), CALIB_METRICS)
    push("anp_calibrated", "fused", "test", "mean_width",
         float(calib["mean_width"]), CALIB_METRICS)
    push("anp_calibrated", "fused", "test", "n_forbidden_flag_violations",
         int(float(calib["n_forbidden_flag_violations"])), CALIB_METRICS)

    # ---- RBF-prior residual hybrid (4A.4 gaussian, 4A.6 calib, 4A.7 CI) --
    hyb = load_metrics_summary(HYBRID_METRICS)["test"]
    push("rbf_prior_hybrid", "gaussian", "test", "mae",
         float(hyb["mae_vs_iv_clean"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "mae_vs_iv_true",
         float(hyb["mae_vs_iv_true"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "delta_vs_rbf",
         float(hyb["delta_vs_rbf"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "coverage_90_vs_iv_true",
         float(hyb["coverage_90_vs_iv_true"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "coverage_90_vs_iv_clean",
         float(hyb["coverage_90"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "hi_conf_mae_keep0.8",
         float(hyb["hi_conf_mae_keep0.8"]), HYBRID_METRICS)
    push("rbf_prior_hybrid", "gaussian", "test", "mean_width",
         float(hyb["mean_width"]), HYBRID_METRICS)

    # ---- CI verdict (4A.7) — assert no drift vs committed json -----------
    with HYBRID_CI.open() as f:
        ci = json.load(f)
    # The hybrid metrics_summary delta must agree with the CI mean_delta.
    if abs(float(hyb["delta_vs_rbf"]) - ci["mean_delta"]) > 1e-9:
        raise SystemExit(
            f"[4A.8] delta drift: metrics_summary {hyb['delta_vs_rbf']} "
            f"vs CI {ci['mean_delta']}")
    if abs(float(hyb["rbf_floor_mae"]) - rbf_floor) > 1e-6:
        raise SystemExit(
            f"[4A.8] RBF floor drift: hybrid {hyb['rbf_floor_mae']} "
            f"vs 3X.6 {rbf_floor}")
    push("rbf_prior_hybrid", "gaussian", "test", "ci_low", ci["ci_low"], HYBRID_CI)
    push("rbf_prior_hybrid", "gaussian", "test", "ci_high", ci["ci_high"], HYBRID_CI)
    push("rbf_prior_hybrid", "gaussian", "test", "ci_significant",
         bool(ci["significant"]), HYBRID_CI)

    # ---- Reliability bar verdict (ADR 0010 §5) ---------------------------
    cov_true = float(hyb["coverage_90_vs_iv_true"])
    coverage_in_band = abs(cov_true - COVERAGE_TARGET) <= COVERAGE_TOL
    hi_conf_below = float(hyb["hi_conf_mae_keep0.8"]) < float(hyb["mae_vs_iv_clean"])
    accuracy_significant = bool(ci["significant"]) and ci["ci_high"] < 0.0
    bar_met = accuracy_significant and coverage_in_band and hi_conf_below

    headline = {
        "substrate": SUB,
        "rbf_floor_mae": rbf_floor,
        "hybrid_mae_vs_iv_clean": float(hyb["mae_vs_iv_clean"]),
        "delta_vs_rbf": ci["mean_delta"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_level": ci["ci_level"],
        "n_groups": ci["n_groups"],
        "n_rows": ci["n_rows"],
        "accuracy_significant": accuracy_significant,
        "coverage_90_vs_iv_true": cov_true,
        "coverage_in_band_pm2pp": coverage_in_band,
        "hi_conf_mae_keep0.8": float(hyb["hi_conf_mae_keep0.8"]),
        "hi_conf_below_no_abstention": hi_conf_below,
        "best_pure_neural_point_mae": neural.get(("anp_single", "point")),
        "calibrated_production_mae": float(calib["mae"]),
        "bar_met": bar_met,
        "flag_count_recomputed": False,
        "flag_count_note": (
            "Hybrid no-arb flag count not recomputed locally (needs the model "
            "checkpoint); hybrid surface = RBF + small smooth residual, so "
            "no-arb behavior tracks RBF's. Deferred per 4A.7 spec."),
        "production_recommendation": (
            "Adopt the RBF-prior gaussian residual hybrid as the production "
            "accuracy surface (first predictor to significantly beat RBF on "
            "the clean OTM substrate), retaining the calibrated reliability "
            "layer. Caveats: coverage vs iv_clean over-covers (0.962) — a "
            "calibrator-refit-on-iv_clean follow-up; no-arb flag count audit "
            "deferred." if bar_met else
            "Ship the calibrated reliability layer on RBF (accuracy-neutral, "
            "reliability-positive); no significant accuracy gain."),
    }

    # ---- Write outputs ---------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_LONG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    OUT_HEADLINE.write_text(json.dumps(headline, indent=2) + "\n")

    # ---- Wide Markdown: accuracy ladder ---------------------------------
    ladder = [
        ("rbf_prior_hybrid · gaussian (Phase 4)", float(hyb["mae_vs_iv_clean"]), rel(HYBRID_METRICS)),
        ("rbf · interp (floor)", rbf_floor, rel(RBF_METRICS)),
        ("anp_single · point (best pure-neural)", neural.get(("anp_single", "point")), rel(NEURAL_WIDE)),
        ("anp_single · quantile", neural.get(("anp_single", "quantile")), rel(NEURAL_WIDE)),
        ("anp_single · gaussian", neural.get(("anp_single", "gaussian")), rel(NEURAL_WIDE)),
        ("anp_calibrated · fused (3X.12 production)", float(calib["mae"]), rel(CALIB_METRICS)),
    ]
    lines = [
        "# Phase 4 — RBF-prior hybrid vs RBF vs pure-neural (matched OTM)",
        "",
        f"**Scope:** `{SUB}` matched clean OTM substrate only "
        "(no claim across the other 10 OTM variants).",
        "",
        "All test MAE vs `iv_clean` on the same fold. The hybrid–RBF delta is "
        "the paired, date-clustered bootstrap (4A.7). Every cell traces to a "
        "committed source bundle.",
        "",
        "| Family / head | OTM test MAE | vs RBF floor | Source |",
        "|---|---:|---:|---|",
    ]
    for label, mae, src in ladder:
        if mae is None:
            continue
        if "floor" in label and "hybrid" not in label:
            vs = "floor"
        else:
            pct = (mae - rbf_floor) / rbf_floor * 100.0
            vs = f"{pct:+.1f}%"
        lines.append(f"| {label} | {mae:.6f} | {vs} | `{src}` |")
    lines += [
        "",
        "## Bar verdict (ADR 0010 §5)",
        "",
        f"- **Accuracy:** hybrid {float(hyb['mae_vs_iv_clean']):.6f} vs RBF "
        f"{rbf_floor:.6f}; mean Δ {ci['mean_delta']:+.6f}, "
        f"{int(ci['ci_level']*100)}% CI "
        f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] "
        f"({'entirely < 0 → significant' if accuracy_significant else 'includes 0'}).",
        f"- **Coverage (vs iv_true):** {cov_true:.4f} "
        f"({'within' if coverage_in_band else 'outside'} ±2pp of 0.90).",
        f"- **Hi-conf MAE:** {float(hyb['hi_conf_mae_keep0.8']):.6f} "
        f"{'<' if hi_conf_below else '≥'} no-abstention "
        f"{float(hyb['mae_vs_iv_clean']):.6f}.",
        f"- **No-arb flag count:** not recomputed (deferred; see headline.json).",
        "",
        f"**Verdict: bar {'MET' if bar_met else 'NOT met'}.** "
        f"{headline['production_recommendation']}",
        "",
        "Long-format companion: `comparison.csv`; machine-readable verdict: "
        "`headline.json`.",
    ]
    OUT_WIDE_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote: {rel(OUT_LONG)} ({len(rows)} rows)")
    print(f"wrote: {rel(OUT_WIDE_MD)}")
    print(f"wrote: {rel(OUT_HEADLINE)}")
    print(f"bar_met={bar_met} accuracy_significant={accuracy_significant} "
          f"coverage_in_band={coverage_in_band} hi_conf_below={hi_conf_below}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
