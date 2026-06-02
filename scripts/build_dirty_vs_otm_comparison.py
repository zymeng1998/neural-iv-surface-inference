#!/usr/bin/env python3
"""3X.13 — Dirty-vs-OTM matched comparison assembler (local, CPU only).

Reads committed source bundles (manifests, metrics_summary.csv, calibrator
reports) for the dirty `spy_phase1_random40_noiselow` and OTM
`spy_phase1_random40_noiselow_otm` substrates, emitting a long-format
comparison table and a wide human-readable view.

No model is re-run; every number is cited from a committed file path.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results/3/spy_phase1_random40_noiselow_otm/3x_compare"
OUT_LONG = OUT_DIR / "comparison.csv"
OUT_WIDE_CSV = OUT_DIR / "comparison_wide.csv"
OUT_WIDE_MD = OUT_DIR / "comparison_wide.md"

LONG_COLUMNS = [
    "family", "head", "data_substrate", "split", "metric", "value", "source", "n",
]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_metrics_summary(path: Path) -> dict[str, dict[str, float | int]]:
    """Return {split: {metric: value, ..., 'n': int}}."""
    out: dict[str, dict[str, float | int]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            d: dict[str, float | int] = {}
            for k, v in row.items():
                if k == "split":
                    continue
                if v == "" or v is None:
                    continue
                try:
                    d[k] = float(v) if k != "n" else int(v)
                except ValueError:
                    continue
            out[split] = d
    return out


def push_manifest_row(rows, *, family, head, substrate, split, metric, value,
                      source_path: Path, n: int | None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"NaN at {family}/{head}/{substrate}/{split}/{metric}")
    rows.append({
        "family": family,
        "head": head,
        "data_substrate": substrate,
        "split": split,
        "metric": metric,
        "value": value,
        "source": rel(source_path),
        "n": "" if n is None else int(n),
    })


def add_manifest_pair(rows, family, head, substrate, manifest_path: Path,
                      bench_n_val: int | None, bench_n_test: int | None):
    """Pull val/test mae from a per-head training manifest."""
    m = load_manifest(manifest_path)
    val_mae = m.get("val_mae_mu") or m.get("ensemble_val_mae")
    test_mae = m.get("test_mae_mu") or m.get("ensemble_test_mae")
    if val_mae is not None:
        push_manifest_row(rows, family=family, head=head, substrate=substrate,
                          split="val", metric="mae", value=val_mae,
                          source_path=manifest_path, n=bench_n_val)
    if test_mae is not None:
        push_manifest_row(rows, family=family, head=head, substrate=substrate,
                          split="test", metric="mae", value=test_mae,
                          source_path=manifest_path, n=bench_n_test)


def add_metrics_summary(rows, family, head, substrate, metrics_path: Path,
                        splits=("val", "test"),
                        metrics=("mae", "coverage_90", "mean_width",
                                 "hi_conf_mae_keep0.8", "abstain_rate",
                                 "mean_tradability", "n_forbidden_flag_violations")):
    data = load_metrics_summary(metrics_path)
    for split in splits:
        if split not in data:
            continue
        n = data[split].get("n")
        for metric in metrics:
            if metric in data[split]:
                push_manifest_row(rows, family=family, head=head,
                                  substrate=substrate, split=split,
                                  metric=metric, value=data[split][metric],
                                  source_path=metrics_path,
                                  n=n if isinstance(n, int) else None)


def add_rbf(rows, substrate, rbf_path: Path):
    """RBF metrics file uses Phase 1 wide format (overall_mae, ...)."""
    with rbf_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            mae = float(row["overall_mae"])
            n = int(row["overall_n"])
            push_manifest_row(rows, family="rbf", head="interp",
                              substrate=substrate, split=split, metric="mae",
                              value=mae, source_path=rbf_path, n=n)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Benchmark row counts (val/test) used to enrich manifest-derived rows.
    # Dirty (spy_phase1_random40_noiselow) test n = 5,805,664 (full-fold);
    # decision-layer view n = 64,610 (subset). OTM equivalents:
    # 3X.6 RBF reports val 2,638,892 / test 2,769,021; 3X.12 calibrated view
    # n = 41,490 / 29,550. We let manifest-derived rows carry the fullfold n
    # by sourcing from the RBF metrics_summary (n column).
    DIRTY_FULLFOLD_N_VAL = 5_593_759   # from 3a_compare precedent
    DIRTY_FULLFOLD_N_TEST = 5_805_664
    OTM_FULLFOLD_N_VAL = 2_638_892
    OTM_FULLFOLD_N_TEST = 2_769_021

    # ---- RBF (no neural head; interpolation baseline)
    # Dirty RBF is a roadmap-cited headline (0.0662). Cite it explicitly.
    push_manifest_row(
        rows, family="rbf", head="interp", substrate="dirty",
        split="test", metric="mae", value=0.0662,
        source_path=REPO / "docs/roadmaps/phase3_accuracy_push.md",
        n=DIRTY_FULLFOLD_N_TEST,
    )
    add_rbf(rows, substrate="otm",
            rbf_path=REPO / "results/3/spy_phase1_random40_noiselow_otm/rbf/metrics_summary.csv")

    # ---- MLP (Phase 1 / 2D masked-MLP baseline — point head)
    add_metrics_summary(
        rows, family="mlp", head="point", substrate="dirty",
        metrics_path=REPO / "results/2D/spy_phase1_random40_noiselow/masked_mlp/metrics_summary.csv",
        metrics=("mae",), splits=("val", "test"),
    )
    # OTM MLP (3X.7) — manifest has nested *_mae_vs_iv_true.overall_mae.
    mlp_otm_manifest = REPO / "artifacts/runs/3X7/mlp_otm/manifest.json"
    mlp_m = load_manifest(mlp_otm_manifest)
    push_manifest_row(
        rows, family="mlp", head="point", substrate="otm",
        split="val", metric="mae",
        value=mlp_m["val_mae_vs_iv_true"]["overall_mae"],
        source_path=mlp_otm_manifest, n=OTM_FULLFOLD_N_VAL,
    )
    push_manifest_row(
        rows, family="mlp", head="point", substrate="otm",
        split="test", metric="mae",
        value=mlp_m["test_mae_vs_iv_true"]["overall_mae"],
        source_path=mlp_otm_manifest, n=OTM_FULLFOLD_N_TEST,
    )

    # ---- DeepSets single (2D.7 dirty vs 3X.8 OTM) — 3 heads
    for head, dirty_dir, otm_dir in [
        ("gaussian", "2D7/gaussian", "3X8/single_gaussian"),
        ("quantile", "2D7/quantile", "3X8/single_quantile"),
        ("point",    "2D7/point",    "3X8/single_point"),
    ]:
        add_manifest_pair(
            rows, family="deepsets_single", head=head, substrate="dirty",
            manifest_path=REPO / f"artifacts/runs/{dirty_dir}/manifest.json",
            bench_n_val=DIRTY_FULLFOLD_N_VAL, bench_n_test=DIRTY_FULLFOLD_N_TEST,
        )
        add_manifest_pair(
            rows, family="deepsets_single", head=head, substrate="otm",
            manifest_path=REPO / f"artifacts/runs/{otm_dir}/manifest.json",
            bench_n_val=OTM_FULLFOLD_N_VAL, bench_n_test=OTM_FULLFOLD_N_TEST,
        )

    # ---- DeepSets K=5 ensemble (2D.8 dirty vs 3X.8 ensemble OTM)
    add_manifest_pair(
        rows, family="deepsets_ensemble", head="point", substrate="dirty",
        manifest_path=REPO / "artifacts/runs/2D8/manifest.json",
        bench_n_val=DIRTY_FULLFOLD_N_VAL, bench_n_test=DIRTY_FULLFOLD_N_TEST,
    )
    add_manifest_pair(
        rows, family="deepsets_ensemble", head="point", substrate="otm",
        manifest_path=REPO / "artifacts/runs/3X8/ensemble/manifest.json",
        bench_n_val=OTM_FULLFOLD_N_VAL, bench_n_test=OTM_FULLFOLD_N_TEST,
    )

    # ---- ANP single (3B.4 dirty vs 3X.9 OTM) — 3 heads
    for head, dirty_dir, otm_dir in [
        ("gaussian", "3B4/gaussian",      "3X9/gaussian"),
        ("quantile", "3B4/quantile",      "3X9/quantile"),
        ("point",    "3B4/point_control", "3X9/point_control"),
    ]:
        add_manifest_pair(
            rows, family="anp_single", head=head, substrate="dirty",
            manifest_path=REPO / f"artifacts/runs/{dirty_dir}/manifest.json",
            bench_n_val=DIRTY_FULLFOLD_N_VAL, bench_n_test=DIRTY_FULLFOLD_N_TEST,
        )
        add_manifest_pair(
            rows, family="anp_single", head=head, substrate="otm",
            manifest_path=REPO / f"artifacts/runs/{otm_dir}/manifest.json",
            bench_n_val=OTM_FULLFOLD_N_VAL, bench_n_test=OTM_FULLFOLD_N_TEST,
        )

    # ---- ANP K=5 ensemble (3B.5 dirty vs 3X.10 OTM)
    add_manifest_pair(
        rows, family="anp_ensemble", head="point", substrate="dirty",
        manifest_path=REPO / "artifacts/runs/3B5/manifest.json",
        bench_n_val=DIRTY_FULLFOLD_N_VAL, bench_n_test=DIRTY_FULLFOLD_N_TEST,
    )
    add_manifest_pair(
        rows, family="anp_ensemble", head="point", substrate="otm",
        manifest_path=REPO / "artifacts/runs/3X10/manifest.json",
        bench_n_val=OTM_FULLFOLD_N_VAL, bench_n_test=OTM_FULLFOLD_N_TEST,
    )

    # ---- Calibrated / decision-layer view (3B.7 dirty vs 3X.12 OTM)
    add_metrics_summary(
        rows, family="anp_calibrated", head="fused", substrate="dirty",
        metrics_path=REPO / "results/3/spy_phase1_random40_noiselow/3b_anp/metrics_summary.csv",
    )
    add_metrics_summary(
        rows, family="anp_calibrated", head="fused", substrate="otm",
        metrics_path=REPO / "results/3/spy_phase1_random40_noiselow_otm/3x_anp/metrics_summary.csv",
    )

    # ---- Write long table
    with OUT_LONG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Validation: no NaN; every row has source; paired substrate coverage
    seen = {}
    for r in rows:
        if r["value"] == "" or r["value"] is None:
            raise SystemExit(f"empty value row: {r}")
        if not r["source"]:
            raise SystemExit(f"empty source row: {r}")
        key = (r["family"], r["head"], r["split"], r["metric"])
        seen.setdefault(key, set()).add(r["data_substrate"])

    # Every (family,head,split,metric=mae) key should have both substrates for
    # shared families (rbf only has test on dirty; mlp dirty has val/test).
    missing_pairs = []
    for key, subs in seen.items():
        family, head, split, metric = key
        if metric != "mae":
            continue
        if family == "rbf" and split == "val":
            continue  # dirty rbf val not cited (test-only headline)
        if {"dirty", "otm"} - subs:
            missing_pairs.append((key, subs))
    if missing_pairs:
        print("[warn] mae pairs missing a substrate:")
        for k, s in missing_pairs:
            print(" ", k, "have", sorted(s))

    # ---- Wide view: pivot test MAE side-by-side
    wide_rows = []
    test_mae = {}
    for r in rows:
        if r["split"] != "test" or r["metric"] != "mae":
            continue
        test_mae.setdefault((r["family"], r["head"]), {})[r["data_substrate"]] = (
            float(r["value"]), r["source"], r["n"])
    for (family, head), sub in sorted(test_mae.items()):
        d = sub.get("dirty")
        o = sub.get("otm")
        delta = (o[0] - d[0]) if (d and o) else None
        ratio = (d[0] / o[0]) if (d and o and o[0]) else None
        wide_rows.append({
            "family": family,
            "head": head,
            "dirty_test_mae": d[0] if d else "",
            "otm_test_mae": o[0] if o else "",
            "delta_otm_minus_dirty": "" if delta is None else delta,
            "dirty_over_otm_ratio": "" if ratio is None else round(ratio, 3),
            "dirty_source": d[1] if d else "",
            "otm_source": o[1] if o else "",
        })

    with OUT_WIDE_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        w.writeheader()
        w.writerows(wide_rows)

    # ---- Markdown wide table
    lines = []
    lines.append("# Dirty vs OTM — matched test-MAE comparison")
    lines.append("")
    lines.append("**Scope:** `spy_phase1_random40_noiselow` matched substrate only "
                 "(no claim across the other 10 OTM variants).")
    lines.append("")
    lines.append("Every cell traces to a committed source bundle.")
    lines.append("Long-format companion: `comparison.csv`.")
    lines.append("")
    lines.append("| Family | Head | Dirty test MAE | OTM test MAE | Δ (OTM − dirty) | Dirty / OTM | Dirty source | OTM source |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for r in wide_rows:
        d = r["dirty_test_mae"]
        o = r["otm_test_mae"]
        delta = r["delta_otm_minus_dirty"]
        ratio = r["dirty_over_otm_ratio"]
        lines.append(
            f"| {r['family']} | {r['head']} | "
            f"{d:.5f}" if isinstance(d, float) else f"| {r['family']} | {r['head']} | "
        )
    # Rebuild lines (the conditional join above is awkward); use a clean loop:
    lines = lines[:-len(wide_rows)]
    for r in wide_rows:
        def fmt(x):
            return f"{x:.5f}" if isinstance(x, float) else ""
        lines.append(
            f"| {r['family']} | {r['head']} | {fmt(r['dirty_test_mae'])} | "
            f"{fmt(r['otm_test_mae'])} | {fmt(r['delta_otm_minus_dirty'])} | "
            f"{r['dirty_over_otm_ratio']} | `{r['dirty_source']}` | `{r['otm_source']}` |"
        )
    lines.append("")
    lines.append("Calibrated / decision-layer view (full metrics, both splits) "
                 "is in `comparison.csv` under `family=anp_calibrated`.")
    OUT_WIDE_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote: {rel(OUT_LONG)} ({len(rows)} rows)")
    print(f"wrote: {rel(OUT_WIDE_CSV)} ({len(wide_rows)} rows)")
    print(f"wrote: {rel(OUT_WIDE_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
