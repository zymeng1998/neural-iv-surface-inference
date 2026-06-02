# Dirty vs OTM — matched test-MAE comparison

**Scope:** `spy_phase1_random40_noiselow` matched substrate only (no claim across the other 10 OTM variants).

Every cell traces to a committed source bundle.
Long-format companion: `comparison.csv`.

| Family | Head | Dirty test MAE | OTM test MAE | Δ (OTM − dirty) | Dirty / OTM | Dirty source | OTM source |
|---|---|---:|---:|---:|---:|---|---|
| anp_calibrated | fused | 0.08135 | 0.01162 | -0.06973 | 7.001 | `results/3/spy_phase1_random40_noiselow/3b_anp/metrics_summary.csv` | `results/3/spy_phase1_random40_noiselow_otm/3x_anp/metrics_summary.csv` |
| anp_ensemble | point | 0.06886 | 0.01220 | -0.05666 | 5.644 | `artifacts/runs/3B5/manifest.json` | `artifacts/runs/3X10/manifest.json` |
| anp_single | gaussian | 0.07256 | 0.01440 | -0.05816 | 5.038 | `artifacts/runs/3B4/gaussian/manifest.json` | `artifacts/runs/3X9/gaussian/manifest.json` |
| anp_single | point | 0.06837 | 0.00987 | -0.05849 | 6.926 | `artifacts/runs/3B4/point_control/manifest.json` | `artifacts/runs/3X9/point_control/manifest.json` |
| anp_single | quantile | 0.06809 | 0.01175 | -0.05634 | 5.794 | `artifacts/runs/3B4/quantile/manifest.json` | `artifacts/runs/3X9/quantile/manifest.json` |
| deepsets_ensemble | point | 0.07477 | 0.01594 | -0.05883 | 4.691 | `artifacts/runs/2D8/manifest.json` | `artifacts/runs/3X8/ensemble/manifest.json` |
| deepsets_single | gaussian | 0.07873 | 0.01530 | -0.06344 | 5.148 | `artifacts/runs/2D7/gaussian/manifest.json` | `artifacts/runs/3X8/single_gaussian/manifest.json` |
| deepsets_single | point | 0.07558 | 0.01752 | -0.05806 | 4.314 | `artifacts/runs/2D7/point/manifest.json` | `artifacts/runs/3X8/single_point/manifest.json` |
| deepsets_single | quantile | 0.07188 | 0.01418 | -0.05770 | 5.069 | `artifacts/runs/2D7/quantile/manifest.json` | `artifacts/runs/3X8/single_quantile/manifest.json` |
| mlp | point | 0.09053 | 0.03006 | -0.06046 | 3.011 | `results/2D/spy_phase1_random40_noiselow/masked_mlp/metrics_summary.csv` | `artifacts/runs/3X7/mlp_otm/manifest.json` |
| rbf | interp | 0.06620 | 0.00613 | -0.06007 | 10.796 | `docs/roadmaps/phase3_accuracy_push.md` | `results/3/spy_phase1_random40_noiselow_otm/rbf/metrics_summary.csv` |

Calibrated / decision-layer view (full metrics, both splits) is in `comparison.csv` under `family=anp_calibrated`.
