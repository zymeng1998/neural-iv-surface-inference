# Phase 4 — RBF-prior hybrid vs RBF vs pure-neural (matched OTM)

**Scope:** `spy_phase1_random40_noiselow_otm` matched clean OTM substrate only (no claim across the other 10 OTM variants).

All test MAE vs `iv_clean` on the same fold. The hybrid–RBF delta is the paired, date-clustered bootstrap (4A.7). Every cell traces to a committed source bundle.

| Family / head | OTM test MAE | vs RBF floor | Source |
|---|---:|---:|---|
| rbf_prior_hybrid · gaussian (Phase 4) | 0.006006 | -2.0% | `results/4/spy_phase1_random40_noiselow_otm/4a_hybrid/metrics_summary.csv` |
| rbf · interp (floor) | 0.006132 | floor | `results/3/spy_phase1_random40_noiselow_otm/rbf/metrics_summary.csv` |
| anp_single · point (best pure-neural) | 0.009871 | +61.0% | `results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.csv` |
| anp_single · quantile | 0.011752 | +91.7% | `results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.csv` |
| anp_single · gaussian | 0.014404 | +134.9% | `results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.csv` |
| anp_calibrated · fused (3X.12 production) | 0.011618 | +89.5% | `results/3/spy_phase1_random40_noiselow_otm/3x_anp/metrics_summary.csv` |

## Bar verdict (ADR 0010 §5)

- **Accuracy:** hybrid 0.006006 vs RBF 0.006132; mean Δ -0.000126, 95% CI [-0.000144, -0.000106] (entirely < 0 → significant).
- **Coverage (vs iv_true):** 0.9181 (within ±2pp of 0.90).
- **Hi-conf MAE:** 0.004710 < no-abstention 0.006006.
- **No-arb flag count:** not recomputed (deferred; see headline.json).

**Verdict: bar MET.** Adopt the RBF-prior gaussian residual hybrid as the production accuracy surface (first predictor to significantly beat RBF on the clean OTM substrate), retaining the calibrated reliability layer. Caveats: coverage vs iv_clean over-covers (0.962) — a calibrator-refit-on-iv_clean follow-up; no-arb flag count audit deferred.

Long-format companion: `comparison.csv`; machine-readable verdict: `headline.json`.
