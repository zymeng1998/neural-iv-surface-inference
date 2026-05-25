# Phase 2 Result Memo — Reliability-Aware Surface Inference

**Date:** 2026-05-25 · **Epic:** 2 (W1 → W5) · **Benchmark:**
`spy_phase1_random40_noiselow` on Alpha Vantage SPY (22.5M strict-cleaned
rows, chronological 11.1M / 5.6M / 5.8M train / val / test).

## TL;DR

Phase 2 closes the reliability layer on top of Phase 1's surface
interpolation floor. The system now emits **six decision-grade outputs per
query** (`sigma_hat`, `confidence_score`, `(lower, upper)` band,
`abstain_flag`, `tradability_score`, `no_arb_risk_flags`), and both
mandatory acceptance numbers pass on real AV data:

| Acceptance bar (§5 / §6)                         | Result                                                                                                                     | Status |
|---|---|---|
| Empirical coverage at α = 0.90 within ±2 pp      | **0.9184** on the 2D.9 test slice (|Δ| = 1.84 pp); **0.8955** on the full 2D.4 test fold (|Δ| = 0.45 pp)                    | PASS   |
| Hi-conf MAE < no-abstention test MAE             | **0.0606** at `keep_fraction=0.8` vs **0.0855** no-abstention test MAE (≈ 29 % reduction)                                  | PASS   |

The calibrated conditional predictor preserves Phase 2C's point-MAE
(0.0855 vs 0.0841) while adding a calibrated band and a ranked
`confidence_score`. The interpolation floor (MAE 0.0730 on the 2D.9
10-date cap, 0.0662 on the full AV fold) remains the headline accuracy
floor; Phase 2's contribution is **reliability**, not accuracy.

Source: `results/2D/comparison_summary.csv`,
`results/2D/spy_phase1_random40_noiselow/conditional_calibrated/metrics_summary.csv`,
`docs/experiments/experiment_journal.md` (2026-05-25T11:00 entry).

## Scope Recap

Phase 2 was framed as **reliability-aware surface inference**, sequenced
W1 → W2 → W3 → W4 → W5:

- **W1 / 2A — Uncertainty evaluation.** A model-agnostic evaluator
  measures coverage, interval width, error-vs-uncertainty rank, the
  abstention curve, and high-confidence MAE for any predictor that emits
  `(pred, lower, upper, uncertainty)`. Interpolation, masked MLP, and
  the conditional model all go through the same evaluator.
- **W2 / 2B — Sensitivity and structure diagnostics.** Masking-
  sensitivity harness produces per-(k, τ) stability metrics;
  no-arbitrage checks (monotonicity, convexity, calendar) produce
  violation counts and per-row `no_arb_risk_flags`. Reports as
  diagnostics, not as a hard constraint.
- **W3 / 2C — Conditional neural surface model.** SetEncoder +
  CoordinateDecoder (85,057 params) ingests the observed chain
  `O_t → z_t` and decodes any `(k, τ)` query conditioned on `z_t`.
  Beats the Phase 1 MLP by ~21 % on test MAE; still ~14 % above the
  RBF interpolation floor.
- **W4 / 2D (uncertainty-aware inference).** Heteroscedastic Gaussian /
  quantile head (2D.2), K = 5 deep ensemble (2D.3 / 2D.8), and a fused
  calibrator (2D.4) that combines Gaussian σ, ensemble disagreement,
  and masking-sensitivity into one `confidence_score` + calibrated
  `(lower, upper)` band.
- **W5 / 2D (decision layer).** Abstention + tradability + no-arb
  forbidden-flag rules (2D.5) yield `abstain_flag`,
  `tradability_score`, `decision_reason`. Pure config-driven NumPy on
  top of the W4 outputs.

## Headline Results — `spy_phase1_random40_noiselow`, test split

Source: [`results/2D/comparison_summary.csv`](../results/2D/comparison_summary.csv)
(2D.9, 10-date cap, 64,610 rows).

| Predictor                | test MAE | hi-conf MAE (keep 0.8) | coverage @ 90 | mean width | abstain rate | mean tradability | forbidden-flag violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| `interpolation` (RBF)    | 0.0730 | 0.0742 | —     | —     | 0.569 | 0.943 | 43,406 |
| `masked_mlp`             | 0.0905 | 0.0919 | —     | —     | 0.015 | 0.998 | 1,004  |
| `conditional_point`      | 0.0841 | 0.0856 | —     | —     | 0.099 | 0.990 | 7,381  |
| **`conditional_calibrated`** | **0.0855** | **0.0606** | **0.9184** | 0.3658 | 1.000 | 0.279 | 7,476  |

Coverage / width are reported only for the calibrated predictor — the
baseline predictors carry no calibrated band (the W1 runner injects a
degenerate full-confidence shim so the W5 decision layer can score them
uniformly).

The calibrated row's `abstain_rate = 1.0` is a **decision-layer
operating-point** artifact, not a quality signal: the current
`configs/decision_layer.yaml` enforces `max_relative_width = 0.5`, which
is tighter than the calibrated Gaussian band (90 % half-width ≈ 1.645 σ,
so relative width ≈ 3.29). Retuning that threshold is the live open
question carried forward from 2D.9 (see Open Questions §1).

## Calibration Evidence

- **2D.4 full-fold calibration** (Gaussian head fused with ensemble
  disagreement and masking-sensitivity):
  - Calibrator parameters: T = 1.087, ensemble_scale = 5.587, u₀ =
    0.0655, u_scale = 0.0391 →
    [`artifacts/calibration/2d4_calibrator.json`](../artifacts/calibration/2d4_calibrator.json).
  - Test coverage **0.8955** at nominal 0.90 (|Δ| = 0.45 pp), mean width
    **0.303**, Pearson(|err|, u) = **0.738**. Source: 2D.4 closing entry
    in `docs/experiments/experiment_journal.md`.
- **2D.9 end-to-end re-score** (10-date cap, 64,610 rows):
  - Test coverage **0.9184** (|Δ| = 1.84 pp ≤ 2 pp tolerance) →
    [`results/2D/spy_phase1_random40_noiselow/conditional_calibrated/metrics_summary.csv`](../results/2D/spy_phase1_random40_noiselow/conditional_calibrated/metrics_summary.csv).
  - Hi-conf MAE **0.0606** at `keep_fraction = 0.8` vs no-abstention
    test MAE **0.0855** → **29 % error reduction** on the
    highest-confidence 80 % of predictions.
- **Calibration plot:** per-predictor
  [`results/2D/spy_phase1_random40_noiselow/<predictor>/calibration_plot.png`](../results/2D/spy_phase1_random40_noiselow/conditional_calibrated/calibration_plot.png).
- **Abstention curve:** per-predictor
  [`results/2D/spy_phase1_random40_noiselow/<predictor>/abstention_curve.png`](../results/2D/spy_phase1_random40_noiselow/conditional_calibrated/abstention_curve.png).

Quantile + conformal head (2D.4 alternate path) under-covers by ~4.3 pp
on the chronological val/test split (exchangeability violation under the
chronological split). This is documented as a known limitation; the
Gaussian + ensemble fusion is the shipped path.

## What Changed vs Phase 1

| Dimension                | Phase 1 (Dubach → AV refresh)                                 | Phase 2 close                                                                                  |
|---|---|---|
| Headline test MAE        | 0.0687 (Dubach) → 0.0662 (AV) interpolation floor             | 0.0730 interp / 0.0855 calibrated conditional on the 2D.9 10-date cap                          |
| Uncertainty signal       | None (point predictors only)                                  | `(lower, upper)` band + `confidence_score` + scalar `uncertainty`                              |
| Per-query decision       | None                                                          | `abstain_flag` + `tradability_score` + `decision_reason`                                       |
| Structural diagnostics   | None                                                          | `calendar_violation` / `monotonicity_violation` / `convexity_violation` per row + region heatmaps |
| Reliability evidence     | None                                                          | Coverage 0.9184 (test), hi-conf MAE 0.0606 < 0.0855                                            |
| Result-artifact discipline | Process gap surfaced at closeout (results not committed)    | Committed `results/2D/<dataset>/<predictor>/` with metrics + figures + region tables           |

Phase 1's interpolation floor still beats every neural predictor on raw
MAE. Phase 2's deliverable is the surrounding reliability scaffolding,
not a lower MAE.

## What Changed vs Phase 2C

| Dimension                | Phase 2C (conditional point)                                                                                   | Phase 2D close (calibrated)                                                              |
|---|---|---|
| Predictor                | SetEncoder + CoordinateDecoder, point head, 85,057 params                                                      | Same backbone + heteroscedastic Gaussian head + 5-seed ensemble                          |
| Outputs per query        | `iv_pred`                                                                                                      | `iv_pred`, `uncertainty`, `(lower, upper)`, `confidence_score`, `abstain_flag`, `tradability_score`, `no_arb_risk_flags`, `decision_reason` |
| Test MAE                 | 0.0841 (point) / 0.0753 on the full 2C.8 AV fold                                                               | 0.0855 calibrated — within ~1 pp of the point baseline                                   |
| Reliability             | None — pure point prediction                                                                                    | Coverage 0.9184, hi-conf MAE 0.0606                                                       |
| Evaluation interface    | W1 evaluator only                                                                                              | W1 + W2 + W5 decision-layer runner; uniform across all four predictors                    |

The calibrated conditional **does not sacrifice point accuracy** for the
reliability scaffolding — the 0.0855 vs 0.0841 gap is within run-to-run
noise. The reliability layer is additive.

## Acceptance-Criteria Map (Phase 2 roadmap §6)

| # | Criterion (abbreviated)                                                                  | Evidence (paths required)                                                                                                                                                                                          | Status |
|---|---|---|---|
| 1 | Model-agnostic uncertainty evaluator exists and is tested.                              | `src/neural_iv_surface_inference/evaluation/` (W1 metrics + runner); `tests/` W1 suite; same evaluator drives all four rows of `results/2D/comparison_summary.csv`.                                                | PASS   |
| 2 | No-arbitrage diagnostics exist and are tested.                                          | `src/neural_iv_surface_inference/diagnostics/` (calendar / monotonicity / convexity); per-row `*_violation` columns and forbidden-flag counts in `results/2D/spy_phase1_random40_noiselow/<predictor>/predictions_decisions.csv`. | PASS   |
| 3 | Masking-sensitivity harness exists.                                                     | W2 harness wired into `scripts/run_decision_layer_eval.py` (`n_draws=5`, `keep_fraction=0.8`); per-row `uncertainty` carries the masking-stability component fused into `confidence_score`.                          | PASS   |
| 4 | Conditional neural surface model evaluated on Phase 1 benchmarks.                       | `artifacts/runs/2D7/<head_kind>/manifest.json` + `training_curve.csv` + `test_predictions.csv`; same benchmark as 2C.8 and the Phase 1 baselines.                                                                  | PASS   |
| 5 | Decision-grade outputs are produced.                                                    | `results/2D/spy_phase1_random40_noiselow/conditional_calibrated/predictions_decisions.csv` columns: `iv_pred`, `confidence_score`, `lower`, `upper`, `abstain_flag`, `tradability_score`, `*_violation`, `decision_reason`. | PASS   |
| 6 | Calibration is demonstrated.                                                            | `results/2D/.../conditional_calibrated/metrics_summary.csv` (test coverage 0.9184); 2D.4 full fold 0.8955; hi-conf MAE 0.0606 < 0.0855 no-abstention. Calibrator: `artifacts/calibration/2d4_calibrator.json`.       | PASS   |
| 7 | Reproducibility and evidence discipline.                                                | Committed manifests under `artifacts/runs/2D7/`, `artifacts/runs/2D8/`; committed `results/2D/` figures + tables; closing journal entry 2026-05-25T11:00 in `docs/experiments/experiment_journal.md`.               | PASS   |

## Open Questions / Known Limitations

1. **Decision-layer `max_relative_width` retuning** (carried from 2D.9).
   Current threshold of 0.5 is tighter than the calibrated 90 % band
   (relative width ≈ 3.29), so `abstain_rate = 1.0` on the calibrated
   predictor. The operating point needs to be retuned — either widen
   `max_relative_width` to align with the calibrated band, or split
   abstention into "structural-flag" vs "width" reasons and report them
   separately. **Next decision:** widen to ~3.5 (consistent with the
   calibrated 90 % half-width) and re-score; gate is whether tradable
   coverage stays meaningful.
2. **Quantile + conformal under-coverage on chronological splits**
   (from 2D.4). Conformal calibration assumes exchangeability; the
   chronological val/test split breaks it (~4.3 pp gap). Gaussian +
   ensemble fusion is the shipped path; the quantile path is retained
   in code as the alternate but is **not** used in the calibrated
   conditional predictor. **Next decision:** keep the quantile head as
   diagnostic-only or remove it from the calibration runner.
3. **2D.9 was run with `max_dates_per_split = 10` on CPU** (the pod's
   RTX PRO 4000 Blackwell GPU was not supported by the installed
   `torch 2.4.1+cu124` kernels). The 64,610-row test slice is a
   representative sample, not the full 5.8M-row fold; the full-fold
   coverage number is **0.8955** from the 2D.4 calibrator verification.
   **Next decision:** re-run 2D.9 on the full test fold once torch is
   upgraded, to get a single-source-of-truth coverage number.
4. **Baseline predictors carry no calibrated band.** Interpolation, MLP,
   and the point conditional flow through the W5 decision layer via a
   degenerate full-confidence shim, so their coverage / width / abstain
   metrics are not directly comparable to the calibrated row. **Next
   decision:** decide whether to add a calibrated band to the
   interpolation / MLP baselines (e.g. residual-bootstrap quantile) or
   document explicitly that the reliability layer is conditional-only.
5. **Forbidden-flag violations cluster at the test-fold tails.** The
   calibrated predictor produces 7,476 structural-flag violations on
   the 64,610-row test slice (~11.6 %). These propagate through the
   `no_arb_risk_flags` column and trigger `decision_reason = "no_arb_*"`
   abstentions independent of width. **Next decision:** decide whether
   structural-flag severity (not just binary) should affect
   `tradability_score`, or whether a hard structural penalty in
   training is needed (Phase 3 framing).

## Reproducing the Headline Numbers

All numbers in this memo trace to committed artifacts:

- `results/2D/comparison_summary.csv` — headline table.
- `results/2D/spy_phase1_random40_noiselow/<predictor>/` — per-predictor
  metrics, region tradability, calibration / abstention figures.
- `artifacts/runs/2D7/<head_kind>/` — Gaussian / point / quantile head
  training manifests + curves + predictions.
- `artifacts/runs/2D8/` — 5-seed ensemble manifest + curves +
  predictions.
- `artifacts/calibration/2d4_calibrator.json` — fitted calibrator.
- `docs/experiments/experiment_journal.md` — 2D.4 (calibration fit) and
  2D.9 (end-to-end eval) entries are the canonical narrative for the
  acceptance numbers.

To regenerate the companion notebook:

```bash
python3 scripts/generate_phase2_results_notebook.py
jupyter nbconvert --to notebook --execute \
  notebooks/05_phase2_results.ipynb \
  --output /tmp/_2D10_check.ipynb
```
