# Experiment Journal

> Append-only log of routine experiment runs, observations, and interpretations.
>
> For major failures or course corrections, create a retrospective in `docs/retrospectives/` instead.
> For stable decisions, create an ADR in `docs/decisions/` instead.

---

## Entry format

Each entry should follow this structure:

```
## YYYY-MM-DDTHH:MM:SS-TZ — Experiment: short name

**Purpose:** Why this experiment was run.

**Changed variables:** What was different from the previous run or baseline.

**Result summary:** Key metrics, observations, or outputs.

**Interpretation:** What the results mean.

**Decision impact:** Does this change what we do next? Does it confirm or challenge a hypothesis?

**Next step:** What to do based on these results.
```

If the entry is recorded after the event, include both:

```
event_at: YYYY-MM-DDTHH:MM:SS-TZ
recorded_at: YYYY-MM-DDTHH:MM:SS-TZ
```

---

## Entries

## 2026-04-03T12:35:00-04:00 — Experiment: baseline run on verified benchmark

**Purpose:** Obtain first real-data baseline metrics after confirming benchmark row-count integrity on RunPod.

**Changed variables:** Used verified benchmark dataset `spy_phase1_random40_noiselow.parquet` and ran interpolation (RBF) plus masked MLP via `scripts/run_baseline.py`.

**Result summary:** Test metrics showed interpolation outperforming current small MLP on this benchmark (`interp_rbf` test overall MAE ≈ 0.0687 vs `mlp` test overall MAE ≈ 0.0967). Both models produced full train/val/test outputs in `artifacts/results/baseline_results.csv`.

**Interpretation:** Current neural baseline is a valid runnable benchmark but is not yet competitive with interpolation under this setting; this is consistent with under-capacity/under-training risk on high-volume noisy data.

**Decision impact:** Confirms Phase 1 stack is functional end-to-end and motivates Phase 2 model-capacity/representation upgrades rather than infrastructure or data-pipeline rework.

**Next step:** Run controlled regime comparisons and finalize S4.3 artifact package (plots + table + memo).

## 2026-04-03T12:50:00-04:00 — Experiment: sampled noise-regime interpolation sweep

**Purpose:** Quantify how reconstruction error shifts across low/medium/high noise regimes with fast turnaround for Phase 1 reporting.

**Changed variables:** Held masking fixed (`random40`) and varied noise regime (`noiselow`, `noisemed`, `noisehigh`) on sampled test subsets (120 dates each), using interpolation baseline with nearest-neighbor mode for runtime efficiency.

**Result summary:** Error increased monotonically with noise:
- low: overall MAE ≈ 0.07013
- medium: overall MAE ≈ 0.07109
- high: overall MAE ≈ 0.07279
with corresponding unobserved MAE increases (0.07875 → 0.07954 → 0.08104). Saved to `artifacts/results/interp_sweep_sampled_test.csv`.

**Interpretation:** The benchmark behaves directionally as expected under noise escalation; this provides sanity-check evidence for Phase 1 diagnostics and reporting.

**Decision impact:** Supports using this sweep as part of S4.3 narrative while scheduling broader/full-range sweeps for stronger confidence.

**Next step:** Generate and review Phase 1 figures/table from baseline + sweep outputs, then unblock/resolve S3.3 vendor-reference path.

## 2026-05-20T03:50:00+00:00 — Repair: regenerate missing MLP baseline rows (eval-only)

**Purpose:** Phase 1 closeout repair. After migrating to a new remote network
volume (operational endpoint details intentionally omitted — see local-only
private runbook), the committed
`artifacts/results/baseline_results.csv` was found to contain only `interp_rbf`
rows. The 2026-04-03 baseline entry documents `mlp` results that no surviving CSV
(committed or in either backup) actually contained — only the trained checkpoint
`artifacts/checkpoints/best_mlp.pt` (epoch 6, val_loss 0.0275) remained.

**Environment:** New pod, NVIDIA RTX A4500, Python 3.11.10. `requirements.txt`
deps (pandas, pyarrow, numpy, matplotlib, PyYAML, pytest) plus `scipy` were
missing from the migrated container image and had to be reinstalled — these live
in the image, not on the network volume.

**Changed variables:** None modelling-wise. The existing checkpoint was
**re-evaluated, not retrained**, via a new dedicated script
`scripts/repair_eval_mlp_baseline.py`. The benchmark
(`spy_phase1_random40_noiselow.parquet`) and the chronological train/val/test
splits are identical to the original baseline run. All splits were evaluated with
`shuffle=False` loaders to guarantee prediction/row alignment (the original
`run_baseline.py` evaluated the train split through a `shuffle=True` loader).

**Result summary:** MLP rows restored; CSV now contains `interp_rbf` and `mlp`.
Test split, overall MAE / RMSE:
- `interp_rbf`: MAE 0.06866, RMSE 0.11374 (observed 0.05624 / unobserved 0.07694)
- `mlp`:        MAE 0.09666, RMSE 0.16619 (observed 0.09660 / unobserved 0.09670)

The restored `mlp` test MAE (0.09666) matches the value recorded in the 2026-04-03
entry (≈0.0967), confirming the checkpoint is the same model that produced the
originally-documented metrics. Output written to
`artifacts/results/baseline_results.csv`; prior file backed up under
`artifacts/results/backups/baseline_results_before_mlp_repair_*.csv`. The existing
`interp_rbf` rows were preserved unchanged.

**Interpretation:** The naive coordinate MLP (inputs = `log_moneyness`, `tau` only)
remains clearly worse than per-date interpolation across every split, and notably
its error is essentially flat between observed and unobserved points — it has not
learned surface structure beyond a smooth global fit, and is weakest exactly where
it matters (short maturity and deep-ITM/deep-OTM wings: test MAE ≈ 0.13 and ≈0.19).

**Decision impact:** Reinforces that Phase 2 should pursue structured / latent /
mask-aware surface-level models rather than a coordinate-regression MLP. The
artifact gap also motivates committing result CSVs (not just checkpoints) as
first-class evidence going forward.

**Next step:** Review diff, commit repair (script + restored CSV + docs), then
proceed to S4.3 artifact finalization.

## 2026-05-22T16:13:00+00:00 — Experiment: Phase 1 surface gallery on real SPY benchmark (RunPod)

**Purpose:** Produce the real-data Phase 1 surface visuals (S4.3) — reference /
sparse-observed / reconstructed triptych, spatial error map, and the joint
maturity x moneyness error heatmap — that cannot be made locally (the benchmark
parquet is RunPod-only).

**Environment:** RunPod pod (host d69d0237e073), NVIDIA RTX PRO 4500 Blackwell
32 GB, Python 3.11.10. Code synced from local via tar-over-SSH (pod had no
GitHub key and lacked rsync); scientific deps (pandas/pyarrow/scipy/matplotlib)
reinstalled into the image (they live in the image, not the network volume).
No GPU compute needed — interpolation baseline is CPU/scipy.

**Changed variables:** None modelling-wise. Ran the new
`scripts/generate_phase1_presentation.py --benchmark
spy_phase1_random40_noiselow.parquet --n-dates 2 --heatmap-max-dates 60` and
executed `notebooks/03_phase1_surface_gallery.ipynb` in place. Reconstruction
uses the per-date RBF interpolation baseline (deterministic, no checkpoint). The
joint heatmap was capped to 60 test dates (389,383 points) for runtime; the full
test split is 678 dates / 5.31M rows.

**Result summary:** 14 presentation figures generated + executed notebook 03
(figures embedded). Joint MAE grid (interp_rbf, 60 test dates) —

|        | deep_itm | itm   | atm   | otm   | deep_otm |
|--------|----------|-------|-------|-------|----------|
| short  | 0.1399   | 0.0895| 0.0228| 0.0483| 0.0258   |
| medium | 0.1489   | 0.0244| 0.0111| 0.0573| 0.0718   |
| long   | 0.1412   | 0.0277| 0.0224| 0.0479| 0.1316   |

Sample date 2023-04-03: 6,351 points, 2,593 observed.

**Interpretation:** On real data the joint grid sharpens the marginal story:
**deep-ITM is the worst region across every maturity** (~0.14–0.15), with a
secondary ridge at long-maturity deep-OTM (0.13); ATM is best (~0.011–0.023).
Error is lowest where quotes are dense (ATM) and worst in the sparse wings —
consistent with the spatial error map.

**Decision impact:** Confirms Phase 2 should target the wings and short maturity
with structure-aware / mask-conditioned models. Closes the visual portion of
S4.3 (figures + both notebooks); only the written Phase 1 memo remains.

**Next step:** Write the Phase 1 memo; pull artifacts to local (done) and
terminate the pod. Then resume Phase 2 (real-data uncertainty-eval run + Epic 2B).

**Artifacts:** `notebooks/03_phase1_surface_gallery.ipynb` (committed, figures
embedded); `artifacts/figures/presentation/fig0[1-9]*.png` (regenerable,
gitignored — pulled to local).

## 2026-05-22T10:06:00-04:00 — Experiment: W1 uncertainty-evaluation runner smoke (synthetic)

**Purpose:** Validate the story 2A.5 end-to-end runner — predictor interface
(2A.2) → core metrics (2A.3) → abstention curves (2A.4) → committed artifacts —
on a tiny synthetic benchmark, since no real benchmark parquet exists locally
(data lives on RunPod). This is an integration smoke run, not a research result.

**Changed variables:** New script `scripts/run_uncertainty_eval.py` and module
`src/neural_iv_surface_inference/eval/report.py`. Predictor: interpolation (RBF)
via `InterpolationPredictor`. Dataset: in-code synthetic smile (20 dates × 80
points, chronological train/val/test). Baseline carries `uncertainty=None`, so
interval coverage/width and error–uncertainty correlation are reported as NaN by
design; abstention uses the oracle (`-abs_error`) confidence ranking.

**Result summary:** `interp_rbf` on synthetic data —
- test: overall MAE 0.00865, RMSE 0.01118 (observed 0.00795 / unobserved 0.00914)
- abstention (oracle): AURC 0.00371, high-confidence MAE 0.00354 at keep=0.5,
  0.00578 at keep=0.8 — retained error falls below overall MAE as coverage drops.
Artifacts written to `artifacts/results/uncertainty_eval_synthetic_demo.csv`
(metrics), `..._curve.csv` (risk–coverage sweep), and
`..._risk_coverage.png` (figure).

**Interpretation:** The W1 measurement layer runs end-to-end through the
model-agnostic interface and emits the documented metric + curve + figure
artifacts. Numbers are not comparable to the real-data baseline (different,
easier synthetic surface); their role here is to confirm wiring and artifact
shape. The oracle abstention curve is the best-case reference until W4 supplies
real model uncertainty.

**Decision impact:** Closes Epic 2A (W1). Real-data uncertainty-eval runs (on the
RunPod benchmark, and wiring the MLP predictor via its checkpoint) can now reuse
this runner. Interval/coverage columns stay NaN until W4 produces uncertainty
signals.

**Next step:** On RunPod, run `run_uncertainty_eval.py --benchmark <parquet>` for
`interp_rbf` and add MLP-predictor wiring (load checkpoint) to compare both
baselines through the same interface; begin Epic 2B decomposition.

## 2026-05-22T20:30:00-04:00 — Experiment: W2 structure-diagnostics runner smoke (synthetic)

**Purpose:** Validate the story 2B.5 end-to-end runner — masking-sensitivity
harness (2B.2) → no-arbitrage diagnostics (2B.3) → risk-flag synthesis +
region heatmaps (2B.4) → committed artifacts — on a tiny synthetic grid
benchmark, since no real benchmark parquet exists locally (data lives on
RunPod). Integration smoke run, not a research result.

**Changed variables:** New script `scripts/run_structure_diagnostics.py` and
module `src/neural_iv_surface_inference/diagnostics/report.py`. Predictor:
interpolation (RBF) via `InterpolationPredictor`. Dataset: in-code synthetic
**grid** smile (8 dates × 35 points = 7 log-moneyness × 5 maturities,
chronological train/val/test). The grid is required so the no-arbitrage checks
(which group by exact coordinate) have evaluable pairs/triples. The smile is
smooth + arbitrage-free by construction.

**Result summary:** `interp_rbf` on synthetic data —
- Structural violations: 0 across all dates/splits for calendar, monotonicity,
  and convexity (arbitrage-free grid, as designed). `risk_flag_rate` = 0 with
  the default `instability_threshold = inf`.
- Masking instability (per-date mean per-point std): ~0.003–0.011, positive and
  finite — the harness exercises its full path.
Artifacts: `artifacts/results/structure_diagnostics_synthetic_demo.csv`
(per-date summary), `..._regions.csv` (long-form (maturity × moneyness) grid),
and per-split region heatmaps `..._<split>_risk.png` / `..._<split>_instability.png`.

**Interpretation:** The W2 diagnostic layer runs end-to-end through the
model-agnostic predictor interface and emits the documented summary + region +
heatmap artifacts. Zero structural violations is the expected outcome on a
smooth arbitrage-free surface; the run confirms wiring and artifact shape, not a
research finding. Region risk scores are instability-driven here (no structural
signal) and are normalized per-surface, so they are surface-relative.

**Decision impact:** Completes Epic 2B (W2). The runner is ready for the real
RunPod benchmark, where genuine structural violations and larger instability are
expected to appear. Tradability/abstention policy on top of these flags is W5
(2D).

**Next step:** On RunPod, run `run_structure_diagnostics.py --benchmark <parquet>`
for `interp_rbf` on the real SPY benchmark; review whether genuine no-arb
violations concentrate in specific (k, tau) regions. Then proceed to Epic 2C.
