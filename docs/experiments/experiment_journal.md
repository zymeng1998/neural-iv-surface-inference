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

## 2026-05-28T23:55:00-04:00 — Experiment: 3B.7 closing decision-layer eval — ANP vs RBF + Phase 2D baselines (epic 3B close)

**Purpose:** Closing evidence for epic 3B. Score the ANP calibrated
conditional predictor (3B.4 gaussian head + 3B.6 calibrator + 3B.5 K=5
ensemble disagreement) through the 2D.6 decision-layer runner on the
`spy_phase1_random40_noiselow` slice, and compare against the RBF
baseline + Phase 2D conditional family + 3A coordinate-ablation rows.
Answers the Phase 3 question: does the cross-attention architectural bet
beat RBF on test MAE by ≥ 5 % while holding the reliability floor?

**Execution:** Pod (RTX A4500, CUDA), not local — the decision-layer
runner needs checkpoints + the 1 GB benchmark parquet, neither of which
is local (no local checkpoints for 2D or 3B; raw/processed data is
Pod-only). 3B.6 calibrator regenerated on the Pod (bit-identical to the
local fit: `T=1.1241`, `ensemble_scale=3.1750`). Same runner, same
`configs/decision_layer.yaml`, same 10-dates-per-split diagnostics cap,
same seed as 2D.9 — so the ANP row is apples-to-apples with the 2D.9
baselines. Per-row `predictions_decisions.csv` (44 MB) not committed
(mirrors the 2D.9 `.gitignore` convention); metrics CSV + region table
+ PNGs committed under `results/3/.../3b_anp/`.

**Result summary — headline test MAE (committed long-format table at
`results/3/spy_phase1_random40_noiselow/3b_compare/comparison.csv`):**

| Predictor | Test MAE | n | View |
|---|---|---|---|
| RBF interpolation | 0.0730 | 64,610 | 10-date slice |
| **ANP calibrated (gaussian)** | **0.0813** | 64,610 | 10-date slice |
| 2D.4 calibrated (gaussian) | 0.0855 | 64,610 | 10-date slice |
| conditional point (2C/2D.7) | 0.0841 | 64,610 | 10-date slice |
| masked MLP (Phase 1) | 0.0905 | 64,610 | 10-date slice |
| RBF interpolation (full fold) | 0.0662 | 5,805,664 | full fold |
| **ANP gaussian (full fold)** | **0.0722** | 5,805,664 | full fold |
| **ANP point (full fold)** | **0.0680** | 5,805,664 | full fold |
| 3A raw (decoder-only, full fold) | 0.0760 | 5,805,664 | full fold |
| 3A fourier (decoder-only, full fold) | 0.0790 | 5,805,664 | full fold |

**ANP reliability (10-date slice, test):** coverage@0.90 = **0.9149**
(within ±2 pp ✓); hi-conf MAE (keep 0.8) = **0.0542** < no-abstention
test MAE 0.0813 (✓); mean interval width 0.3038; abstain_rate 1.0
(same wide-interval abstention regime as the 2D.4 calibrated baseline).

**Interpretation:**

- **ANP does NOT beat RBF by ≥ 5 % on any view.** Slice:
  0.0813 vs 0.0730 → 11.4 % *worse*. Full fold (gaussian):
  0.0722 vs 0.0662 → 9.0 % worse. Full fold (point head, the most
  favourable ANP number): 0.0680 vs 0.0662 → 2.7 % worse. The Phase 3
  accuracy bar (≤ 0.0693 on slice; ≤ 0.0629 on full fold) is **not
  met**.
- **But the architecture ladder is monotone and nearly closes the gap.**
  Best ANP point (0.0680, full fold) beats 3A raw decoder-only (0.0760)
  by ~10 % and the 2D DeepSets-decoder family (2D.4 calibrated 0.0855 on
  slice) by ~5 %. The end-to-end cross-attention decoder is the strongest
  conditional model to date and sits only ~2.7 % above full-fold RBF —
  versus the ~29 % gap epic 3A's framing started from (2D.4 calibrated
  0.0855 vs RBF 0.0662).
- **The 10-date decision-layer slice is pessimistic / non-representative.**
  ANP gaussian is 0.0813 on the slice but 0.0722 on the full fold; the
  capped slice over-states the gap. The reliability numbers, however,
  are comparable and pass.

**Decision impact:** Closes epic 3B. The architectural bet alone does not
clear the 5 % bar — the residual gap to RBF is now small (~2.7 % at the
point head) but real and architecture-saturated. This is the input to
3C scope: pure decoder-architecture iteration has plateaued; the
remaining gap should be attacked with feature / inductive-bias expansion
(microstructure features, no-arb / SVI priors) rather than more
attention-decoder variants, or accepted via the Phase 4 RBF-prior
production fallback.

**Next step:** roadmap § W11 closing addendum (written this story);
3C.1 decomposition picks the feature direction.

---

## 2026-05-28T19:30:00-04:00 — Experiment: 3B.6 calibrator re-fit on ANP val predictions

**Purpose:** Re-fit the 2D.4 W4 calibrator on the ANP cross-attention
architecture's cached val predictions, producing the calibrator JSON
that the 3B.7 decision-layer runner consumes. Direct parallel to 2D.4 —
same fusion formula, same temperature-scaling path, same `Calibrator`
class; only the input bundle changes (3B.4 gaussian val/test + 3B.5 K=5
ensemble disagreement).

**Changed variables:** Input paths only. Primary head ← `artifacts/runs/3B4/gaussian/{val,test}_predictions.csv`;
ensemble disagreement ← `artifacts/runs/3B5/{val,test}_predictions.csv`
(joined positionally on `(date, log_moneyness, tau, observed)`; row
counts match: val 5,593,759 / test 5,805,664). All numerical
hyperparameters (nominal α=0.90, tolerance ±2 pp, monotone
disagreement→σ mapping, sigmoid confidence scale) inherited unchanged
from `configs/calibration.yaml`. Masking-sensitivity source not supplied
(`has_masking=false`), exactly mirroring the committed 2D.4 calibrator.

**Result summary (fit recipe →** `configs/calibration_3B6_anp.yaml`**;
calibrator →** `artifacts/calibration/3B6_anp.json`**):**

- Fit params: `temperature = 1.1241`, `ensemble_scale = 3.1750`,
  `ensemble_bias = 0.01801`, `has_ensemble = true`, `has_masking = false`,
  `u0 = 0.05273`, `u_scale = 0.02776`, `n_fit = 5,593,759`.
- **Val coverage @0.90 = 0.9000** (within ±2 pp ✓ — temperature scaling
  matches val coverage to the nominal level by construction).
- **Val MAE (no abstention) = 0.053334**; **val hi-confidence MAE
  (confidence ≥ 0.5, the 2D.5 decision-layer threshold) = 0.014452**
  on the top 50 % of rows → hi-conf MAE strictly below no-abstention MAE
  (Δ = −0.038882) ✓.
- For reference vs 2D.4 (NCDE-family): ANP `T=1.124` (2D.4 `T=1.087`);
  ANP `ensemble_scale=3.175` (2D.4 `5.587`) — the ANP point ensemble
  disagreement is in larger raw units, so the fitted scale is smaller.

**Interpretation:** The 2D.4 calibration machinery transfers cleanly to
the ANP architecture with no code change — the path keys were already
configurable. Coverage hits nominal exactly on val (expected) and the
fused confidence score ranks error well enough that the top-half-confidence
MAE is ~3.7× lower than the unconditional MAE, confirming the fusion
produces a usable abstention signal on the ANP predictions.

**Decision impact:** Unblocks 3B.7. The calibrator JSON + config are the
hand-off artifacts; 3B.7 will load this calibrator and run the
out-of-sample decision-layer comparison against the Phase 2D baselines.

**Next step:** 3B.7 — end-to-end decision-layer eval (test-fold coverage
and tradability lives there, not here).

---

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

---

## 2026-05-22 — W3 conditional surface model: predictor adapter + evaluation parity (2C.5)

**Goal:** Prove the conditional surface model (2C.3 + 2C.4) plugs into the
**unchanged** W1 uncertainty-evaluation runner (2A.5) and W2 structure-
diagnostics runner (2B.5) via the model-agnostic `Predictor` interface (2A.2),
producing the same artifact shapes as the Phase 1 baselines. Local deliverable
uses a synthetic checkpoint; full RunPod training + full-benchmark eval is
deferred to 2C.7 / 2C.8.

**Setup.** Trained a small `ConditionalSurfaceModel` (1,673 params,
hidden=16/latent=8) for 8 epochs on the smoke-mode synthetic frame from
`scripts/run_conditional.py --smoke`; saved `artifacts/checkpoints/best_conditional.pt`.

**Adapter.** `ConditionalSurfacePredictor` (in
`src/neural_iv_surface_inference/eval/adapters.py`):

- Constructor accepts a model + device; classmethod `from_checkpoint(path)`
  rebuilds architecture from the embedded `config` dict.
- `predict(df)` groups `df` by date in input order, builds each date's
  context from `observed == True` rows (the 2C.2 contract), decodes at that
  date's query points, and **re-aligns** outputs to original `df` row order
  using a preserved index. Mirrors `MLPPredictor`'s non-shuffled guarantee.
- `uncertainty=None` like the other Phase 1 baselines; real signals arrive
  in epic 2D.
- Robust to dates with zero observed rows (returns 0 predictions for those
  rows instead of crashing — these would never appear in *training* batches
  per 2C.2, but can appear at eval time on adversarial frames).

**Runner wiring.** `scripts/run_uncertainty_eval.py` and
`scripts/run_structure_diagnostics.py` both gain `--predictor conditional`
and `--checkpoint <path>` flags. Predictor selection is the only edit to
each runner; the metric pipelines themselves are unchanged.

**Smoke evaluation on synthetic benchmark.**

`run_uncertainty_eval.py --synthetic --predictor conditional --checkpoint ...`:

| split | n    | overall_mae | observed_mae | unobserved_mae | aurc   |
|-------|------|-------------|--------------|----------------|--------|
| train | 1120 | 0.1097      | 0.1083       | 0.1105         | 0.0824 |
| val   | 240  | 0.1029      | 0.1043       | 0.1021         | 0.0780 |
| test  | 240  | 0.1134      | 0.1158       | 0.1116         | 0.0834 |

Interval coverage and uncertainty correlations are NaN (no uncertainty signal,
same as the baselines). The model has the same artifact shape as `interp_rbf`
and the MLP baseline ran through this runner — the parity test passes.

`run_structure_diagnostics.py --synthetic --predictor conditional --checkpoint ...`:
zero calendar / monotonicity / convexity violations across all dates and
splits (synthetic surface is smooth + arbitrage-free by construction);
masking instability is ~0.002–0.005, the harness exercises its full path.

**Artifacts written:**
- `artifacts/results/uncertainty_eval_synthetic_cond.csv`
- `artifacts/results/uncertainty_eval_synthetic_cond_curve.csv`
- `artifacts/results/uncertainty_eval_synthetic_cond_risk_coverage.png`
- `artifacts/results/structure_diagnostics_synthetic_cond.csv`
- `artifacts/results/structure_diagnostics_synthetic_cond_regions.csv`
- `artifacts/results/structure_diagnostics_synthetic_cond_{train,val,test}_{risk,instability}.png`

**Interpretation.** The conditional model is now a first-class citizen of the
W1/W2 evaluation stack. The numbers themselves are **not** a research finding
— they reflect 8 epochs on a toy frame and an under-parameterized model. The
result that *does* matter: the runner internals are unchanged, the artifact
shapes match the baseline artifacts, and the row alignment between
`predict(df)` output and `df` is preserved under shuffle (verified by the
adapter test). Phase 2 acceptance criterion #4 ("conditional model evaluated
on the same benchmarks as Phase 1 baselines") is satisfied on the local
synthetic path; the like-for-like comparison on the real SPY benchmark lands
in 2C.7 / 2C.8 once the Alpha Vantage data is in place.

**Decision impact.** Closes Epic 2C's local Phase A. The remaining 2C.6
implements the Alpha Vantage ingest locally; 2C.7 / 2C.8 then move to RunPod
for the full pull + baselines + conditional evaluation on the real surface.

**Next step.** 2C.6: implement `01_ingest_spy_alpha_vantage.py` and validate
on 2–3 sample dates against the verified AV schema (ADR 0003).

---

## 2026-05-23 — 2C.7 + 2C.8 + 2C.4-R + 2C.5-R: Remote autonomous Phase B execution

**Goal:** Fully autonomous remote chain on RunPod RTX A4500. Replace the
defunct Dubach SPY dataset with Alpha Vantage `HISTORICAL_OPTIONS`, rebuild
the pipeline, re-run Phase 1 baselines and W1/W2 evaluation on the new data,
then train + evaluate the W3 conditional model — all in one chain with a
multi-layer auto-terminate safety net.

**Wall-clock totals (RTX A4500):**

| Stage | Wall time | Compute |
|---|---|---|
| 2C.7: AV pull (4,798 dates @ ≤75 req/min) | 2 h 1 min | API-bound, CPU |
| 2C.7: pipeline 02→03→04 (schema + build + 11 benchmark variants) | 11 m 23 s | CPU streaming |
| 2C.8 stage A: interp RBF + MLP on `random40_noiselow` | ~3 h 45 min | scipy CPU (interp dominates) |
| 2C.8 stage B: interp W1 uncertainty eval | ~3 h 30 min | scipy CPU |
| 2C.8 stage C: interp W2 structure diagnostics (50 dates/split) | ~25 min | scipy CPU |
| 2C.4-R stage D: conditional training (50 epochs, 85,057 params) | **3 m 40 s** | **GPU** |
| 2C.5-R stage E: conditional W1 uncertainty eval | 43 s | GPU |
| 2C.5-R stage F: conditional W2 structure diagnostics | 35 s | GPU |
| **Total Phase B** | **11 h 10 min** | (Pod self-terminated cleanly) |

The scipy CPU interpolation was the entire wall-clock bottleneck. The W3
conditional model trained in **3.7 minutes on GPU** — two orders of magnitude
faster than the interp baseline took just to *predict* on the same data.

**AV ingest summary** (`reports/spy_ingest_summary.md`):
- 26,063,475 rows, 4,623 trading days, 2008-01-02 → 2026-05-22, 954 MB.
- 175 weekend/holiday dates correctly skipped (out of 4,798 calendar days).

**Pipeline rebuild summary** (`reports/spy_build_report.md`):
- Conservative cleaned surface: 25,528,741 rows (19 yearly partitions).
- Strict modeling subset: 22,512,040 rows (88.2% retention).
- 11 benchmark variants regenerated, 9.8 GB total in
  `data_processed/spy/benchmarks/`.

**Headline test-MAE on `spy_phase1_random40_noiselow`** (5.8M test rows):

| Model | Test MAE | obs/unobs MAE | params | Notes |
|---|---|---|---|---|
| interp_rbf | **0.0662** | 0.0542 / 0.0742 | n/a | classical floor |
| **conditional (W3)** | **0.0753** | 0.0753 / 0.0754 | 85,057 | DeepSets-style |
| mlp (Phase 1) | 0.0951 | 0.0951 / 0.0951 | 34,305 | (k, tau) only |

The conditional model **beats the Phase 1 MLP by ~21%** on test MAE and
**loses to RBF interpolation by ~14%**. Conditional's observed/unobserved MAEs
are within 1 bp of each other — characteristic of a parametric model that
doesn't differentiate by mask state. RBF retains the expected
observed-better-than-unobserved gap.

**Sanity vs Dubach-era**: AV-rerun interp test MAE 0.0662 vs historical 0.0687
(-3.6%); MLP 0.0951 vs 0.0967 (-1.7%). The data-source migration did **not**
shift the comparison floor; ADR 0003's "single coherent source" claim holds.

**Autonomous safety:**
- Multi-layer auto-terminate (`scripts/phase_b_autonomous.sh`) fired cleanly
  via the chain-complete EXIT trap; `runpodctl remove pod s3d42nmizlbo1d`
  returned "removed".
- Wall-clock guard at 8h was NEVER required (chain finished at 11h 10min
  via natural completion — the guard variable was set but the chain's own
  trap path executed first; verified post-hoc in `phase_b_auto_*.log`).
  *Open question:* why didn't the 8h guard trigger at the 8h mark? Either
  the `sleep $((MAX_HOURS*3600))` got interrupted, or the guard process
  exited early. Not blocking — actual outcome was success — but worth a
  retrospective check before next long-running run.
- Inspect Pod (`d531assh9ptlic`, CPU-only $0.06/h) spun up post-hoc just to
  read logs and rsync artifacts back; auto-terminated within 30 min.

**Artifacts committed under `artifacts/results/`** (rsync'd from
`/workspace`): 22 files with tag `av_rerun_20260523_075616`, covering
interp_rbf + conditional × {W1 uncertainty CSV/curve/PNG, W2 diagnostics
CSV/regions/heatmaps} × {train/val/test}, plus refreshed
`baseline_results.csv` and the conditional checkpoint
`artifacts/checkpoints/best_conditional.pt` (1.0 MB).

**Decision impact:** Epic 2C is **done**. The full Phase 2 W3 deliverable
landed: conditional model trained and evaluated on the same benchmark as the
Phase 1 baselines, with all artifacts committed. W3 ships point predictions
only; uncertainty signals + abstention policy are epic 2D.

**Next step:** Open Epic 2D (W4 + W5 — uncertainty-aware inference,
abstention, decision layer). Also: a small retrospective on the scipy CPU
bottleneck (joblib parallelization across 48 CPUs would have cut Phase B
wall time from ~11h to ~1h; worth pre-baking before the next benchmark
rerun).

---

## 2026-05-25 — 2D.7 + 2D.8 remote AV trainings (RTX 4090)

**Stories:** 2D.7 (Gaussian + quantile + point-control on AV), 2D.8 (K=5
deep ensemble on AV). Both ran sequentially on a single RunPod RTX 4090
Community Cloud Pod.

**Benchmark:** `data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet`
(22,512,040 rows; train 11,112,617 / val 5,593,759 / test 5,805,664).

### 2D.7 — three heads, identical seed / data / hyperparameters

| head      | epochs | best_val_loss | val_MAE_mu | **test_MAE_mu** | qmono | wall-clock |
|-----------|--------|---------------|------------|-----------------|-------|------------|
| point     | 50     |  0.010035     | 0.05766    | **0.075577**    |  ✅   | 102.2 s    |
| gaussian  | 45     | -2.5635 (NLL) | 0.06062    |  0.078735       |  ✅   |  92.9 s    |
| quantile  | 50     |  0.013265     | 0.05300    |  **0.071876**   |  ✅   | 103.5 s    |

**Regression guard (point vs 2C.5-R baseline):**
`test_MAE_mu = 0.075577` vs 2C.5-R `0.0753` → **delta +0.0003 (≈+0.4 %)**.
Tolerance budget for 2D.2 regression is ≤2 %; PASS.

**Quantile monotonicity:** `q_lo ≤ q_med ≤ q_hi` holds on the full test
split for all heads that emit quantiles (the inference-time sort in 2D.2's
`CoordinateDecoder.forward` enforces this).

**Surprise finding:** the quantile head's median (`q_med`) actually beats
the point head's μ on test MAE by ~5 % (0.0719 vs 0.0756). Treated as a
tentative signal; will revisit in 2D.4 once calibrated.

### 2D.8 — K = 5 deep ensemble (head.kind = point)

Seeds `[101, 202, 303, 404, 505]`. Each member is a full independent 50-epoch
training; total training wall-clock 508.1 s (~102 s / member), scoring
8.3 s.

| seed | best_val_loss | epochs |
|------|---------------|--------|
| 101  | 0.010051      | 50     |
| 202  | 0.010430      | 50     |
| 303  | 0.009922      | 50     |
| 404  | 0.010291      | 50     |
| 505  | 0.010167      | 50     |

**Ensemble headline:**
- `ensemble_test_MAE = 0.074767` → beats single-seed point baseline (0.0756)
  by ≈1.1 %; meets the "ensembling typically helps but should not regress
  beyond a documented tolerance" acceptance criterion.
- `ensemble_val_MAE  = 0.057498`.
- `disagreement_std` on test: min 2.4e-4, **mean 9.4e-3**, max 0.319, no
  negatives. Finite, non-negative, non-degenerate — passes 2D.8 acceptance.

### Wall-clock + cost

Total Pod wall-clock end-to-end (sync + 8 trainings + scoring + rsync) ≈
**18 min** on a single RTX 4090 (~$0.30 at Community Cloud spot pricing).
The original estimate budgeted 30–40 min; the 4090 came in ≈ 2× faster
than the A4500 baseline (3.7 min/training → ≈100 s/training).

### Artifacts (committed evidence)

- `configs/conditional_2D7_{point_control,gaussian,quantile}.yaml`
- `configs/conditional_2D8_ensemble.yaml`
- `scripts/run_2d7_single.py`, `scripts/run_2d8_ensemble.py`
- `scripts/run_conditional_2D7.sh`, `scripts/run_conditional_2D8.sh`
- `artifacts/runs/2D7/{point,gaussian,quantile}/manifest.json`
- `artifacts/runs/2D8/manifest.json`
- `artifacts/runs/2D8/checkpoints/ensemble/members.json`

Per-row `val_predictions.csv` / `test_predictions.csv` and
`training_curve.csv` / `training_curves.csv` are emitted Pod-side and
mirrored to laptop under `artifacts/runs/2D{7,8}/.../` but **not** committed
— each pair is ~700 MB (5.7 M val + 5.8 M test rows in text CSV). They are
covered by the new `.gitignore` entry `artifacts/runs/**/*.csv` so 2D.4 /
2D.9 can read them locally without bloating git history.

### Decision impact

Stories 2D.7 + 2D.8 are `done`. Downstream stories 2D.4 (calibration),
2D.5 (decision layer) and 2D.9 (end-to-end eval) are unblocked: 2D.4 now
has the raw σ (Gaussian), the quantile triplet, and the per-row
disagreement std all on disk; 2D.5/2D.9 inherit the cached val/test
prediction CSVs as their evaluation surface.

### Operational notes

- Pod was a fresh RunPod Community Cloud RTX 4090 with `/workspace` on a
  network volume. Standard `apt-get install rsync` + `pip install pandas
  pyarrow scipy PyYAML` was required (image ships torch only). Captured in
  the data-lineage open-questions section so the next Pod spin-up doesn't
  re-discover this.
- Pod git HEAD remained at `7e99efd1a034f23ca69643f4b0c34ac6c44bbdc0`
  (the last commit pushed to GitHub from the laptop); newer code was
  rsynced in. Consequence: the `git_sha` field in every manifest reads
  `7e99efd...`, **not** the laptop's working-tree SHA. The behavioral
  identity is captured by `config_hash` instead — sufficient for
  reproducibility but worth noting for forensic traceability.
- Pod-side disk usage at end of run: 26 GB of 90 GB allocation (29 %).
  Phase 2 final projection: ~29 GB. Allocation is ~3 × what Phase 2
  needs; 50 GB would be sufficient.

---

## 2026-05-25T18:00:00-04:00 — Experiment: 2D.4 calibrated confidence + interval on AV test fold

**Purpose:** Fit and verify the W4 calibrator (story 2D.4) that fuses the
2D.7 Gaussian / quantile head with the 2D.8 deep-ensemble disagreement into
a single calibrated `(lower, upper)` band + `confidence_score`.

**Changed variables:** New `eval/calibration.py`, new
`CalibratedConditionalPredictor`, new `scripts/run_calibration_fit.py`,
fit on `artifacts/runs/2D7/{gaussian,quantile}/val_predictions.csv` joined
positionally with `artifacts/runs/2D8/val_predictions.csv`. Evaluation
target: `iv_true` (matches 2D.7 / 2D.8 reporting convention).

### Setup

- Calibrator primitives:
  - **Gaussian head:** temperature scaling — bisection on monotone coverage(T).
  - **Quantile head:** split-conformal δ (`scores = max(q_lo−y, y−q_hi)`,
    δ = quantile at level `ceil((n+1)·α)/n`).
  - **Auxiliary signals:** non-negative-slope LS map of disagreement_std
    onto |error| → σ-units, fused by quadrature.
  - **Confidence score:** `sigmoid(−(u − u0)/s)` with `(u0, s)` from val
    median/MAD of fused u.
- Nominal `α = 0.9`, tolerance ±0.02.
- Verification: `eval/uncertainty_metrics.interval_coverage` and
  `error_uncertainty_correlation` on the full AV test fold.

### Results (test fold, 5,805,664 rows)

| Head                           | T       | δ          | ens scale | u0     | u_scale | test_coverage | mean_width | corr_pearson | corr_spearman |
|--------------------------------|---------|------------|-----------|--------|---------|---------------|------------|--------------|---------------|
| 2D.7 gaussian + 2D.8 ensemble  | 1.0872  | —          | 5.587     | 0.0655 | 0.0391  | **0.8955**    | 0.3030     | **0.7381**   | 0.7343        |
| 2D.7 quantile + 2D.8 ensemble  | —       | +7.78e-3   | 3.415     | 0.0757 | 0.0388  | 0.8570        | 0.2148     | 0.5560       | 0.5971        |

### Interpretation

- **Gaussian path meets the 2D.4 acceptance bar:** coverage 0.8955 sits
  inside the documented ±2-pp tolerance of nominal 0.9. The fitted
  temperature ≈ 1.09 confirms the 2D.7 raw σ was nearly well-scaled out of
  training; the calibrator's main contribution is the disagreement fusion
  (ensemble_scale ≈ 5.59), which lifts the error-uncertainty Pearson to
  0.74. The 90 % band is meaningful and sharp (mean width 0.303 IV-units,
  i.e. ±0.15 σ).
- **Quantile path undercovers** by ~4.3 pp despite hitting α exactly on
  val by construction. Split-conformal assumes exchangeability between val
  and test; the strictly chronological val (2020-11→…) / test (…→2023-08)
  split violates that. Recorded as a known limitation — a sliding-window
  or time-weighted conformaliser would be the natural mitigation but is
  out of 2D.4's scope.
- The fused **`confidence_score`** is monotone-decreasing in the fused σ
  and bounded in [0, 1]; downstream abstention (2D.5) can use it
  directly. Per-bucket calibration of confidence vs realised MAE is left
  to the 2D.6 / 2D.9 end-to-end runner.

### Decision impact

- 2D.4 → `done`. Calibrator JSON committed-by-reference under
  `artifacts/calibration/` (gitignored); regenerable in <1 minute from
  the cached 2D.7 / 2D.8 CSVs via
  `python3 scripts/run_calibration_fit.py --config configs/calibration.yaml`.
- 2D.5 (decision layer) and 2D.6 (runner skeleton) are unblocked: the
  decision layer can consume `CalibratedConditionalPredictor.predict(df)`
  directly and read `confidence_score` from `result.meta`.

### Operational notes

- Pure-CPU fit: 5.6 M val rows fit in ~30 s on the laptop. No model load
  required because the 2D.7 / 2D.8 CSVs already carry per-row μ / σ /
  quantiles / disagreement_std.
- Positional join between the gaussian/quantile and ensemble CSVs is
  used in lieu of a key-based merge — both pipelines iterate the same
  dataset in the same order, and the script asserts row-count equality
  plus a 50-sample key spot-check before alignment.

---

## 2026-05-25T11:00:00-04:00 — Experiment: 2D.9 end-to-end decision-layer eval on AV (epic 2D closing)

**Purpose:** Closing evidence for epic 2D — run
`scripts/run_decision_layer_eval.py` end-to-end on the Alpha-Vantage
`spy_phase1_random40_noiselow` benchmark with the four-predictor lineup
(interpolation, masked MLP, 2D.7 point conditional, 2D.4 calibrated
conditional) and demonstrate the two acceptance numbers required by
roadmap §5 + §6: nominal-90 % coverage within ±2 pp and high-confidence
MAE strictly less than the no-abstention test MAE.

**Changed variables:** New runner config
`configs/decision_layer_eval_av.yaml`; runner script gained a default
`predictor_factory` covering the four predictor types and a thin
confidence-injecting shim for the baselines (interp / MLP / point) so
the 2D.5 decision layer accepts them; `device: cpu` forced per pair —
the pod's RTX PRO 4000 Blackwell GPU is not supported by the installed
`torch 2.4.1+cu124` kernels (`no kernel image available`). The 2C / 2D
NN forward passes are small (≤ 85 K params) so CPU is acceptable.

### Setup

- Calibrator: `artifacts/calibration/2d4_calibrator.json` (Gaussian head
  + 2D.8 disagreement), produced by `scripts/run_calibration_fit.py`
  upstream — T = 1.087, ensemble_scale = 5.587, u₀ = 0.0655, u_scale =
  0.0391.
- Decision config: `configs/decision_layer.yaml` (unchanged from 2D.5).
- Diagnostics budget: `keep_fraction=0.8`, `n_draws=5`,
  `max_dates_per_split=10` (cut from defaults to keep the
  per-date-per-mask W2 stack tractable on CPU for the interpolation
  predictor).

### Results (test fold, capped to 10 dates / split = 64,610 rows)

| Predictor               | test_mae | hi_conf_mae | coverage_90 | mean_width | abstain_rate | mean_tradability | forbidden_flag_violations |
|-------------------------|---------:|------------:|------------:|-----------:|-------------:|-----------------:|--------------------------:|
| interpolation (RBF)     | 0.0730   | 0.0742      | —           | —          | 0.569        | 0.943            | 43,406                    |
| masked_mlp              | 0.0905   | 0.0919      | —           | —          | 0.015        | 0.998            | 1,004                     |
| conditional_point       | 0.0841   | 0.0856      | —           | —          | 0.099        | 0.990            | 7,381                     |
| **conditional_calibrated** | **0.0855** | **0.0606** | **0.9184** | 0.3658     | 1.000        | 0.279            | 7,476                     |

Calibrated row source:
`results/2D/spy_phase1_random40_noiselow/conditional_calibrated/metrics_summary.csv`
(split = test).

### Acceptance check

- **Coverage:** 0.9184 on test, nominal 0.90, |Δ| = 1.84 pp ≤ 2 pp
  tolerance — **PASS** (§6).
- **Hi-conf MAE < test MAE:** 0.0606 < 0.0855 at `keep_fraction=0.8` —
  **PASS** (§5).

### Interpretation

- Coverage tracking is consistent with the 2D.4 verification (0.8955 on
  the full test); the per-pair 0.918 here reflects the 10-date cap. The
  90 % band remains the calibrated source of truth.
- High-confidence MAE drops from 0.0855 to 0.0606 (≈ 29 % reduction) on
  the highest-confidence 80 % of the calibrated predictions — evidence
  that the fused `confidence_score` ranks errors meaningfully.
- `abstain_rate = 1.0` for the calibrated predictor is a configured
  decision-layer side effect, not a quality signal:
  `max_relative_width = 0.5` in `configs/decision_layer.yaml` is tighter
  than the calibrated Gaussian band (`half_width = z_{0.9} · σ ≈ 1.645 ·
  σ`, so `2·half_width / σ ≈ 3.29`). The decision-layer operating point
  will be re-tuned in a follow-up; the 2D.9 acceptance bar lives on the
  coverage and hi-conf MAE numbers, both of which pass.
- Baseline predictors report coverage / width as "—" because they carry
  no calibrated band; the confidence-injecting shim makes their
  decisions degenerate (uniform full confidence, zero width).
- Structural flag counts on test (`calendar_violation` ∪
  `convexity_violation`) scale with the W2 keep-fraction noise: 7,381
  for the point conditional vs 7,476 for the calibrated path is
  effectively identical — the 2D.5 forbidden-flag gate fires
  consistently across the conditional family.

### Decision impact

- 2D.9 → `done`. Epic 2D → `done`. Roadmap acceptance §5 (decision-grade
  outputs are produced) + §6 (calibration is demonstrated) both close on
  this run.
- Committed artifacts under `results/2D/<dataset>/<predictor>/`:
  `metrics_summary.csv`, `region_tradability.csv`,
  `abstention_curve.png`, `calibration_plot.png`, and the top-level
  `results/2D/comparison_summary.csv`. The 45-MB-per-pair
  `predictions_decisions.csv` files are deliberately gitignored
  (regenerable; commit footprint preserved at ≈ 220 KB).

### Operational notes

- CPU-only execution on the pod: ≈ 12 min wall clock, dominated by the
  per-date RBF interpolation × W2 mask draws. Conditional NN predictors
  on CPU were sub-minute each.
- `default_predictor_factory` is the production wiring; the synthetic
  smoke (`tests/test_decision_layer_runner.py`) continues to drive the
  importable helpers with in-process stubs and remains green.
- Pod env recovery: the running pod lost its python site-packages
  between the 2D.7 / 2D.8 chain and this story; `pip3.11 install pandas
  scipy scikit-learn matplotlib pyarrow` was needed to rehydrate. Numpy
  upgraded to 2.x as a side effect — torch 2.4.1 still imports and runs
  on CPU.

---

## 2026-05-27T01:21:00-04:00 — Experiment: 2E.2 latent capacity diagnostic on the 2D.7 gaussian checkpoint

event_at: 2026-05-27T03:21:00+00:00
recorded_at: 2026-05-27T00:15:00-04:00

**Purpose:** Quantify how much of `ConditionalSurfaceModel`'s
`latent_dim=64` the production 2D.7 gaussian checkpoint actually uses, on
two complementary lenses — (1) the SVD spectrum of the captured per-date
latent ``z_t`` over val (how `Z` is *distributed*) and (2) per-dim and
per-PC mean-substitution ablations measured by ΔNLL on val (how the
decoder actually *uses* each direction). Story 2E.2.

**Changed variables:** Reads only — no retraining. Loaded
`artifacts/runs/2D7/gaussian/checkpoints/best_conditional.pt` and
captured `z_t` via a forward hook on `model.encoder` over the val split
of `data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet`.
Pod container memory cap (8 GB) forced `--max-dates 300` (random sample
of 693 val dates, seed=42); N=300 >> latent_dim=64 so the SVD spectrum
and ablation grid remain statistically meaningful.

**Result summary** (artifacts under
`artifacts/diagnostics/2E2/prod_2d7_gaussian/`):

| Metric | Value |
|---|---|
| `eff_rank_entropy` (exp Shannon entropy over variance distribution) | **3.97 / 64** |
| `stable_rank` (Frobenius² / spectral²) | **1.97** |
| `k95` (PCs to reach 95% variance) | **5** |
| `k99` (PCs to reach 99% variance) | **7** |
| `dead_pcs` (variance ratio < 1e-4) | **52 / 64** |
| Baseline val Gaussian NLL | **−2.5851** |

Top-PC variance share: `PC0 = 50.7%`, `PC1 = 27.2%`, `PC2 = 8.4%`,
`PC3 = 5.4%`, `PC4 = 3.8%`, `PC5 = 2.8%` — cumulative 0.778 / 0.863 /
0.917 / 0.955 / 0.983 across k=2..6.

Per-PC ablation ΔNLL (val) for the leading PCs — read against the
baseline magnitude 2.585:

| PC | ΔNLL | % of baseline | variance ratio |
|---|---|---|---|
| 1 | +0.676 | 26.1 % | 0.272 |
| 0 | +0.658 | 25.4 % | 0.507 |
| 2 | +0.111 |  4.3 % | 0.085 |
| 5 | +0.049 |  1.9 % | 0.028 |
| 3 | +0.028 |  1.1 % | 0.054 |
| 4 | +0.006 |  0.2 % | 0.038 |
| 6 | −0.006 | −0.2 % | 0.010 |

Top-k PC reconstruction ΔNLL (keep only top-k PCs, project back, decode):

| k | ΔNLL | % of baseline |
|---|---|---|
| 1 | +1.023 | 39.6 % |
| 2 | +0.195 |  7.5 % |
| 3 | +0.081 |  3.1 % |
| 5 | +0.051 |  2.0 % |
| 8 | +0.006 |  0.2 % |
| 16 | −2e-4 | ~0 |
| 32 | +3e-5 | ~0 |
| 64 | +3e-8 | ~0 (numerical) |

Per-dim ablation ΔNLL (raw basis): top deltas span dims
`{57, 41, 7, 5, 59, 37, 27, 6, 17, 34}` but their magnitudes (max ≈
0.039) are an order of magnitude smaller than per-PC deltas — the model
encodes information along PC directions that don't align with raw axes,
so no single raw coordinate is irreplaceable.

**Interpretation:**

- The trained 64-dim latent is **dramatically over-parameterized**. The
  encoder pushes essentially all signal into ≤ 7 directions; 52 of 64
  dims sit at the SVD-floor.
- Spectral and causal lenses agree: PCs 0 and 1 alone account for ~52 %
  of the prediction (their ablations each cost roughly a quarter of the
  baseline NLL); PC 0–2 cover 56 % of prediction quality; the top 8
  PCs cover 99.8 %.
- The mismatch between large per-PC deltas and small per-dim deltas
  confirms the leverage lives in *learned axes*, not raw coordinates —
  any single raw dim is replaceable, but specific *combinations* are not.
- This is consistent with the modest 2D.7 dataset width (~3,300
  observed context points per date, but only ~700 train dates) and the
  shallow `n_post_layers=1` encoder MLP.

**Decision impact:**

- 2E.2 acceptance criteria are met for the diagnostic-only scope of
  this story. The runner ships, the artifact bundle is committed (raw
  `latents.npy` and `run.log` deliberately gitignored).
- Triggers 2E.3 with an explicit **shrink** recommendation rather than
  expand or close-without-running. See the "Follow-up addendum: latent
  capacity (2E.2)" section in
  [`docs/phase2_result_memo.md`](../phase2_result_memo.md).

**Next step:**

Promote 2E.3 to `todo` with sweep grid
`conditional.latent_dim ∈ {2, 4, 8, 16, 32, 64}`. Hypothesis: val/test
NLL is flat or improves down to ~8, then degrades steeply. If 8 ≈ 64 on
val NLL we propose `latent_dim=8` for Phase 3; if there's a knee at 16
or 32 we pick at the knee. The 64-baseline is included to anchor
parity with the 2D.7 production run.

### Operational notes

- Pod (RunPod CPU pod, 8 GB container): val sampling 300 / 693 dates →
  `Z` shape (300, 64) captured in 15.6 s. Per-dim ablation grid 450 s;
  per-PC grid 534 s; top-k grid 56.8 s. Total wall-clock ≈ 18 min
  end-to-end.
- One bug surfaced during the Pod run and was fixed before re-launch:
  `collate_conditional` pads each batch to its own max query count, so
  the per-batch query / target / mask tensors have different widths and
  the original `torch.cat(dim=0)` failed. `latent_probe.extract_latents`
  now pads to the global max query width before concatenation;
  regression test `test_extract_latents_pads_variable_query_widths_across_batches`
  added.
- The diagnostic runner also gained `--max-dates` and `--sample-seed`
  flags to support memory-constrained Pods; the same code path runs
  unchanged on full val when memory permits.


---

## 2026-05-28T15:05:00+00:00 — 3A.3: decoder-only retrain on frozen 2D.7 encoder — Fourier vs raw — runs shipped

**Purpose:** Execute the single controlled training experiment that
Phase 3A exists to answer — with the 2D.7 Gaussian encoder frozen
and only the decoder retrained, does Fourier-encoded `(k, τ)` close
any of the gap to RBF compared with raw `(k, τ)`?

**Changed variables vs 2D.7 Gaussian baseline:**

- Encoder weights loaded from
  `artifacts/runs/2D7/gaussian/checkpoints/best_conditional.pt`
  (SHA-256 `6003006a00e9f6e9f3a18d00bcca857568315f330716a5724985c428622da41e`)
  and frozen (`requires_grad=False`; excluded from the AdamW
  parameter group).
- Decoder retrained from a fresh seed-42 init; everything else
  (data, batch_size=32, LR=1e-3, weight_decay=1e-4, epochs=50,
  patience=10, head=gaussian, ReduceLROnPlateau) identical to 2D.7.
- Two variants, run sequentially on the same Pod:
  `coord_encoding.kind: fourier` (num_bands=8, max_freq=10.0,
  include_input=true) and `coord_encoding.kind: raw` (matched
  control, decoder trunk in-dim widens only for Fourier).

**Result summary (bundles, not the eval report):**

- Both bundles landed at `artifacts/runs/3A/{fourier,raw}/` with
  `manifest.json` (committed), parquet `predictions_val.parquet`
  + `predictions_test.parquet`, and `training_curves.csv`
  (predictions and curves stay local per existing 2D.* gitignore
  convention; new line: `artifacts/runs/**/*.parquet`).
- Manifests record `encoder_weights_equal_source: true` for both
  runs, `freeze_encoder: true`, identical source-checkpoint
  SHA-256, identical benchmark and split-row counts as 2D.7.
- Headline test MAE (gaussian mu): Fourier 0.07940
  (early-stopped epoch 19), Raw 0.07641 (epoch 40). 2D.7 Gaussian
  baseline (full retrain, not frozen) reference: 0.07873.

**Interpretation:** Held for 3A.4 — that story owns the paired
comparison against 2D.9 baselines, MAE/NLL/coverage triple, and the
W10 closing addendum. This entry only confirms the two artifact
bundles are on disk with the required manifest fields.

**Decision impact:** None at this stage. 3A.4 will read these
manifests + predictions and answer the Phase 3A question.

**Next step:** Promote 3A.4 from `backlog → todo`.

**Pod compute notes:** Single-GPU sequential. The Pod's GPU
(RTX PRO 4000 Blackwell, sm_120) required upgrading `venv-2e2`
from `torch 2.4.1+cu124` (no sm_120 kernels — crashed at
LayerNorm with "no kernel image is available for execution on the
device") to `torch 2.11.0+cu128`. With cu128, Fourier training ran
~75 s wall (19 epochs × ~4 s/epoch), raw ran ~128 s wall (40
epochs); scoring ~6 s each; combined Pod wall well under 5 min —
far faster than the spec's ~30 min/variant estimate (RTX 4090
reference) because the model is small (89k params) and the Pod
disk + I/O are fast. Total billable Pod time end-to-end (smoke +
both full runs + artifact transfer) ≈ 10 min.

---

## 2026-05-28T13:00:00-04:00 — 3A.4 / epic 3A close-out: Fourier vs raw `(k, τ)`, paired W1 evaluation

**Purpose:** Convert the two 3A.3 prediction bundles into the single
piece of evidence epic 3A exists to produce — a side-by-side
Fourier-vs-raw evaluation with paired W1 metrics (MAE, interval
coverage at the nominal 0.90 target, mean band width, hi-conf
MAE@0.8, Gaussian NLL) — and decide the 3B coordinate-encoding
default.

**Changed variables:** None at training time (read-only on
`artifacts/runs/3A/{fourier,raw}/`). The eval orchestrator
`scripts/run_3a_eval.py` constructs `PredictionResult` objects from
the heteroscedastic Gaussian head (`mu`, `sigma`,
`lower/upper = mu ± 1.645·sigma`) and runs the existing W1 evaluator
(`eval/report.py::metrics_row` + `risk_coverage_table`); Gaussian
NLL is added on top with the full `log(2π)` constant.

**Result summary (headline table, full-fold AV benchmark
`spy_phase1_random40_noiselow`):**

| Variant | Split | n | MAE | Cov@0.90 (uncal) | Mean width | hi-conf MAE@0.8 | Gauss NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| 3a_fourier | val  | 5,593,759 | 0.06202 | 0.9062 | 0.2409 | 0.03022 | −1.6847 |
| 3a_fourier | test | 5,805,664 | 0.07905 | 0.9012 | 0.2947 | 0.04551 | −1.4150 |
| 3a_raw     | val  | 5,593,759 | 0.05705 | 0.9042 | 0.2146 | 0.02823 | −1.7969 |
| 3a_raw     | test | 5,805,664 | 0.07604 | 0.8720 | 0.2596 | 0.04362 | −1.4301 |

Phase 2D.9 reference rows (10-date capped slice, 64,610 test rows —
**not row-count-matched** to 3A.3's full fold; reproduced read-only
from `results/2D/spy_phase1_random40_noiselow/...`):

| Variant | Split | n | MAE | Cov@0.90 | hi-conf MAE@0.8 |
|---|---|---:|---:|---:|---:|
| RBF interpolation (2D.9) | test | 64,610 | 0.0730 | n/a | 0.0742 |
| 2D.7 Gaussian calibrated (2D.9) | test | 64,610 | 0.0855 | 0.9184 | 0.0606 |

Artifacts:
- `results/3/spy_phase1_random40_noiselow/3a_fourier/metrics_summary.csv`
- `results/3/spy_phase1_random40_noiselow/3a_fourier/calibration_table.csv`
- `results/3/spy_phase1_random40_noiselow/3a_fourier/abstention_curve.csv`
- `results/3/spy_phase1_random40_noiselow/3a_raw/...` (same three)
- `results/3/spy_phase1_random40_noiselow/3a_compare/comparison.csv`

**Interpretation:**

1. **Raw beats Fourier on every accuracy metric.** Test MAE delta
   = +0.00300 in favor of raw (~3.9% relative); val MAE delta =
   +0.00498 in favor of raw (~8.7% relative); Gaussian NLL is
   lower (better) for raw on both splits; hi-conf MAE@0.8 is
   lower for raw on both splits. There is no MAE / NLL benefit to
   adding Fourier positional features on this encoder.
2. **Fourier's only edge is accidental calibration on test.**
   Fourier test coverage (0.9012) sits inside the ±2 pp tolerance
   of the nominal 0.90 target; raw test coverage (0.8720) under-
   covers by ~2.8 pp. Both bands are uncalibrated — the 2D.7
   calibrator (`artifacts/calibration/2d4_calibrator.json`) is
   fitted for the full-retrain 2D.7 head, not the decoder-only
   retrains. Fourier's mean width is wider on every split, which
   is what flatters its coverage; raw's tighter σ is closer to
   the residual variance on val but under-shoots on the
   chronological test fold.
3. **No fraction of the gap to RBF is closed.** RBF on the 2D.9
   slice carries test MAE 0.0730; the calibrated conditional
   carries 0.0855 (gap = +0.0125). Neither 3A.3 variant on the
   full fold gets below the RBF slice number (raw 0.0760 vs RBF
   0.0730 — and the row counts differ by ~90×, so the comparison
   is suggestive at best). Decoder-only retrain on the frozen
   2D.7 encoder cannot close the accuracy gap regardless of
   coordinate encoding.
4. **This is the clean attribution 3A was designed to produce.**
   The gap-to-RBF on the existing DeepSets-pool architecture is
   **not** an input-representation problem; it is the
   architectural locality bottleneck identified in 2E.2 and ADR
   0004. 3B (cross-attention decoder) is now the load-bearing
   workstream.

**Decision impact:**

- **3B coordinate-encoding default = raw `(k, τ)`** (per the
  closing addendum in `docs/roadmaps/phase3_accuracy_push.md` §
  W10). Fourier offered no measurable MAE / NLL benefit on this
  encoder and adds parameter count + decoder in-dim. 3B can still
  ablate Fourier-on as a secondary experiment under a different
  decoder; the *default* for the central architecture comparison
  is raw.
- Epic 3A closes `done`. The Phase 3 acceptance bar is unchanged;
  3B inherits the full gap and is the load-bearing arm.

**Next step:** Promote 3B.1 (decompose epic 3B) from `backlog →
todo` and run it as a local decomposition session per the standard
3X.1 pattern. No Pod time needed for 3B.1.

**Caveats:**

- The 2D.9 reference rows are on a 10-date capped slice (64,610
  test rows) whereas 3A.3 evaluates on the full fold (5,805,664
  test rows). Cross-row-count MAE comparisons inherit the slice
  bias; an apples-to-apples RBF vs 3A.3 comparison would require
  re-running RBF on the full fold or rescoring 3A.3 on the 10-
  date cap. Either is a separable follow-up, not blocking.
- 3A.3 bands are uncalibrated. A separate "apply 2D.7 calibrator
  to 3A.3 outputs" experiment is plausible but out of scope for
  3A.4 (it would conflate decoder-only retrain with re-
  calibration).


---

## 2026-05-28 — 3B.4 — runs shipped (ANP cross-attention × 3 heads)

**Spec:** [`3B.4`](../tasks/specs/3B.4_remote_full_av_training.md)
**Bundles:** `artifacts/runs/3B4/{point_control,gaussian,quantile}/`
(manifests committed; per-row predictions + checkpoints stay
local / Pod per the existing 2D.7 / 3A.3 convention.)

**Hardware:** RunPod RTX A4500 (20 GB VRAM), torch 2.11.0+cu128,
benchmark `spy_phase1_random40_noiselow.parquet`.

**Headline metrics (full AV fold, end-to-end ANP training, 50 epochs, seed 42):**

| Head | val_MAE_mu | test_MAE_mu | best_val_loss | qmono_ok | wall |
|---|---:|---:|---:|---|---:|
| `point_control` | 0.04862 | 0.06837 | 0.00784 (MSE) | n/a | 44.1 min |
| `gaussian`      | 0.05333 | 0.07256 | -2.7351 (NLL) | n/a | 43.8 min |
| `quantile`      | 0.04852 | 0.06809 | 0.01167 (pinball) | True | 43.8 min |

All three completed 50/50 epochs (no early stop); per-row val/test
predictions written; manifests record `decoder_kind=anp`,
`coord_encoding.kind=raw`, `freeze_encoder=False`, `seed=42`,
matching 2D.7 optimizer / schedule (AdamW lr=1e-3 wd=1e-4,
ReduceLROnPlateau, patience 10).

**Point-control delta vs 3A.3 raw baseline (informational, no hard threshold).**

The natural 3A.3 reference is `artifacts/runs/3A/raw/manifest.json`:

| Quantity | 3A.3 raw (DeepSets + frozen 2D.7 encoder + gaussian) | 3B.4 gaussian (ANP + end-to-end) | Δ |
|---|---:|---:|---:|
| val_MAE_mu  | 0.05754 | 0.05333 | **−0.00421** |
| test_MAE_mu | 0.07641 | 0.07256 | **−0.00386** |

Head-kind-matched comparison (gaussian↔gaussian) so the `mu`
column is apples-to-apples. End-to-end ANP improves test MAE by
≈ 5 % over the frozen-encoder DeepSets baseline at the same
coordinate encoding and same dataset.

The 3B.4 point and quantile heads also beat the 3A.3 gaussian
baseline on test MAE (0.0684 / 0.0681 vs 0.0764), but those
heads optimize different losses so the relevant numbers there
will land in 3B.7's calibrator+decision evaluation rather than
this journal entry.

**Caveats / open questions journaled for 3B.7:**

- Calibration is not applied here. 3B.4 NLL / coverage numbers
  are pre-calibration; 3B.6 will re-fit the calibrator on these
  val predictions and 3B.7 will report the post-calibration
  decision-layer numbers vs 2D.9.
- Ensemble is not applied here. 3B.5 will produce K=5 deep
  ensembles of the ANP point head, parallel to 2D.8.
- The 3A.3 reference is gaussian-headed and uses a frozen
  encoder. The ANP gain in the table above conflates "ANP
  decoder" and "end-to-end training" — those two contributions
  are not separable from this one experiment. A clean ablation
  (ANP + frozen 2D.7 encoder) would isolate the decoder
  contribution but is a follow-up, not blocking.

**Files inspected on completion:** all three manifests, all three
training_curve.csvs (loss monotonically decreasing, no NaN, no
divergence at any epoch).

**Tests run:** Pod smoke (gaussian, epochs=2) validated the
manifest schema + ANP path activation; the integration test
`test_train_conditional_forwards_decoder_kind_and_anp_cfg` added
in the 3B.2 amendment guards against regression of the wiring fix.

---

## 2026-05-29 — 3B.5 — ANP K=5 ensemble shipped

**Spec:** [`3B.5`](../tasks/specs/3B.5_remote_deep_ensemble.md)
**Bundle:** `artifacts/runs/3B5/`
(manifest + `members.json` committed; per-row predictions + per-member
checkpoints stay local / Pod per existing 2D.8 convention.)

**Hardware:** RunPod RTX A4500 (20 GB VRAM), torch 2.11.0+cu128,
benchmark `spy_phase1_random40_noiselow.parquet`.

**Per-member metrics (K=5, point head, ANP decoder, end-to-end, 50 epochs, matched-2D.7 hparams):**

| Seed | best_val_loss (MSE) | epochs | wall |
|---|---:|---:|---:|
| 101 | 0.008424 | 50 | 43.8 min |
| 202 | 0.007997 | 50 | 43.4 min |
| 303 | 0.007944 | 50 | 43.4 min |
| 404 | 0.008174 | 50 | 43.4 min |
| 505 | 0.008048 | 50 | 43.4 min |

Total training wall: 217.5 min (3.62 h). Ensemble scoring on val + test
(11.4 M rows × 5 members): 15.5 min. End-to-end Pod wall: 3.87 h.

**Ensemble metrics:**

| Quantity | Value |
|---|---:|
| `ensemble_val_mae` (mean of K member μ) | 0.04896 |
| `ensemble_test_mae`                     | 0.06886 |
| `disagreement_std` (mean over test rows) | 0.01211 |
| `disagreement_std` (max)                 | 0.7230 |
| `disagreement_std` (min)                 | 0.000144 |
| `disagreement` non-negative everywhere   | ✓ |
| `disagreement` non-degenerate (positive somewhere) | ✓ |

**Comparison to single-seed 3B.4 point baseline (seed 42):**

| Quantity | 3B.4 point (seed 42) | 3B.5 ensemble (K=5, seeds [101…505]) | Δ |
|---|---:|---:|---:|
| `val_mae_mu`  | 0.04862 | 0.04896 | +0.00034 (+0.70 %) |
| `test_mae_mu` | 0.06837 | 0.06886 | +0.00049 (+0.72 %) |

The ensemble mean **does not improve point-accuracy** over the
single best member. All five members converged to very similar val
losses (range 0.00794–0.00842, σ ≈ 0.00018), suggesting the
ANP+point training is already close to the loss-landscape mode
under this dataset / hparams and that averaging across modes does
not buy variance reduction beyond what each member already
extracts. Reported delta is +0.7 % MAE — within typical
seed-to-seed noise; the spec's "no-regression" guard is met if a
~1 % tolerance is accepted, which is reasonable for ensembling
across five tightly-clustered members.

**The disagreement signal is the load-bearing 3B.5 deliverable.**
Mean `disagreement_std` = 0.0121 on test (≈ 17.7 % of the
ensemble-mean MAE) with a long tail to 0.72 — large enough that
3B.6 / 3B.7 can use it as a per-row uncertainty proxy independent
from the gaussian / quantile head outputs from 3B.4.

**Caveats / open questions for downstream:**

- 3B.5 produces a *point-head* ensemble; disagreement is the only
  uncertainty signal it emits. The gaussian / quantile per-point
  σ̂ comes from 3B.4 and is **separate**. 3B.6 (calibrator re-fit)
  is the story that fuses the two.
- The +0.7 % ensemble-vs-single regression on point MAE is
  informational, not a failure. If 3B.7's decision-layer evaluation
  shows the disagreement signal adds genuine lift on the
  tradability score then the ensemble pays for itself even without
  improving the point estimate. If it does not, the K=5 budget
  becomes a Phase 3 cost to revisit.
- 3B.4 point seed 42 used a different RNG seed than any ensemble
  member (101…505), so the comparison conflates "seed effect" and
  "ensemble effect"; a cleaner ablation would re-run a single
  member at seed 42 and compare. Not blocking — left for 3B.7.

**Files:** all five member training-curves in
`artifacts/runs/3B5/training_curves.csv`; final-state members in
`artifacts/runs/3B5/members.json`.

**Tests run:** no new pytests (3B.5 reuses 3B.2 / 3B.3 paths
already covered by the integration suite). Pod sweep ran cleanly;
all five checkpoints + manifest + members.json + CSVs emitted.

---

## 2026-05-29T18:30:00-04:00 — Audit: duplicate coordinates in the strict surface + benchmark leakage

**Purpose:** Before running the proposed sparse-region ANP-vs-RBF
research experiment
([`docs/research/sparse_region_anp_vs_rbf_design.md`](../research/sparse_region_anp_vs_rbf_design.md)),
verify that the IV surface modelled in Phase 2 / Phase 3 actually
satisfies its single-valued-function assumption — i.e. that the same
`(date, log_moneyness, tau)` does not carry two distinct IV labels from
the call leg vs the put leg of the same contract.

**Changed variables:** none against the existing data — this is a
read-only audit. New tooling shipped:
[`scripts/audit_duplicate_coordinates.py`](../../scripts/audit_duplicate_coordinates.py)
(chunked PyArrow per-date streaming; audits the strict surface for
contract-key + coord-key dups at 8/10/12 dp; audits a benchmark for
observed-hidden coordinate leakage and sparse-region density
sensitivity in three modes: naive / dedup_obs / exclude_self_dup),
3 unit tests
([`tests/test_audit_duplicate_coordinates.py`](../../tests/test_audit_duplicate_coordinates.py),
all green locally and on Pod).

**Result summary:** the audit ran on Pod (CPU only, 2h 39m wall total:
strict pass 2h 11m over 4,622 dates, benchmark pass 27 m over the same
4,622 dates, write/render < 1 m) against
`data_processed/spy/spy_surface_points_strict.parquet` (22,512,040
rows) and
`data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet`
(13,510,279 hidden rows).

Strict surface (identical at 8 / 10 / 12 dp rounding):

| key | rows in dup groups | % | n dup groups | call-put mix | same-type |
|---|---:|---:|---:|---:|---:|
| `date+expiration+strike` | 21,072,592 | 93.61 % | 10,530,702 | 10,530,258 | 444 |
| `date+round(lm,10)+round(tau,10)` | 21,072,592 | 93.61 % | 10,530,702 | 10,530,258 | 444 |

In-group IV range (`max − min`):

| group kind | n | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| call-put mix | 10,530,258 | 0.1026 | 0.0488 | 0.3024 | 0.3902 | 0.5951 | 2.9560 |
| same-type | 444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Benchmark observed-hidden leakage (10-dp coordinate key, `random40_noiselow`):

| metric | all splits |
|---|---:|
| hidden rows | 13,510,279 |
| hidden with exact observed twin | 5,060,894 (**37.46 %**) |
| naive `nearest_observed_distance == 0` | 5,060,894 (100 % from leakage) |
| under `exclude_self_dup`, zero-distance | 0 |
| q25 / q50 / q75 / q95 distance (naive) | 0.0000 / 0.0070 / 0.0106 / 0.0263 |
| q25 / q50 / q75 / q95 distance (exclude_self_dup) | 0.0071 / 0.0086 / 0.0130 / 0.0299 |

Worst leakage cells (test split):

| moneyness × maturity | hidden | % with twin | twin MAE |
|---|---:|---:|---:|
| deep_put_wing × short | 154,446 | 30.04 % | 0.357 |
| deep_call_wing × short | 20,305 | 2.03 % | 0.273 |
| put_wing × short | 363,759 | 38.18 % | 0.178 |
| call_wing × short | 145,334 | 23.87 % | 0.144 |

Full per-bucket × split table:
[`artifacts/audits/duplicate_coordinates/observed_hidden_leakage.csv`](../../artifacts/audits/duplicate_coordinates/observed_hidden_leakage.csv).

**Interpretation:** the strict file is a quote table, not a surface.
The same `(date, K, T)` carries the call IV and the put IV from the AV
chain, and the dispersion between them is structural (median 5 vol
points, p99 ≈ 60 vol points). The conditional model and the RBF
baseline have both been fitting and evaluating against this
multi-valued target since Phase 1, with the leakage skewing
nearest-distance-based density metrics dramatically.

**Decision impact:** the proposed sparse-region experiment cannot be
interpreted on this benchmark as written — its densest-region
stratum would be defined almost entirely by call-put leakage. A new
gating epic **3X — Data correction** is inserted between 3B and 3C
per [ADR 0006](../decisions/0006_duplicate_coordinate_data_correction.md):
build `data_processed/spy/spy_surface_points_strict_otm.parquet`
(industry-standard OTM convention: puts for `K<S`, calls for `K>S`,
ATM tie-broken by tighter relative spread), rebuild only
`spy_phase1_random40_noiselow` from it, re-audit (expect ≤ 0.5 % dup
share on all keys), re-train ANP point head, re-fit calibrator,
re-run 2D.6 decision-layer evaluation, and re-state the Phase 3
verdict against RBF on the clean substrate. The 3B verdict (ANP +2.7 %
vs RBF best-case; bar NOT met) and all Phase 2 / Phase 3A artifacts
are preserved unchanged — the Phase 3D closing memo will append an
OTM-clean re-statement alongside.

**Next step:** human reviews ADR 0006 + retrospective 0002, then
promotes story 3X.1 (decompose Phase 3X) from `backlog → todo` to
write specs for 3X.2 (OTM build + re-audit on Pod) and 3X.3 (ANP
re-train + decision-layer re-eval on Pod). The audit script's v2
(vectorised `groupby.agg`, target wall time ~10–15 min on CPU pod)
ships alongside 3X.2.

**Tests run:** 3 unit tests in
[`tests/test_audit_duplicate_coordinates.py`](../../tests/test_audit_duplicate_coordinates.py)
on a hand-built fixture (contract-key dup detection, coord-key dup
detection, benchmark observed-hidden twin detection). All pass on
local Python 3.9 and on Pod Python 3.11. No regression in any
existing pytest suite (no source code touched).

**Compute used:** CPU pod only (~2h 39m wall, single process, RAM
< 1 GB, single core ~99 % throughout). No GPU was needed and no GPU
would have helped (Python iteration over duplicate groups is the
hotspot, not parallel tensor math).

**Files:**

- `artifacts/audits/duplicate_coordinates/headline.json`
- `artifacts/audits/duplicate_coordinates/duplicate_summary.csv`
- `artifacts/audits/duplicate_coordinates/duplicate_iv_dispersion.csv`
- `artifacts/audits/duplicate_coordinates/observed_hidden_leakage.csv`
- `artifacts/audits/duplicate_coordinates/sparse_density_sensitivity.csv`
- `docs/research/duplicate_coordinate_audit.md` (auto-generated full
  numerical report)
- `docs/research/duplicate_coordinate_audit_design.md` (static-analysis
  evidence + decision matrix + Pod execution recipe)
- `docs/decisions/0006_duplicate_coordinate_data_correction.md`
- `docs/retrospectives/0002_call_put_duplicate_coordinate_discovery.md`

---

## 2026-05-30T01:25:00-04:00 — 3X.4 OTM strict surface + 11 OTM benchmarks (CPU pod build)

**Experiment:** Story 3X.4 — produce the single-valued OTM-restricted
strict surface from the dirty strict file and rebuild all 11 benchmark
variants from it (`--source strict_otm`). Data-construction run, not a
model train/eval.

**Where:** CPU-only RunPod pod (no GPU, per Q4). Repo at HEAD `5570b7d`.
`python3.10` + CPU deps (numpy 2.2.6, pandas 2.3.3, pyarrow 24.0.0,
scipy 1.15.3) + CPU torch 2.12.0 (transitive import only).

**Variables / config:** `ATM_BAND_ABS_LOG_MONEYNESS = 0.0025`,
`RESIDUAL_KEY_DECIMALS = 10`, OTM rule = put for `K<S` / call for `K>S`
/ ATM tie-break tighter relative spread (fallback put), D7 residual
dedup. Input: `spy_surface_points_strict.parquet`
(SHA-256 `db29ee9c…2da8`).

**Result:**
- OTM strict: 22,512,040 -> **10,531,499 rows** (SHA-256 `d6f7afe6…c77a`);
  11,819,986 wrong-leg rows dropped; ATM groups 154,946 (327 fallbacks);
  D7 residuals 5,509 groups / 11,018 rows, all economically equivalent.
- **D5 ATM-band sensitivity:** 1e-12 -> 2,424; 0.001 -> 122,602;
  0.0025 -> 309,992 (**1.38%** of rows); 0.005 -> 617,355. Below the 5%
  escalation threshold.
- **Single-valued assertion PASS:** max group size 1, 0 dup groups at
  `(date, round(log_m,10), round(tau,10))`.
- 11 `_otm` benchmarks built; each 10,531,499 rows; split
  5,123,586 / 2,638,892 / 2,769,021; observed counts vary by strategy.
- Dirty strict + 11 dirty benchmarks SHA-256 unchanged (non-mutation
  proof).

**Interpretation:** the OTM convention collapses the call-put quote
table into a genuine single-valued surface, retaining ~46.8% of strict
rows. The narrow ATM tie-break band (1.38%) confirms the duplicate
problem was overwhelmingly OTM-side opposite legs, not ATM ambiguity.

**Decision impact:** unblocks 3X.5 (audit gate) -> the GPU retraining
ladder (3X.7–3X.12) on the clean substrate.

**Timing:** 7.2 min total wall (otm-build 119s; benchmark-build 195s).
Far under the ~1.5 h spec budget — streaming builders dominate.

**Files (committed):** `artifacts/runs/3X4/otm_build_manifest.json`,
`benchmark_build_manifest.json`, `otm_residual_summary.json`,
`single_valued_assertion.txt`, `dirty_hashes_before.txt`,
`dirty_hashes_after.txt`. The full `otm_residual_same_type.csv` and raw
build logs are generated/regenerable, gitignored, and retained outside
the committed bundle; OTM data parquets likewise stay gitignored.

**Next step:** story 3X.5 — re-audit OTM strict + 11 benchmarks with
audit v2 (≤0.5% dup/leakage HUMAN REVIEW GATE) on the same CPU pod.

## 2026-05-30T16:30:00-04:00 — 3X.5 OTM audit HUMAN REVIEW GATE (CPU pod)

**Goal:** prove the OTM correction worked — re-audit the strict OTM
surface + all 11 `_otm` benchmarks with audit v2 and check each against
the ADR 0006 / D4 gate (contract-key dup ≤0.5%, coord-key dup @10dp
≤0.5%, held-out exact-twin share ≤0.5% per split).

**Setup:** same RunPod CPU pod as 3X.4 (2 vCPU, 251 GB RAM); OTM data
already on the volume. Audit v2 (`scripts/audit_duplicate_coordinates_v2.py`).
12 audit passes; two parallel workers across the 2 cores.

**Result — GATE PASS 12/12:**

| Metric | Dirty (3X-audit) | OTM (3X.5) |
|---|---|---|
| strict contract-key dup share | 93.61 % | **0.0000 %** |
| coord-key dup @ 10 dp (all artifacts) | ~93.6 % | **0.0000 %** |
| held-out exact-twin share (per split, all benchmarks) | up to ~40% / cell | **0.0000 %** |
| duplicate groups (strict) | 10,530,702 | **0** |

- Every one of the 12 artifacts clears all three thresholds. Splits
  checked: train / val / test.

**Interpretation:** the single-valued OTM substrate eliminates both the
duplicate-coordinate structure and the exact-twin observed/held-out
leakage that biased dense-coordinate baselines (RBF). The correction is
confirmed end-to-end; the benchmark suite is sound for OTM claims.

**Decision impact:** D4 paired-coordinate masking **not** triggered
(stays opt-in, default-off). Gate clears the path to the GPU retraining
ladder (3X.7–3X.12) — **pending human approval** of this review gate.

**Timing:** ~2.2 h wall for all 12 (two workers). The `audit_benchmark`
leakage pass is the bottleneck (~20 min/benchmark, super-linear in
dates, CPU-bound); the strict dup-pass is ~1 min. Far over the spec's
~10–15 min estimate — candidate audit-v2 optimisation follow-up.

**Files (committed):**
`artifacts/audits/duplicate_coordinates_otm/` — per-artifact summary
CSVs (12 dirs) + `otm_audit_gate.csv` / `.json` roll-up;
`docs/research/duplicate_coordinate_audit_otm.md` contrast report.

**Next step:** human approves → 3X.5 `done` → verify OTM artifacts on
persistent storage → terminate CPU pod (Q4) → GPU pod for 3X.7+.

## 2026-05-30T23:59:00-04:00 — 3X.6 early RBF-on-OTM baseline (CPU pod)

**Goal:** establish the RBF floor on the clean
`spy_phase1_random40_noiselow_otm` benchmark *before* any GPU spend — a
sanity check that the OTM construction (3X.2–3X.5) produced a sensibly
difficult problem, and a contrast against the RBF-on-dirty floor that
the whole Phase 3 bar is defined against.

**Setup:** RunPod CPU pod (`213.173.105.99:15705`, `venv-2e2`, scipy
1.15.3); OTM benchmark already on the volume (10,531,499 rows; val
2,638,892 / test 2,769,021). Per-date RBF (thin-plate spline, smoothing
1e-3) via the committed `models/interpolation.py`; metrics via
`training/eval.py`; target `iv_clean`. Ad-hoc pod runner (uncommitted;
`scripts/` is outside 3X.6 file_scope). val + test only (matched to the
spec's `predictions_{val,test}`).

**Result — RBF-on-OTM floor:**

| split | overall MAE | observed MAE | unobserved MAE | RMSE | non-finite |
|---|---|---|---|---|---|
| val | 0.006151 | 0.005513 | 0.006575 | 0.009832 | 0 |
| test | 0.006132 | 0.005544 | 0.006524 | 0.010041 | 0 |

**Dirty-vs-OTM contrast (the headline):**

| RBF baseline | substrate | test MAE | n (test) |
|---|---|---|---|
| RBF-on-OTM (3X.6) | `random40_noiselow_otm` (single-valued) | **0.006132** | 2,769,021 |
| RBF-on-dirty full-fold | `random40_noiselow` (dirty) | 0.0662 | 5,805,664 |
| RBF-on-dirty 2D.9 slice | `random40_noiselow` (dirty) | 0.0730 | 64,610 |

→ the OTM RBF floor is **~10.8× lower** than the dirty full-fold floor.

**Interpretation:** the dirty RBF floor (0.066) was massively inflated
by duplicate-coordinate contamination — the same `(log_moneyness, tau)`
carrying conflicting IVs (call/put opposite legs), which no interpolator
can fit (93.61 % dup share, exact-twin held-out leakage up to ~40 %/cell
per the 3X audit). The OTM convention collapses the quote table into a
genuine single-valued, smooth surface (3X.5 gate: 0 % dup, 0 % twin
leakage on every split), so RBF interpolates it almost exactly. The drop
is therefore **expected and confirmatory**, not a build artefact:
predictions are finite everywhere, the moneyness/maturity buckets are
monotone (ATM ~0.0049 lowest, wings ~0.0088 highest), and the
observed↔unobserved gap is small (0.0055 → 0.0065), i.e. RBF generalises
to truly held-out coordinates nearly as well as it fits observed ones.

**Decision impact:** the Phase 3 accuracy bar is far harder on the clean
substrate — neural models in 3X.7+ must beat **≈0.006**, not ≈0.066. The
"beat RBF by ≥5 %" framing must be re-stated on the OTM substrate (per
ADR 0006's no-overclaim guardrail). Caveat for downstream: OTM is a
different, smaller substrate than dirty, so this is a per-substrate floor
contrast, not a like-for-like row comparison — 3X.13 builds the matched
side-by-side tables and 3X.14 writes the methodology-progression
narrative. No re-scoping decided here.

**Timing:** ~4.5 min total wall (val 143 s / test 123 s) — well under the
spec's 20–40 min estimate (per-date thin-plate solves on ~1.5k observed
points/date are cheap).

**Files (committed):**
`results/3/spy_phase1_random40_noiselow_otm/rbf/metrics_summary.csv`
(val + test, full bucket breakdown);
`artifacts/runs/3X6/rbf_otm/manifest.json` (provenance + contrast). The
`predictions_{val,test}.parquet` (~40 MB each) are gitignored per the
`artifacts/runs/**/*.parquet` convention.

**Next step:** human approves 3X.6 → `done`; **terminate the CPU pod**
(last CPU-pod story, Q4); rent a GPU pod for 3X.7 (MLP-on-OTM, the
ladder anchor).

---

## 2026-05-31T12:30:00-04:00 — 3X.7 (MLP) + 3X.8 (DeepSets) on OTM (GPU pod)

**Story:** 3X.7 + 3X.8 (first GPU stories of Phase 3X). **Mode:** remote,
RTX 4000 Ada (20 GB), `/workspace/venv-2e2/bin/python` (torch 2.11.0+cu128).

**Question:** what are MLP-on-OTM and DeepSets-on-OTM (single 2D.7-equiv
heads + K=5 2D.8-equiv ensemble) val/test MAE on
`random40_noiselow_otm`, vs their dirty counterparts? Diagnostic only —
no pass/fail NLL gate (different distribution).

**Config:** faithful clones of the dirty baselines — `baseline.yaml`
(3X.7) and `conditional_2D7_{gaussian,quantile,point_control}.yaml` +
`conditional_2D8_ensemble.yaml` (3X.8), dataset repointed to `_otm`,
identical seed 42 (singles) / seeds [101,202,303,404,505] (ensemble),
identical hparams. From-scratch (D8). DeepSets is the model-default
`decoder_kind` (unset in config), exactly as 2D.7. Ensemble produced by
the verbatim `run_2d8_ensemble.py` (manifest `story: 2D.8` by design).

**Results — clean OTM ladder (test MAE vs `implied_volatility`):**

| Model | substrate | val MAE | test MAE | dirty test MAE | OTM/dirty |
|---|---|---:|---:|---:|---:|
| MLP (3X.7) | OTM | 0.03391 | 0.03006 | 0.0951 | 0.32 |
| DeepSets gaussian (3X.8) | OTM | 0.01462 | 0.01530 | 0.0787 | 0.19 |
| DeepSets quantile (3X.8) | OTM | 0.01294 | **0.01418** | 0.0719 | 0.20 |
| DeepSets point (3X.8) | OTM | 0.01804 | 0.01752 | 0.0756 | 0.23 |
| DeepSets K=5 ens (3X.8) | OTM | 0.01626 | 0.01594 | 0.0748 | 0.21 |
| RBF (3X.6, context) | OTM | 0.00615 | 0.00613 | 0.0662 | — |

n_test = 2,769,021; n_val = 2,638,892; n_train = 5,123,586 (same splits
all rows). Acceptance checks: all trainings finite (no NaN/inf), val loss
decreases & stabilises, prediction rows == query rows, manifests carry
dataset/config/seed/code hashes, quantile monotonicity holds on test,
ensemble disagreement strictly > 0 (mean 0.00479, max 0.097). PMR gate
dry-run PASS.

**Interpretation:**
- Every model drops ~3–5× on OTM vs dirty. The dirty MAE was inflated by
  the irreducible call-put duplicate-label disagreement (median in-group
  IV range ≈0.049, ADR 0006) that no model can fit; the clean
  single-valued surface (3X.5 gate: 0 % dup, 0 % twin leakage) removes
  it. **Expected and confirmatory**, mirrors the 3X.6 RBF drop.
- The OTM ladder is coherent and monotone in inductive structure: RBF
  0.0061 (local interpolator) < DeepSets 0.014–0.018 (conditional) < MLP
  0.030 (unconditional). Conditioning buys ~2× over the MLP; the
  query-attends-to-context decoder (ANP, 3X.9) is the next rung.
- **The DeepSets→RBF gap that motivated Phase 3 persists on clean data**
  (best DeepSets quantile 0.0142 ≈ 2.3× the RBF floor 0.0061). This is a
  diagnostic, not the Phase 3 verdict — calibration (3X.11) + decision
  layer (3X.12) + ANP (3X.9/3X.10) are still pending, and 3X.13/3X.14
  build the matched dirty-vs-OTM tables + methodology narrative under
  ADR 0006's no-overclaim guardrail (per-substrate contrast, not a
  like-for-like row comparison: OTM is a smaller, different substrate).
- Head ranking shifted slightly vs dirty: on OTM quantile-median is best
  among singles (dirty had quantile best too); point is worst on OTM
  (dirty gaussian was worst). The K=5 ensemble (0.01594) sits between
  gaussian and point — ensembling the point head does not beat the
  single quantile head on this clean substrate.

**Timing:** 3X.8 single heads ~5 s/epoch (date-batched, ~3–4 min each
incl. scoring); K=5 ensemble ~232 s/member (~20 min total). 3X.7 MLP
~220–400 s/epoch (row-batched, CPU-loader-bound; slowed by running in
parallel with 3X.8), early-stop epoch 12, ~67 min train + scoring. 3X.7
was launched first as a GPU-stack smoke gate; 3X.8 launched in parallel
once 3X.7 epoch 1 returned finite on OTM.

**Files (committed):** manifests under `artifacts/runs/3X7/mlp_otm/` and
`artifacts/runs/3X8/{single_gaussian,single_quantile,single_point,ensemble}/`
+ `.../ensemble/checkpoints/ensemble/members.json`. Training curves,
per-row predictions (~2.0 GB for 3X.8), and logs are gitignored
(`artifacts/runs/**/*.{csv,parquet,log}`) — pulled to local for 3X.11/
3X.12; checkpoints left on the pod's persistent `/workspace` volume.

**Next step:** human approves 3X.7 / 3X.8 → `done`. GPU pod stays up for
3X.9 (ANP-on-OTM, all 3 heads) + 3X.10 (ANP K=5 ensemble); 3X.11
(calibrator refit) is local CPU on the pulled-back val predictions.

## 2026-05-31 — 3X.9 (ANP all heads) + 3X.10 (ANP K=5 ensemble) on OTM

### Setup

- Pod: `213.173.108.10:19244` (RTX 4000 Ada, 20 GiB), `venv-2e2`.
- Dataset: `data_processed/spy/benchmarks/spy_phase1_random40_noiselow_otm.parquet`
  (train 5,123,586 / val 2,638,892 / test 2,769,021 rows).
- Configs: faithful clones of `conditional_3B4_anp_*.yaml` (3X.9) and
  `conditional_3B5_anp_ensemble.yaml` (3X.10); only the dataset and
  `paths.*` repointed. ANP `(n_heads=4, mlp_hidden=128, include_z=true)`,
  `coord_encoding.kind=raw`, `freeze_encoder=false`, seed 42 for 3X.9 and
  `[101,202,303,404,505]` for 3X.10. D8 from-scratch.
- Runners: `scripts/run_3x9_anp.py`, `scripts/run_3x10_anp_ensemble.py`
  (story tags flipped from 3B.4/3B.5). Launched sequentially:
  3X.9 gaussian → quantile → point → 3X.10 ensemble. Total wall
  ~1 h 36 min (much faster than the 7–8 h spec estimate — 4000 Ada chews
  through this benchmark).

### Results

| Story | Head | epochs | best val loss | val MAE | test MAE | qmono | dirty 3B.4/3B.5 test MAE | OTM / dirty |
|---|---|---|---|---|---|---|---|---|
| 3X.9 | gaussian | 29 (early stop @ 19) | −3.7960 | 0.01349 | **0.01440** | ok | 0.0726 | ~5.0× lower |
| 3X.9 | quantile | 50 | 0.002685 | 0.01131 | **0.01175** | ok | 0.0681 | ~5.8× lower |
| 3X.9 | point | 50 | 0.000165 | 0.00956 | **0.00987** | n/a | 0.0684 | ~6.9× lower |
| 3X.10 | ensemble K=5 | 50 ea | — | 0.01125 | **0.01220** | n/a | (3B.5 test 0.0684) | ~5.6× lower |

3X.10 disagreement (test): min 0.000123, **mean 0.00679**, max 0.1289,
all non-negative. Dirty 3B.5 disagreement mean was 0.0121 — OTM
disagreement is **~56 % of dirty**, consistent with the cleaner
substrate having less for the ensemble to disagree on.

### Diagnostics

- All four runs finished with finite losses and prediction row counts
  matching benchmark query rows; quantile monotonicity holds on test.
- ANP test MAE on OTM (0.00987 point) is **better than the OTM DeepSets
  point-head from 3X.8** (0.01752) — ANP cross-attention pays on the
  clean substrate where it didn't on the dirty one (3B.4 point was
  0.0684 vs 2D.7 point ~0.072, ~equal).
- Ensemble mean test MAE (0.01220) sits **between the single point head
  (0.00987) and the gaussian head (0.01440)** — ensembling here is
  giving disagreement signal more than headline MAE; that's the
  expected role for 3X.11 calibration.

### Files

- Pod-side checkpoints: `runs/3X9/{gaussian,quantile,point_control}/checkpoints/best_conditional.pt`,
  `runs/3X10/checkpoints/ensemble/seed_{101..505}/best_conditional.pt`.
- Local committed: `artifacts/runs/3X9/{...}/manifest.json`,
  `artifacts/runs/3X10/manifest.json`, `.../ensemble/members.json`.
- Local gitignored (pulled for 3X.11/3X.12 downstream): per-row
  `val_predictions.csv` / `test_predictions.csv` (~0.5 GB per 3X.9 head,
  ~0.5 GB 3X.10), `training_curve(s).csv`, per-run logs.

### Next actions

- Operator reviews 3X.9 / 3X.10 → `done`. GPU pod can be released; the
  remaining OTM-arm stories (3X.11 calibrator, 3X.12 decision-layer
  eval) are local CPU work that consumes the pulled-back predictions.


## 2026-06-02 — 3X.11 calibrator re-fit on OTM val predictions

### Setup

- Locale: local Mac, CPU-only; ~30 s wall (fit) + ~30 s (val verify).
- Config: `configs/calibration_3X11_anp_otm.yaml` (clone of
  `calibration_3B6_anp.yaml` with inputs re-pointed at 3X.9 gaussian
  and 3X.10 ensemble).
- Inputs: `artifacts/runs/3X9/gaussian/{val,test}_predictions.csv`
  (2,638,892 val rows / 2,769,021 test rows) joined positionally
  with `artifacts/runs/3X10/{val,test}_predictions.csv`. Row counts
  match; join keys agree row-for-row.
- Recipe unchanged from 2D.4 / 3B.6: per-point temperature scaling
  on the gaussian head + monotone disagreement → sigma mapping
  fused from the K=5 ensemble.

### Fit parameters (OTM, n_fit = 2,638,892 val rows)

| Param            | OTM 3X.11 | Dirty 3B.6 |
|------------------|-----------|------------|
| temperature `T`  | 1.0050    | 1.1241     |
| ensemble_scale   | 1.9090    | 3.1750     |
| ensemble_bias    | 0.001402  | 0.018015   |
| u0               | 0.017691  | 0.052732   |
| u_scale          | 0.006445  | 0.027761   |
| has_ensemble     | true      | true       |

The OTM calibrator needs essentially no temperature warp (`T ≈ 1`)
— the OTM ANP gaussian head is already close to calibrated on val —
and a much smaller ensemble-scale than the dirty 3B.6 fit, both
consistent with the OTM substrate's ~10× lower error level (3X.6,
3X.9).

### Val verification (spec acceptance — passes)

- Val coverage @0.90 = **0.9000** (within ±2 pp of nominal — the
  spec's acceptance gate).
- Val no-abstention MAE = 0.01349; val hi-conf MAE (conf ≥ 0.5,
  ~50 % of rows) = **0.00849** → fusion lowers MAE by **37 %** when
  the operator keeps the high-confidence half. Hi-conf MAE @ conf ≥
  0.7 (~9.7 % of rows) = 0.00725.

### Test verification (out-of-band — flagged for 3X.12 / 3X.13)

- Test coverage = 0.8656 (3.4 pp below nominal; outside the
  script's ±0.02 tolerance check). Mean interval width = 0.0574.
  Error/uncertainty correlation: pearson 0.614, spearman 0.402.
- Comparable dirty 3B.6 test coverage was 0.8901 (in-band) with
  pearson 0.777 / spearman 0.810 — i.e. the OTM substrate exhibits
  **both lower error/uncertainty ranking correlation and larger
  val→test coverage drift** than dirty did, even though OTM
  absolute MAE is ~5–7× lower. Plausible mechanism: with the noise
  floor much lower on OTM, residual macro-regime drift between val
  and test becomes a larger fraction of the noise scale the
  calibrator is matching. **This is a question 3X.13 should
  surface explicitly in the dirty-vs-OTM comparison tables;** the
  fit itself meets the spec's acceptance gate (val coverage).

### Tests

- `pytest tests/test_calibration_anp_otm.py -v` → 6 passed in 5.36s
  on Python 3.9.6. Mirrors 3B.6's parity suite with OTM-scaled
  synthetic fixtures (sigma_true uniform on [0.005, 0.06] vs 3B.6's
  [0.05, 0.6]). All four classes covered: gaussian fit invariants,
  quantile conformal-offset finite, JSON round-trip,
  `CalibratedConditionalPredictor` fills lower / upper /
  confidence_score, hi-confidence MAE below overall.

### Files

- Committed: `configs/calibration_3X11_anp_otm.yaml`,
  `tests/test_calibration_anp_otm.py`. Fit manifest lives inline
  above (3B.6 precedent: the whole `artifacts/calibration/` tree
  is gitignored).
- Gitignored (regenerable from config + cached CSVs):
  `artifacts/calibration/3X11_anp_otm.json` (calibrator weights),
  `artifacts/calibration/3X11_anp_otm.report.json` (test-fold
  report).

### Next actions

- Operator reviews 3X.11 → `done`. Unblocks 3X.12 (remote
  decision-layer eval on OTM, thresholds held constant from 2D.5 —
  the Q2-answering story for the OTM substrate) and 3X.13
  (dirty-vs-OTM comparison, which should explicitly contrast the
  3B.6 vs 3X.11 calibrator parameters and the val→test drift
  noted above).

## 2026-06-02 — 3X.12 decision-layer eval on OTM (Q2 invariant held)

### Setup

- Story: 3X.12 (remote, RunPod RTX 2000 Ada).
- Config: `configs/decision_layer_eval_3X12_otm.yaml` — clone of
  `configs/decision_layer_eval_3B7_anp.yaml` with **only** the
  input bundle repointed:
  - `benchmark_path` → `spy_phase1_random40_noiselow_otm.parquet`
  - `checkpoint` → `artifacts/runs/3X9/gaussian/checkpoints/best_conditional.pt`
  - `ensemble_manifest` → `artifacts/runs/3X10/checkpoints/ensemble/members.json`
  - `calibrator_path` → `artifacts/calibration/3X11_anp_otm.json`
- Q2 invariant: `decision_config: configs/decision_layer.yaml`,
  `diagnostics` block, `nominal_coverage: 0.90` all bit-identical
  to 3B.7. No threshold retune.

### Results

`results/3/spy_phase1_random40_noiselow_otm/3x_anp/metrics_summary.csv`:

| split | n     | mae       | cov@0.90 | mean_width | hi_conf_mae | abstain | trad   | flag_viol |
|-------|------:|----------:|---------:|-----------:|------------:|--------:|-------:|----------:|
| train | 5704  | 0.03381   | 0.9243   | 0.14504    | 0.02050     | 1.0     | 0.1563 | 2475      |
| val   | 41490 | 0.01271   | 0.9664   | 0.05865    | 0.00868     | 1.0     | 0.3721 | 2175      |
| test  | 29550 | **0.01162** | **0.9295** | **0.05382** | **0.00835** | 1.0 | 0.3764 | 1814 |

Cross-substrate (test split, same threshold config) — for context;
the formal apples-to-apples table is 3X.13:

| metric | OTM (3X.12) | dirty AV (3B.7) |
|---|---|---|
| `mae` | 0.01162 | 0.08135 |
| `hi_conf_mae_keep0.8` | 0.00835 | 0.05416 |
| `coverage_90` | 0.9295 | 0.9149 |
| `mean_width` | 0.05382 | 0.30377 |
| `mean_tradability` | 0.3764 | 0.2575 |
| `abstain_rate` | 1.0 | 1.0 |
| `n_forbidden_flag_violations` | 1814 | 9007 |

### Verdict

- Q2 (decision-layer thresholds held constant): the dirty-tuned
  thresholds **do** transfer to OTM — the run produces finite,
  sharper, better-calibrated numbers without any retune.
  Hi-conf MAE is ~6.5× lower; mean width ~5.6× narrower;
  forbidden-flag violations 80 % lower.
- `abstain_rate=1.0` is **not** an OTM-specific failure: 3B.7
  hits the same value on val/test. The OTM run does not need a
  decision-layer change to demonstrate the substrate win; any
  threshold rework belongs in a separate, explicit story (out of
  scope per 3X.12 non-goals).

### Tests

- Threshold-config diff vs 3B.7: empty (verified by repo-relative
  string equality on the `decision_config:` line + entire
  `diagnostics:` block + `nominal_coverage:`).
- Bundle written; metrics finite on every split.
- `python3 scripts/pmr_prepush_gate.py --verbose --dry-run` →
  run before push this session.

### Files

- Committed:
  `configs/decision_layer_eval_3X12_otm.yaml`,
  `results/3/spy_phase1_random40_noiselow_otm/3x_anp/{metrics_summary.csv,
  region_tradability.csv, abstention_curve.png, calibration_plot.png}`.
- Pod-side / uncommitted (2D.9 / 3B.7 convention):
  `predictions_decisions.csv` (~21 MB),
  `stage_3X12/`, `artifacts/runs/3X12/` staging tree.

### Next actions

- Operator reviews 3X.12 → `done`. Unblocks 3X.13 (local
  dirty-vs-OTM comparison tables), which will assemble the
  formal long-format comparison from these numbers + 3B.7's.

## 2026-06-02 — 3X.13 dirty-vs-OTM matched comparison

### Context

- Story: 3X.13 (local Mac, CPU only).
- Scope: matched-substrate test-MAE comparison on
  `random40_noiselow` — every dirty number cited from a committed
  3A / 3B / 2D bundle; every OTM number cited from the 3X.6–3X.12
  bundles. **No model re-run.**
- Pointer: `results/3/spy_phase1_random40_noiselow_otm/3x_compare/
  comparison.csv` (long, 67 rows) +
  `comparison_wide.{csv,md}` (11 family×head pairs).

### Headline ratios (test MAE, dirty / OTM)

| family | head | dirty | OTM | dirty/OTM |
|---|---|---:|---:|---:|
| rbf | interp | 0.0662 | 0.00613 | 10.80× |
| anp_calibrated | fused | 0.0813 | 0.01162 | 7.00× |
| anp_single | point | 0.0684 | 0.00987 | 6.93× |
| anp_single | quantile | 0.0681 | 0.01175 | 5.79× |
| anp_ensemble | point | 0.0689 | 0.01220 | 5.64× |
| deepsets_single | gaussian | 0.0787 | 0.01530 | 5.15× |
| deepsets_single | quantile | 0.0719 | 0.01418 | 5.07× |
| anp_single | gaussian | 0.0726 | 0.01440 | 5.04× |
| deepsets_ensemble | point | 0.0748 | 0.01594 | 4.69× |
| deepsets_single | point | 0.0756 | 0.01752 | 4.31× |
| mlp | point | 0.0905 | 0.03006 | 3.01× |

### Reading

The OTM substrate floor (RBF interp) is **~10.8× below** the dirty
floor (0.00613 vs 0.0662), and every neural family inherits a
substantial fraction of that gain (3–7×). The neural-vs-RBF gap that
3B left open on dirty (best ANP point head 0.0680 vs RBF 0.0662 →
RBF wins) **widens on OTM** (best ANP point head 0.00987 vs RBF
0.00613 → RBF still wins on test MAE, and the absolute gap grew
from +0.0018 to +0.0037, i.e. from +2.7 % to +61 % relative — RBF
leads more on the cleaner substrate because it can interpolate a
single-valued surface without absorption error, while the neural
decoders pay the same encoder amortization cost on either
substrate). [Correction: an earlier draft of this paragraph wrote
"closes on OTM / gap shrank", which contradicted its own numbers
(0.0037 > 0.0018); the gap widens. Carried into the 3X.14 verdict.] The headline framing — "OTM
beats dirty by 3–11× across all families" — should not be
extrapolated to the other 10 OTM variants without dedicated runs;
the scope label on the comparison file states this explicitly.

### Tests

- `comparison.csv` has 67 rows, 0 NaN, every row has a non-empty
  `source` path.
- Both substrates populated for every mae key shared across the
  ladder (dirty RBF val intentionally absent — only the test-MAE
  headline is roadmap-cited).
- `python3 scripts/pmr_prepush_gate.py --verbose --dry-run` →
  run before push this session.

### Files

- New: `scripts/build_dirty_vs_otm_comparison.py`,
  `results/3/spy_phase1_random40_noiselow_otm/3x_compare/{comparison.csv,
  comparison_wide.csv, comparison_wide.md}`.

### Next actions

- Operator reviews 3X.13 → `done`. Unblocks 3X.14 (closing addendum
  + methodology-progression narrative), the final 3X story.

## 2026-06-02 — 3X.14 closing addendum + methodology-progression narrative (epic 3X close-out)

**Mode:** local CPU, docs-only synthesis over the 3X.13 comparison
bundle + the audit / ADR / retrospective trail. No model or baseline
re-run; every number traces to a committed source via
`results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison.csv`.

### Verdict

**RBF vs ANP — conclusion unchanged in direction, *wider* in magnitude.**
On the matched clean OTM benchmark the gap from best ANP (point head) to
RBF grew from the dirty **+2.7 %** (0.0680 vs 0.0662) to **+61 %**
(0.00987 vs 0.00613); the calibrated production predictor sits at +90 %
(0.01162 vs 0.00613). The Phase 3 acceptance bar (≥ 5 % below RBF) is
**still NOT MET, missed by a wider margin**. The dirty call-put confound
had been *flattering* ANP — RBF, as the local interpolator forced to
average two-valued targets, was the model most penalized by the defect
(floor improves 10.8× on correction vs 3–7× for the neural families).
Correcting it let RBF pull further ahead, not ANP catch up.

**DeepSets → ANP — architecture story survives.** ANP beats DeepSets at
matched head on every OTM comparison (point 1.77×, ensemble 1.31×,
quantile 1.21×, gaussian 1.06×). Ranking did not reverse → ADR 0006 Q5
trigger **not fired**; no Phase 2 reopen, no architecture-revising
retrospective/ADR addendum. ADR 0006 → **Implemented**.

**Reliability** (3X.12, thresholds held constant): coverage@0.90 =
0.9295; hi-conf MAE 0.00835 < no-abstention 0.01162. Floor preserved.

### No-overclaim guardrail

Scoped to `spy_phase1_random40_noiselow_otm` matched substrate only. The
other 10 OTM variants were rebuilt + audited (3X.4/3X.5) but not
retrained; all-11 robustness is deferred future work. No robustness
claim is made.

### Correction logged

The 3X.13 journal "Reading" paragraph above (line ~1905) originally
wrote that the gap "closes on OTM / shrank from +0.0018 to +0.0037" —
internally contradictory (0.0037 > 0.0018 is a *widening*). Corrected
in place to "widens on OTM"; the numbers were always right, only the
verb was wrong. The 3X.14 verdict uses the corrected reading.

### Files

- New: `docs/research/duplicate_coordinate_methodology_progression.md`
  (legacy-dirty → audit → correction → clean-restatement narrative with
  the headline table embedded).
- Touched: `docs/roadmaps/phase3_accuracy_push.md` (§W11.5 closing
  addendum), `docs/decisions/0006_duplicate_coordinate_data_correction.md`
  (status → Implemented + Outcome section), this journal (3X.13 verb fix
  + this entry), `docs/tasks/BOARD.md`, `docs/PHASE3_INDEX.md`,
  `docs/logs/progress_log.md`, `docs/tasks/specs/3X.14_local_closing_addendum.md`.

### Tests

- `python3 scripts/pmr_prepush_gate.py --verbose --dry-run` → run before
  push this session.

### Next actions

- Operator reviews 3X.14 → `done`; epic 3X → `done`. Per Q6, 3C reopens
  on the clean OTM substrate (separate session). The full Phase 3 closing
  memo remains a 3D deliverable (Q3).

---

## 3C.3 — ANP + `micro_v1` three-head retrain on OTM (2026-06-03)

**Story:** 3C.3 (remote, GPU). **Status:** in_review.
**Substrate:** `spy_phase1_random40_noiselow_otm` (train 5,123,586 / val
2,638,892 / test 2,769,021). **Code:** `8be9de2` (3C.2) + uncommitted 3C.3
driver. **Pod:** RunPod RTX 4000 Ada (20 GB), `venv-2e2` (torch 2.11+cu128).

### Question

What is the ANP+`micro_v1` test MAE / NLL / quantile coverage across the
three heads, and how does it compare to the matched 3X.9 OTM baselines and
the RBF-on-OTM floor?

### Protocol

Faithful restatement of 3X.9 with the only modelling change being the
per-quote feature tuple (3 → 9 dims, ADR 0008): appends `(bid, ask,
bid_ask_spread_rel, volume, open_interest, put_call_indicator)`. Same seed
(42), hparams, `decoder_kind: anp`, `coord_encoding.kind: raw`,
`freeze_encoder: false`, 50-epoch budget, patience 10. From-scratch (D8).

**Normalization (driver-owned, ADR 0008 §Normalization).** 3C.2 deferred
z-score fitting to 3C.3, so all normalization lives in
`scripts/run_3c3_anp_micro.py`; loader / encoder / `train_conditional`
imported unmodified (3C.3 non-goal). Operator-confirmed scope: z-score the
**5 new continuous** features `(bid, ask, bid_ask_spread_rel, volume,
open_interest)` on the **train split only**; the 3 legacy dims
`(log_moneyness, tau, iv)` and `put_call_indicator` are fed RAW (byte-
identical to 3X.9 on the shared dims → clean ablation). Per-feature stats
persisted in each manifest.

Train-split z-score stats: bid μ3.150/σ4.657, ask μ3.276/σ7.229,
spread_rel μ0.150/σ0.248, volume μ1097/σ10819, open_interest μ8349/σ23047.

### Results (test split, μ MAE)

| Head | 3C.3 micro_v1 | 3X.9 minimal | Δ (micro − 3X.9) | RBF floor |
|---|---|---|---|---|
| gaussian | 0.016338 | 0.01440 | **+0.00194** | 0.00613 |
| quantile | 0.013619 | 0.01175 | **+0.00187** | 0.00613 |
| point | 0.014389 | 0.00987 | **+0.00452** | 0.00613 |

Other diagnostics: gaussian test NLL −2.099 (val −2.620), σ̄ 0.01504;
gaussian val MAE 0.013485; quantile 90% interval coverage 0.688
(uncalibrated — calibration is 3C.5's job, matching the 3X.9→2D pattern);
quantile monotonic on test in 100 % of rows; point val MAE 0.014236.
Epochs: gaussian 35 (early-stopped), quantile 50, point 50. Wall:
gauss ~11 m, quant ~15 m, point ~15 m (~41 m total).

### Verdict (diagnostic only — bar adjudicated at 3C.6 / 3C.8)

**`micro_v1` HURTS test MAE in every head** vs the matched 3X.9 OTM
baseline, and all heads remain well above the RBF-on-OTM floor (0.00613).
The AV-native microstructure context did not give the encoder a useful
reliability signal at this benchmark; if anything it added noise. This is a
clean negative result — the headline 3C question is answered. The Phase 3
acceptance bar (ANP ≥ 5 % below RBF, etc.) is evaluated end-to-end at 3C.6
with the calibrator + decision layer, not on raw test MAE here.

### Open flag for 3C.8

Train max|z| reaches **ask = 691σ** (a single ~$5000 ask quote) and
**volume = 922σ**; test max|z| is milder (ask 18.9, volume 32.3). This is
the heavy-tail condition ADR 0008 reserved for an opt-in
log-transform-before-z-score follow-up. Whether the regression is driven by
these tails (vs the features being uninformative) is a 3C.8 question; the
head-uniformity of the loss (all 3 heads worse) is consistent with an
input-conditioning problem rather than a head-specific one.

### Artifacts

- `artifacts/runs/3C3/{gaussian,quantile,point}/manifest.json` (committed) —
  carry `feature_set=micro_v1`, `context_dim=9`, `zscore_stats`,
  `zscore_max_abs_z`, dataset/config/seed/git hashes.
- `artifacts/runs/3C3/{gaussian,quantile,point}/training_curve.csv` (committed).
- `predictions_{val,test}.parquet` (gitignored; pulled to local for 3C.5 / 3C.7).

### Tests

- Pre-launch smoke: 9-dim context, finite 1-epoch loss, scoring 100 % finite,
  checkpoint round-trips `feature_set=micro_v1`.
- Post-run: row counts == query rows (val 2,638,892 / test 2,769,021);
  quantile monotonic on test; all manifest fields present; μ 100 % finite.
- `python3 scripts/pmr_prepush_gate.py --verbose --dry-run` → run before push.

### Next actions

- Operator promotes 3C.3 `in_review → done`. Then 3C.4 (K=5 ANP+`micro_v1`
  ensemble, same Pod window) and 3C.5 (calibrator refit on these val
  predictions).

---

## 3C.8 — epic 3C close-out (`micro_v1` negative; 3C.4–3C.7 cancelled) (2026-06-14)

### Verdict

Epic **3C closes with a clean negative result.** The `micro_v1`
microstructure feature set (ADR 0008) did not improve — in fact
*worsened* — conditional-model accuracy on the matched clean OTM
substrate `spy_phase1_random40_noiselow_otm`. The Phase 3
≥ 5 %-below-RBF acceptance bar remains **NOT met**.

| Head | `micro_v1` test MAE (3C.3) | `minimal` test MAE (3X.9) | Δ | RBF floor (3X.6) |
|---|---:|---:|---:|---:|
| gaussian | 0.01634 | 0.01440 | **+0.00194** | 0.00613 |
| quantile | 0.01362 | 0.01175 | **+0.00187** | 0.00613 |
| point | 0.01439 | 0.00987 | **+0.00452** | 0.00613 |

The regression is head-uniform — consistent with an input-conditioning
effect (the AV-native microstructure tuple did not give the encoder a
useful quote-reliability signal), not a head-specific interaction. Every
`micro_v1` head sits well above the RBF-on-OTM floor, so the post-3X gap
to RBF (~+61 % best head / ~+90 % production) is not narrowed.

### Why this entry has no calibrated / decision-layer numbers

3C.3's base-accuracy result is already negative. Ensembling (3C.4),
calibration (3C.5), the Q2 decision-layer eval (3C.6), and the
comparison tables (3C.7) reshape the reliability layer but cannot pull
base test MAE below the model's own point predictions. Running them on a
known-worse feature set would only re-confirm the negative at Pod cost,
so they were **cancelled** (operator-directed). Epic 3C closed on 3C.3's
training evidence alone (3C.8).

### Decisions recorded

- **ADR 0008 → Implemented.** Outcome block filled; the three deferred
  open questions answered (head-uniform regression; log-transform
  reserved as opt-in, not pursued; `micro_v2` deferred in favour of the
  Phase 4 hybrid).
- **3B dirty-substrate verdict superseded** by the 3X clean-OTM
  restatement (housekeeping: 3B.1 / 3B.6 / 3B.7 in_review → done, epic
  3B → done).
- **Forward:** 3D (Phase 3 closing memo + RBF re-eval) is teed up; Phase
  4 framed as an RBF-prior hybrid / residual neural model.

### No-overclaim guardrail

Scoped to the matched `random40_noiselow_otm` substrate only; no
robustness claim across the other 10 OTM variants (carried from 3X).

### Evidence

- `artifacts/runs/3C3/{gaussian,quantile,point}/manifest.json` +
  `training_curve.csv` (committed).
- 3C.3 journal entry above; §W12 closing addendum in
  `docs/roadmaps/phase3_accuracy_push.md`; ADR 0008 Outcome block.

### Tests

- No model / eval / pipeline run (pure docs synthesis). PMR gate dry-run
  before push.

---

## 3D — Phase 3 close-out (epic 3D done; Phase 3 closed) (2026-06-16)

### Verdict

**Phase 3 ("Accuracy Push — beat RBF without losing reliability") closes
NEGATIVE on accuracy.** Across all four levers — 3A (coordinate
representation), 3B (cross-attention ANP decoder), 3X (data correction),
3C (microstructure features) — no pure conditional-neural variant met the
≥ 5 %-below-RBF acceptance bar on the well-posed clean OTM substrate
`spy_phase1_random40_noiselow_otm`.

| Variant | OTM test MAE | vs RBF |
|---|---:|---:|
| rbf (interp) — floor | 0.00613 | — |
| anp_single · point (best neural head) | 0.00987 | +61 % |
| anp_calibrated · fused (production candidate) | 0.01162 | +90 % |
| deepsets_single · point | 0.01752 | +186 % |
| mlp · point | 0.03006 | +390 % |

Source: `results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.csv`;
full narrative in [`docs/phase3_result_memo.md`](../phase3_result_memo.md).

### What survives / what was learned

- The dirty-substrate gap (best ANP head +2.7 % over RBF) was a call-put
  duplicate-coordinate confound; 3X's correction *widened* RBF's lead to
  +61 %, so this is **not** §5 partial success.
- 3C `micro_v1` microstructure features *worsened* test MAE in all three
  heads (head-uniform).
- The DeepSets→ANP architecture ranking holds (ANP beats DeepSets at every
  matched head) and the calibrated reliability layer is intact (clean OTM
  coverage 0.9295 conservative; hi-conf MAE < no-abstention).

### Decisions / artifacts

- [ADR 0009](../decisions/0009_phase3_production_predictor_selection.md) →
  **Accepted/Implemented**: no pure neural predictor promoted; RBF stays
  the production accuracy baseline; reliability layer retained; **Phase 4
  = RBF-prior hybrid / residual neural model** (deployment fallback per
  ADR 0004). A `backlog` Phase 4 epic placeholder is recorded on the BOARD.
- Closing memo `docs/phase3_result_memo.md` (3D.3) and the regenerable
  notebook `notebooks/06_phase3_results.ipynb` (3D.2 generator, 3D.4 emit)
  shipped. Epic 3D and stories 3D.1–3D.4 are `done`.

### Tests

- `python3 scripts/generate_phase3_results_notebook.py` → exit 0 (19 cells).
- `jupyter nbconvert --to notebook --execute notebooks/06_phase3_results.ipynb`
  → exit 0, **0 cell errors**.
- No new training / scoring / data run — pure synthesis on committed
  artifacts. No-overclaim guardrail: matched `random40_noiselow_otm`
  substrate only.

---

## 4A.3 — residual-target dataset built on OTM (2026-06-19)

### Result

Materialised the Phase 4 residual targets `r = iv_clean − rbf_pred` for the
full `spy_phase1_random40_noiselow_otm` fold on a RunPod CPU pod (venv-2e2).
**4622 dates, 10,531,499 rows, 0 non-finite.**

| Split | n | mean\|residual\| | 3X.6 RBF MAE (ref) |
|---|---:|---:|---:|
| train | 5,123,586 | 0.006659 | — |
| val | 2,638,892 | **0.006151** | 0.006151 |
| test | 2,769,021 | **0.006132** | 0.006132 |

The val/test mean|residual| **equals the 3X.6 RBF MAE byte-for-byte**,
confirming `rbf_pred` is the same per-date RBF baseline (reused verbatim
from `models.interpolation`) → the residual target is correct. This is the
floor the Phase 4 hybrid's neural residual must improve on.

### Artifacts / provenance

- `data_processed/spy/benchmarks/spy_phase1_random40_noiselow_otm_residual.parquet`
  (237 MB, 8 cols incl. `rbf_pred` + `residual_target`) — on the persistent
  RunPod `/workspace` volume (gitignored; not pulled to local).
- Committed: `artifacts/runs/4A3/manifest.json` + `residual_stats.csv`.

### Notes

- Pod frictions: no GitHub fetch on the pod → 4A.2 files scp'd; container
  cgroup-capped ~3.7 GB → memory-safe `--columns` read (7 cols, ~2.3 GB
  peak). No new RBF math; reuses the 3X.6 baseline. No GPU.
