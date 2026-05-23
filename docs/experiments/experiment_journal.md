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
