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

**Purpose:** Phase 1 closeout repair. After migrating to the new RunPod network
volume (`/workspace`, pod `0019e3f632c4`, host `213.173.102.225`), the committed
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
