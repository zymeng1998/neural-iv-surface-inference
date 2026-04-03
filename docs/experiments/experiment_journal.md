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
