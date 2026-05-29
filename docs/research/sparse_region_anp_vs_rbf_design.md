# Research note — Sparse-region comparison: ANP vs RBF

---
created_at: 2026-05-29T00:30:00-04:00
last_updated_at: 2026-05-29T00:30:00-04:00
status: proposed   # proposed | in_progress | executed | superseded
type: experiment_design
relates_to:
  - docs/roadmaps/phase3_accuracy_push.md  # §W11 (3B close), §W12 (3C)
  - results/3/spy_phase1_random40_noiselow/3b_compare/comparison.csv
---

> Standalone research note. Proposes a scientifically valid way to test
> whether the best conditional model (ANP, epic 3B) outperforms the RBF
> interpolation baseline **specifically in sparse regions of the surface**
> (deep OTM/ITM wings, extreme maturities), even though ANP loses on the
> aggregate test MAE (Phase 3 bar not met — see roadmap §W11). Not yet
> executed; no Pod run performed for this note.

## 1) Motivation & hypothesis

Epic 3B closed with ANP failing to beat RBF on aggregate test MAE
(full-fold point head 0.0680 vs RBF 0.0662; calibrated gaussian +9–11 %).
But RBF is a **local kernel interpolator**: it weights nearby observed
quotes and degrades into pure *extrapolation* where the local
neighbourhood is empty. A conditional neural model carries a **global
smile/term-structure prior** learned across the full history, which
should degrade more gracefully when local context vanishes.

**Hypothesis (H1):** the ANP−RBF MAE gap *shrinks, and may reverse*, as
the local observed-context density falls — i.e. ANP is relatively (and
possibly absolutely) better than RBF at held-out points that have few or
no nearby observed quotes, concentrated in the deep wings and extreme
maturities.

If H1 holds, the Phase 3 conclusion reframes from "RBF wins" to
"**RBF wins where data is dense; ANP wins where data is sparse → a
density-routed hybrid**" — which is a concrete Phase 4 production angle
(not an RBF-prior scaffold, so it does not concede the Phase 3 research
question).

## 2) Why the comparison is valid (the ground-truth question)

The natural objection — *"we can't know the true IV in sparse regions"* —
is only true for genuine voids, which do not affect the comparison:

- **Benchmark construction.** `spy_phase1_random40_noiselow` keeps 40 %
  of real surface quotes as `observed=1` (context the model sees) and
  holds out 60 % as `observed=0` **targets**. Masking is implemented in
  `src/neural_iv_surface_inference/data/masking.py`; the builder is
  `src/data/04_build_benchmark_tasks.py`.
- **Labels survive masking.** Every held-out point retains its real
  `iv_true` (raw market IV) and `iv_clean` (denoised surface value) from
  the strict surface table. Synthetic noise is injected on *observed*
  points only; held-out targets are scored against `iv_clean` (the same
  ground truth the 2D.6 decision-layer runner uses).
- **Sparsity is in the context, not the label.** A deep-OTM held-out
  point can have an empty *observed* neighbourhood (sparse context) while
  still having a perfectly good *label* (a real quote that was masked
  out). This is exactly the regime that separates a global prior from a
  local interpolator.
- **Genuine voids** — strikes/maturities where no quote ever traded —
  carry no label and are simply absent from the dataset; no model is
  scored there. They cannot bias the comparison. For those, accuracy is
  undefined and only label-free criteria apply (Design C, §4).

**Label-quality caveat:** in the deep wings the *raw* quotes feeding
`iv_clean` are themselves sparse and noisier, so the label has higher
intrinsic variance there. This inflates **both** models' MAE equally
(same target), so it does not bias the **paired** ANP−RBF gap — but it
must be reported (per-stratum quote-count / label-density proxy) so the
gap is read against the right backdrop.

## 3) Feasibility evidence (preliminary, ANP-only)

Computed locally on the cached 3B.4 gaussian test predictions
(`artifacts/runs/3B4/gaussian/test_predictions.csv`), 40 sampled test
dates, held-out points only, scored vs `iv_clean`. Context density =
distance to nearest `observed` point in standardised `(log_moneyness, τ)`.

**ANP MAE by local observed-context density (quintiles):**

| Stratum | n | ANP MAE | mean nn-dist | mean \|log-m\| | mean τ |
|---|---|---|---|---|---|
| Q1 densest | 40,291 | 0.071 | 0.000 | 0.17 | 0.42 |
| Q2 | 40,290 | 0.055 | 0.001 | 0.17 | 0.39 |
| Q3 | 40,290 | 0.049 | 0.006 | 0.07 | 0.14 |
| Q4 | 40,290 | 0.069 | 0.016 | 0.15 | 0.38 |
| **Q5 sparsest** | 40,291 | **0.103** | 0.055 | 0.33 | 0.68 |

**ANP MAE by moneyness region:**

| Region | n | ANP MAE |
|---|---|---|
| **deep OTM put** (log-m < −0.15) | 57,883 | **0.126** |
| OTM put | 40,460 | 0.042 |
| ATM (\|log-m\| < 0.05) | 56,907 | 0.019 |
| OTM call | 23,503 | 0.053 |
| **deep OTM call** (log-m > 0.15) | 22,699 | 0.119 |

**Reading:** sparse / deep-wing strata are well-populated with labels and
are where ANP error concentrates (~7× ATM). This confirms the regions are
real and measurable. It does **not** test H1 — that is *relative* and
needs RBF predictions at the same points (RBF requires the benchmark
parquet, Pod-only). ANP being absolutely worse in the wings is fully
consistent with H1 if RBF is *even worse* there.

## 4) Proposed designs

### Design A — density-stratified paired head-to-head (existing `random40`)

Score ANP and RBF at the **same** held-out points; stratify by
context-density and by moneyness×maturity region; compare per-stratum
MAE. Primary, cheapest, reuses the existing benchmark.

- **Predictors:** ANP calibrated (3B.4 gaussian + 3B.6 calibrator) and/or
  ANP point head (3B.4 point_control); RBF interpolation baseline.
- **Stratifiers:** (i) nearest-observed distance in standardised
  `(k, τ)`; (ii) observed-count in a fixed standardised radius;
  (iii) economic regions (deep-wing / wing / ATM × short / mid / long τ).
- **Statistic:** paired per-point Δ = |err_RBF| − |err_ANP|; report mean Δ
  and the ANP/RBF MAE ratio per stratum, with **date-block bootstrap**
  CIs (resample whole dates — quotes within a date are correlated).
- **Decision rule for H1:** H1 supported in a stratum if the per-stratum
  ANP/RBF MAE ratio drops below 1 (or monotonically toward 1 as density
  falls) with a CI excluding the aggregate ratio.

### Design B — structured-sparsity benchmark (cleanest extrapolation test)

Build a benchmark variant that removes context from the wings but keeps
held-out wing labels, then score both predictors. Isolates pure
interpolation-vs-extrapolation. The masking strategies already exist in
`configs/benchmark_tasks.yaml` (`drop_wings`, `realistic`
liquidity-weighted); needs a new benchmark build on the Pod.

### Design C — label-free criteria (for genuine voids only)

Where no label exists, accuracy is undefined. Use diagnostic, non-scoring
criteria: no-arbitrage adherence (the W2 `diagnostics/no_arbitrage.py`
stack), surface smoothness, and ANP−RBF divergence as a *flag*. Never a
win/lose verdict — purely descriptive.

## 5) Threats to validity / controls

- **Label noise in the wings** — score vs `iv_clean`; report per-stratum
  quote-count; consider a liquidity floor.
- **Density metric leakage** — the stratifier uses only the `observed`
  mask, never the label, so no target leakage.
- **Within-date correlation** — block-bootstrap by date for all CIs.
- **Head choice** — report both ANP point (best accuracy) and ANP
  calibrated gaussian (production predictor); they differ materially.
- **Noise regime** — fix at `noiselow` to match the 3B evidence slice.
- **Multiple strata** — control the comparison count (e.g. Holm) before
  claiming a per-stratum win.

## 6) Compute requirements

| Phase | Locale | Hardware | Wall time | Disk |
|---|---|---|---|---|
| Design A: per-row RBF + ANP predictions over the test fold + stratified analysis | Pod (RBF needs the 1 GB benchmark parquet, Pod-only) | RTX A4500 (ANP forward) | ~15–30 min | per-row CSV ~0.5 GB on Pod (uncommitted); committed stratified-summary CSV <100 KB |
| Design B: build `drop_wings`/`realistic` benchmark variant + score both | Pod | A4500 | ~30–60 min (incl. benchmark build) | new benchmark parquet ~1 GB Pod-only |
| Analysis + write-up | local Mac | CPU | ~1–2 h | <100 KB |

- **Pod note:** Designs A and B require the Pod (benchmark parquet is not
  local). If the Pod is terminated before running, it must be re-provisioned
  and the processed benchmark re-materialised (the pipeline re-runs from
  scratch by design).

## 7) Status & next step

`proposed` — methodology only; no Pod run performed. To execute, promote
to an atomic story (3C/3D-adjacent), keep/again-provision the Pod, run
Design A first (decisive and cheapest), and fold the result into the
Phase 3 closing memo (3D). If H1 holds, record the density-routed-hybrid
implication as a Phase 4 production candidate.
