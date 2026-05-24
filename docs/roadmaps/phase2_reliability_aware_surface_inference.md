# Phase 2 — Reliability-Aware Implied Volatility Surface Inference

---
created_at: 2026-05-22T00:00:00-04:00
last_updated_at: 2026-05-22T00:00:00-04:00
---

## 1) Why Phase 1 Is Not Enough

Phase 1 delivered a reproducible benchmark stack and two baselines: a per-date
interpolation baseline (RBF/griddata) and a global masked MLP. The Phase 1 result
memo (`docs/phase1_result_memo.md`) documents the core limitation:

- The naive MLP is a **coordinate-regression model** of the form
  `(log_moneyness, tau) -> implied_vol`. It does not ingest the observed option
  chain as context, so it cannot adapt its prediction to the specific sparse
  surface present on a given date.
- On every split it **underperforms the interpolation floor** (test MAE 0.0967
  vs 0.0687), and its error is essentially flat between observed and unobserved
  points (≈0.0966 vs ≈0.0967). This indicates it learned a single smooth global
  fit rather than genuine, date-specific surface structure.
- It is weakest exactly where reconstruction matters most: short maturities and
  the deep-ITM / deep-OTM wings.

More fundamentally, Phase 1 produces **point estimates only**. It says nothing
about *whether a prediction should be trusted*. For pricing, hedging, and risk
use, an unqualified point estimate in a sparse, illiquid, or structurally
suspect region is worse than an explicit "I don't know."

## 2) Why Smooth Interpolation Is Not the Final Value

Per-date interpolation is a transparent and useful floor, but it is not the
product:

- It **cannot exploit cross-date or cross-region structure** — each date is fit
  independently from its own observed points.
- It degrades silently as masks become sparser and noisier; it will happily
  extrapolate into regions with no support and return a confident-looking number.
- It carries **no notion of reliability** — no uncertainty, no structural
  validity check, no abstention. It cannot tell a caller that a quoted region is
  untradeable or that the reconstructed surface violates no-arbitrage structure.

The value of this project is not a smoother surface. It is a **reliability-aware
pricing/risk inference system** that knows where it is trustworthy and refuses
to commit where it is not.

## 3) Target System and Outputs

The end-to-end direction:

```text
Raw sparse/noisy option chain
  -> surface inference
  -> uncertainty map
  -> no-arbitrage diagnostics
  -> confidence / tradability / risk flags
  -> abstention decision
```

For a query coordinate `(k, tau)` on a given date `t`, the target system should
eventually emit:

| Output | Meaning |
|---|---|
| `sigma_hat` | Predicted implied volatility |
| `confidence_score` | Calibrated confidence / reliability estimate |
| `uncertainty_band` | Predictive interval or model-disagreement band |
| `tradability_score` | Whether the quote/surface region is suitable for pricing/trading use |
| `no_arb_risk_flags` | Structural / no-arbitrage diagnostic flags |
| `abstain_flag` | Whether the model should refuse to predict in an unreliable region |

This is framed as a **reliability-aware pricing/risk inference system**, not a
smooth surface reconstruction project.

## 4) Workstreams

The phase is decomposed into five workstreams, sequenced so that measurement
precedes modeling: we build the ability to *evaluate* reliability before we
build models that *claim* it.

### W1 — Uncertainty Evaluation (measurement first)

Build a model-agnostic uncertainty-evaluation layer that all current and future
predictors (interpolation, MLP, conditional neural models) can share.

- Prediction-interval coverage (empirical vs nominal)
- Interval width
- Error-vs-uncertainty correlation
- Confidence-bucket metrics (error stratified by confidence)
- Abstention curves (error vs coverage as the abstention threshold varies)
- High-confidence MAE (error on the retained, non-abstained subset)
- A thin model-agnostic predictor interface so interpolation, MLP, and future
  neural models plug into the same evaluator.

Rationale: every downstream reliability claim (confidence, tradability,
abstention) is only meaningful if we can score it. This must exist first.

### W2 — Masking Sensitivity and Structure Diagnostics

Probe reliability through perturbation and structural validity, independent of
any single model.

- Masking-sensitivity experiments: repeated masking / perturbation runs to
  measure prediction stability under different observed subsets.
- Uncertainty heatmaps over `(k, tau)` regions.
- No-arbitrage diagnostics:
  - monotonicity violation checks
  - convexity violation checks
  - calendar (term-structure) sanity checks
  - violation counts and severity scores
- `no_arb_risk_flags` derived from the diagnostics above.

Rationale: diagnostics come **before** hard no-arbitrage constraints. We first
need to observe and quantify violations before deciding whether to penalize or
constrain them inside a model objective.

### W3 — Conditional Neural Surface Model

Introduce the first genuinely useful neural architecture: one that is
**conditional on the observed sparse quotes**, not just on coordinates.

```text
O_t  -> z_t                      (encode observed chain into a latent state)
(k, tau, z_t) -> sigma_hat_t(k, tau)   (decode surface at any coordinate)
```

Candidate architectures:

- **Set Encoder + Coordinate Decoder** (recommended first baseline)
- Conditional / Denoising Autoencoder
- Neural Process / Attentive Neural Process-style model

Start with the simplest reliable conditional model — **Set Encoder + Coordinate
Decoder**. A permutation-invariant set encoder is the natural fit for sparse,
irregular, variable-cardinality observed quotes, and a coordinate decoder lets
the model be queried at arbitrary `(k, tau)`. More expressive variants (CAE,
Neural Process) are deferred unless this baseline proves insufficient.

### W4 — Uncertainty-Aware Neural Inference

Equip the conditional model with reliability signals.

- Ensemble disagreement (multi-seed / multi-model)
- Masking sensitivity as an uncertainty proxy
- Heteroscedastic head or quantile regression for predictive intervals
- Calibrated `confidence_score`

### W5 — Abstention and Tradability Decision Layer

Convert reliability signals into actionable decisions.

- Abstention thresholds and `abstain_flag`
- `tradability_score`
- `no_arb_risk_flags` consumption from W2
- Dashboard / artifact tables summarizing reliability across regions and regimes

## 5) Mapping to Implementation Phases

| Workstream | Implementation phase |
|---|---|
| W1 Uncertainty evaluation | Phase 2A |
| W2 Sensitivity & structure diagnostics | Phase 2B |
| W3 Conditional neural surface model | Phase 2C |
| W4 + W5 Uncertainty-aware inference & decision layer | Phase 2D |

Sequencing principle: **2A before 2C** (we measure reliability before we model
it), and **2B diagnostics before any hard structural constraints** (we quantify
violations before penalizing them).

> **Status (2026-05-24):** Phase 2 is **~75% complete** — three of four epics
> are `done`. Epic 2D (W4 + W5) is the remaining work and is still `backlog`,
> undecomposed.
>
> Epic 2A (W1 — uncertainty evaluation) is `done` — all five stories
> (2A.1–2A.5) complete; the W1 measurement layer runs end-to-end
> (interface → metrics → abstention → committed artifacts).
>
> Epic 2B (W2 — sensitivity & structure diagnostics) is `done` — all five
> stories (2B.1–2B.5) complete: masking-sensitivity harness, no-arbitrage
> diagnostics, risk-flag synthesis + (k, tau) region heatmaps, and the
> end-to-end diagnostics runner with report tables + committed artifacts.
>
> Epic 2C (W3 — conditional neural surface model) is `done` — all eight
> stories (2C.1–2C.8) complete:
> - **Local Phase A** (2C.2–2C.6): dataset + collation, set encoder + coordinate
>   decoder (85,057 params), training loop + config + synthetic smoke,
>   predictor adapter wired into the W1/W2 runners with artifact-shape parity,
>   and the Alpha Vantage `HISTORICAL_OPTIONS` ingest (paid Standard, ADR 0003).
> - **Remote Phase B** (2C.7 + 2C.8 + 2C.4-R + 2C.5-R, executed 2026-05-23
>   as a single autonomous chain): full AV pull (26.06 M rows, 4,623 trading
>   days, 2008-01-02 → 2026-05-22), Dubach snapshot deleted, pipeline 02→04
>   rebuilt, Phase 1 baselines + W1/W2 re-run on the AV data, full conditional
>   training and eval-parity run, Pod self-terminated cleanly after ~11 h 10 m.
> - Results notebook `notebooks/04_phase2c_results.ipynb` (50 cells) ships
>   the executive summary, model I/O contract, live-SPY inference workflow,
>   3D spinning surface, headline MAE / per-region / obs-unobs / risk-coverage
>   tables, and training dynamics — all on real AV data.
>
> Epic 2D (W4 — uncertainty-aware inference + W5 — abstention & tradability
> decision layer) is `backlog` and **undecomposed** (progressive decomposition).
> W3 deliberately ships point predictions only; uncertainty heads, calibration,
> and the `surface_action ∈ {trade, hedge_only, abstain}` decision layer that
> fuses W4 predictive intervals with W2 structure flags are all 2D work.

## 6) Acceptance Criteria

Phase 2 is complete when the following hold:

1. **Model-agnostic uncertainty evaluator exists and is tested.** Interpolation,
   MLP, and at least one conditional neural model all run through the same
   uncertainty-evaluation interface and produce coverage, interval-width,
   error-vs-uncertainty, abstention-curve, and high-confidence-MAE metrics.
2. **No-arbitrage diagnostics exist and are tested.** Monotonicity, convexity,
   and calendar checks produce violation counts, severity scores, and
   `no_arb_risk_flags` on benchmark surfaces.
3. **Masking-sensitivity harness exists.** Repeated-masking / perturbation runs
   produce stability metrics and uncertainty heatmaps for a given model.
4. **A conditional neural surface model exists** that ingests the observed chain
   (`O_t -> z_t`, then `(k, tau, z_t) -> sigma_hat`) and is evaluated on the same
   benchmarks as the Phase 1 baselines.
5. **Decision-grade outputs are produced.** For a query region the system emits
   `sigma_hat`, `confidence_score`, `uncertainty_band`, `tradability_score`,
   `no_arb_risk_flags`, and `abstain_flag`.
6. **Calibration is demonstrated.** Confidence scores are calibrated (empirical
   coverage tracks nominal within a documented tolerance), and abstention
   measurably improves high-confidence error versus the no-abstention baseline.
7. **Reproducibility and evidence discipline.** Each reported result is backed by
   committed result artifacts (CSV/tables/figures) alongside checkpoints, and is
   logged in `docs/experiments/experiment_journal.md`.

## 7) Recommended Decision Records (to be written as decisions are made)

The following ADRs should eventually be created in `docs/decisions/` (not in this
document):

- Why Phase 2 is framed as reliability-aware surface inference rather than
  smoother reconstruction.
- Why uncertainty evaluation (2A) precedes the conditional neural model (2C).
- Why Set Encoder + Coordinate Decoder is the first conditional neural baseline.
- Why no-arbitrage diagnostics (2B) precede hard no-arbitrage constraints.
