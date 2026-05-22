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

> **Status (2026-05-22):** Epic 2A is `done` — all five stories (2A.1–2A.5)
> complete; the W1 measurement layer runs end-to-end (interface → metrics →
> abstention → committed artifacts). Real-data uncertainty-eval run on RunPod and
> MLP-predictor wiring remain a documented follow-up (not blocking the epic).
>
> Epic 2B (W2 — sensitivity & structure diagnostics) is `in_progress`, decomposed
> into stories 2B.1–2B.5: 2B.1 decomposition (done), 2B.2 masking-sensitivity
> harness (`done`), 2B.3 no-arbitrage diagnostics (`in_review`), 2B.4 risk-flag
> synthesis + region heatmaps (`in_review` — `no_arb_risk_flags` + (k, tau)
> region heatmaps implemented + tested), 2B.5 diagnostics runner + artifacts
> (`backlog`). Epics 2C–2D remain undecomposed (progressive decomposition).

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
