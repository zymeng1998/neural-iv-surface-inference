# Neural IV Surface Inference - Structural Roadmap

_Last updated: 2026-04-01_

## 1) End Goal (Decision-Grade Output)

Build an inference system that turns sparse/noisy/irregular option observations into a **decision-grade implied-volatility surface representation** for:
- pricing
- hedging
- risk monitoring
- relative value and scenario analysis

Core quality targets:
- stable and smooth surface behavior
- uncertainty awareness
- arbitrage-aware structure checks
- reproducible benchmark workflow

---

## 2) How Pieces Work Together (High-Level Flow)

```mermaid
flowchart TD
    A[Raw Market Quotes<br/>Sparse + Noisy + Irregular] --> B[Data Pipeline<br/>Ingest -> Clean -> Normalize]
    B --> C[Task Builder<br/>Masking + Noise Regimes + Splits]
    C --> D[Baselines<br/>Simple + Neural]
    D --> E[Evaluation Layer<br/>Metrics + Regional Diagnostics + Plots]
    E --> F{Decision Gate:<br/>Is surface quality good enough for<br/>pricing/risk usage?}
    F -- No --> G[Error Analysis<br/>Failure Regions + Regime Sensitivity]
    G --> H[Method Upgrade<br/>Latent / Uncertainty / Structure Constraints]
    H --> C
    F -- Yes --> I[Decision-Grade Surface Layer<br/>Production-facing research artifact]
```

---

## 3) Task Decomposition Roadmap (Goal -> Subtasks -> Methods)

```mermaid
mindmap
  root((End Goal: Decision-Grade IV Surface Inference))
    Phase 1: Benchmarkable Prototype
      S1 Data Foundation
        S1.1 Ingest real SPY EOD chains
        S1.2 Cleaning and normalization
        S1.3 Coordinate mapping (moneyness, maturity)
      S2 Task Construction
        S2.1 Sparse masking generator
        S2.2 Noise injection regimes
        S2.3 Time-based train/val/test splits
      S3 Baselines
        S3.1 Non-neural interpolation baseline
        S3.2 PyTorch masked MLP baseline
        S3.3 Vendor-style reference alignment (in progress)
      S4 Evaluation
        S4.1 MAE/RMSE on observed and unobserved points
        S4.2 Regional error diagnostics
        S4.3 Visual result package and memo
    Phase 2: Latent Inference Upgrade
      S5 Latent representation and conditional inference
      Methods
        Conditional autoencoder
        Latent bottleneck variants
    Phase 3: Uncertainty + Structure
      S6 Uncertainty calibration
      S7 Arbitrage-aware constraints and diagnostics
      Methods
        Heteroscedastic/quantile modeling
        Penalty-based structure objectives
    Phase 4: Energy-Based Formulation
      S8 Structured energy over observed x, latent z, surface y
      Methods
        Joint inference by energy minimization
```

---

## 4) Subtask-to-Method Matrix

| Subtask ID | Subtask | Method / Implementation Path | Current Status |
|---|---|---|---|
| S1.1 | Ingest real options data | scripted ingestion, Parquet snapshotting | Completed |
| S1.2 | Clean and normalize | quality filters, derived features, conservative/strict subsets | Completed |
| S1.3 | Surface coordinates | moneyness + maturity representation | Completed |
| S2.1 | Sparse observations | random + structured masking schemes | Completed |
| S2.2 | Noisy observations | low/medium/high noise regimes | Completed |
| S2.3 | Evaluation splits | time-based split logic | Completed |
| S3.1 | Simple baseline | interpolation-style baseline | Completed |
| S3.2 | Neural baseline | PyTorch masked MLP reconstruction | Completed |
| S3.3 | Vendor-style reference | external reference integration and alignment | In Progress |
| S4.1 | Core metrics | MAE/RMSE, observed vs unobserved error | Completed |
| S4.2 | Regional diagnostics | by maturity and moneyness bucket | Completed |
| S4.3 | Result artifact | plots + summary table + phase memo | Completed |
| S5 | Latent inference | conditional autoencoder / latent bottleneck | Planned |
| S6 | Uncertainty layer | predictive uncertainty/calibration | Planned |
| S7 | Structure constraints | arbitrage diagnostics + structure-aware losses | Planned |
| S8 | Full EBM stage | structured energy-based inference | Planned |

---

## 5) What Each Existing Work Item Already Fulfilled

| Existing Work Item | What It Fulfilled |
|---|---|
| `scripts/prepare_data.py` + data modules | S1.2, S1.3, S2.1, S2.2, S2.3 |
| `scripts/run_baseline.py` + model/training modules | S3.1, S3.2, S4.1, S4.2 |
| `configs/data.yaml`, `configs/baseline.yaml` | reproducible configuration across S1-S4 |
| `tests/test_data_pipeline.py`, `tests/test_smoke.py` | reproducibility and minimum reliability checks |
| `docs/phase1_result_memo.md` | S4.3 narrative and Phase 2 bridge justification |
| `docs/logs/progress_log.md` + phase action plan | execution traceability and task-level accountability |

---

## 6) Decision Rules (Readable, Practical)

1. **If baseline quality is unstable across sparsity/noise regimes** -> prioritize Phase 2 latent inference.
2. **If point estimates are good but reliability is unclear** -> prioritize Phase 3 uncertainty calibration.
3. **If fit improves but structure risk rises (arbitrage-like artifacts)** -> prioritize Phase 3 structure constraints.
4. **If modular upgrades still fail to unify objectives** -> escalate to Phase 4 structured energy formulation.

---

## 7) Next Milestone Priorities

1. Integrate true vendor-style reference stream (close S3.3 gap).
2. Promote MLP baseline to conditional autoencoder (Phase 2 entry point).
3. Add first uncertainty-aware output head and calibration diagnostics.
4. Add lightweight arbitrage diagnostics to evaluation reports.

