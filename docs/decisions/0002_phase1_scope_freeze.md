# ADR 0002: Phase 1 Scope Freeze

## Status

Accepted

## Date

2026-03-31

## Context

Phase 1 needs a tightly scoped task definition to avoid premature expansion into multiple underlyings, intraday data, or advanced model architectures before a first working end-to-end result exists.

## Decision

Phase 1 scope is frozen as follows:

### Underlying

- **SPY only**
- Do not add QQQ, AAPL, or any other underlying until the SPY pipeline produces a complete result package

### Time granularity

- **Daily / EOD only**
- Do not pursue intraday, minute-bar, or real-time OPRA data in Phase 1

### Task definition

> Given a full real EOD option chain for SPY, construct sparse/noisy/irregular observations and train a model to reconstruct a dense implied-volatility surface.

### Surface representation

- x-axis: moneyness or log-moneyness
- y-axis: time to maturity
- value: implied volatility

### Baselines required

- One simple non-neural baseline (interpolation or smoothing)
- One PyTorch neural baseline (MLP or conditional autoencoder)
- One vendor-style or full-chain reference for comparison

### Deliverables before scope can expand

- At least one working data pipeline (raw → cleaned → processed)
- At least one set of evaluation metrics (MAE / RMSE, observed vs unobserved)
- At least one set of plots (reference surface, sparse input, reconstruction, error)
- One summary table comparing baselines
- One short result memo

## Rationale

- SPY is the most liquid equity option, minimizing data quality issues
- EOD simplifies the pipeline and avoids real-time infrastructure
- A narrow scope forces a complete end-to-end system before adding complexity
- The result package provides a concrete foundation for Phase 2 decisions

## Consequences

Positive:
- Clear stopping condition for Phase 1
- Prevents scope creep into multi-asset, intraday, or advanced architectures
- Every implementation decision can be evaluated against a single concrete task

Trade-offs:
- Some interesting directions (intraday dynamics, cross-asset structure) are deliberately deferred
- The SPY-only constraint may feel limiting, but breadth comes in Phase 2
