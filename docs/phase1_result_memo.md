# Phase 1 Result Memo

## Scope

Phase 1 remained SPY-only and EOD-only with a fixed task:
reconstruct dense IV surfaces from sparse/noisy observations.

## What Worked

- Real data pipeline is running end-to-end (`ingest -> inspect -> build`).
- Benchmark generation now creates reproducible sparse/noisy train/val/test datasets.
- A simple interpolation baseline and a neural baseline (MLP) both run through one evaluation path.
- Required output artifacts are generated in a consistent structure:
  figures, metrics tables, and run metadata.

## What Failed or Is Weak

- The current neural baseline is intentionally small and can underfit harder sparse regimes.
- Vendor-style reference is represented as a placeholder alignment row and still needs a true external surface integration.
- Region-level metrics are still coarse and should be expanded with richer bucket analysis.

## Hardest Part of the Task

The hardest part is robust reconstruction in high-sparsity regions with little observed information,
especially for tail moneyness and short/long maturity edges.

## Is Simple Interpolation Enough?

No. It provides a transparent and useful floor, but it cannot capture richer cross-region structure
once masks become sparse and noisy.

## Why Phase 2 Is Needed

Phase 2 is needed to improve model capacity, add stronger inductive structure,
and incorporate better reference targets and uncertainty-aware evaluation.
