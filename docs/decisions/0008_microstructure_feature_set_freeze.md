# ADR 0008: Microstructure Feature Set Freeze (Phase 3C — `micro_v1`)

## Status

**Implemented (2026-06-14).** Accepted 2026-06-02, locked by Phase 3C
decomposition story [`3C.1`](../tasks/specs/3C.1_decompose_phase_3c.md);
shipped behind the `feature_set ∈ {minimal, micro_v1}` flag in 3C.2 and
trained on OTM in 3C.3. Moved to **Implemented** on the 3C.8 close — see
the Outcome block below. **Result: negative — `micro_v1` worsened test
MAE in all three heads vs the minimal baseline; the feature set is
frozen as specified but is *not* adopted as a production feature set.**

## Date

2026-06-02

## Context

Phase 3X (data correction) closed `done` on 2026-06-02. On the clean
single-valued OTM substrate the original 3B verdict — RBF beats the
end-to-end DeepSets+ANP conditional model — was preserved in direction
and **widened in magnitude**:

| Comparison | Dirty (full fold) | OTM (test) |
|---|---|---|
| RBF | 0.0662 | **0.00613** |
| Best ANP head (point) | 0.0680 | 0.00987 |
| ANP − RBF (relative) | +2.7 % | **+61 %** |
| Calibrated ANP production | +9–11 % | **+90 %** |

(Evidence: [`docs/research/duplicate_coordinate_methodology_progression.md`](../research/duplicate_coordinate_methodology_progression.md);
bundle `results/3/spy_phase1_random40_noiselow_otm/3x_compare/`.)

The §W11 (3B) and §W11.5 (3X) closing addenda both concluded that
**pure decoder-architecture iteration has plateaued** and that 3C should
prioritise **feature / inductive-bias expansion** — give the encoder
real per-quote signal about quote reliability so it can downweight
unreliable quotes (wide spread, low volume, deep wings) when forming
`z_t`, rather than try to recover that information implicitly from
`(log_moneyness, τ, iv_input)` alone.

The current per-quote tuple is the smallest interpretable surface
representation:

```python
# src/neural_iv_surface_inference/data/conditional_loaders.py:27
_CONTEXT_FEATURES = ("log_moneyness", "tau", "implied_volatility")
_QUERY_FEATURES   = ("log_moneyness", "tau")
```

— three dims in, two dims out. Phase 1's masked MLP and Phase 2/3's
conditional models all use this minimal `O_t`. The Phase 2 cleaning
pipeline already lays down a rich superset of microstructure fields on
the strict surface table, but none of them flow into the conditional
loader.

## Decision

**Lock the Phase 3C `micro_v1` feature set as the following six
AV-native per-quote microstructure features, appended (in this order)
to the existing `_CONTEXT_FEATURES`:**

| # | Feature | Definition | Source field |
|---|---|---|---|
| 1 | `bid` | raw bid price, USD | AV `HISTORICAL_OPTIONS.bid` |
| 2 | `ask` | raw ask price, USD | AV `HISTORICAL_OPTIONS.ask` |
| 3 | `bid_ask_spread_rel` | `(ask − bid) / max(mid, ε)`, ε = 1e-4 | derived in `03_build_spy_surface_table.py` |
| 4 | `volume` | reported daily volume | AV `HISTORICAL_OPTIONS.volume` |
| 5 | `open_interest` | reported open interest | AV `HISTORICAL_OPTIONS.open_interest` |
| 6 | `put_call_indicator` | +1 if `type == "call"`, −1 if `put` | derived from `03_build_spy_surface_table.py:type` |

So `micro_v1` `_CONTEXT_FEATURES` =
`("log_moneyness", "tau", "implied_volatility",
   "bid", "ask", "bid_ask_spread_rel",
   "volume", "open_interest", "put_call_indicator")`

— **9 context dims** total (was 3). `_QUERY_FEATURES` is **unchanged**
at `(log_moneyness, tau)`: the model is queried at coordinates, not
quotes, and a query has no microstructure of its own.

### Why these six (and not more)

- All six are **AV-native** — already pulled by `01_ingest_alpha_vantage_spy.py`
  and persisted on the strict surface table via
  `03_build_spy_surface_table.py`. No new ingest path. The 3C.1
  non-goals (and Phase 3's "if a feature needs a new ingest path, raise
  a separate Phase 2-style data-source ADR first" rule) are respected.
- Together they carry **quote reliability** signal (`bid_ask_spread_rel`,
  `volume`, `open_interest`) + **price-level** signal (`bid`, `ask`) +
  **leg-type** signal (`put_call_indicator`). The first three test the
  encoder-downweighting hypothesis; the last three give a complete
  raw-quote tuple in case price-level interacts with downweighting.
- `mid` is **excluded** — collinear with `iv_input` via the
  Black-Scholes inversion that produced `implied_volatility` from the
  mid price during Phase 1 cleaning. Adding it would add no information
  and would risk a redundant degree of freedom.
- `spot` is **excluded** — already implicit in `log_moneyness =
  ln(strike / spot)` and `tau`. It is a per-date constant, not a
  per-quote field, so it belongs in `condition_t`, not `O_t`. A
  spot-level conditioner is deferred to a separate ADR if 3C needs it.

### Features explicitly out of scope for `micro_v1`

These were on the original 3C.1 stub list (2026-05-27, pre-3X) but are
**not** included in this freeze:

- `vix` — requires a new external data source (CBOE VIX history).
  Forbidden by 3C.1 non-goals; would require a Phase 2-style data-
  source ADR first.
- `recent_realized_vol_5d` — derivable from `spot` time series in the
  AV pipeline in principle, but requires either a new ingest path
  (intraday) or a non-trivial transformation on top of the daily
  `spot` column, and adds a model-side hyperparameter (window). Deferred.
- `days_to_next_earnings_or_dividend` — requires an earnings /
  dividends calendar (not in AV `HISTORICAL_OPTIONS`). Forbidden.
- `spot_level` — see above (implicit in `log_moneyness`).

If any of these prove necessary after `micro_v1`'s decision-layer eval
(3C.6), they will be raised as a separate ADR + Phase 2-style data-
source story before any Phase 3C extension.

### Feature-set flag

`micro_v1` ships behind a config flag `feature_set` on the conditional
dataset / loader / `SetEncoder`, with values:

- `minimal` — the legacy three-dim tuple. **Default.** Preserves
  bit-identical behaviour for every committed Phase 2 / Phase 3
  checkpoint (the `_otm` benchmarks, 3B.* and 3X.* runs all stay
  reproducible byte-for-byte).
- `micro_v1` — the nine-dim tuple defined above. Opt-in for 3C training
  runs.

The flag is persisted into the checkpoint config so that
`ConditionalSurfacePredictor.from_checkpoint` can reconstruct the
feature set without operator intervention.

### Normalization

Each new feature is z-scored against **train-split statistics only**,
computed on `spy_phase1_random40_noiselow_otm` (3C's substrate). Stats
are persisted in the checkpoint manifest alongside the existing
`(log_moneyness, tau, iv_input)` stats. Rationale: bid / ask / volume /
open_interest live on very different scales than IV (USD vs vol points
vs counts), and the existing `SetEncoder` MLP expects roughly
unit-variance inputs.

`put_call_indicator` is **not** normalized (already in `{−1, +1}`).

### Missing-value policy

The OTM-restricted strict surface (3X.2 builder) guarantees every
retained row has non-null `bid`, `ask`, `volume`, `open_interest`,
`type` (Phase 1 cleaning rules at
[`docs/data_assumptions_and_cleaning.md`](../data_assumptions_and_cleaning.md)
drop rows where any of these are missing or invalid). `micro_v1`
inherits this; **no in-loader imputation is required**.

If a future variant of the benchmark relaxes the cleaning rules, this
ADR is voided and a successor ADR must define imputation.

## Alternatives considered (and rejected)

- **Quote-quality minimal `(spread, volume, open_interest)` only.**
  Tests the downweighting hypothesis with minimum confound (~3 new
  dims). Rejected because the operator-confirmed scope is "full AV-
  native set" so the head ablation in 3C.3 can attribute any 3C gain
  cleanly against the OTM baseline (3X.9), without leaving "but did
  raw price level matter?" as an open follow-up question.
- **Quote-quality + `put_call_indicator` only (4 dims).**
  Same rationale as above; rejected for the same reason.
- **Include `mid`.** Rejected — collinear with `iv_input` (see above).
- **Include greeks (`delta`, `gamma`, `theta`, `vega`, `rho`).** AV
  returns these but they are themselves model outputs of the AV /
  vendor's pricing model. Including them as inputs risks circular
  conditioning (the model would partly learn to invert the vendor
  pricer). Rejected for 3C; reserved for a Phase 4-style hybrid if
  ever needed.

## Consequences

### Positive

- Encoder gets per-quote reliability signal — directly attacks the
  §W11.5 finding that decoder-architecture iteration plateaued.
- Existing Phase 2 / Phase 3 / 3X checkpoints remain reproducible
  byte-for-byte (the flag defaults to `minimal`).
- Future ablations (drop one feature at a time) are mechanical once
  the loader supports the flag.

### Negative / costs

- One full retrain on OTM is required to measure the effect (3C.3 —
  ANP three-head sweep with `micro_v1`).
- One additional Pod-GPU rental window (3C.3 + 3C.4 ensemble, ~5 h
  total per [`3C.3`](../tasks/specs/3C.3_remote_anp_micro_three_head.md)
  / [`3C.4`](../tasks/specs/3C.4_remote_anp_micro_ensemble.md) compute
  estimates).
- `SetEncoder` input dimension changes; checkpoints with `feature_set
  = micro_v1` are not weight-compatible with `minimal` checkpoints
  (input projection has different `in_features`). This is the *point*
  of the ablation; documented here so it does not surprise a future
  reader of the predictor adapter.

### Neutral

- No data pipeline change. `01_ingest_*` → `02_normalize_*` →
  `03_build_*` → `04_build_benchmark_*` → `05_build_otm_*` all
  already produce these fields on the strict OTM surface. 3C.2 only
  touches the conditional **loader** + `SetEncoder`, not the data
  pipeline.

## Acceptance bar inherited from Phase 3

Phase 3's overall acceptance bar (ANP ≥ 5 % below RBF on test MAE on
`spy_phase1_random40_noiselow_otm`, coverage within ±2 pp of 0.90,
hi-conf MAE strictly below no-abstention MAE, forbidden-flag violations
not meaningfully increased vs the 3X.12 baseline) applies to the
`micro_v1` retrain unchanged. The 3C closing addendum (3C.8) records
the verdict and, if the bar is still NOT met, frames the residual gap
for 3D / Phase 4.

## Open questions deferred to 3C.8 (or later)

- Whether the gain is feature-set-additive or head-dependent (3C.3
  sweeps all three heads so this is observable).
- Whether `volume` and `open_interest` should be log-transformed
  before z-scoring. 3C.2 ships z-score-only; if 3C.3 shows a
  pathological tail on these features it is an opt-in re-fit, not a
  spec change.
- Whether a follow-up `micro_v2` should add `vix` / `realized_vol`
  via a new Phase 2-style data-source ADR. Deferred to 3C.8's
  recommendation block.

---

## Outcome (2026-06-14 — recorded by 3C.8 on close)

**Negative result. `micro_v1` does not help; the Phase 3 bar remains
NOT met.** 3C.3 trained the end-to-end DeepSets+ANP model with
`feature_set: micro_v1` across all three heads on
`spy_phase1_random40_noiselow_otm`, a faithful head-by-head restatement
of 3X.9 whose only delta is the per-quote feature tuple. Evidence:
`artifacts/runs/3C3/{gaussian,quantile,point}/manifest.json` +
`training_curve.csv` (committed) and the 3C.3 `experiment_journal.md`
entry.

| Head | `micro_v1` test MAE (3C.3) | `minimal` test MAE (3X.9) | Δ (micro − minimal) | RBF-on-OTM floor (3X.6) |
|---|---:|---:|---:|---:|
| gaussian | 0.01634 | 0.01440 | **+0.00194** | 0.00613 |
| quantile | 0.01362 | 0.01175 | **+0.00187** | 0.00613 |
| point | 0.01439 | 0.00987 | **+0.00452** | 0.00613 |

- **ANP-vs-ANP-baseline (vs 3X.9):** `micro_v1` is **worse** on test MAE
  in every head. The regression is head-uniform.
- **ANP-vs-RBF (vs the 3X.6 floor 0.00613):** every `micro_v1` head sits
  well above the floor — `micro_v1` does not narrow the post-3X gap to
  RBF (best `minimal` head 0.00987 was already +61 %; `micro_v1` widens
  it). The Phase 3 ≥ 5 %-below-RBF bar is **NOT met**.
- **No calibrated / decision-layer numbers.** Because the base-accuracy
  result is already negative, the downstream chain that would have
  produced calibrated-production + decision-layer numbers (3C.4 ensemble
  / 3C.5 calibrator / 3C.6 Q2 decision-layer / 3C.7 comparison) was
  **cancelled** as no longer informative; the verdict stands on 3C.3
  alone. (Calibration/abstention reshape the reliability layer but
  cannot pull base MAE below the model's own point predictions.)

### Open questions (from the "deferred to 3C.8" block) — answered

- **Feature-set-additive vs head-dependent?** Head-dependent is ruled
  out: all three heads regressed by a similar margin, consistent with an
  input-conditioning effect rather than a head interaction.
- **Log-transform `volume` / `open_interest` before z-scoring?** 3C.3
  flagged a pathological train-split tail (ask 691σ on a single ~$5000
  quote, volume 922σ) — exactly the condition this ADR reserved for an
  opt-in log-transform re-fit. It is **not pursued under 3C**: pure
  feature expansion already missed the bar, so the higher-value forward
  path is the Phase 4 RBF-prior hybrid, not a `micro_v1` rescue. The
  log-transform remains available as an opt-in re-fit if a future epic
  revisits microstructure features.
- **A `micro_v2` with `vix` / `realized_vol` via a new data-source ADR?**
  Not recommended now. Deferred indefinitely in favour of the Phase 4
  hybrid direction (roadmap §W13).

**No-overclaim guardrail:** scoped to the matched `random40_noiselow_otm`
substrate only; no robustness claim across the other 10 OTM variants.

The `micro_v1` feature list and `feature_set` flag remain frozen and
available (default `minimal` keeps every legacy checkpoint reproducible
byte-for-byte), but `micro_v1` is **not** adopted as a production feature
set. Forward path recorded in the §W12 closing addendum and §W13.
