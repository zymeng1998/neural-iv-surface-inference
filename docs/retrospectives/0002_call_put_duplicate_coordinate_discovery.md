# Retrospective 0002: Call-Put Duplicate-Coordinate Discovery in the IV Surface Pipeline

---
created_at: 2026-05-29T23:30:00-04:00
last_updated_at: 2026-05-29T23:30:00-04:00
event_at: 2026-05-29T18:30:00-04:00
---

## Related files

- `scripts/audit_duplicate_coordinates.py`
- `tests/test_audit_duplicate_coordinates.py`
- `artifacts/audits/duplicate_coordinates/duplicate_summary.csv`
- `artifacts/audits/duplicate_coordinates/duplicate_iv_dispersion.csv`
- `artifacts/audits/duplicate_coordinates/observed_hidden_leakage.csv`
- `artifacts/audits/duplicate_coordinates/sparse_density_sensitivity.csv`
- `docs/research/duplicate_coordinate_audit.md`
- `docs/research/duplicate_coordinate_audit_design.md`
- `docs/decisions/0006_duplicate_coordinate_data_correction.md`
- `src/data/03_build_spy_surface_table.py`
- `src/neural_iv_surface_inference/data/conditional_loaders.py`
- `src/neural_iv_surface_inference/models/interpolation.py`
- `src/neural_iv_surface_inference/data/masking.py`

---

## 1. What happened

While designing the sparse-region ANP-vs-RBF research experiment
([`docs/research/sparse_region_anp_vs_rbf_design.md`](../research/sparse_region_anp_vs_rbf_design.md)),
we paused to audit a quiet structural concern: the model and the RBF
baseline both treat IV as a single-valued function of `(log_moneyness,
tau)`, but the raw AV `HISTORICAL_OPTIONS` chain returns both the call
and the put leg for every `(date, expiration, strike)`. The audit
script ([`scripts/audit_duplicate_coordinates.py`](../../scripts/audit_duplicate_coordinates.py),
shipped 2026-05-29) was run on the Pod against the 22.5 M-row strict
surface table and the `spy_phase1_random40_noiselow` benchmark.

It found that **93.61 % of strict-table rows live inside a duplicate
`(date, expiration, strike)` group**, and the **same 93.61 %** also
collide at the model coordinate `(log_m, tau)` at every rounding
tolerance from 8 to 12 decimal places — confirming the call-put pair
shares the exact `(K, T)` mapping at the float level. Of 10,530,702
duplicate groups, **10,530,258 (100.00 %) are call-put pairs**; only
444 are same-type duplicates.

Inside a duplicate group, the **median IV range (max − min) is 0.049**,
mean 0.103, p99 0.595, max 2.96 — i.e. the two reported IVs on the
same `(K, T)` routinely disagree by 5–60 vol points. In the benchmark,
**37.46 % of held-out rows** carry an exact-coordinate observed twin on
the same date; all such rows have `nearest_observed_distance = 0` under
the naive metric, which the proposed sparse-region experiment was
about to interpret as "dense local neighbourhood" and use to define
the densest Q1 stratum.

## 2. Why this was a mistake / why this mattered

The pipeline has carried a silent **structural defect** since Phase 1.
The IV-surface modelling task as defined in
[ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md) — "single-valued IV
surface, beat RBF on test MAE" — was being fitted on data that does
not satisfy the single-valued assumption: at the same model coordinate
the loss saw two distinct labels (the call IV and the put IV reported
by AV), and the RBF kernel was forced to average over them too.

Concrete consequences:

- **The Phase 2 / Phase 3 MAE-vs-RBF comparisons are biased in RBF's
  favour on dense-coordinate regions.** When a held-out put leg has an
  observed call twin (37 % of held-out rows), RBF gets a perfect
  zero-distance neighbour to extrapolate from; the conditional model
  is forced to reconcile two labels through one prediction. RBF's
  apparent lead on the aggregate is inflated by this leakage.
- **The sparse-region experiment design was about to ship with a
  fundamentally wrong density metric.** Its Q1 (densest) bucket is
  defined by smallest `nearest_observed_distance`, and ~100 % of
  zero-distance rows come from call-put leakage, not real density. The
  experiment would have reported a clean "ANP wins / loses in the
  dense regime" verdict that would not have replicated under any
  proper definition of density.
- **Five months of Phase 2 / Phase 3 results, including the Phase 2
  closing acceptance numbers, the 2D.9 decision-layer evaluation, the
  3A coordinate-encoding ablation, and the 3B ANP closing verdict (Δ
  +2.7 % vs RBF), all need an asterisk: they are correct under their
  own (biased) experimental setup, but the cross-model accuracy
  comparison cannot be cleanly defended as written.** This is exactly
  the kind of issue that, surfaced by a reviewer (or an interviewer)
  on an external presentation, would cost the credibility of the
  whole stack.
- **The "label noise" the calibrator absorbs is not noise.** The
  median in-group IV range (0.049) is an order of magnitude larger
  than the synthetic `noise_regime: low` σ (0.005). The Gaussian head
  has been learning to predict the *mean* of the call IV and the put
  IV at every coordinate, and the temperature-scaling calibrator
  ([`configs/calibration_3B6_anp.yaml`](../../configs/calibration_3B6_anp.yaml))
  has been silently widening σ to swallow call-put-quote dispersion as
  if it were a model uncertainty. The reliability numbers may survive
  this re-framing (because both train and test see the same
  dispersion), but that has to be re-verified.

## 3. Root cause

Three independent decisions, each defensible in isolation, combined to
let the defect through:

1. **Phase 1 cleaning rules in
   [`src/data/03_build_spy_surface_table.py`](../../src/data/03_build_spy_surface_table.py)
   focused on per-row finiteness, not contract-level uniqueness.** The
   rules drop nulls, crossed quotes, out-of-range IV / tau, and apply
   strict-subset windows on IV / tau / log-moneyness — sensible, but
   silent on duplicates. There is no OTM convention, no
   `drop_duplicates(["date","expiration","strike"])`, and no
   aggregation step. The cleaning author saw "duplicate row" as a
   data-quality red flag (e.g. ingest re-running a date), not as a
   *structural* outcome of every contract having two legs.
2. **The conditional model interface in
   [`src/neural_iv_surface_inference/data/conditional_loaders.py:27`](../../src/neural_iv_surface_inference/data/conditional_loaders.py)
   silently dropped `type`.** `_CONTEXT_FEATURES = ("log_moneyness",
   "tau", "implied_volatility")`, `_QUERY_FEATURES = ("log_moneyness",
   "tau")`. The interface authors were thinking about the surface as a
   2D function and folded `type` away. Nothing in the loader, training
   loop, or evaluator asserts that the input table is single-valued at
   each `(date, log_m, tau)`. A defensive `assert
   not df.duplicated(["date","log_moneyness","tau"]).any()` at dataset
   construction would have caught this on the first Phase 2C run.
3. **Masking in
   [`src/neural_iv_surface_inference/data/masking.py`](../../src/neural_iv_surface_inference/data/masking.py)
   operates per-row, not per-coordinate.** Sampling `observed ∈ {True,
   False}` independently for each row in
   [`apply_mask`](../../src/neural_iv_surface_inference/data/masking.py)
   makes structural sense for the surface-as-quote-table reading, but
   under the surface-as-function reading it produces guaranteed
   train/test leakage whenever the call leg and the put leg fall on
   opposite sides of the mask. The masking author was working under the
   same implicit "each row is an independent surface point" model the
   loaders assumed.

The deeper cause is one of *uncodified assumption discipline*: the
"surface is a single-valued function of `(log_m, tau)`" assumption
appears in every roadmap, decision record and result memo from Phase 1
onward, but it never landed as a tested invariant in code. Several
documents (`data_lineage.md`, `data_assumptions_and_cleaning.md`,
[ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md)) describe
the surface as a function; none of them check that the dataset
satisfies it.

## 4. What we learned

1. **Assumptions that govern the modelling target deserve a dataset
   invariant, not just prose.** "Single-valued function of `(log_m,
   tau)`" should have been a one-line assertion at the head of the
   strict-subset builder, and re-checked at the head of every
   benchmark builder. We will add this as part of 3X.2 and treat it as
   a CI invariant going forward.
2. **`type` is the canonical lever that distinguishes a quote table
   from a surface.** Every project that models a vol surface has to
   pick one of: (a) OTM-restrict, (b) aggregate, or (c) keep `type`
   as a feature. We had implicitly picked (b) without doing the
   aggregation and without documenting the choice. The OTM convention
   (Correction A in
   [ADR 0006](../decisions/0006_duplicate_coordinate_data_correction.md))
   is the industry standard and what every vendor and the SVI / SSVI
   literature already assume.
3. **Leakage detection in sparse-region experiments is its own
   discipline.** A nearest-observed-distance metric is only meaningful
   if "observed neighbours" mean what they purport to mean. The audit
   script's `exclude_self_dup` mode is a small piece of tooling we
   should keep in the toolbox for any future density-stratified
   evaluation.
4. **A static-analysis audit is a high-leverage first step before any
   data-stratification experiment.** This issue would not have shown
   up in pytest, in `pre-commit`, or in any model-evaluation routine.
   It only shows up when you read the data with the experiment's
   density metric in mind. Future experiment designs should include a
   "what assumption am I about to bake in?" check, executed against
   the data before the experiment is greenlit.

## 5. Improvement plan

Process-level (committed in this same doc-update pass):

- **ADR 0006** locks the data-correction decision and the OTM
  convention, with full numerical evidence.
- **`docs/data_assumptions_and_cleaning.md`** now flags the
  invalidated single-valued assumption and the planned correction.
- **`docs/data/data_lineage.md`** carries the duplicate-coordinate
  status as an explicit open lineage gap (§10).
- **`docs/research/sparse_region_anp_vs_rbf_design.md`** is marked
  `status: blocked` until the OTM benchmark exists.
- **`docs/PHASE3_INDEX.md`**, **`docs/tasks/BOARD.md`**, and
  **`docs/roadmaps/phase3_accuracy_push.md`** all register a new
  Phase 3 epic **3X — Data correction** between 3B and 3C, with the
  decomposition story 3X.1 in `backlog`.

Code-level (deferred to 3X.1 decomposition, not done here):

- A `validate_single_valued()` invariant in the strict-subset builder
  and the benchmark builder.
- A vectorised v2 of `scripts/audit_duplicate_coordinates.py` (the
  current Python `for key, idx in grouped.indices.items()` loop spent
  2h 11m on the strict file; a pandas `groupby.agg` formulation should
  cut this to ~10–15 min on the same CPU pod).
- Paired-coordinate masking option in
  `src/neural_iv_surface_inference/data/masking.py` behind a flag so
  the historical benchmarks remain reproducible bit-for-bit.

Working-norm change:

- **For any remote/Pod work, the compute requirements (CPU vs GPU,
  VRAM, wall time, disk) must be stated up-front.** This audit was
  CPU-bound throughout but I framed it as a "trivial" script and
  didn't flag this — the operator allocated a CPU pod, which was
  correct by coincidence, not by communication. Memory updated.

## 6. Updated implementation plan

The Phase 3 plan is updated as follows:

| Epic | Old status | New status |
|---|---|---|
| 3A | `done` | `done` (unchanged) |
| 3B | `done` (in_review) | `done` (in_review) (unchanged); results carry an asterisk pending re-statement on OTM-clean data in 3D |
| **3X** | n/a | **NEW: `backlog` — data correction (OTM surface + paired masking + re-audit + re-train + re-eval)** |
| 3C | `backlog` | `backlog` (unchanged) — paused until 3X closes |
| 3D | `backlog` | `backlog` (unchanged) — closing memo must include the OTM-clean re-statement |

Phase 3 acceptance bar (≥ 5 % win over RBF on test MAE, no reliability
regression) is **unchanged**, but is now adjudicated on the OTM-clean
benchmark.

## 7. Current decision

Adopt ADR 0006: build an OTM-restricted strict surface
(`spy_surface_points_strict_otm.parquet`), rebuild only the
`spy_phase1_random40_noiselow` benchmark from it, re-audit, re-train
the ANP point head, re-fit the calibrator, re-run the 2D.6
decision-layer evaluation, and re-state the Phase 3 verdict against
RBF on the clean substrate. Original strict file and all historical
benchmarks remain untouched and comparable to themselves.

## 8. Immediate next action

Promote story **3X.1 (Decompose Phase 3X)** from `backlog → todo` and
run it in Plan mode to produce specs for 3X.2 (build OTM surface +
re-audit) and 3X.3 (re-train + re-eval).

## 9. One-sentence summary

The IV-surface pipeline silently violated the single-valued-function
assumption for five months by retaining both call and put legs of
every contract at the same model coordinate, biasing every Phase 2 /
Phase 3 MAE-vs-RBF comparison; the fix is to adopt the industry-
standard OTM-restricted surface convention and re-state Phase 3's
verdict on a clean substrate before any external presentation.
