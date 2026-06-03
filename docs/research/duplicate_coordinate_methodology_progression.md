# Methodology Progression: From the Dirty Call+Put Benchmark to the Clean OTM Restatement

---
created_at: 2026-06-02T00:00:00-04:00
last_updated_at: 2026-06-02T00:00:00-04:00
epic: 3X
story: 3X.14
scope: matched substrate `spy_phase1_random40_noiselow` only
---

> This document is the 3X close-out narrative (story 3X.14). It is **not**
> the full Phase 3 final memo — that remains a 3D deliverable (Q3). It
> tells the single story of how a silent data defect was discovered,
> corrected, and what the corrected benchmark did (and did not) change
> about the Phase 2→3 conclusions. Every number traces to a committed
> bundle via
> [`results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison.csv`](../../results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison.csv).

## 0. One-paragraph summary

A duplicate-coordinate audit found that 93.61 % of the strict surface
table was call-put leg pairs sharing the same `(date, expiration,
strike)` — two distinct IV labels at one coordinate — silently
violating the single-valued-function assumption every model and the RBF
baseline rely on. We adopted the industry-standard OTM-restricted
surface (puts below spot, calls above) as the canonical substrate,
rebuilt and audited it (PASS 12/12, 93.61 %→0 % duplication), and
re-ran the full model-family ladder on the matched clean benchmark
`spy_phase1_random40_noiselow_otm`. **Verdict:** the Phase 3 conclusion
holds in direction but sharpens — RBF still wins, and on clean data it
wins by *more*, not less; the dirty benchmark had been masking RBF's
locality advantage by forcing it to average over call/put pairs. The
DeepSets→ANP architecture story survives intact on clean data. No
architecture reversal; no Phase 2 reopen.

## 1. The legacy dirty benchmark (Phase 1 → Phase 3B)

From Phase 1 onward, the surface table
([`src/data/03_build_spy_surface_table.py`](../../src/data/03_build_spy_surface_table.py))
kept **both** the call leg and the put leg returned by the Alpha Vantage
`HISTORICAL_OPTIONS` endpoint for every listed `(date, expiration,
strike)`. The cleaning gates checked only per-row finiteness (bid / ask
/ IV / tau) — there was no `drop_duplicates`, no OTM convention, no
aggregation. The modelled object was therefore not a function: at almost
every coordinate the loss saw two labels, and the RBF kernel averaged
over them.

All Phase 2 (2A–2E) and Phase 3 (3A, 3B) results — including the
headline 3B verdict that **ANP came within +2.7 % of RBF but did not
clear the +5 % bar** — were computed on this dirty substrate. They are
preserved unchanged; this document does not retract them, it restates
them on corrected data.

## 2. The audit (2026-05-29)

A vectorised audit
([`scripts/audit_duplicate_coordinates.py`](../../scripts/audit_duplicate_coordinates.py);
report [`docs/research/duplicate_coordinate_audit.md`](duplicate_coordinate_audit.md))
quantified the defect on the 22,512,040-row strict table and on
`spy_phase1_random40_noiselow`:

| Metric | Value |
|---|---:|
| Rows in `(date, exp, strike)` duplicate groups | 21,072,592 / 22,512,040 (**93.61 %**) |
| Duplicate groups that are call-put pairs | 100.00 % |
| Median in-group IV range | 0.049 (≈ the ANP test-MAE bar) |
| p90 / max in-group IV range | 0.302 / 2.96 |
| Held-out rows with an exact-coordinate observed twin | **37.46 %** |

The 37.46 % exact-twin leakage forced `nearest_observed_distance = 0`
for over a third of held-out rows regardless of true local density —
which is why the originally-proposed sparse-region ANP-vs-RBF experiment
was *not interpretable as written*: its "densest" stratum was call-put
leakage, not real density.

## 3. The correction (ADR 0006)

[ADR 0006](../decisions/0006_duplicate_coordinate_data_correction.md)
locked the fix: adopt the **OTM-restricted surface** (puts for `K<S`,
calls for `K>S`, tie-broken at ATM by tighter relative spread) as the
canonical modelling substrate, plus paired-coordinate masking as
opt-in, default-off defence-in-depth. **Preservation-first:** no legacy
artifact was mutated; all new artifacts carry an `_otm` suffix.

The OTM strict surface and all 11 OTM benchmark variants were rebuilt
(3X.4) and put through a human-review audit gate (3X.5): **PASS 12/12**,
duplication 93.61 %→0.0000 %, twin leakage 0.0000 % across all splits.

## 4. The clean restatement (3X.6 → 3X.13)

The full model-family ladder was re-run on the matched clean benchmark
`spy_phase1_random40_noiselow_otm` — the apples-to-apples counterpart to
the original dirty `random40_noiselow` — so the correction is isolated
from any masking or noise change. Headline matched test-MAE comparison
(full table:
[`comparison_wide.md`](../../results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.md)):

| Family | Head | Dirty test MAE | OTM test MAE | Dirty / OTM |
|---|---|---:|---:|---:|
| rbf | interp | 0.0662 | **0.00613** | **10.80×** |
| anp_calibrated | fused | 0.0813 | 0.01162 | 7.00× |
| anp_single | point | 0.0684 | 0.00987 | 6.93× |
| anp_single | quantile | 0.0681 | 0.01175 | 5.79× |
| anp_ensemble | point | 0.0689 | 0.01220 | 5.64× |
| deepsets_single | gaussian | 0.0787 | 0.01530 | 5.15× |
| deepsets_single | quantile | 0.0719 | 0.01418 | 5.07× |
| anp_single | gaussian | 0.0726 | 0.01440 | 5.04× |
| deepsets_ensemble | point | 0.0748 | 0.01594 | 4.69× |
| deepsets_single | point | 0.0756 | 0.01752 | 4.31× |
| mlp | point | 0.0905 | 0.03006 | 3.01× |

Every family improves 3–11× on the clean substrate. The RBF floor drops
the most (10.8×) because it was the model most penalized by the dirty
defect: averaging two-valued targets is exactly the error a local
interpolator cannot avoid, whereas the amortized neural decoders paid a
roughly constant encoder cost on either substrate.

## 5. The verdict (3X.14)

### 5.1 RBF vs ANP — unchanged in direction, **wider** in magnitude

| Comparison | Dirty (full fold) | OTM (test) |
|---|---|---|
| RBF | 0.0662 | 0.00613 |
| Best ANP (point head) | 0.0680 | 0.00987 |
| ANP − RBF, absolute | +0.0018 | +0.0037 |
| ANP − RBF, relative | **+2.7 %** | **+61 %** |

The original 3B near-miss (+2.7 %, point head) was **partly an artifact
of the dirty benchmark unfairly penalizing RBF**. Removing the call-put
confound did not let ANP catch RBF — it let RBF pull *further ahead*.
On the clean single-valued surface, RBF's locality advantage is far more
pronounced: the gap widens from +2.7 % to +61 % (best ANP head), and the
calibrated production predictor (`anp_calibrated`) sits at +90 %
(0.01162 vs 0.00613). **The Phase 3 acceptance bar (ANP ≥ 5 % below RBF)
is still NOT MET, and is missed by a wider margin on clean data.**

This is the "conclusion unchanged / data correction was a
scientific-cleanliness win that removed a confounder" branch (ADR 0006
Q5), with one sharpened nuance: the confounder was **flattering ANP**,
not RBF. The honest reading is that ANP never came as close to RBF as
the dirty +2.7 % suggested.

### 5.2 DeepSets → ANP — **architecture story survives**

The Phase 2→3 architecture thesis (cross-attention beats mean-pooling)
holds on clean data. ANP beats DeepSets at matched head on every OTM
comparison:

| Head | DeepSets OTM | ANP OTM | ANP better by |
|---|---:|---:|---:|
| point | 0.01752 | 0.00987 | 1.77× |
| quantile | 0.01418 | 0.01175 | 1.21× |
| gaussian | 0.01530 | 0.01440 | 1.06× |
| ensemble (point) | 0.01594 | 0.01220 | 1.31× |

Because the architecture ranking did **not** reverse, no Phase 2 reopen
and no architecture-revising retrospective/ADR addendum are warranted
(ADR 0006 Q5 trigger not fired). The DeepSets→ANP progression remains
the correct conditional-modelling story; it simply tops out short of the
local-interpolation floor on this substrate.

### 5.3 Reliability

The decision-layer eval on OTM with thresholds held constant (3X.12)
preserved the Phase 3 reliability floor: coverage@0.90 = 0.9295 (within
band), hi-conf MAE 0.00835 < no-abstention 0.01162. Calibration
transfers val→test with more drift than the dirty fit (test coverage
0.866 vs val 0.900 in 3X.11), a known cost of the much tighter OTM error
scale.

## 6. No-overclaim guardrail

- **Correct claim:** *the original Phase 2D / Phase 3B result was
  restated on the matched clean OTM benchmark `random40_noiselow_otm`,
  and the RBF-wins conclusion held and widened.*
- **Forbidden claim:** *the result is robust across all OTM benchmark
  variants.* The other 10 OTM benchmarks were rebuilt and audited for
  substrate completeness but **not** retrained; that all-11 robustness
  study is deferred (roadmap §W11.5 "Future work"). Every claim here is
  scoped to the single matched substrate.

## 7. What this unblocks

- 3C reopens on the clean OTM substrate (per ADR 0006 Q6), now with the
  honest target: the gap to close is ~+61 % (best head) / ~+90 %
  (production), not the dirty +2.7 %. The 3B implication stands and is
  reinforced — pure decoder-architecture iteration has plateaued; 3C
  should prioritise feature / inductive-bias expansion (microstructure
  features, SVI/SSVI priors), with the Phase 4 RBF-prior hybrid as the
  production fallback.
- 3D (the full Phase 3 closing memo) will fold this restatement in
  alongside the preserved dirty 3A/3B verdicts.

## References

- Comparison bundle:
  [`comparison.csv`](../../results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison.csv) /
  [`comparison_wide.md`](../../results/3/spy_phase1_random40_noiselow_otm/3x_compare/comparison_wide.md)
- ADR: [0006 Duplicate-Coordinate Data Correction](../decisions/0006_duplicate_coordinate_data_correction.md)
- Audit report: [`duplicate_coordinate_audit.md`](duplicate_coordinate_audit.md)
- Retrospective: [`0002 Call-Put Duplicate-Coordinate Discovery`](../retrospectives/0002_call_put_duplicate_coordinate_discovery.md)
- Roadmap: [`phase3_accuracy_push.md` §W11.5](../roadmaps/phase3_accuracy_push.md)
- Dirty 3B verdict (preserved): roadmap §W11 closing addendum
