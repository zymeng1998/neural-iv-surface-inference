# Phase 3 — Accuracy Push: Beat RBF Without Losing Reliability

---
created_at: 2026-05-27T00:00:00-04:00
last_updated_at: 2026-06-02T22:30:00-04:00
---

## 1) Why Phase 3 exists

Phase 2 shipped a calibrated, decision-grade reliability layer
([`docs/phase2_result_memo.md`](../phase2_result_memo.md)) but left an
**accuracy gap** versus the per-date RBF interpolation baseline:

| Predictor (2D.9 test slice) | test MAE |
|---|---:|
| RBF interpolation | 0.0730 |
| conditional (calibrated) | 0.0855 |

The gap is ~17 % on the 2D.9 slice and ~29 % on the full 2D.4 fold
(0.0855 vs 0.0662). For a system that wants to *replace* RBF as the
production surface inference, this is the live blocker. Reliability
without accuracy is half a product.

Story **2E.2** (latent capacity diagnostic) ruled out the obvious
capacity-bound explanation:

- Effective rank of `z_t ∈ R^64` ≈ **3.97**; 52 / 64 PCs are dead.
- Top-8 PCs recover val Gaussian NLL to within **0.2 %** of baseline.

The IV-surface dynamics literature (Cont & da Fonseca 2002; Gatheral
2006; Gatheral & Jacquier 2014) supports rank-4 as the *natural*
dimensionality of an IV-surface summary (level / term-slope / skew /
curvature). So the bottleneck is not "summary too small" — it is "the
decoder cannot exploit spatial locality given the summary." RBF beats
us because RBF weights nearby quotes; our DeepSets-pool decoder
answers every query from the same global summary.

Framing decision: **[ADR 0004 — Phase 3 framing](../decisions/0004_phase3_accuracy_push_framing.md)**
locks Phase 3 to interventions that attack locality and inductive bias,
not capacity.

## 2) Target outcomes

Phase 3 is **complete** when all of the following hold on the
`spy_phase1_random40_noiselow` AV benchmark:

1. **Beat RBF by ≥ 5 % on test MAE.** Target:
   - 2D.9 10-date slice: test MAE ≤ **0.0693** (0.95 × 0.0730).
   - Full 2D.4 test fold: test MAE ≤ **0.0629** (0.95 × 0.0662).
2. **No reliability regression.** Coverage stays within ±2 pp of
   nominal 0.90 on the same fold; high-confidence MAE remains strictly
   below no-abstention MAE; forbidden-flag violation count does not
   increase meaningfully versus the Phase 2D close.
3. **Per-query inference latency** stays under a documented budget
   (set in 3B's decomposition). Cross-attention adds compute; we have
   to confirm the decision-layer pipeline still scales.
4. **Reproducibility & evidence discipline.** Each architectural
   variant carries a committed `artifacts/runs/3X/<variant>/` bundle
   (manifest, training curves, val/test predictions) and a
   `results/3/<dataset>/<variant>/` bundle (metrics, figures), with a
   closing entry in `docs/experiments/experiment_journal.md`. The
   Phase 3 closing memo `docs/phase3_result_memo.md` synthesizes these
   into a headline comparison versus RBF and versus the Phase 2D
   baseline.

If condition (1) is not met but condition (2) holds and the gap
narrows by ≥ 50 %, Phase 3 closes as **partial success** with a
documented retrospective in `docs/retrospectives/`.

## 3) Workstreams

The phase is decomposed into four workstreams, sequenced by risk —
cheap-win scaffolding (W10) first, the big architectural bet (W11)
second, feature & inductive-bias expansion (W12) third, closing
synthesis (W13) last.

> Numbering picks up from Phase 2's W1–W5 (epics 2A–2D) and Phase 2E's
> placeholders W6–W9. W7 (pooling / encoder variants) is folded into
> W11 below; W8 (calibration drift) and W9 (decision-layer threshold
> sensitivity) remain Phase 2E placeholders.

### W10 — Coordinate-representation ablation (epic 3A)

A single controlled experiment to diagnose whether the gap-to-RBF is
an **input-representation problem** (the coordinate MLP cannot resolve
high-frequency `(k, τ)` structure due to spectral bias) or an
**architecture problem** (the mean-pool decoder cannot exploit
locality). The answer informs 3B's coordinate-encoding default but
does **not** gate 3B.

- **Fourier positional features for `(k, τ)`.** Concatenated raw
  `(k, τ)` is a known weak input representation for coordinate MLPs
  (Rahaman et al., *ICML* 2019 on spectral bias; Tancik et al.,
  *NeurIPS* 2020 on Fourier features). Add sinusoidal features at
  a band of frequencies, decoder-only retrain on the existing 2D.7
  encoder checkpoint, all else equal versus the production baseline.
- SIREN (Sitzmann et al., *NeurIPS* 2020) is **rejected** — its
  inductive bias targets periodic implicit representations and does
  not match the non-periodic IV surface in `(k, τ)`. See
  [ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md).
- The previously-considered **neural-residual-on-RBF hybrid is
  removed from Phase 3** and reserved as a Phase 4 production fallback
  if Phase 3 closes without meeting the acceptance bar. Rationale:
  the central hypothesis under test is whether a well-designed
  conditional neural model can re-derive RBF-equivalent local-
  weighting behavior *from data*; an RBF-prior scaffolding would
  answer a strictly weaker question and obscure the experimental
  signal we need. See ADR 0004.

3A is **intentionally small**: one ablation, one decoder-only retrain,
one bundle of results. It exists for clean attribution — if Fourier
features close a meaningful fraction of the gap on the current
architecture, the gap is at least partly input-representation; if not,
the bottleneck is purely architectural and 3B has to do all the work.

#### Concrete decomposition (registered 2026-05-28)

| ID | Locale | Title | One artifact bundle |
|---|---|---|---|
| [`3A.2`](../tasks/specs/3A.2_local_fourier_feature_module.md) | local | Fourier-feature module + `coord_encoding` flag on `ConditionalSurfaceModel` + unit / wiring tests | `features/coord_encoding.py` + tests; no training |
| [`3A.3`](../tasks/specs/3A.3_remote_decoder_only_retrain.md) | remote | Decoder-only retrain on frozen 2D.7 encoder, Fourier vs raw, identical seed / optimizer / schedule | `artifacts/runs/3A/{fourier,raw}/` — manifest + training curves + val/test predictions |
| [`3A.4`](../tasks/specs/3A.4_local_eval_and_addendum.md) | local | W1 evaluation of both variants vs the 2D.9 baselines + journal entry + roadmap closing addendum | `results/3/<dataset>/3a_{fourier,raw,compare}/` + this section's addendum |

Each story is atomic: one question, one artifact bundle, one
acceptance check, and no story spans local + remote.

#### 3A closing addendum (filled by 3A.4 — 2026-05-28)

Both decoder-only retrains (Fourier vs raw `(k, τ)`) were scored
end-to-end on the full AV benchmark fold via
`scripts/run_3a_eval.py`. Result bundles:

- `results/3/spy_phase1_random40_noiselow/3a_fourier/`
- `results/3/spy_phase1_random40_noiselow/3a_raw/`
- `results/3/spy_phase1_random40_noiselow/3a_compare/comparison.csv`

**Headline (full-fold AV, n_test = 5,805,664):**

| Variant | test MAE | val MAE | test Cov@0.90 (uncal) | test hi-conf MAE@0.8 | test Gauss NLL |
|---|---:|---:|---:|---:|---:|
| 3a_fourier | 0.07905 | 0.06202 | 0.9012 | 0.04551 | −1.4150 |
| 3a_raw     | 0.07604 | 0.05705 | 0.8720 | 0.04362 | −1.4301 |
| Δ (Fourier − Raw) | **+0.00300** | +0.00498 | +0.0292 | +0.00189 | +0.0151 |

**Measured Fourier-vs-raw delta:** Fourier is **+0.00300** worse
on test MAE (~3.9% relative) and **+0.00498** worse on val MAE
(~8.7% relative). Gaussian NLL also prefers raw on both splits.
Fourier's only edge is *test coverage at the nominal 0.90 band*
(0.9012 vs raw's 0.8720), which is an artefact of Fourier's
wider σ — not an accuracy gain. Both bands are uncalibrated; the
2D.7 calibrator was fitted for the full-retrain head, not the
decoder-only retrains.

**Fraction of the gap to RBF closed:** **zero, in either
direction.** The 2D.9-slice gap was RBF 0.0730 vs calibrated
conditional 0.0855 (+0.0125). Neither 3A.3 variant gets below
the RBF slice number on the full fold (raw 0.0760, Fourier
0.0790), and the row counts differ by ~90× so a strict apples-
to-apples comparison would need RBF re-run on the full fold or
3A.3 rescored on the 10-date cap. Conservative read: the
locality bottleneck identified in 2E.2 + ADR 0004 is
**architectural, not input-representation**.

**Recommended 3B coordinate-encoding default: raw `(k, τ)`.**
Fourier positional features add parameter count and decoder
in-dim with no measurable MAE / NLL benefit on this encoder.
3B's central architecture comparison (cross-attention decoder vs
DeepSets-pool) should default to raw `(k, τ)` to keep the
comparison clean; Fourier-on can be a secondary ablation under
the cross-attention decoder if 3B alone does not close the gap.

Epic 3A closes `done`. Phase 3 acceptance bar unchanged; the
full RBF gap is now carried by 3B.

### W11 — Cross-attention decoder (epic 3B)

The architectural bet **and the central hypothesis test of Phase 3**.
Replace the global pool → coordinate decoder with a
**query-attends-to-context** decoder so each `(k, τ)` query selects
which observations matter for *that* query. Runs in parallel with
3A — 3B does **not** gate on 3A's result. If 3A finishes first and
Fourier features won, 3B inherits Fourier-encoded coordinates by
default; if 3B starts before 3A finishes, it defaults to Fourier-on
as cheap insurance against coordinate-MLP spectral bias.

Candidate architectures, in increasing cost / complexity:

| Name | Reference | Compute (per query) |
|---|---|---|
| Attentive Neural Process (ANP) | Kim et al., *ICLR* 2019 | O(N_context) |
| Set Transformer encoder + cross-attention decoder | Lee et al., *ICML* 2019 | O(N_context²) at encoder |
| Transformer Neural Process (TNP) | Nguyen & Grover, *ICML* 2022 | O((N_context + N_query)²) |
| Perceiver IO with learned latent queries | Jaegle et al., *ICLR* 2022 | O(N_context × L) for L latents |

3B.1 (decomposition story) picks one of these with cost / expected-value
evidence. ANP is the default first cut — direct fix for the locality
bottleneck, lowest incremental compute, well-known training behavior.

The encoder backbone may stay DeepSets (per-element MLP + masked mean)
under ANP, or be promoted to attention-based pooling (Set Transformer
SAB / PMA blocks) if 3B.1 finds encoder expressiveness is a co-limiter.

#### W11 — 3B.1 decomposition outcome (2026-05-28)

3B.1 picked **Attentive Neural Process (ANP)** as the cross-attention
variant, composed end-to-end with the existing DeepSets `SetEncoder`,
on raw `(k, τ)` (per 3A's measured result). Set Transformer / TNP /
Perceiver IO are explicitly rejected for 3B's first pass; rationale
in [ADR 0005](../decisions/0005_cross_attention_architecture_choice.md).

The encoder is **not** promoted to attention-based pooling in 3B's
first pass — 2E.2's effective-rank ≈ 4 evidence argues against
expanding encoder capacity before measuring whether a locality-aware
decoder alone closes the gap. Set Transformer SAB / PMA promotion is
a documented follow-up under 3C only if 3B's ANP run finds the
encoder is a co-limiter.

Concrete story list under W11:

| ID | Where | Title |
|---|---|---|
| [3B.2](../tasks/specs/3B.2_local_anp_cross_attention_decoder.md) | local | ANP cross-attention decoder module + `decoder_kind` flag on `ConditionalSurfaceModel` + unit / smoke / integration tests (no training) |
| [3B.3](../tasks/specs/3B.3_local_predictor_adapter.md) | local | Predictor-adapter wiring so the existing `ConditionalSurfacePredictor` round-trips `decoder_kind` (evaluator parity test) |
| [3B.4](../tasks/specs/3B.4_remote_full_av_training.md) | remote | Full AV training of end-to-end DeepSets+ANP across `head.kind ∈ {gaussian, quantile, point}` — mirrors 2D.7's three-head sweep |
| [3B.5](../tasks/specs/3B.5_remote_deep_ensemble.md) | remote | K=5 deep ensemble of the ANP point head on AV — mirrors 2D.8 |
| [3B.6](../tasks/specs/3B.6_local_calibrator_refit.md) | local | Calibrator re-fit on the ANP val predictions — mirrors 2D.4 |
| [3B.7](../tasks/specs/3B.7_local_decision_layer_eval.md) | local | End-to-end decision-layer eval of ANP vs Phase 2D baselines on the `spy_phase1_random40_noiselow` slice; ships `results/3/.../3b_anp/` + closing comparison CSV + closing addendum on this section |

#### W11 — 3B closing addendum (filled by 3B.7 — 2026-05-28)

End-to-end decision-layer evaluation of the ANP calibrated predictor
(3B.4 gaussian head + 3B.6 calibrator + 3B.5 K=5 ensemble) on the
`spy_phase1_random40_noiselow` slice, scored on the Pod (RTX A4500) with
the 2D.6 runner under the *same* decision config + 10-date diagnostics
cap + seed as 2D.9. Long-format evidence:
`results/3/spy_phase1_random40_noiselow/3b_compare/comparison.csv`;
ANP bundle: `results/3/spy_phase1_random40_noiselow/3b_anp/`.

**Measured ANP test MAE:**

| View | ANP | RBF | ANP/RBF | Bar (0.95×RBF) | Met? |
|---|---|---|---|---|---|
| 10-date slice (calibrated gaussian) | 0.0813 | 0.0730 | +11.4 % | ≤ 0.0693 | ✗ |
| Full fold (gaussian) | 0.0722 | 0.0662 | +9.0 % | ≤ 0.0629 | ✗ |
| Full fold (point head) | 0.0680 | 0.0662 | +2.7 % | ≤ 0.0629 | ✗ |

**ANP-vs-RBF gap:** +2.7 % at best (full-fold point head); the
calibrated gaussian production predictor runs +9–11 %. The 10-date
decision-layer slice is pessimistic (ANP gaussian 0.0813 on the slice vs
0.0722 on the full fold).

**Reliability (10-date slice, test):** coverage@0.90 = 0.9149 (within
±2 pp ✓); hi-conf MAE 0.0542 < no-abstention 0.0813 (✓). The Phase 3
reliability floor holds.

**Verdict against the Phase 3 acceptance bar: NOT MET.** ANP does not
beat RBF by ≥ 5 % on the slice or the full fold. It is, however, the
strongest conditional model produced so far: full-fold point head 0.0680
beats 3A raw decoder-only (0.0760) by ~10 % and the 2D DeepSets-decoder
family (2D.4 calibrated 0.0855 on slice) by ~5 %, narrowing the
gap-to-RBF from the ~29 % Phase-3 starting point to ~2.7 %.

**Implication for 3C scope:** the architecture ladder
(DeepSets-pool → decoder-only raw/Fourier → end-to-end cross-attention)
has plateaued ~2.7 % short of RBF. Further pure decoder-architecture
iteration is unlikely to clear the last few percent. 3C should
prioritise **feature / inductive-bias expansion** — microstructure
features and no-arb / SVI priors — to extract the residual gap, with the
Phase 4 RBF-prior production fallback as the alternative if 3C also
stalls. This addendum *informs* 3C.1 but does not pre-empt its
decomposition.

> **Open research direction (proposed, not yet run):** the aggregate bar
> masks *where* each model wins. RBF is a local interpolator and should
> degrade in sparse-context regions (deep wings, extreme maturities)
> where ANP's global learned prior may hold up — a sparse-region win
> would reframe the conclusion as a density-routed RBF/ANP hybrid (a
> Phase 4 production angle). Scientific design + feasibility evidence
> (held-out labels survive masking, so the comparison is valid) in
> [`docs/research/sparse_region_anp_vs_rbf_design.md`](../research/sparse_region_anp_vs_rbf_design.md).

### W11.5 — Data-correction interlude (epic 3X, NEW 2026-05-29)

A duplicate-coordinate audit on the AV-backed strict surface table
(2026-05-29; full evidence
[`docs/research/duplicate_coordinate_audit.md`](../research/duplicate_coordinate_audit.md))
uncovered a structural violation of the single-valued-function
assumption that has been silent since Phase 1:

- **93.61 %** of the 22,512,040-row strict file lives inside a
  `(date, expiration, strike)` duplicate group; the same 93.61 %
  collides at `(date, round(log_m, 10), round(tau, 10))`. **100.00 %**
  of those duplicates are call-put leg pairs.
- **Median in-group IV range = 0.049** (≈ the current ANP test MAE
  bar); p90 = 0.302; max = 2.96.
- **37.46 %** of held-out rows in `spy_phase1_random40_noiselow` have
  an exact-coordinate observed twin on the same date, forcing
  `nearest_observed_distance = 0` regardless of true local density.

This means the IV surface as it has been modelled is not a
single-valued function: at every coordinate the loss saw two distinct
labels, and the RBF kernel averaged over them. Cross-model MAE
comparisons are biased in RBF's favour on dense-coordinate regions,
and the proposed sparse-region ANP-vs-RBF experiment is **not
interpretable as written** (its densest stratum is dominated by
call-put leakage, not real density).

[ADR 0006](../decisions/0006_duplicate_coordinate_data_correction.md)
locks the correction: adopt the **industry-standard OTM-restricted
surface** (puts for `K<S`, calls for `K>S`, tie-broken by tighter
relative spread at ATM) as the canonical modelling substrate, plus
paired-coordinate masking as a defence-in-depth guarantee.

Epic 3X scope — **14 atomic stories** (decomposed in 3X.1; full D1–D8
+ Q1–Q8 record in the
[ADR 0006 addendum](../decisions/0006_duplicate_coordinate_data_correction.md#addendum-2026-05-29--decomposition-decisions-locked-in-3x1)).
**Preservation-first:** no legacy artifact is mutated; all new artifacts
carry an `_otm` suffix.

| ID | Story | Locale / HW |
|---|---|---|
| 3X.1 | Decompose Phase 3X (ADR addendum + 3X.2–3X.14 specs) | local CPU |
| 3X.2 | OTM-surface builder + step-04 `--source` flag + ATM-band (D5) + residual handling (D7) + tests — **done** (2026-05-30) | local CPU |
| 3X.3 | Vectorised audit v2 + paired-coordinate masking flag (default off) + parity tests — **done** (2026-05-30) | local CPU |
| 3X.4 | Build OTM strict surface + rebuild **all 11** OTM benchmarks — **done** (2026-05-30) | Pod **CPU** |
| 3X.5 | Audit OTM strict + all 11 benchmarks — **HUMAN REVIEW GATE** → **PASS 12/12** (93.61%→0% dup; 0% twin leakage) — **done** (2026-05-30) | Pod **CPU** |
| 3X.6 | Early RBF-on-OTM baseline (floor sanity check before GPU spend) — → test MAE **0.00613** (~10.8× below RBF-on-dirty 0.0662) — **done** (2026-05-31) | Pod **CPU** |
| 3X.7 | MLP-on-OTM baseline (Q1 — ladder anchor) — → test MAE **0.03006** (~3.2× below dirty MLP 0.0951) — **in_review** (2026-05-31) | Pod GPU |
| 3X.8 | DeepSets-on-OTM — single (2D.7-equiv) + K=5 ensemble (2D.8-equiv) (D2) — → test MAE quantile **0.01418** / gaussian 0.01530 / ensemble 0.01594 / point 0.01752 (~5× below dirty 2D 0.072–0.079) — **in_review** (2026-05-31) | Pod GPU |
| 3X.9 | ANP-on-OTM — all three heads (D1) | Pod GPU |
| 3X.10 | ANP K=5 ensemble-on-OTM (mirror 3B.5) | Pod GPU |
| 3X.11 | Calibrator re-fit on OTM val predictions (mirror 3B.6) | local CPU |
| 3X.12 | Decision-layer eval on OTM, thresholds held constant (Q2) | Pod |
| 3X.13 | Dirty-vs-OTM side-by-side comparison tables (matched substrate) | local CPU |
| 3X.14 | 3X closing addendum + methodology-progression narrative (Q3) | local CPU |

**Primary retraining substrate:** the model-family restatement
(3X.6–3X.12) runs on **`spy_phase1_random40_noiselow_otm` only** — the
clean apples-to-apples counterpart to the dirty `random40_noiselow`
experiments — so the correction (raw call+put benchmark → OTM
single-valued benchmark) is isolated from any masking/noise change. All
11 OTM variants are still rebuilt and audited (3X.4 / 3X.5) for
substrate completeness.

**Split Pod rental (Q4):** CPU pod for the build + audit gate + RBF
floor (3X.4–3X.6); human review of the gate; then a GPU pod for
retraining (3X.7–3X.12). Verify OTM artifacts are on persistent storage
before terminating the CPU pod.

**Baseline ladder the closing tables carry (3X.13):** RBF-on-OTM →
MLP-on-OTM → DeepSets-on-OTM (single + ensemble) → ANP-on-OTM (3 heads
+ ensemble) → calibrated / decision-layer — each paired against its
committed dirty counterpart.

#### Future work — full all-11 OTM retraining (deferred robustness study)

Phase 3X performs the full corrected model-family restatement on the
matched `random40_noiselow_otm` benchmark because this is the clean
apples-to-apples counterpart to the original dirty `random40_noiselow`
experiments. All 11 OTM benchmark variants are rebuilt and audited to
make the corrected benchmark suite complete. **Full GPU-heavy retraining
across all 11 OTM variants is deferred as a future robustness study.**
That future study would test whether the dirty-vs-OTM conclusion and the
DeepSets-vs-ANP architecture ranking remain stable across masking
regimes, noise regimes, and structured sparsity patterns (the seven
questions enumerated in the ADR 0006 addendum). Until it runs, the
3X conclusion is scoped to the matched substrate only.

**No-overclaim guardrail.** Correct claim: *"the original Phase 2D /
Phase 3B result was restated on the matched clean OTM benchmark
`random40_noiselow_otm`."* Forbidden claim: *"the result is robust
across all OTM benchmark variants"* — that requires the deferred study.

3C / 3D are **paused** until 3X closes. Phase 3 acceptance bar
(≥ 5 % vs RBF, reliability preserved) is **unchanged** but is now
adjudicated on the OTM-clean benchmark. The original 3B verdict
(ANP +2.7 % vs RBF best-case; bar NOT met) and the §W11 closing
addendum are preserved; the Phase 3 closing memo (3D) will append
an OTM-clean re-statement alongside. Per Q5, if 2D-on-OTM changes the
architecture conclusion, a retrospective / ADR addendum is opened
rather than reopening Phase 2.

#### W11.5 — 3X closing addendum (filled by 3X.14 — 2026-06-02)

Epic 3X is complete. The OTM-restricted surface was built, audited
(3X.5: PASS 12/12, 93.61 %→0 % duplication, 0 % twin leakage), and the
full model-family ladder was re-run on the matched clean benchmark
`spy_phase1_random40_noiselow_otm`. Full narrative:
[`docs/research/duplicate_coordinate_methodology_progression.md`](../research/duplicate_coordinate_methodology_progression.md);
evidence bundle:
`results/3/spy_phase1_random40_noiselow_otm/3x_compare/`.

**Verdict — RBF vs ANP: conclusion unchanged in direction, *wider* in
magnitude.**

| Comparison | Dirty (full fold) | OTM (test) |
|---|---|---|
| RBF | 0.0662 | 0.00613 |
| Best ANP (point head) | 0.0680 | 0.00987 |
| ANP − RBF (relative) | **+2.7 %** | **+61 %** |
| Calibrated ANP (`anp_calibrated`) − RBF | +9–11 % | **+90 %** (0.01162 vs 0.00613) |

The Phase 3 acceptance bar (ANP ≥ 5 % below RBF) is **still NOT MET, and
missed by a wider margin on clean data**. The original 3B near-miss
(+2.7 %) was partly an artifact of the dirty call-put confound, which
penalized the local interpolator (RBF averaged two-valued targets) more
than the amortized neural decoders. Correcting it let RBF pull *further
ahead*, not ANP catch up. Every family improves 3–11× on the clean
substrate; RBF improves the most (10.8×) precisely because it was most
hurt by the defect. This is ADR 0006's "conclusion unchanged / removed a
confounder" branch, with the sharpened nuance that the confounder was
**flattering ANP**.

**Verdict — DeepSets → ANP architecture story: survives.** ANP beats
DeepSets at matched head on every OTM comparison (point 1.77×, ensemble
1.31×, quantile 1.21×, gaussian 1.06×). The cross-attention thesis holds
on clean data; the ranking did **not** reverse, so the Q5 trigger for a
Phase 2 reopen / architecture-revising retrospective is **not fired**.
ADR 0006 moves to **Implemented**.

**Reliability:** 3X.12 preserved the floor with thresholds held constant
(coverage@0.90 = 0.9295; hi-conf MAE 0.00835 < no-abstention 0.01162),
at the cost of more val→test calibration drift (3X.11) on the tighter
OTM error scale.

**No-overclaim guardrail (binds this addendum):** scoped to the matched
`random40_noiselow_otm` substrate only. The other 10 OTM variants were
rebuilt and audited but **not** retrained — the all-11 robustness study
is deferred (§W11.5 "Future work"). No robustness claim is made here.

**Forward:** 3C reopens on the clean OTM substrate (Q6) with the honest
target restored — the gap to close is ~+61 % (best head) / ~+90 %
(production), not the dirty +2.7 %. The 3B implication is reinforced:
decoder-architecture iteration has plateaued; 3C should prioritise
feature / inductive-bias expansion, with the Phase 4 RBF-prior hybrid as
the production fallback. The full Phase 3 closing memo remains a 3D
deliverable (Q3).

### W12 — Feature & inductive-bias expansion (epic 3C)

Two orthogonal directions were originally on the table (microstructure
features in `O_t`; SVI / SSVI parameterized head). 3C.1 (decomposition,
2026-06-02, post-3X) **picks scope = microstructure only**; the SVI
head is deferred to a separate epic if/when `micro_v1` does not narrow
the post-3X gap.

#### Concrete decomposition (registered 2026-06-02)

3C.1 locked the following scope, recorded in
[`3C.1`](../tasks/specs/3C.1_decompose_phase_3c.md) and
[ADR 0008](../decisions/0008_microstructure_feature_set_freeze.md):

- **Track scope:** microstructure only. SVI / SSVI track deferred.
- **Feature set `micro_v1`:** six AV-native per-quote fields appended
  to the existing 3-tuple — `bid`, `ask`, `bid_ask_spread_rel`,
  `volume`, `open_interest`, `put_call_indicator` — for a 9-dim
  context tuple. Frozen by ADR 0008. `mid`, `spot`, `vix`,
  `realized_vol`, earnings calendar are **out of scope** for
  `micro_v1` (collinear or require a new ingest path).
- **Flag:** `feature_set ∈ {minimal, micro_v1}`, default `minimal` —
  every committed 2D / 3A / 3B / 3X checkpoint stays reproducible
  byte-for-byte.
- **Head sweep:** all three heads (gaussian / quantile / point),
  mirrors 3X.9.
- **Substrate:** `spy_phase1_random40_noiselow_otm` only. No all-11
  fan-out (deferred robustness study from §W11.5 carries through).
- **Decision-layer thresholds held constant** against 3X.12 (Q2
  invariant carries forward).
- **No-overclaim guardrail** carries from 3X — claims scoped to the
  matched substrate only.

| ID | Locale / HW | Title | One artifact bundle |
|---|---|---|---|
| [`3C.2`](../tasks/specs/3C.2_local_micro_feature_pipeline.md) | local CPU | Microstructure feature pipeline + `feature_set` flag on loader/encoder/model/predictor/training loop + unit/integration tests | code patch + 3 test files + smoke config; no training |
| [`3C.3`](../tasks/specs/3C.3_remote_anp_micro_three_head.md) | remote GPU | Full AV retrain of DeepSets+ANP with `feature_set: micro_v1` across `head.kind ∈ {gaussian, quantile, point}` on OTM | `artifacts/runs/3C3/{gaussian,quantile,point}/` mirroring 3X.9 |
| [`3C.4`](../tasks/specs/3C.4_remote_anp_micro_ensemble.md) | remote GPU | K=5 ANP+`micro_v1` point-head deep ensemble on OTM (seeds [101,202,303,404,505]) | `artifacts/runs/3C4/ensemble/` mirroring 3X.10 |
| [`3C.5`](../tasks/specs/3C.5_local_calibrator_refit_micro.md) | local CPU | Calibrator re-fit on ANP+`micro_v1` val predictions | `configs/calibration_3C5_anp_micro.yaml` + `artifacts/calibration/3C5_anp_micro.json` + tests |
| [`3C.6`](../tasks/specs/3C.6_remote_decision_layer_eval_micro.md) | remote GPU | End-to-end decision-layer eval on OTM, thresholds held constant (Q2) | `results/3/spy_phase1_random40_noiselow_otm/3c_anp_micro/` + `artifacts/runs/3C6/manifest.json` |
| [`3C.7`](../tasks/specs/3C.7_local_micro_vs_baseline_comparison.md) | local CPU | OTM-baseline vs OTM+`micro_v1` comparison tables (long + wide) on matched substrate | `results/3/spy_phase1_random40_noiselow_otm/3c_compare/` long + wide + `headline.md` |
| [`3C.8`](../tasks/specs/3C.8_local_3c_closing_addendum.md) | local CPU | 3C closing addendum (§W12) + ADR 0008 → Implemented + journal/README sync; **NOT** the full Phase 3 memo (Q3 — 3D) | this section's closing addendum + ADR Outcome + BOARD/INDEX/log/journal/README edits |

Dependency chain: `3C.2 → 3C.3 → 3C.4 → 3C.5 → 3C.6 → 3C.7 → 3C.8`.
3C.3 and 3C.4 should share a single Pod-GPU rental window (~5 h total)
to amortise rental. Each story is atomic — one question, one artifact
bundle, one acceptance check; no story spans local + remote.

#### W12 — 3C closing addendum (filled by 3C.8 on close — placeholder)

Filled by 3C.8 with: headline test MAE per family×head for `micro_v1`
vs the matched OTM baselines (3X.9 / 3X.10) and vs RBF-on-OTM (3X.6
floor 0.00613); calibrated production verdict against the Phase 3 bar
vs 3X.12; forward recommendation (3D close, `micro_v2` data-source
ADR, or Phase 4 RBF-prior hybrid).

### W13 — Closing memo + re-evaluation (epic 3D)

Pure synthesis on committed artifacts. Mirrors story 2D.10:

- `docs/phase3_result_memo.md` with the headline table (all 3A / 3B /
  3C variants versus RBF and versus Phase 2D close).
- A regenerated companion notebook
  `notebooks/06_phase3_results.ipynb` driven by
  `scripts/generate_phase3_results_notebook.py`.
- Closing entry in `docs/experiments/experiment_journal.md`.

No new training, no new eval runs.

## 4) Mapping to implementation phases

| Workstream | Implementation epic | Decomposition story | Sequencing |
|---|---|---|---|
| W10 Coordinate-representation ablation | 3A | 3A.1 | parallel with 3B — independent ablation on the existing architecture |
| W11 Cross-attention decoder | 3B | 3B.1 | parallel with 3A — the central hypothesis test |
| **W11.5 Data-correction interlude** | **3X** | **3X.1** | **between 3B and 3C — gates 3C / 3D / sparse-region experiment** |
| W12 Feature & inductive-bias expansion | 3C | 3C.1 | depends on 3B (builds on the winning architecture) **and on 3X (clean substrate)** |
| W13 Closing memo + re-evaluation | 3D | 3D.1 | last — pure synthesis on committed artifacts, **must include OTM-clean re-statement of 3B verdict** |

Sequencing principle: **diagnose & bet in parallel → inductive-bias
polish → synthesize**. 3A and 3B compare independently against the
same 2D.9 baseline; 3C depends only on 3B's winner. Each epic
decomposes only when entered.

Dependency graph:

```text
3A (Fourier ablation, current arch)  ──┐
                                       ├──→ 3X (OTM data correction) ──→ 3C ──→ 3D
3B (cross-attention, central bet)    ──┘
```

If Phase 3 closes without meeting the acceptance bar after 3D, the
**Phase 4 production-engineering fallback** is opened: RBF-prior +
neural-residual hybrid as a deployment answer (explicitly *not* a
research substitute). See ADR 0004.

> **Status (2026-05-29, post-duplicate-coordinate audit):** Phase 3 is
> **open**, with a new gating epic **3X** (data correction) between 3B
> and 3C/3D. A 2026-05-29 audit
> ([`docs/research/duplicate_coordinate_audit.md`](../research/duplicate_coordinate_audit.md))
> found that 93.61 % of strict-table rows live in `(date, exp, strike)`
> duplicate groups, all 100 % call-put pairs, with median in-group IV
> range 0.049 — invalidating the single-valued-function assumption
> the model and the RBF baseline both rely on. The Phase 3B verdict
> (ANP +2.7 % vs RBF best-case; bar NOT met) and the Phase 2 / Phase
> 3A artifacts are preserved unchanged; the Phase 3D closing memo will
> append an OTM-clean re-statement after 3X.
> [ADR 0006](../decisions/0006_duplicate_coordinate_data_correction.md)
> + [retrospective 0002](../retrospectives/0002_call_put_duplicate_coordinate_discovery.md)
> hold the full narrative. The sparse-region ANP-vs-RBF research note
> (`docs/research/sparse_region_anp_vs_rbf_design.md`) is now `blocked`
> on 3X; it moves into 3C scope on the OTM-clean benchmark. 3X.1
> decomposition is the immediate next action.
>
> **Update (2026-06-02, 3X CLOSED — see §W11.5 closing addendum):**
> epic 3X is `done`. The OTM restatement confirmed the 3B verdict in
> direction and *widened* it — RBF still wins, and on the clean
> single-valued benchmark the best-ANP-head gap grew from +2.7 % to
> +61 % (calibrated production +90 %); the bar remains NOT MET. The
> DeepSets→ANP architecture story survives (ANP beats DeepSets at every
> matched head on OTM), so no Phase 2 reopen (Q5 trigger not fired) and
> ADR 0006 → **Implemented**. Narrative:
> [`duplicate_coordinate_methodology_progression.md`](../research/duplicate_coordinate_methodology_progression.md).
>
> **Update (2026-06-02, 3C ENTERED — decomposition in_review):** epic 3C
> is `in_progress`; story `3C.1` (decompose Phase 3C) is `in_review`.
> Scope locked: **microstructure-only (`micro_v1`)** per ADR 0008 — six
> AV-native per-quote fields (`bid`, `ask`, `bid_ask_spread_rel`,
> `volume`, `open_interest`, `put_call_indicator`) appended to the
> existing 3-tuple → 9-dim context. Three-head sweep (gaussian /
> quantile / point) mirroring 3X.9; K=5 ensemble mirroring 3X.10; Q2
> decision-layer threshold invariant held; matched
> `random40_noiselow_otm` substrate only; SVI / SSVI head **deferred**
> to a separate epic if `micro_v1` does not narrow the gap. Seven new
> atomic stories (3C.2 … 3C.8) registered. **3C.1 closed `done`
> 2026-06-02; all seven downstream stories promoted to `todo`.
> 3C.2 implemented 2026-06-03 (`in_review`): the `feature_set ∈
> {minimal, micro_v1}` flag is wired through loader / encoder / model /
> predictor / training loop (default `minimal` preserves every legacy
> checkpoint byte-for-byte; 16 new tests, full suite 417 passed; no
> training). Immediate next action: operator promotes 3C.2 → `done`,
> then 3C.3 (remote three-head retrain consuming the flag) — strict
> chain 3C.2 → … → 3C.8.**
>
> **Original 2026-05-28 status (preserved for traceability):** Phase 3 is **open**, but both
> diagnostic epics have closed. Epic **3A** is `done` (raw beats
> Fourier on the frozen 2D.7 encoder by Δ +0.00300 full-fold test MAE;
> gap-to-RBF unclosed). Epic **3B** is `done` (in_review): ADR 0005
> picked **ANP** end-to-end with DeepSets, raw `(k, τ)`; stories
> `3B.2 … 3B.7` all shipped. **3B verdict: accuracy bar NOT met** —
> ANP best-case is +2.7 % vs RBF (full-fold point head 0.0680 vs
> 0.0662); the calibrated gaussian production predictor runs +9–11 %;
> reliability holds (coverage 0.9149 ±2 pp; hi-conf MAE 0.0542 <
> 0.0813). See the §W11 closing addendum + `3b_compare/comparison.csv`.
> ANP is the strongest conditional model to date and narrows the gap
> from ~29 % to ~2.7 %, but pure decoder-architecture iteration has
> plateaued. Epics **3C / 3D** remain `backlog` with decomposition
> stories `3C.1 / 3D.1` at `backlog`; 3C is the next epic and should
> prioritise feature / inductive-bias expansion per the §W11 addendum.
> Phase 3 source code touched so far: 3A's
> `features/coord_encoding.py` + `freeze_encoder` / `encoder_init_from`
> flags on `train_conditional`; 3B's `models/anp_decoder.py` +
> `decoder_kind` flag on `ConditionalSurfaceModel`.
>
> Scope was tightened on 2026-05-27 after a planning discussion: the
> originally-proposed neural-residual-on-RBF hybrid was dropped from
> 3A (concedes the central research question) and reserved as a Phase
> 4 production fallback. SIREN was dropped from 3A's coordinate-
> encoding menu (inductive bias mismatch). 3A and 3B were decoupled
> (parallel, not serial) so the central architectural bet does not
> wait on the ablation.

## 5) Acceptance criteria

Phase 3 is **complete** when the following hold:

1. **Beat-RBF target met.** Test MAE on `spy_phase1_random40_noiselow`
   is ≤ 0.0693 on the 2D.9 slice and ≤ 0.0629 on the full fold, for
   at least one Phase 3 variant. Numbers logged in
   `docs/phase3_result_memo.md` and in
   `results/3/<dataset>/<variant>/metrics_summary.csv`.
2. **Reliability preserved.** That same variant's coverage is within
   ±2 pp of 0.90 on the calibrated test fold; high-confidence MAE
   (`keep_fraction = 0.8`) is strictly below no-abstention MAE on
   that fold.
3. **Decision-layer latency budget met.** Per-query inference latency
   on the production decision-layer pipeline stays within the budget
   set by 3B.1.
4. **Reproducibility & evidence discipline.** Each variant has a
   committed `artifacts/runs/3X/<variant>/manifest.json` + curves +
   predictions, and a committed
   `results/3/<dataset>/<variant>/` bundle. Each closes with an
   `experiment_journal.md` entry.
5. **Closing memo + notebook shipped.**
   `docs/phase3_result_memo.md` and
   `notebooks/06_phase3_results.ipynb` exist and are
   regenerated-by-script reproducible.

If (1) misses but the gap narrows by ≥ 50 % and (2)–(5) hold, Phase 3
closes as **partial success** with a numbered doc under
`docs/retrospectives/`.

## 6) Recommended decision records (to be written as decisions land)

- **ADR 0005 — Cross-attention architecture choice** (ANP vs Set
  Transformer vs TNP). Written by 3B.1 with cost / benefit evidence.
  Accepted 2026-05-28.
- **ADR 0006 — Duplicate-coordinate data correction** (OTM-restricted
  surface + paired-coordinate masking guarantee). Written 2026-05-29
  in response to the duplicate-coordinate audit; gates 3X / 3C / 3D.
  Accepted 2026-05-29.
- **ADR 0007 — Microstructure feature set freeze**. Written by 3C.1
  if 3C ships expanded `O_t` features. Locks the feature list for
  reproducibility.
- **ADR 0008 — SVI / SSVI head adoption**. Only if 3C ships the
  SVI-parameterized head. Locks the parameterization (SVI vs SSVI vs
  SABR) and the no-arbitrage projection step.
- **ADR 0009 — Phase 3 production predictor selection**. Written at
  the end of Phase 3 with the headline comparison; locks which
  variant becomes the production surface inference predictor.

## 7) Long-term-memory / parallel-session conventions

Phase 3 introduces a small amount of new infrastructure on top of
[`ai_human_collaboration.md`](../workflows/ai_human_collaboration.md)
and [`session_protocol.md`](../workflows/session_protocol.md) so that
(a) any fresh session can resume cold without reloading the repo, and
(b) multiple sessions can run in parallel safely.

- **[`docs/PHASE3_INDEX.md`](../PHASE3_INDEX.md)** — single entry
  point for any Phase 3 session. Mirrors the BOARD subset for Phase 3,
  but additionally carries per-story "last checkpoint" snippets so a
  cold session sees the exact next concrete action without reading the
  full spec. Updated as part of the PMR pre-push routine.
- **Story template upgrades** — `_template.md` gains
  `parallel_safe_with` (list of story IDs with disjoint file scope)
  and `file_scope` (hard contract: paths this story may write to),
  plus a "Last checkpoint" block updated mid-story.
- **Worktree convention** — for actual concurrent sessions, use
  `git worktree add` per active story. Recommended in
  [`session_protocol.md`](../workflows/session_protocol.md) § 4.
  Not enforced; a solo single-session day does not need a worktree.

These are docs / process changes; no source-code impact.
