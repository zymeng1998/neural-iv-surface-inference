# ADR 0004: Phase 3 Framing — Accuracy Push via Locality and Inductive Bias

## Status

Accepted

## Date

2026-05-27

## Context

Phase 2 closed on 2026-05-25 with both mandatory acceptance numbers green
(coverage 0.9184 within ±2 pp of nominal 0.90; high-confidence MAE 0.0606
< no-abstention test MAE 0.0855). The conditional model carries reliability
that neither the RBF interpolation nor the masked MLP baselines provide
(see [`docs/phase2_result_memo.md`](../phase2_result_memo.md)).

However, on raw point accuracy the calibrated conditional model still
underperforms the per-date RBF interpolation floor:

| Predictor (2D.9 test slice) | test MAE |
|---|---:|
| RBF interpolation | 0.0730 |
| masked MLP | 0.0905 |
| conditional (point) | 0.0841 |
| **conditional (calibrated)** | **0.0855** |

On the full 2D.4 test fold the RBF floor is 0.0662 — the gap is ~17 %.

Story 2E.2 (latent capacity diagnostic on the production 2D.7 Gaussian
checkpoint) found:

- Effective rank of `z_t ∈ R^64` ≈ **3.97** (entropy form).
- **52 / 64** PCs are dead (variance ratio < 1e-4).
- Top-8 PCs recover val Gaussian NLL to within **0.2 %** of baseline.

The natural reading of "rank-4 latent" as a defect is **wrong** in
isolation: the empirical IV-surface dynamics literature (Cont & da Fonseca,
*Quantitative Finance* 2002; Gatheral, *The Volatility Surface* 2006;
Gatheral & Jacquier, *Quantitative Finance* 2014 on SSVI) consistently
finds 3–5 factors (level, term-slope, skew, smile curvature) explain
≥ 95 % of per-date IV-surface variance. The latent collapse is consistent
with the underlying structure of `O_t`.

The actual diagnosis is different: even with a sufficient 4-D summary of
"what surface is today," our decoder cannot beat RBF because **mean
pooling discards spatial locality**. The decoder answers every query
`(k, τ)` from the same global summary; RBF weights nearby quotes more
heavily. The conditional model has no mechanism to "look at the closest
observed quote when predicting here."

Two consequences:

1. Adding capacity (wider latent, deeper trunk) is not expected to close
   the gap. 2E.2 directly confirms this — capacity is already unused.
2. The highest-leverage interventions are architectural (cross-attention
   from query to context), coordinate-feature (Fourier / SIREN on
   `(k, τ)`), and inductive-bias (residual on RBF, SVI-parameterized
   head) — not "more parameters" or "more training steps."

## Decision

**Phase 3 is framed as an accuracy push grounded in spatial locality and
financial inductive bias, not in capacity expansion.**

Operational consequences:

- The Phase 3 acceptance bar is **test MAE ≤ 0.95 × RBF** on both the
  2D.9 10-date slice (target ≤ 0.0693) and the full 2D.4 test fold
  (target ≤ 0.0629), while preserving Phase 2's reliability bar
  (calibration coverage within ±2 pp of nominal; hi-conf MAE strictly
  below no-abstention MAE).
- **The conditional neural model must beat RBF on its own, without
  using RBF as a scaffolding term.** A previously-considered
  neural-residual-on-RBF hybrid (`iv_hat = RBF + Δ`) is explicitly
  **out of scope for Phase 3**, on the grounds that the central
  hypothesis under test is *whether a well-designed conditional
  neural model can re-derive RBF-equivalent local-weighting behavior
  from data*. A hybrid would answer a strictly weaker question
  ("can a neural net learn the residual on top of RBF?" — trivially
  yes) and would obscure the experimental signal we actually need.
- **The RBF-hybrid is reserved as a Phase 4 production-engineering
  fallback**, to be invoked only if Phase 3 closes without meeting
  the acceptance bar. Production hybrids (PINN-style, Kalman + NN)
  are legitimate engineering answers in deployment; they are
  illegitimate as research scaffolding when the research question is
  "is the architecture sufficient?".
- Capacity sweeps (e.g. story 2E.3, `latent_dim ∈ {2, 4, 8, 16, 32, 64}`)
  remain part of the **Phase 2E follow-up** epic, not Phase 3. They
  validate the 2E.2 finding but are not on the Phase 3 critical path.
- Phase 3 is decomposed into four epics aligned to the diagnosis:
  - **3A** Coordinate-representation ablation. Single controlled
    experiment: Fourier-encoded `(k, τ)` vs raw `(k, τ)` on the
    existing SetEncoder + CoordinateDecoder, decoder-only retrain.
    Diagnoses whether the gap-to-RBF is an input-representation
    problem or an architecture problem; informs 3B's coordinate
    defaults but does not gate 3B.
  - **3B** Cross-attention decoder (Attentive Neural Process / Set
    Transformer / TNP style). The big architectural bet. The
    central hypothesis test of Phase 3. Runs in parallel with 3A.
  - **3C** Feature & inductive-bias expansion (microstructure
    features in `O_t`, optionally SVI-parameterized head). Builds
    on 3B's winning architecture.
  - **3D** Closing memo + re-evaluation against RBF.
- SIREN is **rejected** as a coordinate-encoding candidate: its
  inductive bias targets periodic implicit representations (SDFs,
  audio), which does not match the non-periodic IV surface in
  `(k, τ)`. 3A keeps Fourier features only.
- Each epic's stories are written by progressive decomposition (its
  `3X.1` decomposition story, executed in Plan mode), per the
  operating model § 4.

## Consequences

### Positive

- Locks Phase 3 scope to interventions with **prior literature support**
  for closing this exact kind of gap (ANP: Kim et al., *ICLR* 2019;
  Fourier features: Tancik et al., *NeurIPS* 2020; SIREN: Sitzmann et
  al., *NeurIPS* 2020; TNP: Nguyen & Grover, *ICML* 2022; SSVI:
  Gatheral & Jacquier 2014).
- Prevents drift into "let's try a bigger model" rabbit-holes that
  2E.2's evidence already rules out.
- Keeps reliability as a non-negotiable floor — the calibrated
  Gaussian / ensemble / masking-sensitivity fusion from Phase 2D
  remains the production output contract.

### Negative

- The 5 % beat-RBF bar is aggressive. There is real probability Phase 3
  closes the gap (Phase 3 acceptance criterion 1) but does not beat it,
  in which case Phase 3 closes as "partial success" with a documented
  retrospective and the **Phase 4 production-engineering fallback**
  (RBF-prior + neural-residual hybrid) is opened.
- Cross-attention adds inference-time compute (O(N_context × N_query)
  attention per date) versus the current O(N_context + N_query)
  DeepSets-style pipeline. The decision layer's per-query latency budget
  has to be re-validated under 3B.
- The no-RBF-scaffolding rule is binding even if early evidence is
  unfavorable. We accept the risk of "spending a phase to learn the
  bet didn't pay" in exchange for a clean experimental signal.

### Open trade-offs (deferred to per-epic decomposition stories)

- Whether 3B picks ANP (cross-attention only) or full TNP (self-attention
  over `context ∪ query`). 3B.1 will set this with cost / benefit
  evidence.
- Whether 3C ships an SVI-parameterized head. The no-arbitrage benefit
  is structural but risks under-fitting genuinely complex surfaces.
  3C.1 will scope this.
- 3B's coordinate-encoding default (Fourier vs raw) is informed by
  3A's measured ablation if 3A is `done` first. If 3B runs before
  3A finishes, 3B defaults to **Fourier on** as cheap insurance
  against coordinate-MLP spectral bias (Rahaman et al., *ICML*
  2019; Tancik et al., *NeurIPS* 2020).

## Related decisions

- [ADR 0002 — Phase 1 scope freeze](0002_phase1_scope_freeze.md):
  established the chronological-split benchmark on which all Phase 2 and
  Phase 3 numbers are computed.
- [ADR 0003 — SPY options data source migration](0003_spy_options_data_source_migration.md):
  established Alpha Vantage as the canonical data source. Phase 3 inherits.
- Phase 2 closing memo [`docs/phase2_result_memo.md`](../phase2_result_memo.md):
  the headline numbers Phase 3's bar is set against.
- [`docs/roadmaps/phase2_followups.md`](../roadmaps/phase2_followups.md):
  epic 2E. Phase 3 reuses the locality / cross-attention agenda
  previously sketched there as W7; the placeholder is now folded into
  Phase 3 to avoid two parallel architectural plans.
