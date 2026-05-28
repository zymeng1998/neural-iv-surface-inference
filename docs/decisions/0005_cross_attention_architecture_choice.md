# ADR 0005: Phase 3B Cross-Attention Architecture — Attentive Neural Process (ANP)

## Status

Accepted

## Date

2026-05-28

## Context

[ADR 0004](0004_phase3_accuracy_push_framing.md) frames Phase 3 as an
accuracy push grounded in **spatial locality and financial inductive
bias, not capacity expansion**. Epic 3A — the coordinate-representation
ablation under W10 — closed `done` on 2026-05-28 with a clean negative
result on the **frozen 2D.7 encoder**:

| Variant (decoder-only retrain, 2D.7 encoder frozen) | full-fold test MAE |
|---|---:|
| raw `(k, τ)` | 0.0760 |
| Fourier-encoded `(k, τ)` (8 bands, max_freq=10) | 0.0790 |
| RBF interpolation baseline | 0.0662 |

Source: `results/3/spy_phase1_random40_noiselow/3a_compare/comparison.csv`;
write-up in `docs/roadmaps/phase3_accuracy_push.md` § W10 closing addendum.

Fourier positional features added parameter count and decoder in-dim
without measurably closing the gap to RBF. Reading both 2E.2
(latent-capacity diagnostic — top-8 PCs recover val NLL to within 0.2 %
of baseline; effective rank ≈ 4) and 3A together, the residual gap is
**architectural**: a mean-pool DeepSets summary cannot perform
query-local re-weighting of context the way RBF does, regardless of
how `(k, τ)` is encoded into the decoder.

This ADR locks the architectural pick for epic 3B (W11): a
**query-attends-to-context decoder**, replacing the global pool ->
coordinate-decoder pipeline. The candidate set from the Phase 3
roadmap (W11) is:

| Name | Reference | Per-query compute |
|---|---|---|
| Attentive Neural Process (ANP) | Kim et al., *ICLR* 2019 | O(N_context) |
| Set Transformer encoder + cross-attention decoder | Lee et al., *ICML* 2019 | O(N_context²) at encoder + O(N_context) at decoder |
| Transformer Neural Process (TNP) | Nguyen & Grover, *ICML* 2022 | O((N_context + N_query)²) |
| Perceiver IO with learned latent queries | Jaegle et al., *ICLR* 2022 | O(N_context × L) for L latents |

## Decision

**3B ships the Attentive Neural Process (ANP) as the cross-attention
decoder, composed end-to-end with the existing DeepSets `SetEncoder`,
trained from scratch on the AV benchmark.**

Concretely:

- **Encoder:** keep the Phase 2 `SetEncoder` (per-element MLP + masked
  mean pooling producing a global `z_t ∈ R^64`). Per-element
  embeddings `H_t ∈ R^{N_t × D_h}` from the pre-pool layer are
  retained as the cross-attention **keys / values**.
- **Decoder:** for each query `(k_q, τ_q)`, the ANP block computes
  `cross_attention(query=φ(k_q, τ_q), keys=H_t, values=H_t)` →
  context vector `c_q`. The decoder MLP takes `[z_t, c_q, φ(k_q, τ_q)]`
  → predictive head (gaussian / quantile / point per the existing
  `head.kind` flag).
- **Coordinate encoding:** **raw `(k, τ)`** per 3A's measured result.
  (Fourier-on is reserved as a follow-up ablation under the
  cross-attention decoder if 3B alone does not close the gap.)
- **Training:** end-to-end (encoder + decoder jointly), from scratch,
  on the AV benchmark. No frozen-encoder protocol — 3A.3 already
  served that diagnostic purpose.
- **Training-time context / query split:** masked random split of
  `O_t` into `(context, target)` per Kim et al. 2019, matching the
  conditional-NP training recipe.
- **Heads:** all three existing head kinds (`gaussian`, `quantile`,
  `point`) wired identically so 3B's evaluation parallels 2D.7 line
  for line.

The encoder is **not** promoted to attention-based pooling (Set
Transformer SAB / PMA blocks) in 3B's first pass. 2E.2 evidence that
encoder capacity is already unused (effective rank ≈ 4, 52/64 dead PCs)
argues against expanding the encoder before measuring whether a
locality-aware decoder alone closes the gap. Promotion to SAB / PMA is
a possible follow-up under 3C only if 3B's evidence shows the encoder
to be a co-limiter.

Cross-attention is the minimal change that **directly** addresses the
locality bottleneck identified in 2E.2 + ADR 0004: every query gets to
re-weight observed quotes by similarity to its own `(k, τ)`, instead
of consuming a global mean.

## Rejection rationale

### Set Transformer (encoder + cross-attention decoder)

Rejected for 3B's first pass.

- Adds two changes at once (encoder *and* decoder swap), confounding
  attribution. We cannot tell whether the win came from cross-attention
  at the decoder or attention pooling at the encoder.
- 2E.2 evidence (effective rank ≈ 4 on the production encoder; capacity
  already unused) argues that the bottleneck is at the
  encoder→decoder interface, not in encoder expressiveness.
- Compute cost rises (O(N_context²) at the encoder) for unclear
  marginal benefit.
- **If** 3B's ANP run finds the encoder is a co-limiter (e.g. the
  cross-attention decoder still cannot recover RBF-level locality
  even with full access to per-element embeddings), Set Transformer
  encoder is a documented follow-up under 3C.

### Transformer Neural Process (TNP)

Rejected for 3B's first pass.

- Per-query compute is O((N_context + N_query)²) versus ANP's
  O(N_context). On a per-date AV slice with N_context up to ~3000 and
  N_query of the same order, TNP self-attention over the full
  `context ∪ query` set is roughly an order of magnitude more
  expensive per step.
- The marginal benefit over ANP — joint context-query self-attention
  rather than asymmetric cross-attention — is mostly about
  *information sharing across simultaneous queries*. In our task
  queries are evaluated independently per `(k, τ)` location; the
  cross-query signal is weak.
- ANP is a strict subset of TNP's design space. If ANP closes the gap,
  TNP is unnecessary; if ANP does not, TNP is a cleanly-scoped
  follow-up story in 3C with one decision (asymmetric → symmetric
  attention) to evaluate.

### Perceiver IO with learned latent queries

Rejected for 3B's first pass.

- Perceiver IO is designed for *very* large unstructured input sets
  (millions of pixels, long audio). At the per-date scales seen here
  (N_context ≤ ~3 000), the learned-latent compression layer offers
  no meaningful compute saving.
- Training learned latents adds a tuning surface (number of latents,
  initialization, ordering) with no equivalent in the IV-surface
  literature.
- Existing reference implementations target vision / multimodal
  workloads; porting carries integration risk for marginal expected
  value at our problem scale.

## Consequences

### Positive

- Directly addresses the locality bottleneck named in ADR 0004 with
  the smallest viable architectural change.
- Cleanest attribution: only the decoder mechanism changes versus the
  3A.3 raw-`(k,τ)` baseline; the encoder backbone, dataset, optimizer,
  schedule, and head implementations are held fixed.
- Per-query compute stays O(N_context) — within the same order as the
  DeepSets baseline; the Phase 2D decision-layer latency budget is
  expected to remain viable. (3B.4 reports measured inference latency
  in its manifest.)
- ANP training stability is documented (Kim et al. 2019; subsequent
  CNP / ANP papers). No exotic training tricks required.
- Keeps a single architectural variant in flight for 3B.4 / 3B.5,
  matching the non-goal in 3B.1's spec ("Do NOT scope 3B to ship
  more than ONE cross-attention variant in its first pass").

### Negative

- End-to-end retrain means 3B cannot share 2D.7's encoder weights.
  Full training cost per head ≈ 2D.7 wall time; with three heads
  (gaussian / quantile / point-control), expect ~3× the per-head
  Pod wall versus 2D.7's serial sweep.
- If the negative result repeats (cross-attention decoder also fails
  to close the gap), epic 3C inherits the entire RBF gap with two
  failed architectural diagnostics already spent.
- ANP's cross-attention is per-query and per-step; the per-date
  inference call now scales with N_context × N_query at evaluation
  time. The decision-layer runner must batch carefully.

### Open trade-offs (deferred to 3B.2+)

- **Number of attention heads** (default H = 4) and **per-head
  dimension** (default D_h / H). Final pick is in 3B.2's
  implementation spec.
- **Whether to retain the global `z_t` summary** in the decoder
  input alongside the cross-attention context vector. Default: yes
  (the ANP paper's deterministic + latent path; we keep only the
  deterministic path here). Final pick is in 3B.2.
- **Coordinate encoding fall-back ablation** (raw vs Fourier under
  the new decoder). If 3B's first end-to-end run does not close the
  gap, 3B.4 can register a Fourier-on follow-up; not in scope for the
  initial cut.

## Related decisions

- [ADR 0004 — Phase 3 framing](0004_phase3_accuracy_push_framing.md):
  defines the acceptance bar and the no-RBF-scaffolding rule that 3B
  inherits unchanged.
- [ADR 0002 — Phase 1 scope freeze](0002_phase1_scope_freeze.md): the
  chronological-split benchmark all Phase 3 numbers are computed
  against.
- 3A closing addendum in
  [`docs/roadmaps/phase3_accuracy_push.md`](../roadmaps/phase3_accuracy_push.md)
  § W10: the measured Fourier-vs-raw delta that fixes 3B's
  coordinate-encoding default at raw `(k, τ)`.
- 2E.2 capacity diagnostic memo addendum in
  `docs/phase2_result_memo.md`: the evidence that argues against
  promoting the encoder before measuring the cross-attention decoder.
