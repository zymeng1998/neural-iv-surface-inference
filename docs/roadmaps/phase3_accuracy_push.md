# Phase 3 — Accuracy Push: Beat RBF Without Losing Reliability

---
created_at: 2026-05-27T00:00:00-04:00
last_updated_at: 2026-05-28T13:00:00-04:00
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

### W12 — Feature & inductive-bias expansion (epic 3C)

Two orthogonal directions; 3C.1 decides whether to ship one or both.

- **Microstructure features in `O_t`.** The current per-quote tuple is
  `(log_moneyness, τ, iv_input)` — 3 dims. Adding `bid`, `ask`,
  `bid_ask_spread`, `mid_price`, `volume`, `open_interest`,
  `put_call_indicator`, and exogenous state (`spot_level`, `vix`,
  `recent_realized_vol_5d`, `days_to_next_earnings_or_dividend`) gives
  the encoder real signal about quote reliability and vol regime.
  Note: this does **not** require latent rank to rise — most of these
  are correlated with the surface itself. It lets the encoder
  *downweight* unreliable quotes (wide spread, low volume) when
  forming `z_t`.
- **SVI / SSVI parameterized head.** Instead of free
  `decoder(z_t, k, τ) → σ`, output `θ_t ∈ R^5` (SVI per slice) and
  evaluate the SVI formula at `(k, τ)`. Built-in arbitrage-free
  structure under SSVI (Gatheral & Jacquier 2014); strong inductive
  bias; what practitioners actually use. Risk: under-fitting genuinely
  complex surfaces.

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
| W12 Feature & inductive-bias expansion | 3C | 3C.1 | depends on 3B (builds on the winning architecture) |
| W13 Closing memo + re-evaluation | 3D | 3D.1 | last — pure synthesis on committed artifacts |

Sequencing principle: **diagnose & bet in parallel → inductive-bias
polish → synthesize**. 3A and 3B compare independently against the
same 2D.9 baseline; 3C depends only on 3B's winner. Each epic
decomposes only when entered.

Dependency graph:

```text
3A (Fourier ablation, current arch)  ──┐
                                       ├──→ 3C ──→ 3D
3B (cross-attention, central bet)    ──┘
```

If Phase 3 closes without meeting the acceptance bar after 3D, the
**Phase 4 production-engineering fallback** is opened: RBF-prior +
neural-residual hybrid as a deployment answer (explicitly *not* a
research substitute). See ADR 0004.

> **Status (2026-05-28, post-3B.1):** Phase 3 is **open**. Epic
> **3A** is `done` (raw beats Fourier on the frozen 2D.7 encoder by
> Δ +0.00300 full-fold test MAE; gap-to-RBF unclosed). Epic **3B**
> is `in_progress`; ADR 0005 picks **ANP** end-to-end with DeepSets,
> raw `(k, τ)`; decomposition story `3B.1` is `in_review`; six
> atomic stories `3B.2 … 3B.7` are registered at `backlog`. Epics
> **3C / 3D** remain `backlog` with decomposition stories `3C.1 /
> 3D.1` still at `backlog`. Phase 3 source code touched so far:
> 3A's `features/coord_encoding.py` + `freeze_encoder` /
> `encoder_init_from` flags on `train_conditional`; no 3B code yet.
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
- **ADR 0006 — Microstructure feature set freeze**. Written by 3C.1
  if 3C ships expanded `O_t` features. Locks the feature list for
  reproducibility.
- **ADR 0007 — SVI / SSVI head adoption**. Only if 3C ships the
  SVI-parameterized head. Locks the parameterization (SVI vs SSVI vs
  SABR) and the no-arbitrage projection step.
- **ADR 0008 — Phase 3 production predictor selection**. Written at
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
