# Phase 2 Follow-ups — Roadmap

---
created_at: 2026-05-26T00:00:00-04:00
last_updated_at: 2026-05-27T00:00:00-04:00
---

## Why this phase exists

Phase 2 (epics 2A–2B–2C–2D) closed on 2026-05-25 with both headline acceptance
numbers green (coverage 0.9184 within ±2 pp, hi-conf MAE 0.0606 < 0.0855; see
[docs/phase2_result_memo.md](../phase2_result_memo.md)). Closure surfaced
several follow-up questions that do **not** belong in Phase 3 scoping — they
are localized diagnostics or small-scope architectural sweeps that test
assumptions baked into Phase 2's `ConditionalSurfaceModel` and decision layer.

Epic **2E — Phase 2 follow-ups** collects this work. Stories are added by
progressive decomposition as follow-ups are identified. The epic is open-ended:
it stays `in_progress` while new follow-up stories are added, and only flips
`done` if and when the human decides the follow-up set is complete (e.g. at
the start of Phase 3).

## Workstreams

### W6 — Capacity & representation diagnostics

Question: how much of the `ConditionalSurfaceModel`'s latent / hidden capacity
is actually carrying signal at the production hyperparameters
(`latent_dim=64`, `hidden_dim=128`, `n_post_layers=1`)?

Outputs sought:

- Effective rank (entropy form) of the `SetEncoder` latent `z_t` on val data.
- PCA spectrum and cumulative variance: how many PCs explain 95% / 99% of
  variance.
- Per-dimension variance share — explicit count of dead / near-dead units.
- Val/test NLL across a latent-dim sweep `{8, 16, 32, 64, 96, 128}`.

Stories (atomic — each one decision, one artifact bundle):

- [`2E.2 Latent capacity diagnostic`](../tasks/specs/2E.2_latent_capacity_diagnostic.md) —
  effective rank + PCA + per-dim / per-PC ablation on the existing 2D.7
  gaussian checkpoint. No retraining. Outputs a recommendation that
  scopes 2E.3.
- [`2E.3 latent_dim sweep`](../tasks/specs/2E.3_latent_dim_sweep.md) —
  retrain at widths chosen by 2E.2's recommendation (shrink, expand, or
  close-without-running). Sweep grid intentionally deferred until 2E.2
  is done.

### Future workstreams (placeholders)

- **W7 — Pooling / encoder architecture variants.** **Folded into Phase 3
  (epic 3B, workstream W11)** on 2026-05-27. The cross-attention-decoder
  bet covers the same surface (encoder/decoder swap that fixes the
  spatial-locality bottleneck identified in 2E.2) and supersedes a W7-only
  effort. See [`phase3_accuracy_push.md`](phase3_accuracy_push.md) and
  [ADR 0004](../decisions/0004_phase3_accuracy_push_framing.md).
- **W8 — Calibration drift.** Re-fit the temperature / split-conformal
  calibrator on a held-out window further from the training cutoff to test
  whether confidence calibration drifts under realistic operational gaps.
  Still a placeholder; orthogonal to Phase 3.
- **W9 — Decision-layer threshold sensitivity.** Sweep the abstention and
  tradability thresholds in [`configs/decision_layer.yaml`](../../configs/decision_layer.yaml)
  to surface the operating-point Pareto frontier. Still a placeholder;
  orthogonal to Phase 3.

## Operating model

- Epic 2E uses the same progressive-decomposition rule as 2A–2D
  ([docs/workflows/ai_human_collaboration.md](../workflows/ai_human_collaboration.md) §4).
- All 2E stories must specify whether they run **local** (Mac, synthetic /
  cached artifacts) or **remote** (RunPod, full AV data). No story straddles.
- All 2E stories must comply with the PMR pre-push routine ([CLAUDE.md](../../CLAUDE.md)).

## Status

| Workstream | Status | Story trigger |
|---|---|---|
| W6 Capacity & representation diagnostics | `in_progress` | 2E.2 (diagnostic) `done`; 2E.3 (sweep) `backlog` |
| W7 Pooling / encoder variants | `folded into Phase 3 (epic 3B / W11)` | — |
| W8 Calibration drift | `not yet committed` | — |
| W9 Decision-layer threshold sensitivity | `not yet committed` | — |
