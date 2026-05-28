# Phase 3 — Single Entry Point for Fresh Sessions

---
created_at: 2026-05-27T00:00:00-04:00
last_updated_at: 2026-05-28T11:10:00-04:00
---

> **Read this first if you are picking up Phase 3 work cold.** It mirrors
> the relevant subset of [`docs/tasks/BOARD.md`](tasks/BOARD.md) plus
> per-story "last checkpoint" snippets and a parallel-safety matrix.
> The goal: a fresh session learns the exact next concrete action in
> under 60 seconds without loading the rest of the repo.

## 30-second orientation

- **Phase 3 framing:** beat RBF (the per-date interpolation baseline)
  on test MAE by ≥ 5 %, without losing the Phase 2D reliability layer
  and **without using RBF as a scaffolding term**. The conditional
  neural model must beat RBF on its own; RBF-as-prior is reserved as
  a Phase 4 production fallback. See
  [ADR 0004](decisions/0004_phase3_accuracy_push_framing.md).
- **Why the gap exists:** mean-pool DeepSets decoder cannot exploit
  spatial locality (story 2E.2 confirmed latent capacity is not the
  bottleneck).
- **How we close it:** 3A (coordinate-representation ablation:
  Fourier vs raw `(k, τ)`) and 3B (cross-attention decoder) run in
  **parallel** as independent diagnostics; 3C (feature & inductive-
  bias expansion) builds on 3B's winning architecture; 3D synthesizes.
  See [`roadmaps/phase3_accuracy_push.md`](roadmaps/phase3_accuracy_push.md).
- **Bar:** test MAE ≤ 0.0693 on 2D.9 slice, ≤ 0.0629 on full 2D.4
  fold; coverage stays within ±2 pp of 0.90; hi-conf MAE strictly
  below no-abstention MAE.

## Live status (synced manually with `BOARD.md`)

> Update this table whenever a Phase 3 status changes. The PMR pre-push
> routine flags it if `BOARD.md` is touched but this index is not.

| ID | Type | Title | Status | Spec | Updated |
|---|---|---|---|---|---|
| 3A | Epic | Coordinate-representation ablation (Fourier vs raw `(k, τ)`, decoder-only) | `in_progress` | [`roadmaps/phase3_accuracy_push.md`](roadmaps/phase3_accuracy_push.md) §W10 | 2026-05-28 |
| 3A.1 | Story | Decompose Phase 3A | `in_review` | [`3A.1`](tasks/specs/3A.1_decompose_phase_3a.md) | 2026-05-28 |
| 3A.2 | Story | Local: Fourier-feature module + `coord_encoding` flag + unit / wiring tests | `in_review` | [`3A.2`](tasks/specs/3A.2_local_fourier_feature_module.md) | 2026-05-28 |
| 3A.3 | Story | Remote: decoder-only retrain on frozen 2D.7 encoder (Fourier vs raw) | `in_review` | [`3A.3`](tasks/specs/3A.3_remote_decoder_only_retrain.md) | 2026-05-28 |
| 3A.4 | Story | Local: eval of both variants vs 2D.9 baselines + journal + roadmap addendum | `backlog` | [`3A.4`](tasks/specs/3A.4_local_eval_and_addendum.md) | 2026-05-28 |
| 3B | Epic | Cross-attention decoder (ANP / Set Transformer / TNP) | `backlog` | [`roadmaps/phase3_accuracy_push.md`](roadmaps/phase3_accuracy_push.md) §W11 | 2026-05-27 |
| 3B.1 | Story | Decompose Phase 3B | `backlog` | [`3B.1`](tasks/specs/3B.1_decompose_phase_3b.md) | 2026-05-27 |
| 3C | Epic | Feature & inductive-bias expansion (microstructure, optional SVI) | `backlog` | [`roadmaps/phase3_accuracy_push.md`](roadmaps/phase3_accuracy_push.md) §W12 | 2026-05-27 |
| 3C.1 | Story | Decompose Phase 3C | `backlog` | [`3C.1`](tasks/specs/3C.1_decompose_phase_3c.md) | 2026-05-27 |
| 3D | Epic | Closing memo + re-evaluation vs RBF | `backlog` | [`roadmaps/phase3_accuracy_push.md`](roadmaps/phase3_accuracy_push.md) §W13 | 2026-05-27 |
| 3D.1 | Story | Decompose Phase 3D | `backlog` | [`3D.1`](tasks/specs/3D.1_decompose_phase_3d.md) | 2026-05-27 |

## Parallel-safety matrix

Two stories can be worked **concurrently** (in separate worktrees, see
[`workflows/session_protocol.md`](workflows/session_protocol.md) §4)
only if neither writes to a path in the other's `file_scope`.

| Story | parallel_safe_with | Hard conflicts (must serialize) |
|---|---|---|
| 3A.1 | 3B.1, 3C.1, 3D.1 | none |
| 3A.2 | 3B.1, 3C.1, 3D.1 | 3A.3 (consumes the module 3A.2 ships), 3A.4 (chains through 3A.3) |
| 3A.3 | 3B.1, 3C.1, 3D.1 | 3A.2 (must be `done` first), 3A.4 (chains through 3A.3) |
| 3A.4 | 3B.1, 3C.1, 3D.1 | 3A.3 (must be `done` first; reads its artifact bundles) |
| 3B.1 | 3A.1, 3A.2, 3A.3, 3A.4, 3C.1, 3D.1 | none |
| 3C.1 | 3A.*, 3B.1, 3D.1 | none |
| 3D.1 | 3A.*, 3B.1, 3C.1 | none |

> All four decomposition stories touch `BOARD.md`, `PHASE3_INDEX.md`,
> the roadmap, and `progress_log.md`. They are listed as
> `parallel_safe_with` because each touches **disjoint sections** of
> those shared files; if two sessions actually race them, expect a
> trivial merge on the shared files. Implementation stories under the
> epics (3A.2+, 3B.2+, 3C.2+, 3D.2+) must declare their own
> `parallel_safe_with` once written, based on their actual
> `file_scope`.

## Per-story last checkpoint (mirror of spec's tail block)

> Updated by the executing session as part of every checkpoint. A fresh
> session reads only the latest entry for the story it is picking up.

### 3A.1 — Decompose Phase 3A

- **2026-05-28** decomposition executed. Epic 3A → `in_progress`;
  3A.1 → `in_review`. Three atomic stories registered: `3A.2` (local
  Fourier module), `3A.3` (remote decoder-only retrain), `3A.4`
  (local eval + roadmap addendum). Each story is atomic and does
  not span local + remote.
- **2026-05-27 (rev. 2)** scope tightened: RBF-residual and SIREN
  dropped. 3A is now strictly a Fourier-vs-raw coordinate ablation
  on the frozen 2D.7 encoder, decoder-only retrain. See ADR 0004.
- **Next concrete action:** human reviews the three new specs and
  promotes `3A.2` from `backlog → todo` to start implementation.
- **Open blocker:** none.

### 3A.2 — Local: Fourier-feature module + `coord_encoding` flag

- **2026-05-28** registered via 3A.1 decomposition (not yet
  executed). Scope: new `features/coord_encoding.py` (raw + Fourier
  encodings, builder), `coord_encoding=…` kwarg on
  `ConditionalSurfaceModel`, unit + wiring tests. No training.
- **2026-05-28 (impl)** landed `features/coord_encoding.py`
  (`RawCoordEncoding`, `FourierCoordEncoding` with log-spaced bands
  per Tancik et al., `build_coord_encoding`); added
  `coord_encoding=…` kwarg + `coord_encoding_cfg` persistence on
  `ConditionalSurfaceModel` with `decoder coord_dim` resolved from
  `encoded_dim`. Tests: 12 unit + 6 integration, all green; existing
  conditional-surface suite passes unchanged (26 selected, 0
  regression). Raw-default path is bit-for-bit equal to the
  pre-change model. PMR gate dry-run passes. No training; no
  checkpoint mutated.
- **Next concrete action:** flip BOARD row to `in_review`; human
  reviews diff and promotes `3A.3` (remote decoder-only retrain).
- **Open blocker:** none.

### 3A.3 — Remote: decoder-only retrain on frozen 2D.7 encoder

- **2026-05-28 (executed)** both runs shipped on RunPod. Bundles:
  `artifacts/runs/3A/{fourier,raw}/` — `manifest.json` (committed),
  `training_curves.csv`, `predictions_val.parquet`,
  `predictions_test.parquet`, `checkpoints/best_conditional.pt`
  (last three local-only, gitignored). Manifests record
  `encoder_weights_equal_source: true` and source SHA-256
  `6003006a00e9f6e9f3a18d00bcca857568315f330716a5724985c428622da41e`.
- **Headline test MAE (held for 3A.4 to write up):** Fourier
  0.07940 (epochs_completed=19), Raw 0.07641 (epochs_completed=40).
  2D.7 Gaussian baseline reference (full retrain): 0.07873.
- **Code shipped:** `train_conditional` gained `coord_encoding`
  passthrough + `freeze_encoder` + `encoder_init_from`;
  `scripts/run_3a_decoder_only.{py,sh}` + two YAML configs +
  `tests/integration/test_3a_decoder_only_wiring.py`.
- **Next concrete action:** human reviews diff; promotes 3A.4
  (local eval + W10 addendum). Pod can be terminated.
- **Open blocker:** none.

### 3A.4 — Local: eval + journal + roadmap addendum

- **2026-05-28** registered via 3A.1 decomposition (not yet
  executed). Scope: `scripts/run_3a_eval.py` (read-only on 3A.3
  bundles), three result CSVs per variant + paired comparison,
  experiment-journal entry, "3A closing addendum" under § W10 of
  the Phase 3 roadmap.
- **Next concrete action:** 3A.3 must close first; then human
  promotes `3A.4` and a fresh local session executes per spec.
- **Open blocker:** 3A.3.

### 3B.1 — Decompose Phase 3B

- **2026-05-27 (rev. 2)** decoupled from 3A. 3B runs in parallel
  with 3A; coordinate-encoding default is Fourier-on per ADR 0004
  if 3A has not measured the delta yet.
- **Next concrete action:** human promotes `3B.1` from `backlog →
  todo` whenever ready.
- **Open blocker:** none (parallel-safe with 3A).

### 3C.1 — Decompose Phase 3C

- **2026-05-27** initial scaffold (not yet executed).
- **Next concrete action:** gate on 3A and 3B closing, then human
  promotes `3C.1` to `todo`.
- **Open blocker:** depends on 3A and 3B closing.

### 3D.1 — Decompose Phase 3D

- **2026-05-27** initial scaffold (not yet executed).
- **Next concrete action:** gate on 3A / 3B / 3C closing, then human
  promotes `3D.1` to `todo`.
- **Open blocker:** depends on all three sibling epics closing.

## Resume snippet (copy-paste into a fresh session)

```text
You are resuming Phase 3 of the Neural IV Surface Inference project.

Start of session ritual:
1. Read docs/PHASE3_INDEX.md (this file) end-to-end.
2. Identify the active story from the "Live status" table — the one
   marked in_progress, or the top todo if none is in_progress.
3. Open ONLY its spec in docs/tasks/specs/<ID>_*.md.
4. Read ONLY the files its "Context files to read first" section
   names. Do not load the whole repo.
5. Read the spec's "Last checkpoint" block — that is the exact next
   concrete action.
6. Restate scope, acceptance criteria, non-goals, and your proposed
   step plan. Wait for human approval before any edit.
7. Stay inside the spec's file_scope. If a new path is needed, stop
   and either expand file_scope (re-check parallel_safe_with) or
   split the work into a new backlog story.
```

## How to keep this index honest

These four rules keep the index from going stale:

1. **Every status change on `BOARD.md` for a 3X.* row is mirrored
   here in the same commit.** The PMR pre-push routine flags
   asymmetric updates.
2. **Every checkpoint a session writes into a spec's "Last checkpoint"
   block is also added here under "Per-story last checkpoint"** in
   abbreviated form. The spec is the long form; this index is the
   short form.
3. **When a new 3X.* story spec is added, the decomposition story
   that created it also adds the row here and the empty checkpoint
   block.** This is part of every 3X.1 decomposition story's
   acceptance criteria.
4. **When an epic closes, mark its row `done` here and add a one-line
   results summary** (e.g. "→ test MAE 0.0712, coverage 0.901,
   evidence: results/3/.../metrics_summary.csv"). Do not delete rows.
