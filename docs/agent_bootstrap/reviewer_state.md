# Project Memory Reviewer — State

> Human-readable summary of the reviewer system's state.
> Machine-readable counterpart: `reviewer_state.json` (same directory).

---
created_at: 2026-04-02T01:46:00-04:00
last_updated_at: 2026-05-29T07:00:00-04:00
---

## System status

| Field | Value |
|---|---|
| Bootstrap completed | Yes |
| Bootstrap completed at | 2026-04-02T01:34:00-04:00 |
| Baseline review completed | Yes |
| Baseline completed at | 2026-04-02T01:35:00-04:00 |
| Last review run at | 2026-05-29T07:00:00-04:00 |
| Last review run type | ongoing |
| Last processed git HEAD | `99ec6fa` |

## Phase status snapshot (2026-05-29)

| Phase / Epic | Status | Notes |
|---|---|---|
| Phase 1 (structural roadmap) | `done` | Interpolation floor + masked MLP baseline; AV-era rerun refreshed memo (2C.8). |
| Epic 2A — W1 uncertainty evaluation | `done` | Stories 2A.1–2A.5 complete. |
| Epic 2B — W2 sensitivity & structure diagnostics | `done` | Stories 2B.1–2B.5 complete. |
| Epic 2C — W3 conditional neural surface model | `done` | Stories 2C.1–2C.8 complete; Phase B autonomous remote run 2026-05-23. |
| Epic 2D — W4 + W5 uncertainty-aware inference & decision layer | `done` | Stories 2D.1–2D.9 complete (closed 2026-05-25 by 2D.9). |
| Epic 2E — Phase 2 follow-ups (open-ended) | `in_progress` | 2E.1 decomposition `done`; 2E.2 latent-capacity diagnostic `done` (2026-05-27); 2E.3 latent_dim sweep `backlog` awaiting human review of the 2E.2 memo addendum. |
| Epic 3A — Coordinate encoding ablation (raw vs Fourier) | `done` | 3A.1 decomposition `done`; 3A.2 local module + integration test `done`; 3A.3 remote decoder-only retrain on frozen 2D.7 encoder, raw + fourier bundles shipped `done`; 3A.4 closing addendum `done`. **Headline:** Fourier coord-encoding offers no measurable MAE / NLL benefit on the frozen DeepSets encoder; **3B coordinate-encoding default = raw (k, τ)**. The Phase-3 accuracy gap is an architectural locality bottleneck, not an input-representation problem. |
| Epic 3B — Cross-attention decoder (ANP) | `in_progress` | 3B.1 decomposition `done`; 3B.2 local ANP decoder module + `decoder_kind` flag `done` (+ amendment fixing `train_conditional` forwarding); 3B.3 predictor-adapter wiring `done`; 3B.4 remote full AV training across {gaussian, quantile, point} `done` (2026-05-28); 3B.5 K=5 ANP point-head deep ensemble on AV `done` (2026-05-29). 3B.6 calibrator re-fit `backlog`; 3B.7 decision-layer eval `backlog`. **3B.4 headline:** end-to-end ANP gaussian test_MAE 0.0726 vs 3A.3 raw gaussian 0.0764 → ≈5 % MAE reduction (conflates ANP decoder + end-to-end retrain). **3B.5 headline:** ensemble test_MAE 0.0689 vs single-seed 3B.4 point 0.0684 — ensembling did not improve point accuracy (members converged to similar modes); the disagreement signal (mean 0.0121, max 0.723) is the load-bearing deliverable for 3B.6 / 3B.7. |

## Known project-memory artifacts

### Retrospectives

| ID | File | Status |
|---|---|---|
| 0001 | `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` | Exists |

### Decisions (ADRs)

| ID | File | Status |
|---|---|---|
| 0001 | `docs/decisions/0001_remote_dev_stack.md` | Exists |
| 0002 | `docs/decisions/0002_phase1_scope_freeze.md` | Exists |
| 0003 | `docs/decisions/0003_spy_options_data_source_migration.md` | Exists |
| 0004 | `docs/decisions/0004_phase3_accuracy_push_framing.md` | Exists |
| 0005 | `docs/decisions/0005_cross_attention_architecture_choice.md` | Exists |

### Append-only logs

| File | Last entry timestamp | Entry count |
|---|---|---|
| `docs/logs/progress_log.md` | 2026-05-29 | 50 entries |
| `docs/experiments/experiment_journal.md` | 2026-05-29 | 17 entries |

## Bootstrap files

All 14 bootstrap files exist. See `reviewer_state.json` for the full list.

## Notes

- Reviewer state refreshed 2026-05-29 after 3B.5 close-out.
- HEAD advanced from `90c46fa` (2026-05-27, 2E.2 close) to `99ec6fa`
  (2026-05-29, 3B.5 close) across eight commits spanning epic 3A
  closure and the first five 3B stories:
  - `619896a` — 3A.2: coord_encoding module + `ConditionalSurfaceModel`
    flag.
  - `482345c` — 3A.3: decoder-only retrain on frozen 2D.7 encoder;
    Fourier + raw bundles shipped (raw → 3B default).
  - (3A.4 closing addendum landed in the same series; see PHASE3 index.)
  - `2cc186f` — 3B.2: ANP cross-attention decoder module + `decoder_kind`
    flag + unit/smoke/integration tests.
  - `dd48f1b` — 3B.3: predictor-adapter parity (13-line patch).
  - `a601a59` — 3B.4 local prep + **3B.2 amendment**: fixed
    `train_conditional` silently dropping `decoder_kind` / `anp` from
    config; added regression test
    `test_train_conditional_forwards_decoder_kind_and_anp_cfg`. Without
    this fix every 3B.4 / 3B.5 YAML would have trained DeepSets.
  - `536e535` — 3B.4 remote sweep: three ANP heads × 50 epochs on AV,
    manifests committed.
  - `789a56a` — 3B.5 local prep (config + runner clone of 2D.8's).
  - `99ec6fa` — 3B.5 remote sweep: K=5 ANP point ensemble on AV,
    manifest + members.json + per-member training curves committed.
- **Headline 3B.4 finding:** end-to-end ANP gaussian on AV gives
  test_MAE 0.0726 vs 3A.3 raw gaussian (frozen 2D.7 encoder +
  DeepSets) test_MAE 0.0764 → ≈ 5 % MAE reduction. The number
  conflates "ANP decoder" with "end-to-end retrain"; clean
  ablation is deferred to 3B.7.
- **Headline 3B.5 finding:** five-member ANP point ensemble does
  **not** improve point accuracy over the single-seed baseline
  (ensemble test_MAE 0.0689 vs single-seed 0.0684, +0.7 %; within
  noise). Members converged to tightly-clustered val losses (range
  0.00794–0.00842). The disagreement signal is the load-bearing
  deliverable for 3B.6 (calibrator re-fit) and 3B.7 (decision-layer
  scoring).
- 3B remains `in_progress`; next stories are 3B.6 (local-only,
  calibrator re-fit on ANP val predictions) and 3B.7 (local-only,
  full decision-layer evaluation vs Phase 2D baselines).
- HEAD advanced from `345dffb` (2026-05-25, 2D.9 close) to `90c46fa`
  (2026-05-27, 2E.2 close) across two 2E.2 commits:
  - `e5022de` — local diagnostic modules (`effective_rank`,
    `contribution`, `latent_probe`), CLI runner
    `scripts/run_latent_diagnostic.py`, and 18 unit tests; 2E.1
    decomposition story; 2E.3 latent_dim-sweep story stub at `backlog`.
  - `90c46fa` — Pod run on the 2D.7 gaussian checkpoint over val
    (300 / 693 dates sampled, seed=42), 12-file artifact bundle under
    `artifacts/diagnostics/2E2/prod_2d7_gaussian/`, experiment-journal
    entry, `phase2_result_memo.md` follow-up addendum, plus two
    bug-fix backports (`latent_probe` global-Q padding +
    `run_latent_diagnostic` memory-aware `--max-dates` flag).
- **Headline 2E.2 finding:** the 64-dim latent in the production 2D.7
  gaussian model is dramatically over-parameterized.
  `eff_rank_entropy = 3.97 / 64`; `stable_rank = 1.97`;
  `dead_pcs = 52 / 64`; `k95 = 5`, `k99 = 7`. Top-2 PCs hold 78 % of
  latent variance and 51 % of prediction quality (per-PC ΔNLL); top-8
  PC reconstruction recovers val Gaussian NLL to within 0.2 %.
- Memo addendum ends with the explicit **shrink-sweep** recommendation
  for 2E.3 — widths `{2, 4, 8, 16, 32, 64}`. 2E.3 stays `backlog`
  awaiting the human's accept.
- All four Phase 2 epics 2A–2D remain `done`; epic 2E is `in_progress`
  (open-ended follow-up backlog).
- This file and `reviewer_state.json` should be updated at the end of
  every reviewer run that produces material changes.
