# Project Memory Reviewer — State

> Human-readable summary of the reviewer system's state.
> Machine-readable counterpart: `reviewer_state.json` (same directory).

---
created_at: 2026-04-02T01:46:00-04:00
last_updated_at: 2026-06-02T14:30:00-04:00
---

## System status

| Field | Value |
|---|---|
| Bootstrap completed | Yes |
| Bootstrap completed at | 2026-04-02T01:34:00-04:00 |
| Baseline review completed | Yes |
| Baseline completed at | 2026-04-02T01:35:00-04:00 |
| Last review run at | 2026-06-02T14:30:00-04:00 |
| Last review run type | manual-doc-sync (see caveat in Notes) |
| Last processed git HEAD | `7788f12` |

> **Caveat (2026-06-02):** this refresh is a **manual documentation
> sync** performed by the session that closed epic 3X — not a
> diff-driven, line-level reviewer pass. `last_processed_git_head` is
> advanced to `7788f12` so the status snapshot is current, but the
> `99ec6fa..7788f12` implementation commits (epic 3X + epic M1) were
> **not** code-reviewed here. If a substantive code review is wanted,
> treat that range as unreviewed.

## Phase status snapshot (2026-06-02)

| Phase / Epic | Status | Notes |
|---|---|---|
| Phase 1 (structural roadmap) | `done` | Interpolation floor + masked MLP baseline; AV-era rerun refreshed memo (2C.8). |
| Epic 2A — W1 uncertainty evaluation | `done` | Stories 2A.1–2A.5 complete. |
| Epic 2B — W2 sensitivity & structure diagnostics | `done` | Stories 2B.1–2B.5 complete. |
| Epic 2C — W3 conditional neural surface model | `done` | Stories 2C.1–2C.8 complete. |
| Epic 2D — W4 + W5 uncertainty-aware inference & decision layer | `done` | Stories 2D.1–2D.10 complete (closed 2026-05-25). |
| Epic 2E — Phase 2 follow-ups (open-ended) | `in_progress` | 2E.1 `done`; 2E.2 latent-capacity diagnostic `done` (over-parameterized 64-dim latent: eff_rank 3.97, k95=5); **2E.3 latent_dim sweep `cancelled`** (scope-killed by 2E.2 findings — see board). |
| Epic 3A — Coordinate encoding ablation (raw vs Fourier) | `done` | Raw beats Fourier on the frozen 2D.7 encoder (0.0760 vs 0.0790 full-fold test MAE); 3B default = raw `(k, τ)`. The Phase-3 gap is an architectural locality bottleneck, not input representation. |
| Epic 3B — Cross-attention decoder (ANP) | `done` | 3B.1–3B.7 complete. **Bar NOT met (dirty):** best ANP point head +2.7 % vs RBF (0.0680 vs 0.0662); calibrated gaussian +9–11 %. Reliability holds (cov 0.9149; hi-conf MAE 0.0542 < no-abst 0.0813). Strongest conditional model so far; decoder-architecture iteration plateaued. |
| Epic M1 — Multi-agent collaboration infrastructure | `done` | Shipped under [ADR 0007](../decisions/0007_multi_agent_handoff.md) + roadmap `meta1_agent_collaboration.md`. Git-hook gates (`check_story_dependencies` / `check_file_scope` / `check_commit_trailer` + `install_hooks.sh`) enforce the AGENTS.md hard rules. |
| Epic 3X — Data correction (OTM-restricted surface) | `done` | **CLOSED 2026-06-02.** Built + audited OTM substrate (PASS 12/12, 93.61 %→0 % dup, 0 % twin leakage); restated the full model-family ladder on `spy_phase1_random40_noiselow_otm`. **Verdict:** RBF still wins, gap WIDENED (+2.7 %→+61 % best ANP head, +90 % calibrated; bar still NOT met); DeepSets→ANP architecture story survives (no Q5 reopen). ADR 0006 → Implemented. See [methodology progression](../research/duplicate_coordinate_methodology_progression.md). |
| Epic 3C — Feature & inductive-bias expansion | `backlog` | **NEXT.** Reopens on the clean OTM substrate; promote 3C.1 (decompose) to `todo`. Honest gap-to-close is now +61 % (best head) / +90 % (production), not the dirty +2.7 %. |
| Epic 3D — Phase 3 closing memo + re-evaluation | `backlog` | Must fold the OTM-clean re-statement in alongside the original dirty numbers. |

## Known project-memory artifacts

### Retrospectives

| ID | File | Status |
|---|---|---|
| 0001 | `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md` | Exists |
| 0002 | `docs/retrospectives/0002_call_put_duplicate_coordinate_discovery.md` | Exists |

### Decisions (ADRs)

| ID | File | Status |
|---|---|---|
| 0001 | `docs/decisions/0001_remote_dev_stack.md` | Exists |
| 0002 | `docs/decisions/0002_phase1_scope_freeze.md` | Exists |
| 0003 | `docs/decisions/0003_spy_options_data_source_migration.md` | Exists |
| 0004 | `docs/decisions/0004_phase3_accuracy_push_framing.md` | Exists |
| 0005 | `docs/decisions/0005_cross_attention_architecture_choice.md` | Exists |
| 0006 | `docs/decisions/0006_duplicate_coordinate_data_correction.md` | **Implemented** (2026-06-02) |
| 0007 | `docs/decisions/0007_multi_agent_handoff.md` | Exists |

### Append-only logs

| File | Last entry timestamp | Entry count |
|---|---|---|
| `docs/logs/progress_log.md` | 2026-06-02 | 90 entries |
| `docs/experiments/experiment_journal.md` | 2026-06-02 | 31 entries |

## Bootstrap files

All 14 bootstrap files exist. See `reviewer_state.json` for the full list.

## Notes

- **This refresh (2026-06-02) is a manual documentation sync, not a
  code-review pass.** See the System-status caveat above. HEAD advanced
  `99ec6fa` → `7788f12`.
- **Epic M1 (multi-agent collaboration infrastructure)** shipped in this
  window under ADR 0007: pre-push git-hook gates enforcing the AGENTS.md
  hard rules (story-dependency check, file-scope check, commit-trailer
  check, `install_hooks.sh`). Epic closed after review.
- **Epic 3X (data correction)** opened on the 2026-05-29
  duplicate-coordinate audit (ADR 0006 + retrospective 0002): 93.61 % of
  strict-table rows are call-put paired duplicates (median in-group IV
  range 0.049) violating the single-valued-function assumption, and
  37.46 % of held-out rows in `random40_noiselow` have an exact-twin
  leak. Correction = industry-standard OTM-restricted surface + opt-in
  (default-off) paired-coordinate masking.
- **3X ladder (matched `random40_noiselow_otm`, test MAE):** RBF
  **0.00613** → MLP 0.03006 → DeepSets (best quantile 0.01418, ensemble
  0.01594) → ANP (point **0.00987** / quantile 0.01175 / gaussian
  0.01440, ensemble 0.01220) → calibrated decision-layer 0.01162. Every
  family improves 3–11× over its dirty counterpart; RBF improves the
  most (10.8×) because it was the model most penalised by the dirty
  two-valued-target confound.
- **Headline verdict (3X.14):** RBF still wins and the neural-vs-RBF gap
  *widened* on clean data — best ANP head goes from +2.7 % (dirty) to
  **+61 %** vs RBF (calibrated production +90 %); Phase 3 bar still NOT
  met, by a wider margin. The dirty benchmark had been flattering ANP.
  The **DeepSets→ANP architecture story survives** (ANP beats DeepSets
  at every matched head, 1.06–1.77×), so the ADR 0006 Q5 reopen trigger
  did not fire and Phase 2 is not reopened. ADR 0006 → Implemented.
- **No-overclaim guardrail:** the verdict is scoped to the matched
  `random40_noiselow_otm` substrate only; all-11-variant robustness
  retraining is deferred future work.
- **Reliability on OTM (3X.12):** coverage@0.90 = 0.9295, hi-conf MAE
  0.00835 < no-abstention 0.01162; floor preserved, with larger val→test
  calibration drift (3X.11) on the tighter OTM error scale.
- **Next:** epic **3C** reopens on the clean OTM substrate — promote
  **3C.1** (decompose Phase 3C) to `todo`. Epic **2E** remains
  `in_progress` (2E.3 cancelled).
- This file and `reviewer_state.json` should be updated at the end of
  every reviewer run that produces material changes.
