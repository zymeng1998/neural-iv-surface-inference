# Phase 4 — Single Entry Point for Fresh Sessions

---
created_at: 2026-06-18T00:00:00-04:00
last_updated_at: 2026-06-18T22:00:00-04:00
---

> Read this first if you are picking up Phase 4 work cold. Mirrors the
> Phase 4 subset of [`docs/tasks/BOARD.md`](tasks/BOARD.md) plus per-story
> checkpoints. For the full plan see
> [`roadmaps/phase4_hybrid_residual.md`](roadmaps/phase4_hybrid_residual.md)
> and [ADR 0010](decisions/0010_rbf_prior_residual_hybrid.md).

## 30-second orientation

- **Why Phase 4:** Phase 3 closed **negative on accuracy** — no pure
  conditional-neural variant beat RBF on the clean OTM substrate (best head
  +61 %; ADR 0009). Phase 4 stops fighting RBF and **stands on it**.
- **The bet:** `σ̂(k,τ) = RBF_t(k,τ) + f_θ(k,τ | context_t)` — RBF carries
  the local interpolation it already wins; the neural model learns the
  **residual** + the calibrated reliability/abstention layer RBF lacks
  (ADR 0010).
- **Substrate:** `spy_phase1_random40_noiselow_otm`. **Floor to beat:** RBF
  test MAE **0.00613** (3X.6).
- **Success bar (operator-set):** any *statistically meaningful* gain over
  RBF (paired bootstrap 95 % CI on per-query MAE delta excludes 0) **plus**
  reliability preserved (coverage ±2 pp of 0.90; hi-conf < no-abstention;
  flags not worse than 3X.12). **Negative branch is acceptable:** if no
  significant gain, ship the calibrated reliability layer on RBF.
- **Status:** epic 4A `in_progress`; 4A.1 `done`; **4A.2 `in_review`**
  (residual-target builder + `target_mode` flag, 20 tests green); 4A.3–4A.8
  `backlog`. **Next action: promote 4A.2 → `done`, then run 4A.3** (remote
  CPU: build the full OTM residual dataset). GPU stories (4A.4/4A.5) gated on
  operator Pod go-ahead.

## Live status (sync with BOARD.md)

| ID | Type | Title | Status |
|---|---|---|---|
| 4A | Epic | RBF-prior residual hybrid | `in_progress` |
| 4A.1 | Story | Decompose Phase 4A + ADR 0010 | `done` |
| 4A.2 | Story | Residual-target builder + `target_mode` flag (local) | `in_review` |
| 4A.3 | Story | Build full residual dataset on OTM (remote CPU) | `backlog` |
| 4A.4 | Story | Train residual hybrid, 3 heads (remote GPU) | `backlog` |
| 4A.5 | Story | K=5 residual ensemble (remote GPU) | `backlog` |
| 4A.6 | Story | Calibrator re-fit on hybrid val (local) | `backlog` |
| 4A.7 | Story | Decision-layer eval + bootstrap CI vs RBF | `backlog` |
| 4A.8 | Story | Comparison + closing memo + ADR 0010 Outcome | `backlog` |

Chain: `4A.1 → 4A.2 → 4A.3 → 4A.4 → 4A.5 → 4A.6 → 4A.7 → 4A.8`. 4A.4+4A.5
share one Pod-GPU window; 4A.3 is a CPU pre-step.

## Per-story last checkpoint

- **4A.1 — decompose (`done`, 2026-06-18).** Roadmap + ADR 0010 (Proposed)
  + 4A.2–4A.8 specs authored; epic entered. Backbone fork (ANP-residual
  default vs MLP-residual ablation) open for the 4A.4 review.
- **4A.2 — implemented (`in_review`, 2026-06-18).** `data/residual_targets.py`
  (reuses the 3X.6 RBF verbatim) + `target_mode ∈ {absolute, residual}` loader
  flag (absolute byte-identical) + `scripts/build_residual_targets.py` + tests
  (20 passed). **Next: operator promotes 4A.2 → `done`, then 4A.3** (remote
  CPU: build the full OTM residual dataset).
- **4A.3–4A.8** — registered (`backlog`); not yet executed. See each spec.
  GPU stories (4A.4/4A.5) gated on operator Pod go-ahead.
