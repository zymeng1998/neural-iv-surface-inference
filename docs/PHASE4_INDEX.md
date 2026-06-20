# Phase 4 — Single Entry Point for Fresh Sessions

---
created_at: 2026-06-18T00:00:00-04:00
last_updated_at: 2026-06-20T13:30:00-04:00
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
- **Status:** epic 4A `in_progress`; 4A.1–4A.4 `done`; **4A.5 `in_review`**
  (K=5 residual point-head ensemble: ties RBF at 0.006141, like the 4A.4
  single point; disagreement mean 0.000209 for the calibrator). **The Phase 4
  accuracy win is the 4A.4 gaussian (0.006006) / quantile (0.005906) heads —
  below the RBF floor 0.006132; significance adjudicated by 4A.7.** 4A.5 was
  the last GPU run; 4A.6/4A.7/4A.8 are local. 4A.6–4A.8 `backlog`. **Next
  action: promote 4A.5 → `done`, then 4A.6** (calibrator re-fit, local).

## Live status (sync with BOARD.md)

| ID | Type | Title | Status |
|---|---|---|---|
| 4A | Epic | RBF-prior residual hybrid | `in_progress` |
| 4A.1 | Story | Decompose Phase 4A + ADR 0010 | `done` |
| 4A.2 | Story | Residual-target builder + `target_mode` flag (local) | `done` |
| 4A.3 | Story | Build full residual dataset on OTM (remote CPU) | `done` |
| 4A.4 | Story | Train residual hybrid, 3 heads (remote GPU) — **hybrid BEATS RBF (gaussian/quantile below floor; point ties)** | `done` |
| 4A.5 | Story | K=5 residual ensemble (remote GPU) — ties RBF (0.006141); disagreement 0.000209 | `done` |
| 4A.6 | Story | Calibrator re-fit on hybrid val (local) — fitted; test coverage 0.9181 | `in_review` |
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
- **4A.3 — built (`in_review`, 2026-06-19).** Full OTM residual dataset on
  the RunPod CPU pod: 10.53M rows, 0 non-finite, val/test mean|residual| ==
  3X.6 RBF MAE (0.006151 / 0.006132). Parquet (237 MB) on persistent
  `/workspace`; `artifacts/runs/4A3/{manifest.json,residual_stats.csv}`
  committed. **Next: promote 4A.3 → `done`, then 4A.4** (GPU, Pod-gated).
- **4A.4 — trained (`in_review`, 2026-06-20).** ANP-residual hybrid, 3 heads
  on OTM. Hybrid test MAE (vs iv_clean): gaussian 0.006006 (Δ −0.000126) /
  quantile 0.005906 (Δ −0.000225) / point 0.006138 (ties); RBF floor
  0.006132. All ≪ 3X.9. qmono ok. Backbone fork resolved: ANP-residual
  confirmed (no MLP ablation). Point-estimate gain → 4A.7 tests significance.
- **4A.5 — trained (`in_review`, 2026-06-20).** K=5 residual point-head
  ensemble: test MAE 0.006141 (ties RBF, like the 4A.4 single point);
  disagreement mean 0.000209 (all positive) — the uncertainty signal for
  4A.6/4A.7. Last GPU run; pod terminable.
- **4A.6 — fitted (`in_review`, 2026-06-20).** Calibrator re-fit on the
  hybrid val predictions (recipe unchanged from 3X.11): T=1.147,
  ens_scale=438; held-out test coverage 0.9181 (within ±2 pp). 4 tests
  green. Prediction CSVs pulled local (gitignored) from a fresh
  volume-mounted pod (213.173.110.22) after the prior pod was terminated —
  **4A.7/4A.8 now run fully locally, no pod needed.**
- **4A.7–4A.8** — registered (`backlog`); local. 4A.7 decision-layer +
  **bootstrap CI (the bar)** → 4A.8 close.
