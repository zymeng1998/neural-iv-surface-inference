# Phase 4 — Single Entry Point for Fresh Sessions

---
created_at: 2026-06-18T00:00:00-04:00
last_updated_at: 2026-06-20T17:00:00-04:00
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
- **HEADLINE (2026-06-20): Phase 4 accuracy bar MET.** The RBF-prior
  residual hybrid (gaussian head, calibrated) is **statistically
  significantly more accurate than RBF** on clean OTM (test MAE 0.006006 vs
  0.006132; paired date-clustered 95% CI [−0.000144, −0.000106]) — the first
  neural-based predictor to beat RBF here. Reliability holds in direction
  (hi-conf < no-abstention); coverage is conservative (over-covers iv_clean
  — a calibrator-refit follow-up). 4A.8 writes the close + ADR 0010 Outcome.
- **Status:** epic 4A `in_progress`; 4A.1–4A.7 `done` (**4A.7 accuracy bar
  MET**); 4A.8 `in_progress`. Accuracy win: 4A.4 gaussian (0.006006,
  production head) / quantile (0.005906) below the RBF floor 0.006132; 4A.7's
  bootstrap CI confirms the gaussian gain is significant. **Next action:
  4A.8** (Phase 4 close + ADR 0010 Outcome). GPU runs are all complete;
  4A.8 is local.

## Live status (sync with BOARD.md)

| ID | Type | Title | Status |
|---|---|---|---|
| 4A | Epic | RBF-prior residual hybrid | `in_progress` |
| 4A.1 | Story | Decompose Phase 4A + ADR 0010 | `done` |
| 4A.2 | Story | Residual-target builder + `target_mode` flag (local) | `done` |
| 4A.3 | Story | Build full residual dataset on OTM (remote CPU) | `done` |
| 4A.4 | Story | Train residual hybrid, 3 heads (remote GPU) — **hybrid BEATS RBF (gaussian/quantile below floor; point ties)** | `done` |
| 4A.5 | Story | K=5 residual ensemble (remote GPU) — ties RBF (0.006141); disagreement 0.000209 | `done` |
| 4A.6 | Story | Calibrator re-fit on hybrid val (local) — fitted; test coverage 0.9181 | `done` |
| 4A.7 | Story | Decision-layer + bootstrap CI vs RBF (the bar) — **MET: hybrid significantly beats RBF (95% CI [−0.000144,−0.000106])** | `done` |
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
- **4A.7 — executed (`done`, 2026-06-20). PHASE 4 ACCURACY BAR MET.**
  Date-clustered bootstrap CI: gaussian hybrid 0.006006 vs RBF 0.006132,
  Δ −0.000126, **95% CI [−0.000144, −0.000106]** (entirely < 0) →
  significantly more accurate than RBF (first neural-based predictor to do
  so on clean OTM). Reliability: hi-conf MAE 0.004710 < no-abstention
  0.006006 ✓; coverage vs iv_true 0.9181 (in-band), vs iv_clean 0.962
  (conservative/over-covers — calibrator-refit-on-iv_clean follow-up). Flag
  count needs the model (deferred). Computed fully local from cached
  predictions.
- **4A.8** — in progress (`in_progress`, 2026-06-20); local. Phase 4 closing
  memo + ADR 0010 Outcome + production recommendation; flips epic 4A `done`.
