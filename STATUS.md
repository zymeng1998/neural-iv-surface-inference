# STATUS — Phase 4 closed; 4B.3 swept inputs done; next 4B.4 (remote hybrid)

**Updated:** 2026-06-21
**Branch:** main
**Mode:** local CPU. Docs/planning only — no GPU, no data/code/config/test edits.

## Headline

Phase 4 is **closed positive** (epic 4A `done`; RBF-prior gaussian hybrid is
the adopted production estimator; ADR 0010 Implemented). The win over RBF is
**statistically real but economically small** (≈ 2 % relative MAE on the dense
clean OTM benchmark), so the project will **not** keep grinding that benchmark.
The forward direction is staged and recorded in
[ADR 0011](docs/decisions/0011_forward_strategy_accuracy_reliability_pricing.md)
(**Accepted**):

- **`4B` — sparsity-sweep diagnostic (the decision gate).** RBF vs hybrid over
  increasing sparsity / thin wings / missing maturities. Edge grows under
  sparsity ⇒ accuracy story lives; stays ~0–2 % ⇒ pivot fully to reliability.
- **`5A` — reliability-first surface inference / quote-risk layer.** Primary
  contribution if 4B is negative; worth building regardless.
- **`6A` — structured-product pricing / FCN monetization demo.** Sell-side
  framing; constrained demo, not a full FCN pricer.

## Where things stand

- Forward-strategy planning pass committed to the working tree: ADR 0011 +
  three roadmaps (`phase4b_sparsity_sweep`, `phase5_reliability_first_*`,
  `phase6_structured_product_pricing`) + decomposition specs `4B.1 / 5A.1 /
  6A.1` (all `backlog`) + BOARD rows `4B / 4B.1 / 5A / 5A.1 / 6A / 6A.1` +
  README Next Direction + progress_log entry.
- PMR gate dry-run: **PASS** (0 evidence-source files; docs-only).
- No experiment was run; the Phase 4A result is unchanged.

## Current task — 4B.3 done (remote); next 4B.4

**4B.1** decomposed Phase 4B (`done`, `0b5de5f`). **4B.2** sparsity-sweep harness
(`done`, `27b1c79`). **4B.3** swept eval inputs (`done`) — ran on the RunPod CPU
pod: per-rung RBF re-fit over 4 regimes × 5 rungs on the OTM **test split** (694
dates, 2.769M rows, wall 12 min). Epic 4B `in_progress`.

- **Audit MET:** 0 non-finite; RBF MAE monotone non-decreasing per regime; dense
  rung overall MAE == 4A RBF floor 0.0061318 exactly (|err| 5.2e-18); dense
  query-only == committed `unobserved_mae`. Sparsest-rung degradation: combined
  ×14.4 / thin_wings ×14.1 / missing_maturities ×4.9 / fewer_quotes ×1.5.
- Committed: `artifacts/runs/4B3/{manifest.json, rbf_sweep_stats.csv}`.
  Pod handoff (gitignored): `artifacts/runs/4B3/swept_rbf_test.parquet` (~492 MB).

### Pod status (IMPORTANT)

**Keep the Pod up.** The 492 MB swept parquet lives on `/workspace` and is the
input to 4B.4; 4B.4 also needs the 4A checkpoint volume. Pod: `213.173.111.74:34574`,
key `~/.ssh/id_ed25519_runpod`, interpreter `/workspace/venv-2e2/bin/python`,
repo `/workspace/Neural-IV-Surface-inference` (updated via scp — no GitHub access).

Spec: [`docs/tasks/specs/4B.1_decompose_phase_4b.md`](docs/tasks/specs/4B.1_decompose_phase_4b.md).
Roadmap: [`docs/roadmaps/phase4b_sparsity_sweep.md`](docs/roadmaps/phase4b_sparsity_sweep.md).

## Resolved design forks (operator, 2026-06-21)

1. **Approach → staged (eval-first).** Eval-time sweep on the existing 4A
   checkpoint first; per-regime retraining held as a *conditional* escalation
   (4B.7) only if the read is ambiguous.
2. **Axes → all four regimes:** `fewer_quotes`, `thin_wings`,
   `missing_maturities`, `combined_quotes_wings` (operator-added). Parameters of
   the sweep, not separate stories.
3. **Gate threshold** pre-registered inside 4B.5 before the trajectory is
   computed.

## Child stories authored (all `backlog`)

`4B.2` harness+tests (local CPU) → `4B.3` build swept inputs (remote CPU) →
`4B.4` hybrid forward sweep (remote; needs the 4A checkpoint volume) → `4B.5`
trajectory + CIs + **gate verdict** (local CPU) → `4B.6` close + ADR 0011
Outcome (local CPU). `4B.7` per-regime retrain is **conditional** off 4B.5
(`ambiguous` → run; else `cancelled`) — no Pod spend unless triggered.

## Next concrete action

- **4B.4** (remote): run the 4A RBF-prior hybrid checkpoint **forward** across
  all regime × rung swept inputs (`artifacts/runs/4B3/swept_rbf_test.parquet`)
  → σ̂ = RBF_rung + f_θ(residual). Eval-time only (no retrain); the only
  checkpoint-dependent story — needs the 4A checkpoint volume on the Pod. Then
  **4B.5** (local) computes the edge-vs-sparsity trajectory + date-clustered
  bootstrap CIs and adjudicates the ADR 0011 gate.
