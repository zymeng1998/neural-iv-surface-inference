# STATUS — Phase 4 closed; 4B.4 hybrid forward done; next 4B.5 (local gate)

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

## Current task — 4B.4 done (remote); next 4B.5 (local)

**4B.1**→**4B.2**→**4B.3** all `done`. **4B.4** hybrid forward sweep (`done`) —
ran the 4A gaussian hybrid + 5-seed point ensemble **forward** (no retrain) over
4 regimes × 5 rungs on the OTM test split (694 dates, 2.769M rows, wall 49 min).
Epic 4B `in_progress`.

- **Audit MET:** 0 non-finite; hybrid MAE monotone per regime; dense rung hybrid
  MAE 0.0060062 == 4A hybrid floor exactly (|err| 2.8e-9). Edge vs RBF (preview,
  NOT the gate): hybrid below RBF at every rung; edge widens with sparsity for
  fewer_quotes (+0.000126→+0.000407) and missing_maturities (+0.000126→+0.000471);
  small absolute edge on a catastrophic base for thin_wings / combined.
- Committed: `artifacts/runs/4B4/{manifest.json, hybrid_sweep_stats.csv}`.
  Predictions parquet (~1.38 GB, gitignored) **pulled local** →
  `artifacts/runs/4B4/swept_hybrid_test.parquet` (55,380,420 rows = 2.77M × 20).
- 4B.3 RBF substrate: `artifacts/runs/4B3/rbf_sweep_stats.csv` (committed).

### Pod status

**Pod now terminable** — 4B.5/4B.6 run fully local from the cached prediction
parquet (mirrors 4A.7/4A.8). Keep it only if 4B.5 returns `ambiguous` (→ the
conditional 4B.7 fair-retrain escalation). Pod: `213.173.111.74:34574`, key
`~/.ssh/id_ed25519_runpod`, interpreter `/workspace/venv-2e2/bin/python`.

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

- **4B.5** (local CPU): from the cached prediction parquet
  `artifacts/runs/4B4/swept_hybrid_test.parquet` + the 4B.3 RBF stats, apply the
  4A.6 calibrator, compute the **edge-vs-sparsity trajectory** (hybrid − RBF MAE
  per regime × rung) with **date-clustered bootstrap CIs** (mirror 4A.7), add
  reliability (coverage from calibrated bands), **pre-register the gate
  threshold**, and adjudicate the ADR 0011 verdict (edge grows ⇒ accuracy story
  lives; stays ~0–2 % ⇒ pivot to reliability; `ambiguous` ⇒ conditional 4B.7).
