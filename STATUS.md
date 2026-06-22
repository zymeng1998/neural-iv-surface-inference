# STATUS — Phase 4 closed; 4B.5 gate = AMBIGUOUS; operator decides 4B.7 vs 4B.6

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

## Current task — 4B.5 done (gate = AMBIGUOUS); operator decides 4B.7 vs 4B.6

**4B.1**→**4B.2**→**4B.3**→**4B.4**→**4B.5** all `done`. Epic 4B `in_progress`.
**4B.5** (decision-bearing, local) computed the edge-vs-sparsity trajectory +
date-clustered bootstrap CIs + ADR 0011 gate verdict.

### Gate verdict: `accuracy_survives = ambiguous` → triggers 4B.7

- Relative edge (RBF−hybrid)/RBF, overall MAE: `fewer_quotes` grows 2.05→4.32 %
  (sub-5 % bar); `missing_maturities` 2.05→1.56 % (peaks ~3 %); `thin_wings`
  collapses 2.05→0.31 %; `combined` collapses 2.05→0.46 %. **All deltas
  significant**, but the relative edge does **not** survive the wing/maturity
  stresses — it grows only under benign random thinning.
- Wing collapse is confounded by the **eval-time OOD caveat** (full-context
  checkpoint scored on wing-less context); only a fair retrain (4B.7) can
  disambiguate → rule lands on `ambiguous`.
- Secondary: calibrated coverage collapses 0.962 → 0.35–0.39 (wings) — Phase 5
  reliability motivation.
- Dense provenance exact (rbf 0.006132 / hybrid 0.006006 / Δ −0.000126 / CI
  [−1.44e-4,−1.06e-4] all == 4A.7).
- Artifacts: `results/4/spy_phase1_random40_noiselow_otm/4b_sweep/{trajectory.csv,
  trajectory_wide.md, gate_verdict.json}` + `artifacts/runs/4B5/manifest.json`.

### Pod status

**Pod terminable now.** 4B.7 (if chosen) needs a **GPU** pod (per-regime retrain),
not this CPU pod — so terminate the current pod regardless; rent GPU only if you
choose 4B.7. Old pod: `213.173.111.74:34574`, key `~/.ssh/id_ed25519_runpod`.

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

## Next concrete action — operator decision (GPU spend)

The gate is `ambiguous`. Two principled paths (the gate is informational; the
GPU spend is the operator's call):

- **(a) Run 4B.7** (remote **GPU**, conditional escalation): per-regime fair
  retrain to test whether the wing-collapse is a fundamental ceiling or just the
  eval-time OOD artifact. Resolves the ambiguity; costs a GPU pod.
- **(b) Cancel 4B.7, go to 4B.6** (local close): accept the
  ambiguous-leaning-negative read (relative edge doesn't survive the stresses
  that matter; even a fair retrain's upside looks economically marginal at ~2–4 %
  on a ×14-inflated base), mark 4B.7 `cancelled`, write the Phase 4B close +
  ADR 0011 Outcome, and pivot to **Phase 5 reliability-first** (which the
  coverage collapse independently motivates).
