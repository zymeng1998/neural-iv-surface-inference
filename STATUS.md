# STATUS — Phase 4 4A.2 (residual-target builder) implemented; next = 4A.3

**Updated:** 2026-06-18 (late)
**Branch:** main
**Mode:** local, CPU only — no GPU, no model/eval run

## Where things stand

Phase 4 (epic `4A`) is in progress. 4A.1 (decompose) `done` and on origin
(`97d9299`). **4A.2 (residual-target builder + `target_mode` flag) is
implemented and `in_review`.** Next: 4A.3 (remote CPU — build the full OTM
residual dataset), then the GPU training chain (4A.4/4A.5, Pod-gated).

origin/main is at **97d9299**. The 4A.2 commit below is local, NOT pushed.

## What was just completed (4A.2 — one local commit)

- `src/.../data/residual_targets.py` (new): per-date RBF at query coords
  (reuses the 3X.6 baseline verbatim; lazy import to dodge a circular
  import) + `residual_target = iv_clean − rbf_pred` + `add_rbf_pred_column`.
- `ConditionalIVSurfaceDataset` gains `target_mode ∈ {absolute, residual}`
  (default `absolute` = byte-identical legacy; verified by regression).
- `scripts/build_residual_targets.py` (new): materialiser (full-fold run is
  4A.3); fails loudly on non-finite RBF.
- `tests/test_residual_targets.py` + loader regression → **20 passed**;
  builder smoke exit 0. `data_lineage.md` updated.

## Push readiness (4A.2 — designed zero-waiver, NOT pushed)

- **PMR (live):** `src/`+`scripts/`+`tests/` touched; data_lineage +
  progress_log updated → PASS.
- **DEP:** 4A.2 dep 4A.1 = `done` → PASS.
- **SCOPE:** only active spec 4A.2; all touched files in its `file_scope`
  (`STATUS.md` added) → PASS.
- **Do NOT set `WAIVE_DEPS` / `WAIVE_SCOPE`.** Single self-contained commit.

## Next concrete action

- **Verify all three gates standalone, stop before push for review** (4A.2
  is real `src/` code — operator reviews before push).
- After approval: push (clean env); promote 4A.2 → `done`.
- Then **4A.3** (remote CPU): build the full residual dataset on
  `random40_noiselow_otm`. GPU stories (4A.4/4A.5) gated on a Pod go-ahead.
