# STATUS — Phase 4 4A.5 ensemble done; next = 4A.6 (local, no GPU)

**Updated:** 2026-06-20
**Branch:** main
**Mode:** GPU runs complete for Phase 4; remaining stories are local/CPU.

## Where things stand

Phase 4 (epic `4A`) in progress. 4A.1–4A.4 `done` (origin `01d3b9b`).
**4A.5 (K=5 residual ensemble) done on the GPU pod and `in_review`.** 4A.5
was the **last GPU run**; 4A.6/4A.7/4A.8 are local. **The GPU pod can be
terminated now.**

origin/main is at **01d3b9b**. The 4A.5 commit below is local, NOT pushed.

## Phase 4 accuracy picture (significance pending 4A.7)

Hybrid test MAE vs `iv_clean`, RBF floor **0.006132**:

| variant | test MAE | Δ vs RBF |
|---|---:|---:|
| 4A.4 gaussian | 0.006006 | −0.000126 |
| **4A.4 quantile** | **0.005906** | **−0.000225** |
| 4A.4 point | 0.006138 | +0.000006 (ties) |
| 4A.5 ensemble (point) | 0.006141 | +0.000009 (ties) |

**The win is the gaussian/quantile heads (below the floor).** Point head
(single + ensemble) ties. The ensemble's role is the disagreement signal
(mean 0.000209) for the calibrator/decision-layer. **4A.7's bootstrap CI
decides whether the gaussian/quantile beat is statistically significant —
that is the formal Phase 4 bar.**

## Push readiness (4A.5 — designed zero-waiver, NOT pushed)

- **PMR (live):** `configs/`+`scripts/`+`artifacts/runs/4A5/`; journal +
  progress_log updated → PASS.
- **DEP:** 4A.5 dep 4A.4 = `done` → PASS.
- **SCOPE:** only active spec 4A.5; touched files in its `file_scope`
  (`STATUS.md` added) → PASS. **No `WAIVE_*`.**

## Next concrete action

- **Verify gates standalone, stop before push for review.**
- After approval: push; promote 4A.5 → `done`.
- **4A.6** (local, no GPU): re-fit the 3X.11 calibrator on the hybrid val
  predictions (4A.4 gaussian/quantile + 4A.5 disagreement) →
  `artifacts/calibration/4A6_hybrid.json`. Then **4A.7** (decision-layer +
  bootstrap CI — the bar) → **4A.8** close. The val/test prediction CSVs are
  on the persistent `/workspace` volume (pull to local for 4A.6/4A.7).
