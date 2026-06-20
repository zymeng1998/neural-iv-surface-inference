# STATUS — Phase 4 4A.6 calibrator fit done; next = 4A.7 (the bar)

**Updated:** 2026-06-20
**Branch:** main
**Mode:** local CPU. No GPU / pod needed for the rest of Phase 4.

## Where things stand

Phase 4 (epic `4A`) in progress. 4A.1–4A.5 `done` (origin `125ff32`).
**4A.6 (calibrator re-fit) done locally and `in_review`.** Next is 4A.7
(decision-layer eval + bootstrap CI — the formal Phase 4 bar), then 4A.8
close. **All remaining work is local — no pod needed.**

origin/main is at **125ff32**. The 4A.6 commit below is local, NOT pushed.

## 4A.6 result

Re-fit the 3X.11 calibrator (recipe unchanged) on the hybrid val predictions
(4A.4 gaussian + 4A.5 ensemble disagreement): **T=1.147, ens_scale=438.4,
held-out test coverage 0.9181** (within ±2 pp of 0.90), error↔uncertainty
corr +0.46. `target_col=iv_true` (matches 3X.11/3X.12 so 4A.7 reliability is
comparable to 3X.12). 4 tests green. Calibrator JSON gitignored (regenerable).

## Data-access recovery (resolved)

The prior GPU pod was terminated before I pulled the hybrid prediction CSVs
(I'd only pulled manifests — a miss). A fresh volume-mounted pod
(213.173.110.22) was provided; the gaussian + ensemble val/test CSVs are now
**pulled to local** (gitignored, ~1.5 GB). **4A.7/4A.8 run fully locally;
the pod can be released.**

## Push readiness (4A.6 — designed zero-waiver, NOT pushed)

- **PMR (live):** `configs/` + `tests/`; progress_log + spec updated → PASS.
- **DEP:** 4A.6 dep 4A.4 + 4A.5 = `done` → PASS.
- **SCOPE:** only active spec 4A.6; touched files in its `file_scope`
  (`STATUS.md` added) → PASS. **No `WAIVE_*`.**

## Next concrete action

- **Verify gates standalone, stop before push for review.**
- After approval: push; promote 4A.6 → `done`.
- **4A.7** (local): apply the 4A6 calibrator to the hybrid test predictions,
  3X.12 thresholds held; compute coverage / hi-conf / abstain / flag
  metrics + the **paired bootstrap 95% CI on the per-query MAE delta
  (hybrid − rbf_pred) vs iv_clean** — decides whether the Phase 4 accuracy
  gain (gaussian 0.006006 / quantile 0.005906 vs RBF 0.006132) is
  statistically significant. Then **4A.8** close + ADR 0010 outcome.
