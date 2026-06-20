# STATUS — Phase 4 4A.4 trained: residual hybrid BEATS RBF; next = 4A.5

**Updated:** 2026-06-20
**Branch:** main
**Mode:** remote GPU work done; now local docs. Pod terminable.

## Where things stand

Phase 4 (epic `4A`) in progress. 4A.1/4A.2/4A.3 `done` (origin `7b896e4`).
**4A.4 (residual-hybrid training) done on the GPU pod and `in_review` — and
the hybrid beats RBF** on the clean OTM substrate. Next is 4A.5 (K=5 residual
point-head ensemble) — **gated on a Pod go-ahead**.

origin/main is at **7b896e4**. The 4A.4 commit below is local, NOT pushed.

## 4A.4 result (headline)

ANP-residual hybrid `σ̂ = rbf_pred + f_θ`, hybrid test MAE vs `iv_clean`
(RBF floor 0.006132):

| head | hybrid | Δ vs RBF |
|---|---:|---:|
| gaussian | 0.006006 | −0.000126 |
| **quantile** | **0.005906** | **−0.000225 (~3.7 %)** |
| point | 0.006138 | +0.000006 (ties) |

**First neural-based predictor in the project to beat RBF on clean OTM**
(gaussian + quantile below the floor; point ties; all ≪ 3X.9). qmono ok.
**Caveat:** point estimates, small margins — **statistical significance is
4A.7's bootstrap-CI job** (the formal bar). ADR 0010 backbone fork resolved:
ANP-residual confirmed.

## Pod / artifacts

- GPU pod (RTX 4000 Ada) shares the `/workspace` volume with the 4A.3 CPU
  pod (residual parquet already present). Connect with `id_ed25519_runpod`;
  pod can't `git fetch` → scp'd loader + 4A.4 code.
- 3 manifests pulled local (committed); curves + val/test prediction CSVs on
  the persistent volume for 4A.5/4A.6/4A.7. **GPU pod terminable now.**

## Push readiness (4A.4 — designed zero-waiver, NOT pushed)

- **PMR (live):** `configs/`+`scripts/`+`artifacts/runs/4A4/`; journal +
  progress_log updated → PASS.
- **DEP:** 4A.4 dep 4A.3 = `done` → PASS.
- **SCOPE:** only active spec 4A.4; all touched files in its `file_scope`
  (`STATUS.md` added) → PASS. **No `WAIVE_*`.**

## Next concrete action

- **Verify gates standalone, stop before push for review** (4A.4 ships new
  configs + a training driver — operator reviews before push).
- After approval: push; promote 4A.4 → `done`.
- **4A.5** (remote GPU, Pod-gated): K=5 residual point-head ensemble
  (seeds 101..505), mirror 3X.10. Then 4A.6 → 4A.7 (the bar) → 4A.8 close.
