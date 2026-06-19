# STATUS — Phase 4 4A.3 (residual dataset) built on OTM; next = 4A.4 (GPU, Pod-gated)

**Updated:** 2026-06-19
**Branch:** main
**Mode:** remote CPU work done; now local docs. No GPU spend yet.

## Where things stand

Phase 4 (epic `4A`) in progress. 4A.1 `done`; 4A.2 `done` (origin
`c64b5e9`); **4A.3 (residual-target dataset) built on the RunPod CPU pod and
`in_review`.** Next is 4A.4 (remote GPU residual-hybrid training) — **gated
on an operator Pod go-ahead.**

origin/main is at **c64b5e9**. The 4A.3 commit below is local, NOT pushed.

## 4A.3 result

Full OTM residual targets `r = iv_clean − rbf_pred`: **4622 dates,
10,531,499 rows, 0 non-finite.** Per-split mean|residual|: train 0.006659 /
**val 0.006151 / test 0.006132** — val/test **match the 3X.6 RBF MAE
byte-for-byte** (`rbf_pred` is the same RBF baseline; target validated).

- Residual parquet (237 MB) on the **persistent `/workspace` volume**
  (gitignored; not pulled). Committed: `artifacts/runs/4A3/manifest.json` +
  `residual_stats.csv`.
- **⇒ The CPU pod can be terminated now** (parquet on persistent volume,
  stats local).

## Pod frictions handled (carry forward to 4A.4)

- Connect with **`~/.ssh/id_ed25519_runpod`** (not the default key).
- Pod **can't `git fetch`** (SSH origin, no GitHub key) → **scp** code to the
  pod, or set up a deploy key.
- Container **cgroup-capped ~3.7 GB** → memory-safe reads (the builder's
  `--columns` subset). 4A.4 training must respect this if it runs on a
  similarly capped pod.

## Push readiness (4A.3 — designed zero-waiver, NOT pushed)

- **PMR (live):** `scripts/` + `artifacts/runs/4A3/` touched; journal +
  lineage + progress_log updated → PASS.
- **DEP:** 4A.3 dep 4A.2 = `done` → PASS.
- **SCOPE:** only active spec 4A.3; all touched files in its `file_scope`
  (`STATUS.md` added) → PASS. **No `WAIVE_*`.**

## Next concrete action

- **Verify gates standalone, stop before push for review.**
- After approval: push (clean env); promote 4A.3 → `done`.
- **4A.4** (remote GPU) on operator go-ahead: train the residual hybrid on
  the OTM residual parquet (ANP-residual default, `target_mode=residual`),
  3 heads; report summed σ̂ MAE vs RBF 0.00613 + 3X.9. Ensure the GPU pod
  mounts the same `/workspace` volume.
