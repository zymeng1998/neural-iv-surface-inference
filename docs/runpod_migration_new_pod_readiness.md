# RunPod New Network Volume — Migration Readiness Report

## 1. Check Metadata

- **Check date/time (local):** 2026-05-19T23:40:22-0400
- **Check date/time (pod UTC):** 2026-05-20T03:40:21+0000
- **Performed by:** Automated readiness verification over SSH alias `runpod-iv-new`
- **Scope:** Verification, documentation, and safe setup checks only. No destructive
  operations, no GPU training, no data modification.

## 2. New Pod SSH Information

| Field | Value |
|---|---|
| SSH alias | `runpod-iv-new` |
| Host / IP | 213.173.102.225 |
| Port | 15963 |
| User | root |
| Local private key | `~/.ssh/id_ed25519_runpod` |
| Project path | `/workspace/Neural-IV-Surface-inference` |
| Pod hostname (observed) | `0019e3f632c4` |
| Volume mount (observed) | `/workspace` on `mfs#euro.runpod.net:9421` |

Local `~/.ssh/config` was backed up to `~/.ssh/config.backup.20260519_225204`
before the `runpod-iv-new` block was appended (no existing block was overwritten —
the alias did not previously exist).

## 3. Verification Summary

| Item | Result |
|---|---|
| SSH alias connectivity | PASS — `hostname` = `0019e3f632c4` |
| `/workspace` mount | PASS — 2.3P filesystem, mounted on euro.runpod.net |
| `/workspace` total size | ~15 GB used (project = 12 GB, `.cache` = 3.0 GB) |
| Repo branch | `main` |
| Repo git status | Clean — "nothing to commit, working tree clean" |
| Repo remote | `git@github.com:zymeng1998/Neural-IV-Surface-inference.git` |
| `artifacts/results/baseline_results.csv` | PRESENT (4 lines: header + train/val/test, `interp_rbf` only) |
| `artifacts/results/interp_sweep_sampled_test.csv` | PRESENT (3 noise variants: low/med/high) |
| Parquet count (`data_raw` + `data_processed`) | 33 files |
| MLP checkpoint | PRESENT — `artifacts/checkpoints/best_mlp.pt` |
| Python | `/usr/bin/python` 3.11.10 |
| pip | 24.2 (`/usr/local/bin/pip`) |
| Dependency spec | `requirements.txt` present (numpy, pandas, pyarrow, matplotlib, torch, PyYAML, pytest); no `pyproject.toml`, no `environment.yml` |
| GitHub SSH | **NOT READY** — `ssh -T git@github.com` → Permission denied (publickey) |
| `git fetch` (dry-run) | **FAILED** — could not read from remote (publickey) |

### Backup / evidence artifacts found

- `/workspace/Neural-IV-Surface-inference/artifacts/results/baseline_results.csv`
- `/workspace/merge_safe_backup_20260404_233051/artifacts/results/baseline_results.csv`
- `/workspace/run_artifacts_backup_20260404_233156/artifacts/results/baseline_results_random40_noisemed_interp.csv`
- `/workspace/merge_safe_backup_20260404_233051/docs/phase1_result_memo.md`
- `/workspace/phase1_baseline_noisemed.log`

### Artifact integrity (committed vs backup)

- Committed `baseline_results.csv` and `merge_safe_backup` copy are **byte-identical**
  (md5 `b25dde5555bdb653987e1edfa6cc24d7`).
- `run_artifacts_backup` CSV (`baseline_results_random40_noisemed_interp.csv`) holds the
  same `interp_rbf` train/val/test values, just under a different filename.
- **All three CSVs contain only `interp_rbf` rows. No `mlp` row exists anywhere on the volume.**

## 4. Cyberduck Manual Settings (SFTP)

| Field | Value |
|---|---|
| Protocol | SFTP |
| Server | 213.173.102.225 |
| Port | 15963 |
| Username | root |
| Private key | `/Users/mengziyue/.ssh/id_ed25519_runpod` |
| Remote path | `/workspace/Neural-IV-Surface-inference` |

## 5. Cursor Manual Settings (Remote-SSH)

- Remote-SSH host: `runpod-iv-new`
- Open folder: `/workspace/Neural-IV-Surface-inference`

## 6. GitHub SSH — Action Required

GitHub SSH auth on the new pod is **not yet configured**:

- The GitHub key pair exists at `/workspace/.ssh/id_ed25519_github(.pub)`.
- It is **not** copied into `/root/.ssh/`, and `/root/.ssh/config` has no `github.com` block.
- `/workspace/setup_ssh.sh` exists and, on inspection, would safely:
  copy the key into `~/.ssh/`, `chmod 600/644`, and append a `github.com` host block.
- It was **not run automatically** (per safety rules). Run it manually on the pod, then
  re-test `ssh -T git@github.com`. If auth still fails afterward, the public key
  (`id_ed25519_github.pub`) likely needs to be registered on the GitHub account/repo.

## 7. Safety Note

The old Pod may be stopped/terminated, **but the old network volume must NOT be deleted**
until Phase 1 closeout repair confirms there are no missing local artifacts — specifically
until the missing `mlp` baseline row is regenerated and committed (see closeout notes below).

## 8. Phase 1 Closeout — Findings

1. **New pod is ready as the primary working environment** for everything except GitHub
   push/pull (see item 6). Mount, repo, data, checkpoint, and Python env all verified.
2. **Missing artifact:** the documented `mlp` baseline result is absent from
   `baseline_results.csv` (and from every backup). The experiment journal entry
   (2026-04-03) records `mlp` test overall MAE ≈ 0.0967 vs `interp_rbf` ≈ 0.0687, but no
   `mlp` row survives. The trained checkpoint `artifacts/checkpoints/best_mlp.pt` is
   present, so the model exists — only its evaluation row was lost/overwritten.
3. **No value mismatches** between committed and backup `interp_rbf` artifacts (identical).
   The only discrepancy is documentation-vs-artifact: journal claims an `mlp` row that the
   CSV does not contain.

### Recommended Phase 1 closeout repair commands (run on pod, lightweight)

```bash
ssh runpod-iv-new
cd /workspace/Neural-IV-Surface-inference

# 1. Configure GitHub SSH (safe; copies key + appends github config block)
bash /workspace/setup_ssh.sh
ssh -T git@github.com           # expect: "Hi zymeng1998! ..."

# 2. Sync repo (after GitHub SSH works)
git fetch origin && git pull --ff-only

# 3. Inspect run_baseline.py to confirm it evaluates the existing MLP checkpoint
#    and appends an `mlp` row (avoid retraining if checkpoint is reusable)
sed -n '1,200p' scripts/run_baseline.py

# 4. Regenerate the missing mlp baseline row (CPU eval if possible; avoid heavy GPU train)
#    Re-run on the SAME benchmark used in the journal: spy_phase1_random40_noiselow
python scripts/run_baseline.py   # confirm flags first from step 3

# 5. Verify both models now present
cut -d, -f1 artifacts/results/baseline_results.csv | sort -u   # expect interp_rbf + mlp

# 6. Update PMR docs (progress_log, experiment_journal) then commit per repo policy
```

> Do not delete the old volume until step 5 confirms the `mlp` row is restored.
