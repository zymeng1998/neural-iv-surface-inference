# Progress Log

## Logging Convention

Each entry should capture:
- what was completed
- important notes or observations
- unresolved items
- immediate next actions

---

## 2026-03-31

### Completed

- Provisioned the remote RunPod development environment for Phase 1 work
- Verified that the persistent workspace is mounted and writable
- Confirmed the remote project path under `/workspace/neural-iv-surface-inference`
- Established shell access to the remote environment
- Verified basic remote environment usability
- Configured GitHub SSH authentication for remote Git operations
- Confirmed passwordless Git clone capability on remote
- Connected Cyberduck for file browsing and transfer
- Connected Cursor via Remote-SSH for primary development work
- Verified Python 3.11, PyTorch 2.4.1+cu124, CUDA 12.4, and RTX A5000 GPU on remote
- Installed core scientific packages (NumPy, SciPy, Matplotlib, Pandas, Scikit-learn, PyArrow, Fastparquet, Boto3, Seaborn)
- Configured Git user identity on remote
- Established the initial documentation plan for the repository

### Notes

- Cursor Remote-SSH is the primary development interface
- Cyberduck is mainly a convenience tool for file transfer and quick inspection
- Tracked documentation must stay sanitized
- Sensitive operational details belong only in a local-only private runbook
- Network volume permissions do not support chmod; SSH keys must be copied (not symlinked) to `~/.ssh` after pod restart
- A setup script at `/workspace/setup_ssh.sh` handles post-restart SSH key restoration

### Open Items

- Initialize project subdirectories on remote
- Add a minimal GPU smoke test
- Define the first-pass data and experimentation layout
- Set up AWS S3 cold backup

### Next Actions

- Create the remaining project folder scaffold on remote
- Run a GPU smoke test from within a Python script
- Begin SPY EOD options data acquisition planning
- Continue with Phase 1 project initialization

---

## 2026-03-31 (update 2)

### Completed

- Added minimal ML project scaffold: `src/`, `scripts/`, `configs/`, `tests/`, `artifacts/`, `data/`, `notebooks/`
- Created Python package `src/neural_iv_surface_inference/` with submodules for data, features, models, training, and utils
- Moved `smoke_test.py` from repo root to `scripts/smoke_test.py`
- Added placeholder scripts: `prepare_data.py`, `run_baseline.py`
- Added YAML configs: `data.yaml`, `baseline.yaml`
- Added placeholder tests: `test_smoke.py`, `test_data_pipeline.py`
- Added `.gitkeep` files to preserve empty directory structure
- Updated `.gitignore` with data/artifact ignore rules and `.gitkeep` exceptions
- Updated `README.md` with repo structure section

### Notes

- All Python stubs are import-safe placeholders with TODO comments
- `data/samples/` is not globally ignored so small committed sample files are possible
- `baseline_mlp.py` contains a minimal class skeleton ready for implementation

### Next Actions

- Begin SPY EOD options data acquisition
- Implement data loading and cleaning logic
- Run GPU smoke test on remote to verify scaffold

---

## 2026-03-31 (update 3)

### Completed

- Formally froze Phase 1 scope via ADR 0002: SPY-only, EOD-only, fixed task definition
- Defined required deliverables before scope can expand

### Next Actions

- Choose primary SPY EOD option chain data source
- Acquire first raw data and load into `data/raw/`
- Begin data cleaning and preprocessing implementation

---

## 2026-03-31 (update 4)

### Completed

- Built complete Phase 1 SPY data pipeline (3 scripts + config + notebook + cleaning doc)
- Data source: Philipp Dubach historical options dataset (static.philippdubach.com)
- `src/data/01_ingest_spy_github_dataset.py` — downloads SPY options + underlying Parquet
- `src/data/02_inspect_spy_schema.py` — schema inspection + quality report
- `src/data/03_build_spy_surface_table.py` — join, clean, derive, produce conservative + strict subsets
- `src/data/config.py` — centralized paths, URLs, thresholds
- `notebooks/01_spy_data_firstlook.ipynb` — EDA notebook
- `docs/data_assumptions_and_cleaning.md` — cleaning rules and assumptions documented
- Updated `.gitignore` for pipeline data directories

### Notes

- Primary source uses the broader repo (richer schema: includes mark, last, bid_size, ask_size, in_the_money)
- Spot = unadjusted close (NOT adjusted_close) for moneyness computation
- Conservative dataset keeps quality flags; strict subset enforces tighter thresholds
- Pipeline designed to run end-to-end from command line on RunPod

### Next Actions

- Run the pipeline on RunPod: ingest → inspect → build
- Review schema report and EDA plots
- Begin sparse masking and baseline implementation

---

## 2026-03-31 (update 5)

### Completed

- Ran SPY data ingestion on RunPod: 24,681,665 option rows + 4,529 underlying rows downloaded successfully
- Ran schema inspection on RunPod: all 21 expected columns present, minimal quality issues confirmed
- Step 3 (build processed table) failed due to OOM — eager pandas workflow on the full 24.7M-row dataset exceeded Pod memory
- Added retrospective note: `docs/retrospectives/0001_spy_step3_oom_and_pipeline_fix.md`

### Notes

- The failure was a pipeline design mistake, not a data-quality mistake
- Schema validation succeeded but execution scalability was not tested before full-scale run
- Multiple concurrent SSH sessions compounded the OOM condition

### Next Actions

- Refactor `src/data/03_build_spy_surface_table.py` to a year-by-year partitioned workflow
- Test the refactored build on a single year before running full scale
- Produce two outputs: broad processed table (conservative drops + flags) and strict modeling subset

---

## 2026-03-31T22:28:00-04:00

### Completed

- Refactored `src/data/03_build_spy_surface_table.py` to a year-by-year partitioned workflow as planned in retrospective 0001
- Each year is read, joined, cleaned, and written to disk independently before proceeding to the next
- Added memory-safe concatenation and strict subset generation (`fa82e71`)
- Step 3 OOM issue from retrospective 0001 is now resolved

### Notes

- The Yahoo Finance direct download URL for underlying data was blocked; added a `yfinance` fallback (`b0cfde5`, `e6b31de`)
- The fix produces two outputs as planned: `spy_surface_points.parquet` (broad) and `spy_surface_points_strict.parquet` (strict subset)

### Next Actions

- Run the full refactored pipeline end-to-end on RunPod
- Review output quality and row counts
- Begin task construction (sparse masking, noise injection, evaluation splits)

---

## 2026-04-01T05:30:00+00:00

### Completed

- Created `docs/roadmaps/phase1_structural_roadmap.md` with end-goal task decomposition, subtask-to-method matrix, and decision rules
- Added Mermaid dependency flowchart (`docs/roadmaps/flowchart_deps.mmd`) and gantt chart (`docs/roadmaps/gantt_chart.mmd`) with rendered PNGs (`cec12b1`)

### Notes

- Roadmap marks S1.1 through S4.3 as completed or in progress, with S3.3 (vendor-style reference) as the open Phase 1 gap
- Phase 2–4 subtasks (latent inference, uncertainty, structure constraints, EBM) are marked as planned

### Next Actions

- Close the S3.3 vendor-style reference gap
- Prepare for first full evaluation run

---

## 2026-04-02T01:35:00-04:00

### Completed

- Installed repo-local Project Memory Reviewer system (`docs/agent_bootstrap/`)
- Created project memory registry, update policy, runbook, and change event taxonomy
- Created experiment journal (`docs/experiments/experiment_journal.md`)
- Created retrospective guidance and template (`docs/retrospectives/README.md`, `docs/retrospectives/_template.md`)
- Created review packet builder script (`scripts/build_project_memory_review_packet.py`)
- Ran first baseline project memory review

### Notes

- The reviewer system extends the existing documentation structure without replacing it
- Timestamp policy (ISO 8601 with timezone) is enforced for all new project-memory files; existing files will be updated when next materially touched
- No existing files were modified except this progress log

### Next Actions

- Commit the bootstrap system and this progress log update
- Use `python scripts/build_project_memory_review_packet.py` before future reviewer runs
- Begin routine experiment tracking in `docs/experiments/experiment_journal.md` when training runs start

---

## 2026-04-02T02:00:00-04:00

### Completed

- Created data lineage document (`docs/data/data_lineage.md`) tracing the full raw → processed → modeling-ready pipeline for Phase 1 SPY data
- Built pre-push PMR gate (`scripts/pmr_prepush_gate.py`) that blocks pushes when evidence-source files change without corresponding PMR doc updates
- Added `.pre-commit-config.yaml` for hook installation via pre-commit framework
- Installed the pre-push hook in the current clone
- Added idempotence hardening to the PMR system (`reviewer_state.json`, packet fingerprinting)
- Created operator manual and quickstart guide in `docs/private/`

### Notes

- The pre-push hook is clone-local — must be installed in any clone from which `git push` is run
- Remediation path when push is blocked: run PMR reviewer via Claude/Cursor on the same remote repo, commit doc updates, retry push
- Bypass: `git push --no-verify` or `SKIP_PMR_GATE=1 git push`
- Data lineage doc explicitly calls out open questions: whether full pipeline was re-run after OOM fix, where S2.1–S2.3 masking logic lives, config path mismatch between `configs/data.yaml` and `src/data/config.py`

### Next Actions

- Install the pre-push hook on the RunPod clone if pushes happen there
- Resolve the data lineage open questions when the pipeline is next run
- Begin routine experiment tracking when training runs start

---

## 2026-04-02T15:00:00-04:00

### Completed

- Implemented task construction modules S2.1–S2.3 as standalone, tested modules:
  - **S2.1 Masking** (`src/neural_iv_surface_inference/data/masking.py`): 7 strategies — random uniform, drop short/long/random maturity, drop wings/ATM, realistic liquidity-weighted. High-level `apply_mask()` API adds an `observed` boolean column.
  - **S2.2 Noise** (`src/neural_iv_surface_inference/data/noise.py`): 4 regimes (none/low/med/high) with i.i.d. Gaussian and heteroscedastic modes. Noise applied only to observed points; `iv_clean` always preserved for evaluation.
  - **S2.3 Splits & versioning** (`src/neural_iv_surface_inference/data/splits.py`): chronological train/val/test split (70/15/15), canonical benchmark naming (e.g. `spy_phase1_random20_noise0`), Parquet save/load with embedded provenance metadata.
- Built orchestration script `src/data/04_build_benchmark_tasks.py` — config-driven, reads `configs/benchmark_tasks.yaml`, produces one Parquet per variant
- Defined 11 benchmark variants in `configs/benchmark_tasks.yaml` covering core grid (random masking × noise levels), realistic liquidity masking, and structured stress tests
- Updated `src/neural_iv_surface_inference/data/__init__.py` with public API exports
- Wrote 32 unit tests (`tests/test_task_construction.py`) — all passing
- Reviewed `src/data/03_build_spy_surface_table.py` for RunPod re-run readiness — script is solid post-OOM refactor, no changes needed

### Notes

- Raw data only exists on RunPod (`data_raw/spy/` is empty locally); Step 3 must be re-run on RunPod before Step 4 can execute
- Data lineage open question resolved: S2.1–S2.3 masking/noise/splits now live in `src/neural_iv_surface_inference/data/` as proper modules
- Config path mismatch persists: `configs/data.yaml` is legacy; `src/data/config.py` is source of truth for pipeline paths; `configs/benchmark_tasks.yaml` is source of truth for task construction

### Next Actions

- Run Step 3 on RunPod to produce `spy_surface_points_strict.parquet`
- Run Step 4 on RunPod: `python src/data/04_build_benchmark_tasks.py`
- Begin baseline implementation (S3.1 interpolation, S3.2 neural MLP)

---

## 2026-04-02T17:00:00-04:00

### Completed

- Implemented S3.1 interpolation baseline (`src/neural_iv_surface_inference/models/interpolation.py`):
  - Per-date interpolation using scipy RBF (thin-plate spline), linear, cubic, or nearest-neighbor
  - Automatic fallback: RBF → linear → nearest for edge cases; mean IV when < 3 observed points
  - `run_interpolation_baseline()` runs across all dates in a benchmark dataset

- Implemented S3.2 masked MLP baseline (`src/neural_iv_surface_inference/models/baseline_mlp.py`):
  - Global model trained across all dates on observed points only
  - Architecture: Linear → LayerNorm → SiLU × n_layers, softplus output (ensures positive IV)
  - Kaiming weight initialization
  - Configurable hidden_dim, n_layers, dropout

- Implemented loss functions (`src/neural_iv_surface_inference/models/losses.py`):
  - MaskedMSELoss, MaskedMAELoss, CombinedLoss (weighted MSE+MAE)
  - All losses accept a boolean mask to restrict computation to observed points

- Implemented PyTorch Dataset and loaders (`src/neural_iv_surface_inference/data/loaders.py`):
  - IVSurfaceDataset wraps benchmark Parquet output from Step 4
  - Each sample: (log_moneyness, tau) features, iv_clean target, observed flag, date index
  - `load_benchmark_splits()` factory returns train/val/test DataLoaders + raw DataFrames

- Implemented training loop (`src/neural_iv_surface_inference/training/train.py`):
  - Full training with AdamW, ReduceLROnPlateau scheduler, early stopping, checkpointing
  - Per-epoch logging of train loss, val observed MSE, val unobserved MAE

- Implemented evaluation framework (`src/neural_iv_surface_inference/training/eval.py`):
  - Core metrics: MAE, RMSE, MAPE
  - Split by observed vs unobserved points
  - Regional diagnostics: 3 maturity buckets (short/medium/long), 5 moneyness buckets (deep ITM → deep OTM)
  - `metrics_to_dataframe()` for tabular comparison, `print_evaluation()` for console output

- Wired up entry point (`scripts/run_baseline.py`):
  - Config-driven via `configs/baseline.yaml`
  - Runs interpolation + MLP, evaluates both, saves comparison CSV to `artifacts/results/`
  - CLI flags: `--interp-only`, `--mlp-only`, `--benchmark <path>`

- Updated `configs/baseline.yaml` with full settings (data, interpolation, MLP, paths)
- Updated `utils/io.py` with YAML config loader
- Wrote 25 unit tests (`tests/test_baselines.py`) — all passing; 59/59 total tests pass

### Notes

- Both baselines are tested on synthetic data locally. Real-data execution requires Steps 3+4 on RunPod first.
- The MLP is a global model (trained across all dates), not per-date. This is intentional — per-date fitting has too few points for meaningful neural training.
- The interpolation baseline is per-date by design (it doesn't learn, just interpolates).
- S4.3 (result artifact: plots, summary memo) is not yet implemented — `run_baseline.py` only produces a CSV comparison table so far.

### Next Actions

- Run Steps 3 + 4 on RunPod to produce benchmark datasets
- Run `python scripts/run_baseline.py` on RunPod to get first real results
- Implement S4.3: visualization plots and Phase 1 result memo

---

## 2026-04-03T00:00:00-04:00

### Completed

- Rewrote `src/data/04_build_benchmark_tasks.py` for memory-safe streaming after OOM on RunPod (21M-row strict dataset caused ~11GB peak memory with eager loading)
  - Now streams via PyArrow `iter_batches` (500K rows/batch) instead of loading full dataset
  - Precomputes global date→split map by scanning partition files (reads only date column)
  - Per-batch: mask → noise → split assignment → ParquetWriter, with `gc.collect()` after each batch
  - Fallback: if no partition files exist, reads only the date column from strict file
- Added `compute_date_split_map()` to `src/neural_iv_surface_inference/data/splits.py` — scans partition files to build date→split mapping without loading full data
- Optimized `src/neural_iv_surface_inference/data/loaders.py`:
  - Column pruning on load (only reads 8 needed columns instead of all)
  - `del df; gc.collect()` after splitting into train/val/test
- Changed `run_interpolation_baseline()` in `models/interpolation.py` to return `np.ndarray` instead of a full DataFrame copy
- Updated `scripts/run_baseline.py` to assign `iv_pred` from the returned array
- Updated 2 tests in `tests/test_baselines.py` for new return type
- All 59 tests pass

### Notes

- Root cause of Step 4 OOM: the original script loaded 21M rows at once, then for each of 11 variants, called `.copy()` 3 times (mask, noise, split) creating ~11GB peak per variant
- Masking and noise functions are stateless array operations — they work on any chunk size with no cross-chunk dependencies
- The only global dependency is the date→split mapping, which is precomputed once (~4,500 dates, trivial memory)
- The seed for masking/noise now includes a batch offset (`seed + batch_num`) to ensure different random draws per batch while remaining deterministic

### Next Actions

- Commit and push changes
- Pull on RunPod and re-run Step 4 with memory monitoring
- Run Step 5 (baselines) on RunPod
- Verify benchmark parquet row counts match expectations

---

## 2026-04-03T12:10:00-04:00

### Completed

- Verified benchmark parquet row counts on RunPod against strict source output.
- Confirmed strict source row count: `21,386,789` (`data_processed/spy/spy_surface_points_strict.parquet`).
- Confirmed benchmark outputs: `11/11` expected files in `data_processed/spy/benchmarks/`.
- Confirmed each benchmark file has `21,386,789` rows (no row-loss/duplication across variants).
- Confirmed split counts are internally consistent for each file:
  - train: `10,458,763`
  - val: `5,617,807`
  - test: `5,310,219`
  - split sum equals total rows in every benchmark file.

### Notes

- Verification was run directly on RunPod using Parquet metadata and split-column counts.
- This resolves the previously listed action item to verify benchmark parquet row counts.

### Next Actions

- Run Step 5 on RunPod: `python scripts/run_baseline.py` on the verified benchmark datasets.
- Save and review first real-data baseline outputs in `artifacts/results/` and `artifacts/checkpoints/`.
- Implement S4.3 deliverables: visualization plots and a Phase 1 summary memo.
