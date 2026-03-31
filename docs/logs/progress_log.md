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
