# Neural IV Surface Inference

## Overview

Neural IV Surface Inference is an ML × Finance project focused on recovering implied-volatility surfaces from sparse, noisy, and irregular option observations. The longer-term direction includes structured inference, uncertainty-aware modeling, and arbitrage-aware constraints. The current focus is Phase 1: establishing a reproducible remote development workflow and baseline experimentation stack.

## Current Phase

**Phase 1 — remote development environment and baseline pipeline setup**

## Current Status

Completed so far:
- RunPod-based remote development environment provisioned
- Persistent workspace verified and writable
- Remote GitHub SSH authentication configured
- Git clone on remote working without password prompts
- Cyberduck connection working
- Cursor Remote-SSH connection working
- Python, PyTorch, CUDA, and core scientific packages verified on remote
- Initial project documentation scaffold established
- Minimal ML project scaffold in place (src, scripts, configs, tests, artifacts)

## Documentation Map

- `docs/setup/remote_dev.md` — sanitized remote development workflow and environment notes
- `docs/setup/private_runbook_template.md` — template for local-only private ops notes
- `docs/logs/progress_log.md` — chronological project progress log
- `docs/decisions/0001_remote_dev_stack.md` — architecture / workflow decision record

## Repository Structure

```text
src/neural_iv_surface_inference/   Python package (data, features, models, training, utils)
scripts/                           Entry-point scripts (smoke test, data prep, training)
configs/                           YAML configuration files
notebooks/                         Jupyter notebooks
tests/                             Test suite
data/                              Data directories (raw, interim, processed, samples)
artifacts/                         Output artifacts (figures, tables, checkpoints)
docs/                              Project documentation
```

## Immediate Next Steps

- Define the first-pass data pipeline layout
- Begin SPY EOD options data acquisition
- Implement data loading and cleaning stubs

## Security Note

Tracked documentation intentionally excludes sensitive operational details such as IPs, ports, usernames, SSH config details, private key paths, and tokens. Those details belong only in a local-only private runbook that must not be committed.
