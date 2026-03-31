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

## Documentation Map

- `docs/setup/remote_dev.md` — sanitized remote development workflow and environment notes
- `docs/setup/private_runbook_template.md` — template for local-only private ops notes
- `docs/logs/progress_log.md` — chronological project progress log
- `docs/decisions/0001_remote_dev_stack.md` — architecture / workflow decision record

## Immediate Next Steps

- Initialize project directories for code, data, notebooks, figures, and artifacts
- Add a minimal GPU smoke test
- Define the first-pass data pipeline layout
- Begin SPY EOD options data acquisition

## Security Note

Tracked documentation intentionally excludes sensitive operational details such as IPs, ports, usernames, SSH config details, private key paths, and tokens. Those details belong only in a local-only private runbook that must not be committed.
