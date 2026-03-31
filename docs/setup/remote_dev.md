# Remote Development Setup

## Purpose

This document records the sanitized remote development workflow for the Neural IV Surface Inference project. It is intended to preserve reproducibility and team memory without exposing sensitive operational details.

## Infrastructure Summary

- Remote provider: RunPod
- Current purpose: Phase 1 development workstation
- Persistent workspace path: `/workspace`
- Project repository path on remote: `/workspace/neural-iv-surface-inference`
- GitHub SSH authentication on remote: configured
- Primary coding interface: Cursor Remote-SSH
- Secondary file transfer / browsing interface: Cyberduck

## Current Workflow

The current workflow is centered around a remote development machine with persistent storage. Git operations are performed against the repository cloned in the remote workspace. Cursor Remote-SSH is the primary interface for editing and development, while Cyberduck is used mainly for file browsing, upload, and download.

## Access Methods

- SSH is used for shell access to the remote development environment
- Cursor Remote-SSH is used for editing, coding, and project navigation
- Cyberduck is used for file transfer and quick directory inspection

This tracked document intentionally omits literal connection details.

## Git Workflow

- The canonical working repository is the clone under `/workspace/neural-iv-surface-inference`
- GitHub SSH authentication is already functioning on the remote machine
- Clone / pull / push operations do not require repeated password entry
- Project documentation and code should be committed regularly in small, meaningful increments

## Recommended Usage Pattern

1. Use Cursor Remote-SSH as the main coding environment
2. Use Cyberduck only when file transfer or quick visual inspection is more convenient
3. Run Git commands from the repository working tree
4. Update the progress log after each meaningful milestone
5. Keep sensitive operational details out of tracked documentation

## Security Boundary

The following must never be committed:
- IP addresses
- ports
- usernames
- SSH config details
- private key paths
- tokens, secrets, or credentials

Those details belong only in a local-only private runbook.

## Next Setup Tasks

- Create project subdirectories for code, notebooks, data, figures, and artifacts
- Add a minimal GPU smoke test
- Begin the first-pass data pipeline scaffold
