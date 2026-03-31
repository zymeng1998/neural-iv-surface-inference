# ADR 0001: Remote Development Stack for Phase 1

## Status

Accepted

## Date

2026-03-31

## Context

Phase 1 of the project requires a GPU-capable, persistent, remotely accessible development environment with a workflow that supports coding, Git operations, and file transfer without relying on the local machine for compute.

## Decision

Use a RunPod-based remote workstation as the primary development environment.
Use Cursor Remote-SSH as the primary coding interface.
Use Cyberduck as a secondary file transfer / directory browsing tool.
Use GitHub SSH authentication for Git operations on the remote machine.

## Rationale

This combination provides:
- persistent remote storage
- GPU availability
- a practical coding workflow
- clean Git integration
- a lightweight file transfer path when needed

## Consequences

Positive:
- Development is centered in a reproducible remote environment
- Git operations are streamlined on remote
- The workflow is practical for iterative ML experimentation

Trade-offs:
- Documentation must clearly separate safe project notes from sensitive operational details
- The local machine is no longer the primary source of truth for the working environment
