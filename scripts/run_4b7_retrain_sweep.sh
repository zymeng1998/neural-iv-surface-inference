#!/usr/bin/env bash
# Story 4B.7 (remote GPU) — fair per-(regime,rung) retrain of the RBF-prior hybrid.
#
# Usage:
#   bash scripts/run_4b7_retrain_sweep.sh [PYTHON]
# PYTHON defaults to the RunPod venv interpreter.
#
# Resumable (per-cell JSON skip). Checkpoints + per-cell test preds stay on the
# Pod (gitignored); committed bundle is artifacts/runs/4B7/{manifest.json,
# fair_retrain_stats.csv, cells/*.json}. Logs to /workspace/4B7_run.log.
set -euo pipefail

PY="${1:-/workspace/venv-2e2/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOG="/workspace/4B7_run.log"
echo "[4b7] python=$PY" | tee "$LOG"
PYTHONPATH=src "$PY" scripts/run_4b7_retrain_sweep.py \
    --config configs/conditional_4B7_retrain_gaussian_otm.yaml \
    --ckpt-root /workspace/4B7_checkpoints \
    --pred-dir artifacts/runs/4B7/preds 2>&1 | tee -a "$LOG"
