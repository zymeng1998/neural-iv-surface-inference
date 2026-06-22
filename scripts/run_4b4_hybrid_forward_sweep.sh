#!/usr/bin/env bash
# Story 4B.4 (remote) — run the 4A hybrid checkpoint forward across the 4B
# sparsity sweep and emit per-rung sigma_hat predictions + audit.
#
# Usage:
#   bash scripts/run_4b4_hybrid_forward_sweep.sh [PYTHON] [SPLIT]
# PYTHON defaults to the RunPod venv interpreter; SPLIT defaults to test.
#
# Logs to /workspace/4B4_run.log. Predictions stream to a Pod parquet
# (gitignored); the committed bundle is artifacts/runs/4B4/{manifest.json,
# hybrid_sweep_stats.csv}.
set -euo pipefail

PY="${1:-/workspace/venv-2e2/bin/python}"
SPLIT="${2:-test}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

INPUT="data_processed/spy/benchmarks/spy_phase1_random40_noiselow_otm_residual.parquet"
GAUSS="artifacts/runs/4A4/gaussian/checkpoints/best_conditional.pt"
ENS_DIR="artifacts/runs/4A5/checkpoints/ensemble"
PARQUET_OUT="artifacts/runs/4B4/swept_hybrid_${SPLIT}.parquet"
LOG="/workspace/4B4_run.log"

echo "[4b4] python=$PY split=$SPLIT" | tee "$LOG"
PYTHONPATH=src "$PY" scripts/run_4b4_hybrid_forward_sweep.py \
    --input "$INPUT" \
    --split "$SPLIT" \
    --gaussian-ckpt "$GAUSS" \
    --ensemble-dir "$ENS_DIR" \
    --parquet-out "$PARQUET_OUT" \
    --verbose 2>&1 | tee -a "$LOG"
