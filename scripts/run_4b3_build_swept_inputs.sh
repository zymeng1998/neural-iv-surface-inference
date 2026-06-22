#!/usr/bin/env bash
# Story 4B.3 (remote CPU) — build the Phase 4B swept eval inputs (RBF re-fit per
# rung) on the OTM benchmark and emit the audit bundle.
#
# Usage:
#   bash scripts/run_4b3_build_swept_inputs.sh [PYTHON] [SPLIT]
# PYTHON defaults to the RunPod venv interpreter; SPLIT defaults to test.
#
# Re-run safe: regenerates artifacts/runs/4B3/{manifest.json,rbf_sweep_stats.csv}
# and the Pod handoff parquet. Logs to /workspace/4B3_build.log.
set -euo pipefail

PY="${1:-/workspace/venv-2e2/bin/python}"
SPLIT="${2:-test}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

INPUT="data_processed/spy/benchmarks/spy_phase1_random40_noiselow_otm_residual.parquet"
PARQUET_OUT="artifacts/runs/4B3/swept_rbf_${SPLIT}.parquet"
LOG="/workspace/4B3_build.log"

echo "[4b3] python=$PY split=$SPLIT input=$INPUT" | tee "$LOG"
PYTHONPATH=src "$PY" scripts/run_4b3_build_swept_inputs.py \
    --input "$INPUT" \
    --split "$SPLIT" \
    --parquet-out "$PARQUET_OUT" \
    --verbose 2>&1 | tee -a "$LOG"
