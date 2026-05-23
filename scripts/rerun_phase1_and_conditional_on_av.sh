#!/usr/bin/env bash
# Phase 2C.8 + 2C.4-R + 2C.5-R — remote-only rerun of Phase 1 baselines and
# the conditional model on the AV-sourced dataset.
#
# Runs sequentially on a single Pod, logs each stage to its own file under
# artifacts/logs/, and aborts on first error. Per CLAUDE.md, full output is
# in the log files; only summary numbers should be reported back.
#
# Prerequisite: 2C.7 must be done (data_processed/spy/benchmarks/ regenerated
# from the new AV options Parquet).
#
# Run from the repo root:
#   bash scripts/rerun_phase1_and_conditional_on_av.sh

set -euo pipefail

ROOT=/workspace/Neural-IV-Surface-inference
cd "$ROOT"

LOG_DIR=artifacts/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/av_rerun_master_${STAMP}.log"
TAG="av_rerun_${STAMP}"
BENCH="data_processed/spy/benchmarks/spy_phase1_random40_noiselow.parquet"
CKPT="artifacts/checkpoints/best_conditional.pt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "=== 2C.8 + 2C.4-R + 2C.5-R START  tag=$TAG ==="

if [ ! -f "$BENCH" ]; then
  log "FATAL: benchmark not found at $BENCH — run scripts/rebuild_pipeline_on_av.sh first."
  exit 2
fi
log "benchmark size: $(stat -c %s "$BENCH") bytes"

# --- Stage A: Phase 1 baselines (interp + MLP) ---
log "--- 2C.8 stage A: run_baseline.py (interp + MLP on random40_noiselow) ---"
PYTHONPATH=src python3 scripts/run_baseline.py --config configs/baseline.yaml \
  > "$LOG_DIR/baseline_${STAMP}.log" 2>&1
log "stage A done; tail:"
tail -10 "$LOG_DIR/baseline_${STAMP}.log" | tee -a "$MASTER_LOG"

# --- Stage B: W1 uncertainty eval (interp) ---
log "--- 2C.8 stage B: run_uncertainty_eval.py --predictor interp ---"
PYTHONPATH=src python3 scripts/run_uncertainty_eval.py \
    --benchmark "$BENCH" --predictor interp --method rbf \
    --tag "${TAG}_interp_rbf" --splits train,val,test \
    > "$LOG_DIR/w1_interp_${STAMP}.log" 2>&1
log "stage B done; metrics:"
tail -15 "$LOG_DIR/w1_interp_${STAMP}.log" | tee -a "$MASTER_LOG"

# --- Stage C: W2 structure diagnostics (interp) ---
log "--- 2C.8 stage C: run_structure_diagnostics.py --predictor interp ---"
PYTHONPATH=src python3 scripts/run_structure_diagnostics.py \
    --benchmark "$BENCH" --predictor interp --method rbf \
    --tag "${TAG}_interp_rbf" --splits train,val,test --max-dates 50 \
    > "$LOG_DIR/w2_interp_${STAMP}.log" 2>&1
log "stage C done; summary tail:"
tail -10 "$LOG_DIR/w2_interp_${STAMP}.log" | tee -a "$MASTER_LOG"

# --- Stage D: conditional training on real AV data (2C.4-R) ---
log "--- 2C.4-R stage D: run_conditional.py (full real-data training) ---"
PYTHONPATH=src python3 scripts/run_conditional.py \
    --config configs/conditional.yaml \
    --benchmark "$BENCH" \
    > "$LOG_DIR/cond_train_${STAMP}.log" 2>&1
log "stage D done; tail:"
tail -15 "$LOG_DIR/cond_train_${STAMP}.log" | tee -a "$MASTER_LOG"

if [ ! -f "$CKPT" ]; then
  log "FATAL: conditional checkpoint not found at $CKPT after training."
  exit 3
fi
log "checkpoint size: $(stat -c %s "$CKPT") bytes"

# --- Stage E: conditional W1 eval (2C.5-R) ---
log "--- 2C.5-R stage E: run_uncertainty_eval.py --predictor conditional ---"
PYTHONPATH=src python3 scripts/run_uncertainty_eval.py \
    --benchmark "$BENCH" --predictor conditional --checkpoint "$CKPT" \
    --tag "${TAG}_conditional" --splits train,val,test \
    > "$LOG_DIR/w1_cond_${STAMP}.log" 2>&1
log "stage E done; metrics:"
tail -15 "$LOG_DIR/w1_cond_${STAMP}.log" | tee -a "$MASTER_LOG"

# --- Stage F: conditional W2 eval (2C.5-R) ---
log "--- 2C.5-R stage F: run_structure_diagnostics.py --predictor conditional ---"
PYTHONPATH=src python3 scripts/run_structure_diagnostics.py \
    --benchmark "$BENCH" --predictor conditional --checkpoint "$CKPT" \
    --tag "${TAG}_conditional" --splits train,val,test --max-dates 50 \
    > "$LOG_DIR/w2_cond_${STAMP}.log" 2>&1
log "stage F done; summary tail:"
tail -10 "$LOG_DIR/w2_cond_${STAMP}.log" | tee -a "$MASTER_LOG"

# --- Summary ---
log "=== ALL STAGES COMPLETE  tag=$TAG ==="
log "Artifacts under artifacts/results/ with tag prefix '$TAG':"
ls -la artifacts/results/*${TAG}* 2>/dev/null | tee -a "$MASTER_LOG"
log "Master log: $MASTER_LOG"
