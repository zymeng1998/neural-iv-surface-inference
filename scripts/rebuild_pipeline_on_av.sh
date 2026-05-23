#!/usr/bin/env bash
# Phase 2C.7 — post-ingest pipeline rebuild on the AV dataset.
#
# Idempotent-ish: if Dubach backup already exists, we skip the rename step.
# Stops at first error.
#
# Run from the repo root:
#   bash scripts/rebuild_pipeline_on_av.sh

set -euo pipefail

ROOT=/workspace/Neural-IV-Surface-inference
cd "$ROOT"

LOG_DIR=artifacts/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/rebuild_pipeline_${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "=== 2C.7 pipeline rebuild starting ==="

# --- Step 0: sanity ---
AV_RAW=data_raw/spy/spy_options_av.parquet
if [ ! -f "$AV_RAW" ]; then
  log "FATAL: AV options Parquet not found at $AV_RAW"
  exit 2
fi
AV_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet(\"$AV_RAW\")))")
log "AV options rows: $AV_ROWS"

# --- Step 1: backup Dubach raw + processed ---
BACKUP=data_raw/spy_dubach_pre_av_${STAMP}
if [ -f data_raw/spy/spy_options.parquet ] && [ ! -d "$BACKUP" ]; then
  log "moving Dubach raw -> $BACKUP/"
  mkdir -p "$BACKUP"
  mv data_raw/spy/spy_options.parquet "$BACKUP/"
  # underlying: replaced by AV pull (yfinance dynamic-end), so move old aside
  if [ -f data_raw/spy/spy_underlying.parquet ]; then
    # spy_underlying.parquet was overwritten by --with-underlying in the AV pull
    # If size differs from a freshly-pulled file, it is the AV version. Just leave it.
    log "underlying parquet present (size=$(stat -c %s data_raw/spy/spy_underlying.parquet))"
  fi
fi

PROC_BACKUP=data_processed/spy_dubach_pre_av_${STAMP}
if [ -f data_processed/spy/spy_surface_points.parquet ] && [ ! -d "$PROC_BACKUP" ]; then
  log "moving Dubach processed -> $PROC_BACKUP/"
  mkdir -p "$PROC_BACKUP"
  [ -f data_processed/spy/spy_surface_points.parquet ] && mv data_processed/spy/spy_surface_points.parquet "$PROC_BACKUP/"
  [ -f data_processed/spy/spy_surface_points_strict.parquet ] && mv data_processed/spy/spy_surface_points_strict.parquet "$PROC_BACKUP/"
  [ -d data_processed/spy/partitions ] && mv data_processed/spy/partitions "$PROC_BACKUP/"
  [ -d data_processed/spy/benchmarks ] && mv data_processed/spy/benchmarks "$PROC_BACKUP/"
  log "Dubach processed backed up to $PROC_BACKUP"
fi

# --- Step 2: AV parquet -> canonical name ---
if [ ! -f data_raw/spy/spy_options.parquet ]; then
  log "moving AV pull -> canonical data_raw/spy/spy_options.parquet"
  mv "$AV_RAW" data_raw/spy/spy_options.parquet
fi
log "active options parquet size: $(stat -c %s data_raw/spy/spy_options.parquet) bytes"

# --- Step 3: step 02 schema inspection ---
log "--- step 02: schema inspection ---"
python3 src/data/02_inspect_spy_schema.py >> "$MASTER_LOG" 2>&1
log "step 02 done"

# --- Step 4: step 03 surface build ---
log "--- step 03: surface build (year-by-year) ---"
python3 src/data/03_build_spy_surface_table.py >> "$MASTER_LOG" 2>&1
log "step 03 done"
ls -la data_processed/spy/spy_surface_points*.parquet | tee -a "$MASTER_LOG"

# --- Step 5: step 04 benchmark variants ---
log "--- step 04: benchmark tasks (all 11 variants) ---"
python3 src/data/04_build_benchmark_tasks.py >> "$MASTER_LOG" 2>&1
log "step 04 done"
ls -la data_processed/spy/benchmarks/ | tee -a "$MASTER_LOG"

# --- Step 6: summary ---
log "=== 2C.7 rebuild SUCCESS ==="
log "Outputs:"
du -sh data_raw/spy/* | tee -a "$MASTER_LOG"
du -sh data_processed/spy/* | tee -a "$MASTER_LOG"
log "Total persistent data: $(du -sh data_raw/spy data_processed/spy 2>/dev/null | awk \"{s+=\\$1}END{print s\\\"M ish\\\"}\")"
log "Backups preserved at: $BACKUP and $PROC_BACKUP"
log "Master log: $MASTER_LOG"
