#!/usr/bin/env bash
# Chain: 2C.7 rebuild  -->  2C.8 + 2C.4-R + 2C.5-R rerun  -->  Pod terminate.
#
# Designed to run on the RunPod remote. The actual terminate is gated by:
#   - all preceding stages exiting 0
#   - a 5-minute abort window (touch /tmp/ABORT_TERMINATE to cancel)
#   - the --dry-run-terminate flag (echoes the API call instead of firing)
#
# Run from the repo root on the Pod:
#   bash scripts/terminate_after_phase_b.sh                  # real terminate
#   bash scripts/terminate_after_phase_b.sh --dry-run-terminate
#
# Pre-flight: `runpodctl config --apiKey <rpa_...>` must already be set up.

set -euo pipefail

POD_ID="${POD_ID:-s3d42nmizlbo1d}"
ABORT_WINDOW_SEC="${ABORT_WINDOW_SEC:-300}"
ABORT_SENTINEL="${ABORT_SENTINEL:-/tmp/ABORT_TERMINATE}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run-terminate) DRY_RUN=1 ;;
    *) ;;
  esac
done

ROOT=/workspace/Neural-IV-Surface-inference
cd "$ROOT"

LOG_DIR=artifacts/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG="$LOG_DIR/chain_terminate_${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CHAIN_LOG"; }

log "=== chain start  pod_id=$POD_ID  dry_run=$DRY_RUN ==="

# --- Stage 1: 2C.7 rebuild ---
log "--- chain stage 1: rebuild_pipeline_on_av.sh ---"
if [ "$DRY_RUN" = "1" ] && [ -n "${SKIP_STAGES:-}" ]; then
  log "DRY-RUN MODE: skipping real rebuild (SKIP_STAGES set)"
else
  bash scripts/rebuild_pipeline_on_av.sh 2>&1 | tee -a "$CHAIN_LOG"
fi

# --- Stage 2: 2C.8 + 2C.4-R + 2C.5-R ---
log "--- chain stage 2: rerun_phase1_and_conditional_on_av.sh ---"
if [ "$DRY_RUN" = "1" ] && [ -n "${SKIP_STAGES:-}" ]; then
  log "DRY-RUN MODE: skipping real rerun (SKIP_STAGES set)"
else
  bash scripts/rerun_phase1_and_conditional_on_av.sh 2>&1 | tee -a "$CHAIN_LOG"
fi

# --- Stage 3: terminate Pod (gated) ---
log "--- chain stage 3: terminate ---"
log "All Phase B stages succeeded."

# Verify auth + Pod still reachable BEFORE the abort window so we fail fast.
log "verifying runpodctl can see Pod $POD_ID ..."
if ! runpodctl get pod "$POD_ID" > /dev/null 2>&1; then
  log "WARN: runpodctl get pod $POD_ID failed; auth or Pod ID may be wrong."
  log "Skipping terminate. Investigate manually."
  exit 4
fi
log "runpodctl get pod $POD_ID OK."

log "Auto-terminate in $ABORT_WINDOW_SEC seconds."
log "To abort: touch $ABORT_SENTINEL (any shell on the Pod)."
rm -f "$ABORT_SENTINEL"

for i in $(seq 1 "$ABORT_WINDOW_SEC"); do
  if [ -f "$ABORT_SENTINEL" ]; then
    log "ABORTED by sentinel ($ABORT_SENTINEL exists). Terminate skipped."
    exit 0
  fi
  # heartbeat every 60s
  if [ $((i % 60)) -eq 0 ]; then
    log "abort window: ${i}/${ABORT_WINDOW_SEC} s elapsed"
  fi
  sleep 1
done

if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] Would call: runpodctl remove pod $POD_ID"
  log "[dry-run] No API call fired. Chain complete."
  exit 0
fi

log "Firing terminate: runpodctl remove pod $POD_ID"
runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$CHAIN_LOG"
log "terminate call returned; the Pod will go away shortly. SSH sessions may drop."
