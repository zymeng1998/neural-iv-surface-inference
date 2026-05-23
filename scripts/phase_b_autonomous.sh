#!/usr/bin/env bash
# Phase B fully-autonomous runner — fire once, walk away.
#
# Layers of safety so the Pod cannot accidentally bill overnight:
#   1) Chain: wait for AV pull PID -> rebuild -> rerun -> terminate.
#      Each stage's exit code gates the next via `set -euo pipefail`.
#   2) trap EXIT: regardless of how the chain exits (success, failure,
#      kill signal), a final `runpodctl remove pod` is fired.
#   3) Wall-clock guard: a parallel background killer terminates the Pod
#      no later than MAX_HOURS from start (default 8).
#
# Run once on the Pod (nohup'd):
#   nohup bash scripts/phase_b_autonomous.sh > artifacts/logs/phase_b_auto.log 2>&1 &
#
# Env overrides:
#   POD_ID          — RunPod Pod ID (default s3d42nmizlbo1d)
#   AV_PID          — process to wait for (default: detect via pgrep)
#   MAX_HOURS       — wall-clock guard (default 8)
#   ABORT_WINDOW_SEC — last-chance human abort window (default 0; pure
#                     autonomous mode). Set >0 to enable the sentinel.

set -uo pipefail

POD_ID="${POD_ID:-s3d42nmizlbo1d}"
MAX_HOURS="${MAX_HOURS:-8}"
ABORT_WINDOW_SEC="${ABORT_WINDOW_SEC:-0}"
ABORT_SENTINEL="${ABORT_SENTINEL:-/tmp/ABORT_TERMINATE}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_STAGES="${SKIP_STAGES:-0}"

ROOT=/workspace/Neural-IV-Surface-inference
cd "$ROOT"

LOG_DIR=artifacts/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
AUTO_LOG="$LOG_DIR/phase_b_auto_${STAMP}.log"
SENT_DIR=artifacts/logs/sentinels
mkdir -p "$SENT_DIR"
START_EPOCH=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$AUTO_LOG"; }

# ---------------------------------------------------------------------------
# Always-terminate trap. Fires on EVERY exit path (success or failure).
# ---------------------------------------------------------------------------
TERMINATE_FIRED=0
final_terminate() {
  if [ "$TERMINATE_FIRED" = "1" ]; then
    return 0
  fi
  TERMINATE_FIRED=1
  log "=== final terminate path  pod_id=$POD_ID ==="
  # Kill the wall-clock guard so it doesn't keep the script alive after exit
  if [ -n "${GUARD_PID:-}" ]; then
    kill "$GUARD_PID" 2>/dev/null || true
  fi

  if [ "$ABORT_WINDOW_SEC" -gt 0 ]; then
    rm -f "$ABORT_SENTINEL"
    log "abort window: ${ABORT_WINDOW_SEC} s (touch $ABORT_SENTINEL to cancel)"
    for i in $(seq 1 "$ABORT_WINDOW_SEC"); do
      if [ -f "$ABORT_SENTINEL" ]; then
        log "ABORTED by sentinel. terminate skipped. Pod stays alive."
        return 0
      fi
      sleep 1
    done
  fi

  # Verify auth once before firing
  if runpodctl get pod "$POD_ID" > /dev/null 2>&1; then
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] Would call: runpodctl remove pod $POD_ID"
    else
      log "firing terminate: runpodctl remove pod $POD_ID"
      runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$AUTO_LOG" || \
        log "WARN: terminate command returned non-zero; check console."
    fi
  else
    log "WARN: runpodctl get pod failed; not firing terminate (auth or Pod ID issue)."
  fi
  log "=== end of final_terminate ==="
}
trap final_terminate EXIT
trap 'log "caught SIGINT"; exit 130' INT
trap 'log "caught SIGTERM"; exit 143' TERM

# ---------------------------------------------------------------------------
# Wall-clock guard. Backgrounded, kills the Pod after MAX_HOURS regardless.
# ---------------------------------------------------------------------------
wall_clock_guard() {
  local seconds=$(( MAX_HOURS * 3600 ))
  sleep "$seconds"
  log "WALL-CLOCK GUARD: ${MAX_HOURS}h elapsed; force-terminating Pod"
  if runpodctl get pod "$POD_ID" > /dev/null 2>&1; then
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] WALL-CLOCK GUARD would call: runpodctl remove pod $POD_ID"
    else
      runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$AUTO_LOG"
    fi
  fi
  # Also kill the parent chain so trap doesn't double-fire (defensive)
  kill -TERM "$$" 2>/dev/null || true
}
wall_clock_guard &
GUARD_PID=$!
log "wall-clock guard armed: PID=$GUARD_PID  max_hours=$MAX_HOURS"

# ---------------------------------------------------------------------------
# Stage 0: wait for the in-flight AV pull to finish.
# ---------------------------------------------------------------------------
if [ -z "${AV_PID:-}" ]; then
  AV_PID=$(pgrep -f "alpha_vantage.*--start" | head -1 || true)
fi
log "=== Phase B autonomous start  pod_id=$POD_ID  max_hours=$MAX_HOURS ==="
log "av pull pid: ${AV_PID:-<none>}"

if [ -n "${AV_PID:-}" ] && ps -p "$AV_PID" > /dev/null 2>&1; then
  log "waiting for AV pull PID $AV_PID to exit ..."
  while ps -p "$AV_PID" > /dev/null 2>&1; do
    sleep 30
    elapsed=$(( $(date +%s) - START_EPOCH ))
    log "still waiting on AV pull ... elapsed=${elapsed}s"
  done
  log "AV pull PID gone."
else
  log "no AV pull PID found alive; assuming pull is already done."
fi

# Sanity-check the AV pull's output before kicking the rebuild.
AV_RAW=data_raw/spy/spy_options_av.parquet
if [ "$SKIP_STAGES" = "1" ]; then
  log "SKIP_STAGES=1 -> skipping AV parquet existence check"
elif [ ! -f "$AV_RAW" ]; then
  log "FATAL: $AV_RAW not found after AV pull. Aborting chain; terminate will still fire via trap."
  exit 2
else
  AV_SIZE=$(stat -c %s "$AV_RAW")
  log "AV options parquet present: $AV_SIZE bytes"
fi

# ---------------------------------------------------------------------------
# Stage 1: rebuild (2C.7).
# ---------------------------------------------------------------------------
log "--- chain stage 1: rebuild_pipeline_on_av.sh ---"
if [ "$SKIP_STAGES" = "1" ]; then
  log "SKIP_STAGES=1 -> skipping rebuild"
else
  set -e
  bash scripts/rebuild_pipeline_on_av.sh
  set +e
fi
log "stage 1 done."
touch "$SENT_DIR/stage1_rebuild_done.${STAMP}"

# ---------------------------------------------------------------------------
# Stage 2: 2C.8 + conditional rerun (2C.4-R + 2C.5-R).
# ---------------------------------------------------------------------------
log "--- chain stage 2: rerun_phase1_and_conditional_on_av.sh ---"
if [ "$SKIP_STAGES" = "1" ]; then
  log "SKIP_STAGES=1 -> skipping rerun"
else
  set -e
  bash scripts/rerun_phase1_and_conditional_on_av.sh
  set +e
fi
log "stage 2 done."
touch "$SENT_DIR/stage2_rerun_done.${STAMP}"

# ---------------------------------------------------------------------------
# Stage 3: terminate via trap on exit.
# ---------------------------------------------------------------------------
log "=== Phase B chain complete. exit -> trap fires terminate. ==="
log "Total elapsed: $(( $(date +%s) - START_EPOCH )) s"
exit 0
