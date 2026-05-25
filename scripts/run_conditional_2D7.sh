#!/usr/bin/env bash
# 2D.7 orchestration — runs all three head variants sequentially.
# Point-control first (regression guard); aborts if it fails.
#
# Usage:  bash scripts/run_conditional_2D7.sh
# Logs:   artifacts/runs/2D7/<head>/run.log

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/runs/2D7/point artifacts/runs/2D7/gaussian artifacts/runs/2D7/quantile

run_one() {
  local head="$1"
  local cfg="$2"
  local logdir="artifacts/runs/2D7/${head}"
  local log="${logdir}/run.log"
  echo "[2D.7-orch] $(date -u +%FT%TZ)  starting head=${head}  cfg=${cfg}"
  python3 scripts/run_2d7_single.py --config "${cfg}" > "${log}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[2D.7-orch] FAILED head=${head} rc=${rc}  see ${log}"
    tail -30 "${log}"
    return $rc
  fi
  echo "[2D.7-orch] $(date -u +%FT%TZ)  done head=${head}"
  tail -3 "${log}"
}

run_one point     configs/conditional_2D7_point_control.yaml || exit $?
run_one gaussian  configs/conditional_2D7_gaussian.yaml      || exit $?
run_one quantile  configs/conditional_2D7_quantile.yaml      || exit $?

echo "[2D.7-orch] all three runs complete"
