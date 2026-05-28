#!/usr/bin/env bash
# 3B.4 orchestration — runs all three ANP head variants sequentially.
# Point-control first (regression guard); aborts if it fails.
#
# Usage:  bash scripts/run_conditional_3B4.sh
# Logs:   artifacts/runs/3B4/<head>/run.log

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/runs/3B4/point_control artifacts/runs/3B4/gaussian artifacts/runs/3B4/quantile

run_one() {
  local head="$1"
  local cfg="$2"
  local logdir="artifacts/runs/3B4/${head}"
  local log="${logdir}/run.log"
  echo "[3B.4-orch] $(date -u +%FT%TZ)  starting head=${head}  cfg=${cfg}"
  python3 scripts/run_3b4_single.py --config "${cfg}" > "${log}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[3B.4-orch] FAILED head=${head} rc=${rc}  see ${log}"
    tail -30 "${log}"
    return $rc
  fi
  echo "[3B.4-orch] $(date -u +%FT%TZ)  done head=${head}"
  tail -3 "${log}"
}

run_one point_control configs/conditional_3B4_anp_point_control.yaml || exit $?
run_one gaussian      configs/conditional_3B4_anp_gaussian.yaml      || exit $?
run_one quantile      configs/conditional_3B4_anp_quantile.yaml      || exit $?

echo "[3B.4-orch] all three runs complete"
