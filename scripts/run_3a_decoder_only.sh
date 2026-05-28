#!/usr/bin/env bash
# 3A.3 orchestration — decoder-only retrain on the frozen 2D.7 encoder.
# Runs the Fourier variant then the raw control, sequentially.
#
# Usage:  bash scripts/run_3a_decoder_only.sh
# Logs:   artifacts/runs/3A/<variant>/run.log

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/runs/3A/fourier artifacts/runs/3A/raw

run_one() {
  local variant="$1"
  local cfg="$2"
  local logdir="artifacts/runs/3A/${variant}"
  local log="${logdir}/run.log"
  echo "[3A.3-orch] $(date -u +%FT%TZ)  starting variant=${variant}  cfg=${cfg}"
  python3 scripts/run_3a_decoder_only.py --config "${cfg}" > "${log}" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[3A.3-orch] FAILED variant=${variant} rc=${rc}  see ${log}"
    tail -40 "${log}"
    return $rc
  fi
  echo "[3A.3-orch] $(date -u +%FT%TZ)  done variant=${variant}"
  tail -5 "${log}"
}

run_one fourier configs/conditional_3A3_fourier.yaml || exit $?
run_one raw     configs/conditional_3A3_raw.yaml     || exit $?

echo "[3A.3-orch] both variants complete"
