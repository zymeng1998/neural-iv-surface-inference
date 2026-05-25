#!/usr/bin/env bash
# 2D.8 orchestration — trains K=5 ensemble + scores + emits artifacts.
#
# Usage:  bash scripts/run_conditional_2D8.sh
# Log:    artifacts/runs/2D8/run.log

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/runs/2D8
log="artifacts/runs/2D8/run.log"
echo "[2D.8-orch] $(date -u +%FT%TZ)  starting ensemble training"
python3 scripts/run_2d8_ensemble.py --config configs/conditional_2D8_ensemble.yaml \
  > "${log}" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[2D.8-orch] FAILED rc=${rc}  see ${log}"
  tail -40 "${log}"
  exit $rc
fi
echo "[2D.8-orch] $(date -u +%FT%TZ)  done"
tail -5 "${log}"
