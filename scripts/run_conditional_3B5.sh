#!/usr/bin/env bash
# 3B.5 orchestration — trains K=5 ANP ensemble + scores + emits artifacts.
#
# Usage:  bash scripts/run_conditional_3B5.sh
# Log:    artifacts/runs/3B5/run.log

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/runs/3B5
log="artifacts/runs/3B5/run.log"
echo "[3B.5-orch] $(date -u +%FT%TZ)  starting ANP ensemble training"
python3 scripts/run_3b5_ensemble.py --config configs/conditional_3B5_anp_ensemble.yaml \
  > "${log}" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[3B.5-orch] FAILED rc=${rc}  see ${log}"
  tail -40 "${log}"
  exit $rc
fi
echo "[3B.5-orch] $(date -u +%FT%TZ)  done"
tail -5 "${log}"
