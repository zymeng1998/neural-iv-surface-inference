#!/usr/bin/env bash
# 3X.10 launcher: K=5 ANP point-head deep ensemble on OTM.
# Faithful restatement of 3B.5 on random40_noiselow_otm.
# Usage:  bash scripts/run_3x10_anp_ensemble.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
LOGDIR="artifacts/runs/3X10/logs"
mkdir -p "$LOGDIR"

echo "===== [3X.10] START ensemble $(date -u +%FT%TZ) ====="
"$PY" -u scripts/run_3x10_anp_ensemble.py \
  --config configs/conditional_3X10_anp_ensemble_otm.yaml \
  2>&1 | tee "${LOGDIR}/ensemble.log"
echo "===== [3X.10] END   ensemble $(date -u +%FT%TZ) ====="
