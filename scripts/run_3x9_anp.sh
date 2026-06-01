#!/usr/bin/env bash
# 3X.9 sequential launcher: ANP cross-attention on OTM, all three heads.
# Faithful restatement of 3B.4 on random40_noiselow_otm. Sequential on a
# single GPU. Each run redirects to its own log.
# Usage:  bash scripts/run_3x9_anp.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
LOGDIR="artifacts/runs/3X9/logs"
mkdir -p "$LOGDIR"

run() {
  local name="$1"; shift
  echo "===== [3X.9] START ${name} $(date -u +%FT%TZ) ====="
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "===== [3X.9] END   ${name} $(date -u +%FT%TZ) ====="
}

run gaussian      "$PY" -u scripts/run_3x9_anp.py \
  --config configs/conditional_3X9_anp_gaussian_otm.yaml
run quantile      "$PY" -u scripts/run_3x9_anp.py \
  --config configs/conditional_3X9_anp_quantile_otm.yaml
run point_control "$PY" -u scripts/run_3x9_anp.py \
  --config configs/conditional_3X9_anp_point_control_otm.yaml

echo "===== [3X.9] ALL DONE $(date -u +%FT%TZ) ====="
