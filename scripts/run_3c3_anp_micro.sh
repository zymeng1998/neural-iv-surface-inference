#!/usr/bin/env bash
# 3C.3 sequential launcher: ANP cross-attention on OTM with the micro_v1
# feature set, all three heads. Faithful restatement of 3X.9 with the only
# modelling change being data.feature_set: micro_v1 (ADR 0008). Sequential on
# a single GPU. Each run redirects to its own log.
#
# Usage:  bash scripts/run_3c3_anp_micro.sh
#   PY=/workspace/venv-2e2/bin/python bash scripts/run_3c3_anp_micro.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
LOGDIR="artifacts/runs/3C3/logs"
mkdir -p "$LOGDIR"

run() {
  local name="$1"; shift
  echo "===== [3C.3] START ${name} $(date -u +%FT%TZ) ====="
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "===== [3C.3] END   ${name} $(date -u +%FT%TZ) ====="
}

run gaussian "$PY" -u scripts/run_3c3_anp_micro.py \
  --config configs/conditional_3C3_anp_micro_gaussian_otm.yaml
run quantile "$PY" -u scripts/run_3c3_anp_micro.py \
  --config configs/conditional_3C3_anp_micro_quantile_otm.yaml
run point    "$PY" -u scripts/run_3c3_anp_micro.py \
  --config configs/conditional_3C3_anp_micro_point_otm.yaml

echo "===== [3C.3] ALL DONE $(date -u +%FT%TZ) ====="
