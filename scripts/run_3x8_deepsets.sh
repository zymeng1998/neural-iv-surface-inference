#!/usr/bin/env bash
# 3X.8 sequential launcher: DeepSets/mean-pool conditional on OTM.
#
#   single heads  : run_3x8_deepsets.py over {gaussian, quantile, point}
#   K=5 ensemble  : run_2d8_ensemble.py (verbatim 2D.8 recipe) on the OTM
#                   ensemble config — manifest story reads "2D.8" by design
#                   (faithful restatement; bundle lives under runs/3X8/ensemble).
#
# Runs strictly sequentially (single GPU). Each run redirects to its own log.
# Usage:  bash scripts/run_3x8_deepsets.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
# Interpreter: override with PY=... ; default to the Pod's complete env.
PY="${PY:-python3}"
LOGDIR="artifacts/runs/3X8/logs"
mkdir -p "$LOGDIR"

run() {
  local name="$1"; shift
  echo "===== [3X.8] START ${name} $(date -u +%FT%TZ) ====="
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "===== [3X.8] END   ${name} $(date -u +%FT%TZ) ====="
}

run single_gaussian "$PY" -u scripts/run_3x8_deepsets.py \
  --config configs/conditional_3X8_deepsets_gaussian_otm.yaml
run single_quantile "$PY" -u scripts/run_3x8_deepsets.py \
  --config configs/conditional_3X8_deepsets_quantile_otm.yaml
run single_point "$PY" -u scripts/run_3x8_deepsets.py \
  --config configs/conditional_3X8_deepsets_point_control_otm.yaml
run ensemble "$PY" -u scripts/run_2d8_ensemble.py \
  --config configs/conditional_3X8_deepsets_ensemble_otm.yaml

echo "===== [3X.8] ALL DONE $(date -u +%FT%TZ) ====="
