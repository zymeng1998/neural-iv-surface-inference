#!/usr/bin/env bash
# 4A.4 — RBF-prior residual hybrid on OTM: train gaussian → quantile → point
# sequentially, each logged. Run from the repo root on the GPU pod.
#   bash scripts/run_4a4_residual.sh [PYTHON]
# PYTHON defaults to the venv interpreter.
set -euo pipefail

PY="${1:-/workspace/venv-2e2/bin/python}"
mkdir -p artifacts/runs/4A4

for head in gaussian quantile point; do
  cfg="configs/conditional_4A4_residual_${head}_otm.yaml"
  log="artifacts/runs/4A4/${head}/train.log"
  mkdir -p "artifacts/runs/4A4/${head}"
  echo "=== 4A.4 ${head} :: $(date -u +%FT%TZ) ==="
  "$PY" scripts/run_4a4_residual.py --config "$cfg" 2>&1 | tee "$log"
done
echo "=== 4A.4 all heads done :: $(date -u +%FT%TZ) ==="
