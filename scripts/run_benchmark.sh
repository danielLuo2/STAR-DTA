#!/usr/bin/env bash
# Table 1: STAR-DTA on the standard Davis and KIBA split, three seeds each.
# Usage: bash scripts/run_benchmark.sh [davis|kiba|all] [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

TARGET="${1:-all}"
shift || true
case "$TARGET" in
  all) DATASETS=(davis kiba) ;;
  davis|kiba) DATASETS=("$TARGET") ;;
  *) echo "unknown dataset: $TARGET" >&2; exit 2 ;;
esac

for ds in "${DATASETS[@]}"; do
  for seed in ${SEEDS:-42 2026 3407}; do
    echo "==== [$(date)] $ds seed=$seed"
    python -u train.py --dataset "$ds" --data-root "${DATA_ROOT:-data}" --device "${DEVICE:-cuda:0}" \
      --seed "$seed" "$@" > "logs/${ds}_seed${seed}.log" 2>&1
    grep '^BEST' "logs/${ds}_seed${seed}.log"
  done
done
