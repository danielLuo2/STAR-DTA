#!/usr/bin/env bash
# Table 2: STAR-DTA trained from scratch on the pancreatic-cancer ChEMBL set, three seeds.
# Usage: bash scripts/run_pancreatic.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

for seed in ${SEEDS:-42 2026 3407}; do
  echo "==== [$(date)] pancreatic seed=$seed"
  python -u train_pancreatic.py --data-dir data/pancreatic --device "${DEVICE:-cuda:0}" --seed "$seed" \
    > "logs/pancreatic_seed${seed}.log" 2>&1
  grep '^\[TEST\]' "logs/pancreatic_seed${seed}.log"
done
