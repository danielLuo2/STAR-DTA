#!/usr/bin/env bash
# Table 3: Davis ablations, three seeds each, all with the same settings as the full model.
# Usage: bash scripts/run_ablation.sh [variant ...]
#   variants: full gin_only no_multiscale_residue no_drug_res_attention no_protein_atom_attention no_y_standardization
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/ablation

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(full gin_only no_multiscale_residue no_drug_res_attention no_protein_atom_attention no_y_standardization)
fi

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    full) FLAGS=() ;;
    gin_only) FLAGS=(--gin-only) ;;
    no_multiscale_residue) FLAGS=(--no-multiscale-residue) ;;
    no_drug_res_attention) FLAGS=(--no-drug-res-attention) ;;
    no_protein_atom_attention) FLAGS=(--no-protein-smiles-attention) ;;
    no_y_standardization) FLAGS=(--no-standardize-y) ;;
    *) echo "unknown variant: $variant" >&2; exit 2 ;;
  esac
  for seed in ${SEEDS:-42 2026 3407}; do
    echo "==== [$(date)] $variant seed=$seed"
    python -u train.py --dataset davis --data-root "${DATA_ROOT:-data}" --device "${DEVICE:-cuda:0}" \
      --run-name "ablation_${variant}" --seed "$seed" "${FLAGS[@]}" \
      > "logs/ablation/${variant}_seed${seed}.log" 2>&1
    grep '^BEST' "logs/ablation/${variant}_seed${seed}.log"
  done
done
