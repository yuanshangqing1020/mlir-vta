#!/usr/bin/env bash
# Stage vta-translate outputs for fsim_single_layer (suffix-aware).
# Usage: stage_fsim_single_layer.sh <out_dir> <layer_name> <compiler_output_dir>
set -euo pipefail
OUT=$1
NAME=$2
CO=${3:-/mnt/c/MLIR-VTA/standalone-vta/compiler_output}

for f in layers_name.csv \
         "instructions${NAME}.bin" "uop${NAME}.bin" \
         "input${NAME}.bin" "weight${NAME}.bin" \
         "accumulator${NAME}.bin" \
         "metadata${NAME}.csv" "memory_addresses${NAME}.csv"; do
  if [[ -f "$OUT/$f" ]]; then
    cp "$OUT/$f" "$CO/$f"
  fi
done
: > "$CO/add_accumulator${NAME}.bin"
if [[ -f "$OUT/out_init.bin" ]]; then
  cp "$OUT/out_init.bin" "$CO/out_init.bin"
fi
