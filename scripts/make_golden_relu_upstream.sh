#!/usr/bin/env bash
# Golden for 16x16 ReLU (MAX_IMM 0) via upstream VTA compiler.
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/relu_16x16"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
JSON=$SVTA/examples/vta_ir/relu_16x16.json

mkdir -p "$GD"

# Fixed accumulator for reproducible golden (mixed signs → relu changes negatives).
$PY - <<'PY'
import numpy as np
from pathlib import Path
acc = np.array([
    [-5, 3, -1, 7, 0, -9, 2, 4, -2, 1, 6, -8, 3, -4, 5, 0],
    [1] * 16,
    [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    *[[(-1)**(i+j) * (i + j) for j in range(16)] for i in range(3, 16)],
], dtype=np.int32)
Path("/mnt/c/MLIR-VTA/standalone-vta/compiler_output/accumulator_16x16.bin").write_bytes(
    acc.tobytes()
)
print("accumulator_16x16.bin written", acc.shape)
PY

cp "$JSON" "$CO/Relu1.json"
cd "$SVTA/examples"
$PY ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
  ../config/vta_config.json ../compiler_output/Relu1.json >/dev/null

: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

NAME=Relu1
cp "$CO/instructions${NAME}.bin" "$GD/instructions.bin" 2>/dev/null || cp "$CO/instructions.bin" "$GD/instructions.bin"
cp "$CO/uop${NAME}.bin" "$GD/uop.bin" 2>/dev/null || cp "$CO/uop.bin" "$GD/uop.bin"
cp "$CO/accumulator${NAME}.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/accumulator.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/accumulator_16x16.bin" "$GD/accumulator_raw.bin"
cp "$CO/accumulator_16x16.bin" "$GD/accumulator_raw.bin"
cp "$CO/metadata${NAME}.csv" "$GD/metadata.csv" 2>/dev/null || cp "$CO/metadata.csv" "$GD/metadata.csv"
cp "$CO/memory_addresses${NAME}.csv" "$GD/memory_addresses.csv" 2>/dev/null || \
  cp "$CO/memory_addresses.csv" "$GD/memory_addresses.csv"
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"
echo "Upstream relu golden -> $GD"
ls "$GD"
