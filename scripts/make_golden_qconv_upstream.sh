#!/usr/bin/env bash
# Golden from upstream NN+VTA compiler (unpadded qlinearconv logical dims).
set -euo pipefail
ROOT=/mnt/c/MLIR-VTA
SVTA=$ROOT/standalone-vta
MLIRVTA=$ROOT/mlir-vta
GD="$MLIRVTA/test/golden/qlinearconv_upstream"
CO=$SVTA/compiler_output
PY=/home/john/miniconda3/envs/standalone-vta/bin/python3
ONNX=$SVTA/examples/onnx/qlinearconv_debug.onnx

mkdir -p "$GD"
cd "$SVTA/examples"
make clean >/dev/null 2>&1 || true
$PY ../src/compiler/nn_compiler/vta_backend.py True False ../config/vta_config.json "$ONNX" >/dev/null
$PY ../src/compiler/vta_compiler/main_vta_compiler.py False True False \
  ../config/vta_config.json ../compiler_output/QLinearConv1.json >/dev/null

: > "$CO/add_accumulator.bin"
rm -f "$SVTA/log_output/fsim_report.txt"
make fsim_compile_single_layer >/dev/null 2>&1
make fsim_single_layer >/dev/null 2>&1

NAME=QLinearConv1
cp "$CO/instructions${NAME}.bin" "$GD/instructions.bin" 2>/dev/null || cp "$CO/instructions.bin" "$GD/instructions.bin"
cp "$CO/uop${NAME}.bin" "$GD/uop.bin" 2>/dev/null || cp "$CO/uop.bin" "$GD/uop.bin"
cp "$CO/input${NAME}.bin" "$GD/input.bin" 2>/dev/null || cp "$CO/input.bin" "$GD/input.bin"
cp "$CO/weight${NAME}.bin" "$GD/weight.bin" 2>/dev/null || cp "$CO/weight.bin" "$GD/weight.bin"
cp "$CO/accumulator${NAME}.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/accumulator.bin" "$GD/accumulator.bin" 2>/dev/null || \
  cp "$CO/${NAME}accumulator_1x3.bin" "$GD/accumulator.bin" 2>/dev/null || true
cp "$CO/accumulator${NAME}.bin" "$GD/accumulator_raw.bin" 2>/dev/null || \
  cp "$CO/${NAME}accumulator_1x3.bin" "$GD/accumulator_raw.bin" 2>/dev/null || true
cp "$CO/metadata${NAME}.csv" "$GD/metadata.csv" 2>/dev/null || cp "$CO/metadata.csv" "$GD/metadata.csv"
cp "$CO/memory_addresses${NAME}.csv" "$GD/memory_addresses.csv" 2>/dev/null || cp "$CO/memory_addresses.csv" "$GD/memory_addresses.csv"
cp "$CO/layers_name.csv" "$GD/layers_name.csv"
cp "$CO/dependency.csv" "$GD/dependency.csv" 2>/dev/null || true
sed -n '/Final result/,/^} $/p' "$SVTA/log_output/fsim_report.txt" > "$GD/fsim_result.txt"
echo "Upstream golden -> $GD"
ls "$GD"
